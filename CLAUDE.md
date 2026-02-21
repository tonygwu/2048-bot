# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

On a fresh clone, create the virtual environment and install dependencies.
Use PyPy by default for faster strategy/search performance:

```bash
pypy3.10 -m venv .venv
.venv/bin/pip install playwright pillow
.venv/bin/playwright install chromium
```

## Commands

```bash
# Run the bot (auto depth, adaptive to board state)
.venv/bin/python bot.py

# Common flag combinations
.venv/bin/python bot.py --headless --depth 4          # fast, headless
.venv/bin/python bot.py --headless --depth 5          # stronger (~5× slower per move)
.venv/bin/python bot.py --headless --depth auto       # adaptive depth (default)
.venv/bin/python bot.py --headless --depth 4 --games 5  # multi-game stats

# Smoke-test the browser interface (opens visible browser)
.venv/bin/python game.py

# Sanity-check that swap/delete power-ups work in browser
.venv/bin/python test_powers.py

# ── Strategy test harness (no browser required) ───────────────────────────────
# List all fixture boards
python3 tests/run.py

# Run 10 moves on a fixture (auto depth, random tile spawns)
python3 tests/run.py mid_game

# Common flag combinations
python3 tests/run.py mid_game --moves 20 --depth 4 --seed 42
python3 tests/run.py jammed --scores          # print expectimax score per direction
python3 tests/run.py swap_test --peek         # show only the first action, then stop
python3 tests/run.py late_game --no-random    # skip tile placement (pure decision trace)

# Depth calibration benchmark (no browser required)
.venv/bin/python benchmark_depth.py
.venv/bin/python benchmark_depth.py --moves 15 --seeds 3         # quick pass
.venv/bin/python benchmark_depth.py --depths 3,4,5,6 --seeds 8   # deeper comparison

# Phase-1 regression checks for strategy action APIs
python3 -m unittest tests/test_strategy_actions.py
```

### Board fixtures (`tests/boards/*.json`)

| File | max tile | Notes |
|------|----------|-------|
| `early_game` | 32 | Basic move ordering |
| `mid_game` | 512 | Snake gradient heuristic |
| `late_game` | 2048 | Deep search, monotonicity |
| `jammed` | 1024 | Delete power-up (D=1) |
| `swap_test` | 1024 | Swap power-up (S=1) |
| `corner_trap` | 512 | Misaligned snake, recovery |

**Board JSON format** — add new fixtures in `tests/boards/`:
```json
{
  "name": "my_test",
  "description": "What this tests",
  "board": [
    [1024, 512, 256, 128],
    [  32,  64,  32,  16],
    [   4,   8,   4,   2],
    [   0,   2,   0,   0]
  ],
  "score": 30000,
  "powers": {"undo": 0, "swap": 1, "delete": 0}
}
```

## Architecture

Core runtime is centered on three files (plus cache/test tooling), with no external game logic libraries:

**`game.py`** — Playwright async interface to https://play2048.co/

- `launch_browser()` injects `WORKER_INTERCEPT_JS` via `add_init_script` before any page JS runs. This overrides the `Worker` constructor to capture all `postMessage` traffic in `window._workerMsgs`.
- Board state is read from the most recent Worker `"update"` message (exact, fast). Screenshot pixel-sampling (`read_board_from_screenshot`) is a fallback when no Worker messages are found.
- Worker message format: `{"type":"call","call":"update","args":[{"state":"playing","board":[[null|{value,position:{x,y}}]...],"powerups":{...}}]}`
- `GameState` dataclass holds `board` (4×4 int grid, 0=empty), `score`, `best`, `over`, `won`, `powers`.
- Power-up execution identifies the three `button[class*=aspect-square]` elements and clicks them plus canvas tiles.
- Score is read from DOM: `span.shrink-1.truncate` (first=score, second=best). Score briefly drops to 0 when the win overlay is visible — callers must track `best_score_seen`.

**`strategy.py`** — Pure-Python Expectimax AI, no I/O

- `apply_move(board, direction)` → `(new_board, score_delta, changed)`. Does NOT place a new tile; that is done by the chance node.
- `best_action_obj(board, powers, depth)` evaluates all regular moves plus swap/delete power-ups and returns typed actions: `MoveAction`, `SwapAction`, `DeleteAction`.
- `best_action(board, powers, depth)` is the backwards-compatible tuple wrapper over `best_action_obj`: `("move", dir)`, `("swap", r1,c1,r2,c2)`, or `("delete", value, row, col)`.
- `_expectimax(board, depth, is_max, powers)`: max nodes try all 4 directions; chance nodes place 2 (90%) or 4 (10%) in empty cells. When more than 6 cells are empty, cells are subsampled to cap branching. `powers` is threaded through unchanged (no power-ups are used within the tree).
- `score_board(board, powers)` is cached by `(board_bb, swap_uses, delete_uses)` and currently uses: empty cells, snake gradient, direction-aware monotonicity (rows follow snake direction), roughness (subtracted), merge potential, max tile log2, `log2(sum_tiles+1)`, legal move count, corner bonus when max tile is at `(0,0)`, and power-up value.
- Power-up proximity is intentionally damped/gated: swap proximity starts at max tile `>=128`, delete proximity at `>=256`, each scaled by `0.25`, and proximity goes to `0` once unlock threshold is reached (`256`/`512`) so the heuristic does not carry phantom inventory.
- Snake weight matrix puts upper-left corner at weight 16, zigzagging down to 1 at lower-left.

**`bot.py`** — Async game loop

- `auto_depth(board)` chooses depth from board-state features (max tile, fullness, mobility, roughness), with depth-6 reserved for near-death late-game boards.
- `play_one_game()` handles the full lifecycle: win overlay dismissal (guard flag so DOM element persisting hidden doesn't re-trigger), stuck-board detection (5 unchanged consecutive moves), and power-up tracking.
- Prints board and status every 25 moves; announces depth bumps in auto mode.

## Transposition Table System

`score_board` results are cached to avoid recomputing the same board position multiple times within and across game runs.

**In-memory cache** (`strategy.py`):
- `_TRANS_TABLE`: `{(board_bb, swap_uses, delete_uses) → score}`. Normally cleared (LRU-free eviction) when it reaches `_TRANS_CAP` (500 000 entries).
- `_NEW_ENTRIES`: same keys, only entries added in the current run — drained after each game and flushed to SQLite.
- `board_to_bb(board)` encodes the 4×4 grid as a 64-bit int (4 bits per cell = log2(tile value), row-major).
- Oversized preload behavior: if startup preload from SQLite is already above `_TRANS_CAP`, `_TRANS_TABLE` is preserved and new lookups are not inserted into memory; they are still tracked in `_NEW_ENTRIES` and flushed to SQLite.

**SQLite persistence** (`cache.py`, `cache/transposition.db`):
- Schema: `entries(board_bb INTEGER, swap_uses INTEGER, delete_uses INTEGER, version TEXT, score REAL, PK on all four)`.
- `board_bb` is stored as signed int64 (`_to_signed`/`_from_signed` helpers handle values > 2^63).
- `load_version(version)` → dict loaded into `_TRANS_TABLE` at bot startup.
- `save_entries(entries, version)` writes new rows after each game.
- DB path can be overridden with `TRANS_DB_PATH` (default is `cache/transposition.db` in this repo).

**Versioning** — **CRITICAL**:
- `SCORE_BOARD_VERSION` in `strategy.py` must be bumped (e.g. `"1.0"` → `"1.1"`) every time heuristic weights or eval logic changes.
- Old version rows remain in the DB but are ignored; `populate_cache.py --recompute` re-scores them under the new version.

**`populate_cache.py`** — Cache seeding/maintenance:
```bash
# Re-score all known board states under the current version (after a heuristic bump):
.venv/bin/python populate_cache.py --recompute

# Generate N self-play positions at depth D and cache them:
.venv/bin/python populate_cache.py --generate 500 --depth 3

# Inspect DB contents:
.venv/bin/python populate_cache.py --list
```

**Bot output** — cache stats are printed after each game and included in the summary table (`CacheHit%` column).

## Key Implementation Notes

- The site uses Svelte + Tailwind CSS with an OffscreenCanvas in a Web Worker. DOM inspection does not reveal tile values directly.
- `add_init_script` runs before any page scripts, so the Worker constructor override is guaranteed to be in place before the game's Worker is created.
- `_roughness` must guard `if v <= 0: continue` (not just `== 0`) because board values are never negative but the guard matters for correctness.
- Monotonicity is penalty-based and mixed-direction: rows are direction-aware (must follow snake direction), columns still use the minimum penalty across both directions, then negate.
- `new_game()` tries several button text variants and falls back to force-click + Escape to handle the post-game-over overlay.

## Refactor Roadmap

This section tracks active refactor phases so agents can align changes without re-discovering intent.

### Phase 1 (low risk, in progress): typed actions + regression tests
- Introduce typed action models in `strategy.py` (`MoveAction`, `SwapAction`, `DeleteAction`) and a new `best_action_obj(...)` API.
- Keep `best_action(...)` tuple API for backward compatibility; it wraps typed actions.
- Deterministic regression checks live in `tests/test_strategy_actions.py` (tuple back-compat + fixture action expectations).

### Phase 2 (medium risk): evaluation decomposition
- Split `score_board` into `extract_features(...)` + weighted scorer.
- Keep transposition-key semantics stable and preserve current behavior behind tests.
- Move heuristic weights to a central config object (still loaded in-process by default).

### Phase 3 (higher leverage): cache/search modularization
- Extract transposition-table behavior into a dedicated cache component with explicit policies (cap, preload, flush semantics).
- Remove duplicated simulation/depth-schedule logic across scripts by introducing shared harness utilities.
- Add CI quality gates (`ruff`, unit tests, targeted benchmark smoke checks) before larger search optimizations.
