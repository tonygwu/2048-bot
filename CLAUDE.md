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
# Run the bot (auto depth, adaptive to max tile)
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
- `best_action(board, powers, depth)` evaluates all regular moves plus swap/delete power-ups. Returns a tuple: `("move", dir)`, `("swap", r1,c1,r2,c2)`, or `("delete", value, row, col)`. Power-up cost is priced implicitly: the tree receives a decremented `powers` dict, so every leaf evaluation reflects the lost use.
- `_expectimax(board, depth, is_max, powers)`: max nodes try all 4 directions; chance nodes place 2 (90%) or 4 (10%) in empty cells. When more than 6 cells are empty, cells are subsampled to cap branching. `powers` is threaded through unchanged (no power-ups are used within the tree).
- `score_board(board, powers)` is an 8-term heuristic: empty cells, snake gradient, penalty-based monotonicity (returns 0 for perfect, negative for imperfect), roughness (subtracted), merge potential, max tile log2, tile sum, and power-up value. Power-up value = `_W_PU_BASE × log2(max_tile) × effective_uses` where effective_uses = actual uses + proximity bonus (0–0.95 based on how close max_tile is to the unlock threshold: 256 for swap, 512 for delete). Proximity bonus is zeroed when at the cap (2 uses), incentivising spending a use right before earning a replacement.
- Snake weight matrix puts upper-left corner at weight 16, zigzagging down to 1 at lower-left.

**`bot.py`** — Async game loop

- `auto_depth(max_tile)` maps tile thresholds to search depths: `<512→2`, `<2048→3`, `<4096→4`, `<8192→5`, `≥8192→6`, increasing depth as the board gets more complex and less branchy.
- `play_one_game()` handles the full lifecycle: win overlay dismissal (guard flag so DOM element persisting hidden doesn't re-trigger), stuck-board detection (5 unchanged consecutive moves), and power-up tracking.
- Prints board and status every 25 moves; announces depth bumps in auto mode.

## Transposition Table System

`score_board` results are cached to avoid recomputing the same board position multiple times within and across game runs.

**In-memory cache** (`strategy.py`):
- `_TRANS_TABLE`: `{(board_bb, swap_uses, delete_uses) → score}`. Cleared (LRU-free eviction) when it reaches `_TRANS_CAP` (500 000 entries).
- `_NEW_ENTRIES`: same keys, only entries added in the current run — drained after each game and flushed to SQLite.
- `board_to_bb(board)` encodes the 4×4 grid as a 64-bit int (4 bits per cell = log2(tile value), row-major).

**SQLite persistence** (`cache.py`, `cache/transposition.db`):
- Schema: `entries(board_bb INTEGER, swap_uses INTEGER, delete_uses INTEGER, version TEXT, score REAL, PK on all four)`.
- `board_bb` is stored as signed int64 (`_to_signed`/`_from_signed` helpers handle values > 2^63).
- `load_version(version)` → dict loaded into `_TRANS_TABLE` at bot startup.
- `save_entries(entries, version)` writes new rows after each game.

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
- Monotonicity is penalty-based: take the minimum drop penalty across both traversal directions for each row/col, then negate. Do NOT sum both directions.
- `new_game()` tries several button text variants and falls back to force-click + Escape to handle the post-game-over overlay.
