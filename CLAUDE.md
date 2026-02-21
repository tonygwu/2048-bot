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
.venv/bin/python tests/run.py

# Run 10 moves on a fixture (auto depth, random tile spawns)
.venv/bin/python tests/run.py mid_game

# Common flag combinations
.venv/bin/python tests/run.py mid_game --moves 20 --depth 4 --seed 42
.venv/bin/python tests/run.py jammed --scores          # print expectimax score per direction
.venv/bin/python tests/run.py swap_test --peek         # show only the first action, then stop
.venv/bin/python tests/run.py late_game --no-random    # skip tile placement (pure decision trace)

# Canonical evaluator (shared metrics for heuristic + depth experiments)
.venv/bin/python tests/evaluator.py
.venv/bin/python tests/evaluator.py --suite fixtures --depths 3,4,5 --seeds 30 --moves 80
.venv/bin/python tests/evaluator.py --suite fixtures --boards late_game,jammed --depths 4 --per-fixture
.venv/bin/python tests/evaluator.py --suite depth_calibration --depths 2,3,4,5 --seeds 5 --moves 25

# Compatibility wrapper (delegates to tests/evaluator.py depth_calibration suite)
# Deprecated for new experiments: prefer tests/evaluator.py directly.
.venv/bin/python benchmark_depth.py
.venv/bin/python benchmark_depth.py --moves 15 --seeds 3         # quick pass
.venv/bin/python benchmark_depth.py --depths 3,4,5,6 --seeds 8   # deeper comparison

# Phase-1 regression checks for strategy action APIs
python3 -m unittest tests/test_strategy_actions.py

# Phase-2/3 regression checks (eval decomposition + cache policy)
python3 -m unittest tests/test_strategy_eval.py tests/test_transposition_cache.py
python3 -m unittest tests/test_auto_depth_policy.py

# Canonical evaluator (shared metrics + A/B + CIs + progress logs)
python3 tests/evaluator.py --suite fixtures --depths 3,4,5 --seeds 30 --moves 80
python3 tests/evaluator.py --suite fixtures --boards late_game,jammed --depths 4 --per-fixture
python3 tests/evaluator.py --suite fixtures --depths 3,4 --ab-depths 3,4 --ab-metric score
python3 tests/evaluator.py --suite fixtures --depths 3,4 --json-out /tmp/eval_summary.json --jsonl-out /tmp/eval_runs.jsonl
python3 tests/evaluator.py --suite fixtures --depths 3,4 --jsonl-out /tmp/eval_runs.jsonl --resume
python3 tests/evaluator.py --suite fixtures --depths 3,4 --checkpoint-out /tmp/eval_ckpt.json --checkpoint-every 50
python3 tests/evaluator.py --suite fixtures --depths 3,4 --module strategy --jobs 4
python3 tests/evaluator.py --suite fixtures --depths 3 --module strategy --candidate-module strategy_alt --ab-metric score

# Lint (mirrors CI)
ruff check .
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

Core runtime is centered on three files (`game.py`, `strategy.py`, `bot.py`) plus config/cache/simulation utilities:

**`game.py`** — Playwright async interface to https://play2048.co/

- `launch_browser()` injects `WORKER_INTERCEPT_JS` via `add_init_script` before any page JS runs. This overrides the `Worker` constructor to capture all `postMessage` traffic in `window._workerMsgs`.
- Board state is read from the most recent Worker `"update"` message (exact, fast). Screenshot pixel-sampling (`read_board_from_screenshot`) is a fallback when no Worker messages are found.
- Worker message format: `{"type":"call","call":"update","args":[{"state":"playing","board":[[null|{value,position:{x,y}}]...],"powerups":{...}}]}`
- `GameState` dataclass holds `board` (4×4 int grid, 0=empty), `score`, `best`, `over`, `won`, `powers`.
- Power-up execution identifies the three `button[class*=aspect-square]` elements and clicks them plus canvas tiles.
- Score is read from DOM: `span.shrink-1.truncate` (first=score, second=best). Score briefly drops to 0 when the win overlay is visible — callers must track `best_score_seen`.

**`strategy.py`** — Compatibility facade for modularized strategy code

- Keep importing from `strategy.py`; it re-exports the stable API while implementation lives in focused modules:
- `strategy_core.py`: deterministic board mechanics (`apply_move`, `empty_cells`, `is_game_over`, `board_to_bb`, `DIRECTIONS`).
- `strategy_actions.py`: typed actions (`MoveAction`, `SwapAction`, `DeleteAction`) + tuple compatibility helpers.
- `strategy_eval.py`: eval features/scoring and transposition cache integration (`score_board`, `extract_eval_features`, `score_from_features`, cache load/drain/stats APIs).
- `strategy_search.py`: adaptive depth, expectimax, and action selection (`auto_depth`, `_expectimax`, `best_action_obj`, `best_action`).

### Canonical evaluation metrics (`tests/evaluator.py`)

Use these definitions consistently when comparing strategy changes:

| Metric | Definition |
|---|---|
| `avg_score` | Average final in-game score across runs |
| `avg_max` | Average highest tile reached by end of run |
| `survive%` | Percent of runs that reached the configured move cap |
| `reach2048%` | Percent of runs whose max tile reached at least 2048 |
| `reach4096%` | Percent of runs whose max tile reached at least 4096 |
| `reach8192%` | Percent of runs whose max tile reached at least 8192 |
| `avg_moves` | Average number of executed actions per run |
| `avg_eval` | Average final `score_board(board, powers)` value |
| `avg_think_ms` | Average `best_action` compute time per executed action |

**`bot.py`** — Async game loop

- `auto_depth(board)` chooses depth from board-state features (max tile, fullness, mobility, roughness), with depth-6 reserved for near-death late-game boards.
- `play_one_game()` handles the full lifecycle: win overlay dismissal (guard flag so DOM element persisting hidden doesn't re-trigger), stuck-board detection (5 unchanged consecutive moves), and power-up tracking.
- Prints board and status every 25 moves; announces depth bumps in auto mode.

**`strategy_config.py`** — Centralized tuning knobs
- `DepthPolicy`: adaptive-depth policy constants (open/jammed/surgery/near-death thresholds and weights).
- `EvalWeights`: coefficients for the decomposed eval feature scorer.
- `PowerUpPolicy`: power-up valuation/unlock/proximity constants.
- `strategy.py` imports defaults from here so tuning changes are centralized.

## Transposition Table System

`score_board` results are cached to avoid recomputing the same board position multiple times within and across game runs.

**In-memory cache** (`strategy.py`, `transposition_cache.py`):
- `TranspositionCache` owns cache policy and stats; `strategy.py` keeps a singleton `_TRANS_CACHE`.
- Cache key remains `{(board_bb, swap_uses, delete_uses) → score}` with cap `_TRANS_CAP = 500_000`.
- `drain_new_entries()` still returns newly-added keys for SQLite flush after each game.
- `board_to_bb(board)` encodes the 4×4 grid as a 64-bit int (4 bits per cell = log2(tile value), row-major).
- Oversized preload behavior: if startup preload from SQLite is already above `_TRANS_CAP`, the preloaded table is preserved and new lookups are not inserted into the in-memory table; they are still tracked for DB flush.

## Shared Simulation Utilities

- `sim_utils.py` currently owns `place_random_tile(...)`.
- Both `tests/run.py` and `benchmark_depth.py` use this helper to keep spawn semantics aligned.

## Evaluator Notes

- `tests/evaluator.py` is the canonical offline harness for depth/heuristic experiments.
- Core summary metrics: `avg_score`, `avg_max`, `survive%`, `reach2048%`, `reach4096%`, `reach8192%`, `avg_moves`.
- Diagnostics: `avg_eval`, `avg_think_ms`, `think_p50/think_p90/think_p99`, `cache_hit_rate%`, action mix (`move/swap/delete`).
- CI bands: summary includes bootstrap 95% CIs for `avg_score`, `avg_max`, `avg_eval` (configured by `--bootstraps`).
- Paired A/B mode: `--ab-depths baseline,candidate --ab-metric score|max_tile|final_eval` compares identical fixture+seed samples.
- A/B significance: permutation test p-value is reported (`--ab-permutations` controls sample count).
- Module A/B mode: `--module`, `--baseline-module`, `--candidate-module` compares two strategy implementations with paired seeds.
- Long-run visibility: status logs print depth/fixture/seed progress (`--progress-every`).
- ETA is printed during seed progress logs (`overall=X/Y eta=...`).
- Resume support: `--resume` reuses completed `(depth, fixture, seed)` rows from `--jsonl-out`.
- Resume safety: `--resume` requires a manifest alongside JSONL; evaluator validates config/fixture fingerprint/module name before resuming.
- Parallelism: `--jobs N` uses multiprocessing when available and auto-falls back to single-process if the environment blocks process pools.
- Checkpoint snapshots: `--checkpoint-out` writes periodic and final progress snapshots (`--checkpoint-every` controls cadence).
- Structured output:
  - `--json-out`: aggregate summary/per-fixture payload (includes git SHA and timestamp).
  - `--jsonl-out`: one row per run (depth + fixture + seed + run metrics).

**SQLite persistence** (`cache.py`, `cache/transposition.db`):
- Schema: `entries(board_bb INTEGER, swap_uses INTEGER, delete_uses INTEGER, version TEXT, score REAL, PK on all four)`.
- `board_bb` is stored as signed int64 (`_to_signed`/`_from_signed` helpers handle values > 2^63).
- `load_version(version)` → dict loaded into `_TRANS_TABLE` at bot startup.
- `save_entries(entries, version)` writes new rows after each game.
- DB path can be overridden with `TRANS_DB_PATH` (default is `cache/transposition.db` in this repo).

**Versioning** — **CRITICAL**:
- `SCORE_BOARD_VERSION` in `strategy.py` must be bumped (e.g. `"1.0"` → `"1.1"`) every time heuristic weights or eval logic changes.
- After bumping `SCORE_BOARD_VERSION`, immediately run `.venv/bin/python populate_cache.py --recompute` so the shared DB is refreshed for the latest version.
- Old version rows remain in the DB but are ignored by runtime loads.

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
- Shared simulation helper `sim_utils.place_random_tile(...)` is used by both `tests/run.py` and `benchmark_depth.py`; keep spawn semantics there to avoid drift.

## Refactor Roadmap

This section tracks active refactor phases so agents can align changes without re-discovering intent.

### Phase 1 (low risk, complete): typed actions + regression tests
- Introduce typed action models in `strategy.py` (`MoveAction`, `SwapAction`, `DeleteAction`) and a new `best_action_obj(...)` API.
- Keep `best_action(...)` tuple API for backward compatibility; it wraps typed actions.
- Deterministic regression checks live in `tests/test_strategy_actions.py` (tuple back-compat + fixture action expectations).

### Phase 2 (medium risk, complete): evaluation decomposition
- Split `score_board` into `extract_eval_features(...)` + weighted scorer (`score_from_features(...)`).
- Keep transposition-key semantics stable and preserve current behavior behind tests.
- Move heuristic weights to a central config object (still loaded in-process by default).

### Phase 3 (higher leverage, complete): cache/search modularization
- Extract transposition-table behavior into a dedicated cache component with explicit policies (cap, preload, flush semantics).
- Remove duplicated simulation/depth-schedule logic across scripts by introducing shared harness utilities (`sim_utils.py` done for tile-spawn path).
- Add CI quality gates (`ruff`, unit tests, targeted benchmark smoke checks) before larger search optimizations.

### Phase 4 (in progress): structural decomposition
- Split strategy internals into smaller modules (`strategy_actions.py`, `strategy_core.py`, `strategy_eval.py`, `strategy_search.py`) while preserving compatibility exports from `strategy.py`.
- Keep fixture-driven regression tests as a hard gate for any behavioral drift.
