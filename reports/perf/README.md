# Perf Reports

This directory stores committed, compact performance artifacts derived from raw bot profiling logs.

## Why this exists

Raw `.bot_logs/*.jsonl` files are large and noisy. They are useful for local debugging, but not ideal for long-term git history.

`reports/perf/*.json` and `reports/perf/*.md` provide stable summaries so agents can compare runs across commits.

## Workflow

1. Run the bot with profiling enabled (raw log remains local):

```bash
.venv/bin/python bot.py --headless --profile-log auto
```

2. Generate a committed summary artifact from the latest log:

```bash
python3 scripts/perf_report.py --label baseline
```

3. Compare against a prior committed report:

```bash
python3 scripts/perf_report.py \
  --label candidate \
  --baseline-report reports/perf/<baseline_report>.json
```

This writes:

- `reports/perf/<run_id>.json` (machine-readable, schema-versioned)
- `reports/perf/<run_id>.md` (human-readable summary with optional baseline deltas)

`run_id` format: `<UTC_TIMESTAMP>_<label>_<short_commit>`

## Output format

Schema version: `perf_report.v1`

Top-level JSON keys:

- `schema_version`
- `meta`:
  - `run_id`
  - `label`
  - `generated_at`
  - `git` (`commit`, `short_commit`, `branch`, `dirty`)
- `source`:
  - `log_path`
  - `events`
  - `move_events`
  - `run_start` (`depth_mode`, `headless`, `games`)
  - `run_end_present`
  - `observed_wall_s`
- `signals`:
  - `mean_think_ms`, `p95_think_ms`, `p99_think_ms`
  - `mean_loop_ms`, `p95_loop_ms`, `p99_loop_ms`
  - `think_share_pct`
  - `eval_hit_rate_pct`, `search_hit_rate_pct`
  - `search_size_max`
- `timings`:
  - distributions for `read_state`, `best_action`, `execute_action`, `loop_total`
- `cache`:
  - eval/search hit/miss totals and rates
- `groups`:
  - `depth` (mean timing by depth)
  - `max_tile_bucket` (mean timing by max-tile bucket)
  - `max_tile` (mean timing by exact max tile)
- `slow_moves`:
  - top-N by `loop_total_ms`
  - top-N by `best_action_ms`
- `maxrss_raw`
- `comparison` (only when `--baseline-report` is provided):
  - signal deltas
  - depth deltas
  - bucket deltas

## Recommendations

- Commit only selected report pairs that represent meaningful checkpoints.
- Use consistent labels (for example: `baseline`, `undo-tuning-v2`, `depth-policy-a`).
- Keep raw `.bot_logs` local and disposable; use these reports for cross-commit learning.
