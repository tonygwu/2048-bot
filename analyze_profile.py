#!/usr/bin/env python3
"""
Analyze bot profiling JSONL logs emitted by bot.py.

Usage:
  python analyze_profile.py                   # analyze latest .bot_logs file
  python analyze_profile.py path/to/log.jsonl
  python analyze_profile.py --top 15
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_LOG_DIR = Path(".bot_logs")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    s = sorted(values)
    pos = (len(s) - 1) * p
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "n": float(len(values)),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _bucket_floor(max_tile: int) -> int:
    if max_tile < 512:
        return 0
    return max(512, 1 << (max_tile.bit_length() - 1))


def _bucket_label(bucket_floor: int) -> str:
    if bucket_floor == 0:
        return "<512"
    return f"{bucket_floor}-{bucket_floor * 2 - 1}"


def _fmt_ms(value: float) -> str:
    return f"{value:8.1f}"


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _parse_ts(ts: str | None) -> dt.datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _resolve_log_path(path_arg: str | None) -> Path:
    if path_arg:
        p = Path(path_arg).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p)
    if not DEFAULT_LOG_DIR.exists():
        raise FileNotFoundError(f"No log directory found: {DEFAULT_LOG_DIR}")
    logs = sorted(DEFAULT_LOG_DIR.glob("bot_profile_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"No profile logs found in {DEFAULT_LOG_DIR}")
    return logs[-1]


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payload["_lineno"] = lineno
                yield payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze bot profiling JSONL logs")
    parser.add_argument("log", nargs="?", help="Path to profile JSONL (default: latest in .bot_logs)")
    parser.add_argument("--top", type=int, default=10, help="Show top N slow moves (default: 10)")
    args = parser.parse_args()

    log_path = _resolve_log_path(args.log)
    if not log_path.exists():
        raise SystemExit(f"log file not found: {log_path}")

    event_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    games_seen: set[int] = set()
    game_end_reasons: Counter[str] = Counter()
    run_start = None
    run_end = None
    first_ts = None
    last_ts = None
    rows = []

    for ev in _iter_jsonl(log_path):
        event = str(ev.get("event", "unknown"))
        event_counts[event] += 1
        ts = _parse_ts(ev.get("ts"))
        if ts is not None:
            first_ts = ts if first_ts is None or ts < first_ts else first_ts
            last_ts = ts if last_ts is None or ts > last_ts else last_ts

        if event == "run_start":
            run_start = ev
        elif event == "run_end":
            run_end = ev
        elif event == "game_start":
            game = ev.get("game")
            if isinstance(game, int):
                games_seen.add(game)
        elif event == "game_end":
            reason = str(ev.get("reason", "unknown"))
            game_end_reasons[reason] += 1
            game = ev.get("game")
            if isinstance(game, int):
                games_seen.add(game)
        elif event == "move_profile":
            t = ev.get("timings_ms", {}) or {}
            c = ev.get("cache_delta", {}) or {}
            status = str(ev.get("status", "unknown"))
            status_counts[status] += 1
            max_tile = int(ev.get("max_tile", 0) or 0)
            bucket = _bucket_floor(max_tile)
            game = ev.get("game")
            if isinstance(game, int):
                games_seen.add(game)
            row = {
                "game": int(game or 0),
                "move": int(ev.get("move", 0) or 0),
                "depth": int(ev.get("depth", 0) or 0),
                "status": status,
                "action_kind": str(ev.get("action_kind", "unknown")),
                "max_tile": max_tile,
                "bucket": bucket,
                "read_state": float(t.get("read_state", 0.0) or 0.0),
                "best_action": float(t.get("best_action", 0.0) or 0.0),
                "execute_action": float(t.get("execute_action", 0.0) or 0.0),
                "loop_total": float(t.get("loop_total", 0.0) or 0.0),
                "eval_hits": int(c.get("eval_hits", 0) or 0),
                "eval_misses": int(c.get("eval_misses", 0) or 0),
                "search_hits": int(c.get("search_hits", 0) or 0),
                "search_misses": int(c.get("search_misses", 0) or 0),
                "search_size": int(c.get("search_size", 0) or 0),
                "maxrss_raw": ev.get("maxrss_raw"),
            }
            rows.append(row)

    if not rows:
        raise SystemExit(f"No move_profile events found in {log_path}")

    by_max_tile = defaultdict(list)
    by_bucket = defaultdict(list)
    by_depth = defaultdict(list)
    for r in rows:
        by_max_tile[r["max_tile"]].append(r)
        by_bucket[r["bucket"]].append(r)
        by_depth[r["depth"]].append(r)

    total_eval_hits = sum(r["eval_hits"] for r in rows)
    total_eval_misses = sum(r["eval_misses"] for r in rows)
    total_search_hits = sum(r["search_hits"] for r in rows)
    total_search_misses = sum(r["search_misses"] for r in rows)
    eval_total = total_eval_hits + total_eval_misses
    search_total = total_search_hits + total_search_misses

    print(f"Log file: {log_path}")
    print(f"Lines analyzed: {_fmt_int(sum(event_counts.values()))} events")
    print(f"Games seen: {sorted(games_seen) if games_seen else 'n/a'}")
    if run_start:
        print(
            "Run config: "
            f"depth_mode={run_start.get('depth_mode')} "
            f"headless={run_start.get('headless')} "
            f"games={run_start.get('games')}"
        )
    if first_ts and last_ts:
        elapsed = (last_ts - first_ts).total_seconds()
        print(f"Observed wall time in log: {elapsed:.1f}s")
    if run_end is None:
        print("Run end marker: missing (run likely interrupted)")
    print()

    print("Event counts:")
    for name, count in sorted(event_counts.items()):
        print(f"  {name:16} {_fmt_int(count)}")
    if game_end_reasons:
        print("Game end reasons:")
        for name, count in sorted(game_end_reasons.items()):
            print(f"  {name:16} {_fmt_int(count)}")
    print()

    timing_metrics = ["read_state", "best_action", "execute_action", "loop_total"]
    print("Overall timing summary (ms):")
    print("  metric           n      mean      p50      p95      p99      max")
    for metric in timing_metrics:
        values = [r[metric] for r in rows]
        s = _summary(values)
        print(
            f"  {metric:12} {_fmt_int(int(s['n'])):>8}"
            f"{_fmt_ms(s['mean'])}{_fmt_ms(s['p50'])}{_fmt_ms(s['p95'])}"
            f"{_fmt_ms(s['p99'])}{_fmt_ms(s['max'])}"
        )
    print()

    print("Move statuses:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:16} {_fmt_int(count)}")
    print()

    print("By max-tile state (mean ms):")
    print("  max_tile         n read_mean think_mean exec_mean loop_mean think_share")
    for max_tile in sorted(by_max_tile.keys()):
        tile_rows = by_max_tile[max_tile]
        n = len(tile_rows)
        read_mean = sum(r["read_state"] for r in tile_rows) / n if n else 0.0
        think_mean = sum(r["best_action"] for r in tile_rows) / n if n else 0.0
        exec_mean = sum(r["execute_action"] for r in tile_rows) / n if n else 0.0
        loop_mean = sum(r["loop_total"] for r in tile_rows) / n if n else 0.0
        think_share = (think_mean / loop_mean * 100.0) if loop_mean > 0 else 0.0
        print(
            f"  {max_tile:14} {_fmt_int(n):>8}"
            f"{_fmt_ms(read_mean)}{_fmt_ms(think_mean)}{_fmt_ms(exec_mean)}"
            f"{_fmt_ms(loop_mean)}  {think_share:8.1f}%"
        )
    print()

    print("By max-tile bucket (mean ms):")
    print("  bucket           n read_mean think_mean exec_mean loop_mean think_share")
    for bucket in sorted(by_bucket.keys()):
        bucket_rows = by_bucket[bucket]
        n = len(bucket_rows)
        read_vals = [r["read_state"] for r in bucket_rows]
        think_vals = [r["best_action"] for r in bucket_rows]
        exec_vals = [r["execute_action"] for r in bucket_rows]
        loop_vals = [r["loop_total"] for r in bucket_rows]
        read_mean = sum(read_vals) / n if n else 0.0
        think_mean = sum(think_vals) / n if n else 0.0
        exec_mean = sum(exec_vals) / n if n else 0.0
        loop_mean = sum(loop_vals) / n if n else 0.0
        think_share = (think_mean / loop_mean * 100.0) if loop_mean > 0 else 0.0
        print(
            f"  {_bucket_label(bucket):14} {_fmt_int(n):>8}"
            f"{_fmt_ms(read_mean)}{_fmt_ms(think_mean)}{_fmt_ms(exec_mean)}"
            f"{_fmt_ms(loop_mean)}"
            f"  {think_share:8.1f}%"
        )
    print()

    print("By search depth (mean ms):")
    print("  depth            n read_mean think_mean exec_mean loop_mean think_share")
    for depth in sorted(by_depth.keys()):
        depth_rows = by_depth[depth]
        n = len(depth_rows)
        read_vals = [r["read_state"] for r in depth_rows]
        think_vals = [r["best_action"] for r in depth_rows]
        exec_vals = [r["execute_action"] for r in depth_rows]
        loop_vals = [r["loop_total"] for r in depth_rows]
        read_mean = sum(read_vals) / n if n else 0.0
        think_mean = sum(think_vals) / n if n else 0.0
        exec_mean = sum(exec_vals) / n if n else 0.0
        loop_mean = sum(loop_vals) / n if n else 0.0
        think_share = (think_mean / loop_mean * 100.0) if loop_mean > 0 else 0.0
        print(
            f"  {depth:14} {_fmt_int(n):>8}"
            f"{_fmt_ms(read_mean)}{_fmt_ms(think_mean)}{_fmt_ms(exec_mean)}"
            f"{_fmt_ms(loop_mean)}"
            f"  {think_share:8.1f}%"
        )
    print()

    eval_hit_rate = (total_eval_hits / eval_total * 100.0) if eval_total else 0.0
    search_hit_rate = (total_search_hits / search_total * 100.0) if search_total else 0.0
    print("Cache deltas across move_profile events:")
    print(
        f"  eval   hits={_fmt_int(total_eval_hits)} misses={_fmt_int(total_eval_misses)} "
        f"hit_rate={eval_hit_rate:.1f}%"
    )
    print(
        f"  search hits={_fmt_int(total_search_hits)} misses={_fmt_int(total_search_misses)} "
        f"hit_rate={search_hit_rate:.1f}%"
    )
    print(f"  search_size max={_fmt_int(max(r['search_size'] for r in rows))}")
    print()

    if any(isinstance(r["maxrss_raw"], int) for r in rows):
        maxrss_vals = [int(r["maxrss_raw"]) for r in rows if isinstance(r["maxrss_raw"], int)]
        if maxrss_vals:
            print(
                "maxrss_raw observed "
                f"(platform units; KiB on Linux, bytes on macOS): "
                f"min={_fmt_int(min(maxrss_vals))} "
                f"p50={_fmt_int(int(_percentile([float(v) for v in maxrss_vals], 0.50)))} "
                f"max={_fmt_int(max(maxrss_vals))}"
            )
            print()

    top_n = max(1, args.top)
    print(f"Top {top_n} slow moves by loop_total (ms):")
    print("  rank  game  move  max_tile   status           read    think    exec    loop")
    slow_loop = sorted(rows, key=lambda r: r["loop_total"], reverse=True)[:top_n]
    for idx, r in enumerate(slow_loop, start=1):
        print(
            f"  {idx:>4} {r['game']:>5} {r['move']:>5} {r['max_tile']:>9} "
            f"{r['status'][:14]:14} {_fmt_ms(r['read_state'])}{_fmt_ms(r['best_action'])}"
            f"{_fmt_ms(r['execute_action'])}{_fmt_ms(r['loop_total'])}"
        )
    print()

    print(f"Top {top_n} slow moves by best_action (ms):")
    print("  rank  game  move  max_tile   status           read    think    exec    loop")
    slow_think = sorted(rows, key=lambda r: r["best_action"], reverse=True)[:top_n]
    for idx, r in enumerate(slow_think, start=1):
        print(
            f"  {idx:>4} {r['game']:>5} {r['move']:>5} {r['max_tile']:>9} "
            f"{r['status'][:14]:14} {_fmt_ms(r['read_state'])}{_fmt_ms(r['best_action'])}"
            f"{_fmt_ms(r['execute_action'])}{_fmt_ms(r['loop_total'])}"
        )


if __name__ == "__main__":
    main()
