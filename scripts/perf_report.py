#!/usr/bin/env python3
"""Generate stable performance reports from bot profiling JSONL logs.

Workflow:
1) Run bot with profiling enabled:
   .venv/bin/python bot.py --headless --profile-log auto
2) Generate report artifact:
   python3 scripts/perf_report.py --label baseline
3) Generate a comparison vs a prior report:
   python3 scripts/perf_report.py --label candidate --baseline-report reports/perf/<baseline>.json

Outputs:
- reports/perf/<run_id>.json  (machine-readable, versioned schema)
- reports/perf/<run_id>.md    (human-readable summary + optional comparison)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "perf_report.v1"
DEFAULT_LOG_DIR = Path(".bot_logs")
DEFAULT_OUT_DIR = Path("reports/perf")


@dataclass(frozen=True)
class MoveRow:
    game: int
    move: int
    depth: int
    action_kind: str
    status: str
    max_tile: int
    bucket_floor: int
    read_state_ms: float
    best_action_ms: float
    execute_action_ms: float
    loop_total_ms: float
    eval_hits: int
    eval_misses: int
    search_hits: int
    search_misses: int
    search_size: int


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _bucket_floor(max_tile: int) -> int:
    if max_tile < 512:
        return 0
    return max(512, 1 << (max_tile.bit_length() - 1))


def _bucket_label(bucket_floor: int) -> str:
    if bucket_floor == 0:
        return "<512"
    return f"{bucket_floor}-{bucket_floor * 2 - 1}"


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
        return {
            "n": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "n": float(len(values)),
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "max_ms": max(values),
    }


def _parse_ts(ts: Any) -> dt.datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                obj["_lineno"] = lineno
                yield obj


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


def _git_output(args: list[str]) -> str:
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
    except Exception:
        return ""
    return out.decode("utf-8", errors="replace").strip()


def _git_meta() -> dict[str, Any]:
    commit = _git_output(["rev-parse", "HEAD"])
    short_commit = commit[:8] if commit else "unknown"
    branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(_git_output(["status", "--porcelain"]))
    return {
        "commit": commit,
        "short_commit": short_commit,
        "branch": branch or "unknown",
        "dirty": dirty,
    }


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", label.strip())
    cleaned = cleaned.strip("-_.")
    return cleaned or "run"


def _build_group_rows(rows: list[MoveRow], key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[MoveRow]] = defaultdict(list)
    for r in rows:
        grouped[getattr(r, key)].append(r)

    out: list[dict[str, Any]] = []
    for group_key in sorted(grouped.keys()):
        g = grouped[group_key]
        n = len(g)
        read_vals = [x.read_state_ms for x in g]
        think_vals = [x.best_action_ms for x in g]
        exec_vals = [x.execute_action_ms for x in g]
        loop_vals = [x.loop_total_ms for x in g]
        think_mean = sum(think_vals) / n if n else 0.0
        loop_mean = sum(loop_vals) / n if n else 0.0
        row: dict[str, Any] = {
            "n": n,
            "read_mean_ms": sum(read_vals) / n if n else 0.0,
            "think_mean_ms": think_mean,
            "exec_mean_ms": sum(exec_vals) / n if n else 0.0,
            "loop_mean_ms": loop_mean,
            "think_share_pct": (think_mean / loop_mean * 100.0) if loop_mean > 0 else 0.0,
        }
        if key == "depth":
            row["depth"] = int(group_key)
        elif key == "bucket_floor":
            row["bucket_floor"] = int(group_key)
            row["bucket"] = _bucket_label(int(group_key))
        elif key == "max_tile":
            row["max_tile"] = int(group_key)
        out.append(row)
    return out


def _format_delta(new: float, old: float) -> str:
    d = new - old
    sign = "+" if d >= 0 else "-"
    return f"{sign}{abs(d):.1f}"


def _build_comparison(report: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "mean_think_ms",
        "p95_think_ms",
        "mean_loop_ms",
        "p95_loop_ms",
        "eval_hit_rate_pct",
        "search_hit_rate_pct",
    ]
    new_sig = report["signals"]
    base_sig = baseline.get("signals", {})
    signal_delta = {}
    for k in keys:
        signal_delta[k] = {
            "new": _safe_float(new_sig.get(k, 0.0)),
            "baseline": _safe_float(base_sig.get(k, 0.0)),
            "delta": _safe_float(new_sig.get(k, 0.0)) - _safe_float(base_sig.get(k, 0.0)),
        }

    def _index_by(rows: list[dict[str, Any]], name: str) -> dict[Any, dict[str, Any]]:
        return {r.get(name): r for r in rows if name in r}

    new_depth = _index_by(report.get("groups", {}).get("depth", []), "depth")
    base_depth = _index_by(baseline.get("groups", {}).get("depth", []), "depth")
    depth_delta = []
    for depth in sorted(set(new_depth.keys()) & set(base_depth.keys())):
        nrow = new_depth[depth]
        brow = base_depth[depth]
        depth_delta.append(
            {
                "depth": depth,
                "new_think_mean_ms": _safe_float(nrow.get("think_mean_ms", 0.0)),
                "baseline_think_mean_ms": _safe_float(brow.get("think_mean_ms", 0.0)),
                "delta_think_mean_ms": _safe_float(nrow.get("think_mean_ms", 0.0))
                - _safe_float(brow.get("think_mean_ms", 0.0)),
                "new_n": _safe_int(nrow.get("n", 0)),
                "baseline_n": _safe_int(brow.get("n", 0)),
            }
        )

    new_bucket = _index_by(report.get("groups", {}).get("max_tile_bucket", []), "bucket")
    base_bucket = _index_by(baseline.get("groups", {}).get("max_tile_bucket", []), "bucket")
    bucket_delta = []
    for bucket in sorted(
        set(new_bucket.keys()) & set(base_bucket.keys()),
        key=lambda s: (s != "<512", int(str(s).split("-")[0]) if s and s != "<512" else 0),
    ):
        nrow = new_bucket[bucket]
        brow = base_bucket[bucket]
        bucket_delta.append(
            {
                "bucket": bucket,
                "new_think_mean_ms": _safe_float(nrow.get("think_mean_ms", 0.0)),
                "baseline_think_mean_ms": _safe_float(brow.get("think_mean_ms", 0.0)),
                "delta_think_mean_ms": _safe_float(nrow.get("think_mean_ms", 0.0))
                - _safe_float(brow.get("think_mean_ms", 0.0)),
                "new_n": _safe_int(nrow.get("n", 0)),
                "baseline_n": _safe_int(brow.get("n", 0)),
            }
        )

    return {
        "baseline_report": baseline.get("meta", {}).get("run_id", ""),
        "signal_delta": signal_delta,
        "depth_delta": depth_delta,
        "bucket_delta": bucket_delta,
    }


def _render_markdown(report: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    meta = report["meta"]
    src = report["source"]
    sig = report["signals"]
    lines: list[str] = []
    lines.append(f"# Perf Report: {meta['run_id']}")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- Generated: {meta['generated_at']}")
    lines.append(f"- Commit: `{meta['git']['short_commit']}`")
    lines.append(f"- Branch: `{meta['git']['branch']}`")
    lines.append(f"- Dirty: `{meta['git']['dirty']}`")
    lines.append(f"- Log: `{src['log_path']}`")
    lines.append(f"- Move events: `{src['move_events']}`")
    lines.append("")

    lines.append("## Key Signals")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| mean_think_ms | {sig['mean_think_ms']:.1f} |")
    lines.append(f"| p95_think_ms | {sig['p95_think_ms']:.1f} |")
    lines.append(f"| mean_loop_ms | {sig['mean_loop_ms']:.1f} |")
    lines.append(f"| p95_loop_ms | {sig['p95_loop_ms']:.1f} |")
    lines.append(f"| eval_hit_rate_pct | {sig['eval_hit_rate_pct']:.1f} |")
    lines.append(f"| search_hit_rate_pct | {sig['search_hit_rate_pct']:.1f} |")
    lines.append("")

    lines.append("## By Depth (mean ms)")
    lines.append("| depth | n | think_mean | loop_mean | think_share% |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in report.get("groups", {}).get("depth", []):
        lines.append(
            f"| {row['depth']} | {row['n']} | {row['think_mean_ms']:.1f} | "
            f"{row['loop_mean_ms']:.1f} | {row['think_share_pct']:.1f} |"
        )
    lines.append("")

    lines.append("## By Max-Tile Bucket (mean ms)")
    lines.append("| bucket | n | think_mean | loop_mean | think_share% |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in report.get("groups", {}).get("max_tile_bucket", []):
        lines.append(
            f"| {row['bucket']} | {row['n']} | {row['think_mean_ms']:.1f} | "
            f"{row['loop_mean_ms']:.1f} | {row['think_share_pct']:.1f} |"
        )
    lines.append("")

    lines.append("## Slowest Moves (by think ms)")
    lines.append("| rank | game | move | max_tile | depth | action | think_ms | loop_ms |")
    lines.append("|---:|---:|---:|---:|---:|---|---:|---:|")
    for i, row in enumerate(report.get("slow_moves", {}).get("by_think", []), start=1):
        lines.append(
            f"| {i} | {row['game']} | {row['move']} | {row['max_tile']} | "
            f"{row['depth']} | {row['action_kind']} | {row['best_action_ms']:.1f} | {row['loop_total_ms']:.1f} |"
        )
    lines.append("")

    if baseline is not None and "comparison" in report:
        cmp = report["comparison"]
        lines.append("## Comparison vs Baseline")
        lines.append(f"- Baseline run: `{cmp.get('baseline_report', 'unknown')}`")
        lines.append("")
        lines.append("| Signal | Baseline | New | Delta |")
        lines.append("|---|---:|---:|---:|")
        for k, row in cmp["signal_delta"].items():
            lines.append(
                f"| {k} | {row['baseline']:.1f} | {row['new']:.1f} | {_format_delta(row['new'], row['baseline'])} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stable perf reports from bot profile logs")
    parser.add_argument(
        "--log",
        "--log-file",
        dest="log",
        default=None,
        help="Path to bot profile JSONL (default: latest in .bot_logs)",
    )
    parser.add_argument("--label", default="run", help="Short label for output artifact filenames")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory (default: reports/perf)")
    parser.add_argument("--baseline-report", default=None, help="Optional baseline report JSON for delta section")
    parser.add_argument("--top", type=int, default=12, help="Rows for slow-move tables (default: 12)")
    args = parser.parse_args()

    log_path = _resolve_log_path(args.log)
    if not log_path.exists():
        raise SystemExit(f"log file not found: {log_path}")

    events = Counter()
    move_rows: list[MoveRow] = []
    first_ts = None
    last_ts = None
    run_start: dict[str, Any] | None = None
    run_end: dict[str, Any] | None = None
    maxrss_vals: list[int] = []

    for ev in _iter_jsonl(log_path):
        event = str(ev.get("event", "unknown"))
        events[event] += 1

        ts = _parse_ts(ev.get("ts"))
        if ts is not None:
            first_ts = ts if first_ts is None or ts < first_ts else first_ts
            last_ts = ts if last_ts is None or ts > last_ts else last_ts

        if event == "run_start":
            run_start = ev
        elif event == "run_end":
            run_end = ev

        maxrss = ev.get("maxrss_raw")
        if isinstance(maxrss, int):
            maxrss_vals.append(int(maxrss))

        if event != "move_profile":
            continue
        t = ev.get("timings_ms", {}) or {}
        c = ev.get("cache_delta", {}) or {}
        max_tile = _safe_int(ev.get("max_tile"), 0)
        move_rows.append(
            MoveRow(
                game=_safe_int(ev.get("game"), 0),
                move=_safe_int(ev.get("move"), 0),
                depth=_safe_int(ev.get("depth"), 0),
                action_kind=str(ev.get("action_kind", "unknown")),
                status=str(ev.get("status", "unknown")),
                max_tile=max_tile,
                bucket_floor=_bucket_floor(max_tile),
                read_state_ms=_safe_float(t.get("read_state"), 0.0),
                best_action_ms=_safe_float(t.get("best_action"), 0.0),
                execute_action_ms=_safe_float(t.get("execute_action"), 0.0),
                loop_total_ms=_safe_float(t.get("loop_total"), 0.0),
                eval_hits=_safe_int(c.get("eval_hits"), 0),
                eval_misses=_safe_int(c.get("eval_misses"), 0),
                search_hits=_safe_int(c.get("search_hits"), 0),
                search_misses=_safe_int(c.get("search_misses"), 0),
                search_size=_safe_int(c.get("search_size"), 0),
            )
        )

    if not move_rows:
        raise SystemExit(f"No move_profile events found in {log_path}")

    read_vals = [r.read_state_ms for r in move_rows]
    think_vals = [r.best_action_ms for r in move_rows]
    exec_vals = [r.execute_action_ms for r in move_rows]
    loop_vals = [r.loop_total_ms for r in move_rows]

    eval_hits = sum(r.eval_hits for r in move_rows)
    eval_misses = sum(r.eval_misses for r in move_rows)
    search_hits = sum(r.search_hits for r in move_rows)
    search_misses = sum(r.search_misses for r in move_rows)
    eval_total = eval_hits + eval_misses
    search_total = search_hits + search_misses
    think_mean = sum(think_vals) / len(think_vals)
    loop_mean = sum(loop_vals) / len(loop_vals)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = _sanitize_label(args.label)
    git = _git_meta()
    run_id = f"{_utc_stamp()}_{label}_{git['short_commit']}"

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "run_id": run_id,
            "label": label,
            "generated_at": _utc_now(),
            "git": git,
        },
        "source": {
            "log_path": str(log_path),
            "events": dict(events),
            "move_events": len(move_rows),
            "run_start": {
                "depth_mode": run_start.get("depth_mode") if run_start else None,
                "headless": run_start.get("headless") if run_start else None,
                "games": run_start.get("games") if run_start else None,
            },
            "run_end_present": run_end is not None,
            "observed_wall_s": (last_ts - first_ts).total_seconds() if (first_ts and last_ts) else 0.0,
        },
        "signals": {
            "mean_read_ms": sum(read_vals) / len(read_vals),
            "mean_think_ms": think_mean,
            "mean_exec_ms": sum(exec_vals) / len(exec_vals),
            "mean_loop_ms": loop_mean,
            "p95_think_ms": _percentile(think_vals, 0.95),
            "p95_loop_ms": _percentile(loop_vals, 0.95),
            "p99_think_ms": _percentile(think_vals, 0.99),
            "p99_loop_ms": _percentile(loop_vals, 0.99),
            "think_share_pct": (think_mean / loop_mean * 100.0) if loop_mean > 0 else 0.0,
            "eval_hit_rate_pct": (eval_hits / eval_total * 100.0) if eval_total else 0.0,
            "search_hit_rate_pct": (search_hits / search_total * 100.0) if search_total else 0.0,
            "search_size_max": max(r.search_size for r in move_rows),
        },
        "timings": {
            "read_state": _summary(read_vals),
            "best_action": _summary(think_vals),
            "execute_action": _summary(exec_vals),
            "loop_total": _summary(loop_vals),
        },
        "cache": {
            "eval_hits": eval_hits,
            "eval_misses": eval_misses,
            "eval_hit_rate_pct": (eval_hits / eval_total * 100.0) if eval_total else 0.0,
            "search_hits": search_hits,
            "search_misses": search_misses,
            "search_hit_rate_pct": (search_hits / search_total * 100.0) if search_total else 0.0,
        },
        "groups": {
            "depth": _build_group_rows(move_rows, "depth"),
            "max_tile_bucket": _build_group_rows(move_rows, "bucket_floor"),
            "max_tile": _build_group_rows(move_rows, "max_tile"),
        },
        "slow_moves": {
            "by_loop": [
                {
                    "game": r.game,
                    "move": r.move,
                    "max_tile": r.max_tile,
                    "depth": r.depth,
                    "action_kind": r.action_kind,
                    "status": r.status,
                    "read_state_ms": r.read_state_ms,
                    "best_action_ms": r.best_action_ms,
                    "execute_action_ms": r.execute_action_ms,
                    "loop_total_ms": r.loop_total_ms,
                }
                for r in sorted(move_rows, key=lambda x: x.loop_total_ms, reverse=True)[: max(1, int(args.top))]
            ],
            "by_think": [
                {
                    "game": r.game,
                    "move": r.move,
                    "max_tile": r.max_tile,
                    "depth": r.depth,
                    "action_kind": r.action_kind,
                    "status": r.status,
                    "read_state_ms": r.read_state_ms,
                    "best_action_ms": r.best_action_ms,
                    "execute_action_ms": r.execute_action_ms,
                    "loop_total_ms": r.loop_total_ms,
                }
                for r in sorted(move_rows, key=lambda x: x.best_action_ms, reverse=True)[: max(1, int(args.top))]
            ],
        },
        "maxrss_raw": {
            "observed": bool(maxrss_vals),
            "min": min(maxrss_vals) if maxrss_vals else None,
            "p50": int(_percentile([float(v) for v in maxrss_vals], 0.50)) if maxrss_vals else None,
            "max": max(maxrss_vals) if maxrss_vals else None,
        },
    }

    baseline_report: dict[str, Any] | None = None
    if args.baseline_report:
        baseline_path = Path(args.baseline_report).expanduser()
        if not baseline_path.exists():
            raise SystemExit(f"baseline report not found: {baseline_path}")
        baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
        report["comparison"] = _build_comparison(report, baseline_report)

    json_path = out_dir / f"{run_id}.json"
    md_path = out_dir / f"{run_id}.md"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report, baseline_report), encoding="utf-8")

    print(f"Report JSON: {json_path}")
    print(f"Report MD:   {md_path}")
    if baseline_report is not None:
        print(f"Compared against baseline: {args.baseline_report}")


if __name__ == "__main__":
    main()
