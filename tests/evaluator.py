#!/usr/bin/env python3
"""
Canonical evaluation harness for strategy changes.

This script defines and computes the shared metrics used across experiments:
  - avg_score      : average in-game score at end of run
  - avg_max        : average max tile reached at end of run
  - survive%       : percentage of runs that reached the move cap
  - reach2048%     : percentage of runs that reached max tile >= 2048
  - reach4096%     : percentage of runs that reached max tile >= 4096
  - reach8192%     : percentage of runs that reached max tile >= 8192
  - avg_moves      : average number of actions executed

Additional diagnostics are also reported (avg_eval, avg_think_ms, action mix).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Allow importing from project root regardless of execution cwd.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from strategy import (
    apply_delete,
    apply_move,
    apply_swap,
    best_action,
    get_trans_stats,
    is_game_over,
    reset_trans_stats,
    score_board,
)
from sim_utils import place_random_tile


DEFAULT_MOVES = 80
DEFAULT_SEEDS = 30
DEFAULT_DEPTHS = [3, 4, 5]
DEFAULT_BOOTSTRAPS = 400
DEFAULT_AB_PERMUTATIONS = 2000


# Backwards-compatible board suite for depth tuning (folded from benchmark_depth.py).
DEPTH_CALIBRATION_BOARDS: dict[str, dict] = {
    "early_open": {
        "description": "12 empty, max=64, very open",
        "board": [[64, 0, 0, 0], [32, 16, 0, 0], [8, 0, 0, 0], [0, 0, 0, 0]],
        "score": 0,
        "powers": {"undo": 0, "swap": 0, "delete": 0},
    },
    "mid_open": {
        "description": "8 empty, max=512, clean monotone",
        "board": [[512, 256, 128, 64], [32, 16, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        "score": 0,
        "powers": {"undo": 0, "swap": 0, "delete": 0},
    },
    "mid_messy": {
        "description": "6 empty, max=512, scattered",
        "board": [[512, 32, 64, 0], [16, 256, 4, 0], [8, 32, 2, 0], [4, 0, 0, 0]],
        "score": 0,
        "powers": {"undo": 0, "swap": 0, "delete": 0},
    },
    "late_clean": {
        "description": "5 empty, max=4096, monotone snake",
        "board": [[4096, 2048, 1024, 512], [128, 256, 512, 256], [64, 32, 16, 0], [0, 0, 0, 0]],
        "score": 0,
        "powers": {"undo": 0, "swap": 0, "delete": 0},
    },
    "late_messy": {
        "description": "4 empty, max=2048, scrambled",
        "board": [[2048, 64, 512, 1024], [256, 32, 128, 16], [8, 64, 4, 0], [0, 0, 0, 2]],
        "score": 0,
        "powers": {"undo": 0, "swap": 0, "delete": 0},
    },
    "crisis_low_max": {
        "description": "3 empty, max=1024, crisis",
        "board": [[1024, 512, 256, 128], [64, 32, 16, 8], [4, 2, 4, 2], [0, 0, 0, 2]],
        "score": 0,
        "powers": {"undo": 0, "swap": 0, "delete": 0},
    },
    "crisis_high_max": {
        "description": "1 empty, max=4096, near-death",
        "board": [[8, 2, 4096, 2048], [256, 8, 128, 1024], [8, 64, 16, 8], [4, 2, 32, 0]],
        "score": 0,
        "powers": {"undo": 0, "swap": 0, "delete": 0},
    },
    "big_tile_open": {
        "description": "10 empty, max=8192, very open",
        "board": [[8192, 4096, 2048, 1024], [512, 256, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        "score": 0,
        "powers": {"undo": 0, "swap": 0, "delete": 0},
    },
}


@dataclass
class Fixture:
    name: str
    board: list[list[int]]
    score: int
    powers: dict
    description: str = ""


def _parse_depths(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("--depths must provide at least one depth")
    if any(v <= 0 for v in vals):
        raise ValueError("--depths values must be positive integers")
    return vals


def _parse_names(raw: str | None) -> set[str] | None:
    if raw is None or not raw.strip():
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def _load_fixture_suite(suite: str, selected_names: set[str] | None = None) -> list[Fixture]:
    fixtures: list[Fixture] = []

    if suite == "fixtures":
        boards_dir = Path(__file__).parent / "boards"
        for p in sorted(boards_dir.glob("*.json")):
            data = json.loads(p.read_text())
            name = p.stem
            if selected_names and name not in selected_names:
                continue
            fixtures.append(
                Fixture(
                    name=name,
                    board=[row[:] for row in data["board"]],
                    score=int(data.get("score", 0)),
                    powers=dict(data.get("powers", {})),
                    description=data.get("description", ""),
                )
            )
    elif suite == "depth_calibration":
        for name, spec in DEPTH_CALIBRATION_BOARDS.items():
            if selected_names and name not in selected_names:
                continue
            fixtures.append(
                Fixture(
                    name=name,
                    board=[row[:] for row in spec["board"]],
                    score=int(spec.get("score", 0)),
                    powers=dict(spec.get("powers", {})),
                    description=spec.get("description", ""),
                )
            )
    else:
        raise ValueError(f"Unknown suite: {suite}")

    if not fixtures:
        raise ValueError("No fixtures selected")
    return fixtures


def _load_existing_jsonl(path: str | None) -> tuple[set[tuple[int, str, int]], list[dict]]:
    if not path:
        return set(), []
    p = Path(path)
    if not p.exists():
        return set(), []
    completed: set[tuple[int, str, int]] = set()
    rows: list[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(k in row for k in ("depth", "fixture", "seed")):
                completed.add((int(row["depth"]), str(row["fixture"]), int(row["seed"])))
            rows.append(row)
    return completed, rows


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    k = (len(xs) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    w = k - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def _bootstrap_mean_ci(
    values: list[float],
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Return percentile CI for the mean."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = max(0, int((alpha / 2.0) * n_bootstrap) - 1)
    hi_idx = min(n_bootstrap - 1, int((1 - alpha / 2.0) * n_bootstrap) - 1)
    return means[lo_idx], means[hi_idx]


def _paired_permutation_test(
    deltas: list[float],
    n_permutations: int = DEFAULT_AB_PERMUTATIONS,
    seed: int = 0,
) -> float:
    """Two-sided paired sign-flip permutation test p-value for mean delta."""
    if not deltas:
        return 1.0
    obs = abs(sum(deltas) / len(deltas))
    if obs == 0.0:
        return 1.0
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_permutations):
        s = 0.0
        for d in deltas:
            s += d if rng.random() < 0.5 else -d
        if abs(s / len(deltas)) >= obs:
            ge += 1
    return (ge + 1) / (n_permutations + 1)


def _simulate_one(
    fixture: Fixture,
    depth: int,
    n_moves: int,
    seed: int,
    no_random: bool,
) -> dict:
    rng = random.Random(seed)
    board = [row[:] for row in fixture.board]
    score = int(fixture.score)
    powers = dict(fixture.powers)

    moves = 0
    think_ms_total = 0.0
    action_counts = {"move": 0, "swap": 0, "delete": 0}
    think_samples: list[float] = []
    reset_trans_stats()

    for _ in range(n_moves):
        if is_game_over(board):
            break

        t0 = time.perf_counter()
        action = best_action(board, powers, depth=depth)
        think_ms = (time.perf_counter() - t0) * 1000.0
        think_ms_total += think_ms
        think_samples.append(think_ms)

        if action is None:
            break

        kind = action[0]
        action_counts[kind] += 1

        if kind == "move":
            _, direction = action
            nb, delta, changed = apply_move(board, direction)
            if not changed:
                break
            board = nb
            score += delta
            if not no_random:
                board = place_random_tile(board, rng)
        elif kind == "swap":
            _, r1, c1, r2, c2 = action
            board = apply_swap(board, r1, c1, r2, c2)
            powers = dict(powers)
            powers["swap"] = max(0, powers.get("swap", 0) - 1)
            if not no_random:
                board = place_random_tile(board, rng)
        elif kind == "delete":
            _, value, _, _ = action
            board = apply_delete(board, value)
            powers = dict(powers)
            powers["delete"] = max(0, powers.get("delete", 0) - 1)
            if not no_random:
                board = place_random_tile(board, rng)
        else:
            break

        moves += 1

    max_tile = max(v for row in board for v in row)
    cache_stats = get_trans_stats()
    cache_total = cache_stats["hits"] + cache_stats["misses"]
    cache_hit_rate = cache_stats["hits"] * 100.0 / cache_total if cache_total else 0.0
    return {
        "moves": moves,
        "score": score,
        "max_tile": max_tile,
        "game_over": is_game_over(board),
        "survived_to_cap": moves >= n_moves,
        "reach2048": max_tile >= 2048,
        "reach4096": max_tile >= 4096,
        "reach8192": max_tile >= 8192,
        "final_eval": score_board(board, powers),
        "think_ms_total": think_ms_total,
        "think_ms_samples": think_samples,
        "think_ms_mean": think_ms_total / max(1, moves),
        "cache_hits": cache_stats["hits"],
        "cache_misses": cache_stats["misses"],
        "cache_hit_rate": cache_hit_rate,
        "actions": action_counts,
    }


def _aggregate(runs: list[dict], bootstrap_count: int = DEFAULT_BOOTSTRAPS) -> dict:
    n = len(runs)
    if n == 0:
        return {}

    total_moves = sum(r["moves"] for r in runs)
    total_actions = sum(r["actions"][k] for r in runs for k in ("move", "swap", "delete"))
    think_samples = [x for r in runs for x in r.get("think_ms_samples", [])]
    total_cache_hits = sum(r.get("cache_hits", 0) for r in runs)
    total_cache_misses = sum(r.get("cache_misses", 0) for r in runs)
    total_cache = total_cache_hits + total_cache_misses

    score_ci = _bootstrap_mean_ci([r["score"] for r in runs], n_bootstrap=bootstrap_count)
    max_ci = _bootstrap_mean_ci([r["max_tile"] for r in runs], n_bootstrap=bootstrap_count)
    eval_ci = _bootstrap_mean_ci([r["final_eval"] for r in runs], n_bootstrap=bootstrap_count)

    return {
        "runs": n,
        "avg_score": sum(r["score"] for r in runs) / n,
        "avg_max": sum(r["max_tile"] for r in runs) / n,
        "survive_pct": sum(1 for r in runs if r["survived_to_cap"]) * 100.0 / n,
        "reach2048_pct": sum(1 for r in runs if r["reach2048"]) * 100.0 / n,
        "reach4096_pct": sum(1 for r in runs if r["reach4096"]) * 100.0 / n,
        "reach8192_pct": sum(1 for r in runs if r["reach8192"]) * 100.0 / n,
        "avg_moves": total_moves / n,
        "avg_eval": sum(r["final_eval"] for r in runs) / n,
        "avg_think_ms": (sum(r["think_ms_total"] for r in runs) / max(1, total_moves)),
        "think_p50_ms": _percentile(think_samples, 50),
        "think_p90_ms": _percentile(think_samples, 90),
        "think_p99_ms": _percentile(think_samples, 99),
        "cache_hits": total_cache_hits,
        "cache_misses": total_cache_misses,
        "cache_hit_rate_pct": (total_cache_hits * 100.0 / total_cache) if total_cache else 0.0,
        "avg_score_ci95": [score_ci[0], score_ci[1]],
        "avg_max_ci95": [max_ci[0], max_ci[1]],
        "avg_eval_ci95": [eval_ci[0], eval_ci[1]],
        "move_pct": (sum(r["actions"]["move"] for r in runs) * 100.0 / max(1, total_actions)),
        "swap_pct": (sum(r["actions"]["swap"] for r in runs) * 100.0 / max(1, total_actions)),
        "delete_pct": (sum(r["actions"]["delete"] for r in runs) * 100.0 / max(1, total_actions)),
    }


def evaluate_suite(
    fixtures: list[Fixture],
    depths: list[int],
    n_moves: int,
    n_seeds: int,
    seed_start: int,
    no_random: bool,
    progress_every: int = 5,
    bootstrap_count: int = DEFAULT_BOOTSTRAPS,
    jsonl_out: str | None = None,
    resume: bool = False,
    checkpoint_out: str | None = None,
    checkpoint_every: int = 25,
) -> tuple[list[dict], dict[int, dict[str, dict]], dict[int, dict[str, list[dict]]]]:
    summary_rows: list[dict] = []
    per_fixture_depth: dict[int, dict[str, dict]] = {}
    per_fixture_runs: dict[int, dict[str, list[dict]]] = {}

    existing_completed: set[tuple[int, str, int]] = set()
    existing_rows: list[dict] = []
    existing_row_by_key: dict[tuple[int, str, int], dict] = {}
    if resume and jsonl_out:
        existing_completed, existing_rows = _load_existing_jsonl(jsonl_out)
        for r in existing_rows:
            if all(k in r for k in ("depth", "fixture", "seed")):
                existing_row_by_key[(int(r["depth"]), str(r["fixture"]), int(r["seed"]))] = r
        if existing_completed:
            print(f"[status] resume enabled: found {len(existing_completed)} completed runs in JSONL")

    jsonl_path = Path(jsonl_out) if jsonl_out else None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            jsonl_path.write_text("")

    checkpoint_path = Path(checkpoint_out) if checkpoint_out else None
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    total_units = len(depths) * len(fixtures) * n_seeds
    completed_units = 0
    total_start = time.perf_counter()
    rows_since_checkpoint = 0
    for depth_idx, depth in enumerate(depths, start=1):
        print(f"[status] depth {depth} ({depth_idx}/{len(depths)}) starting...")
        depth_start = time.perf_counter()
        per_fixture_depth[depth] = {}
        per_fixture_runs[depth] = {}
        all_runs: list[dict] = []

        for fx_idx, fx in enumerate(fixtures, start=1):
            print(f"[status]   fixture {fx.name} ({fx_idx}/{len(fixtures)})")
            runs: list[dict] = []
            for i in range(n_seeds):
                seed = seed_start + i
                key = (depth, fx.name, seed)
                if key in existing_completed:
                    match = existing_row_by_key.get(key)
                    if match is None:
                        continue
                    run = {
                        "moves": match["moves"],
                        "score": match["score"],
                        "max_tile": match["max_tile"],
                        "game_over": match["game_over"],
                        "survived_to_cap": match["survived_to_cap"],
                        "reach2048": match["reach2048"],
                        "reach4096": match["reach4096"],
                        "reach8192": match["reach8192"],
                        "final_eval": match["final_eval"],
                        "think_ms_total": match["think_ms_total"],
                        "think_ms_samples": match.get("think_ms_samples", []),
                        "think_ms_mean": match.get("think_ms_mean", 0.0),
                        "cache_hits": match.get("cache_hits", 0),
                        "cache_misses": match.get("cache_misses", 0),
                        "cache_hit_rate": match.get("cache_hit_rate", 0.0),
                        "actions": match["actions"],
                    }
                else:
                    run = _simulate_one(fx, depth, n_moves, seed, no_random)
                runs.append(run)
                if jsonl_path:
                    if key not in existing_completed:
                        row = {
                            "depth": depth,
                            "fixture": fx.name,
                            "seed": seed,
                            **run,
                        }
                        with jsonl_path.open("a") as f:
                            f.write(json.dumps(row) + "\n")
                        rows_since_checkpoint += 1
                completed_units += 1
                elapsed = time.perf_counter() - total_start
                rate = completed_units / elapsed if elapsed > 0 else 0.0
                remaining = total_units - completed_units
                eta = remaining / rate if rate > 0 else float("inf")
                if (i + 1) % progress_every == 0 or (i + 1) == n_seeds:
                    print(
                        f"[status]     seeds {i + 1}/{n_seeds} "
                        f"(latest score={run['score']}, max_tile={run['max_tile']}) "
                        f"overall={completed_units}/{total_units} eta={_format_eta(eta)}"
                    )

                if checkpoint_path and rows_since_checkpoint >= checkpoint_every:
                    checkpoint = {
                        "meta": {
                            "git_sha": _git_sha(),
                            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "incomplete": True,
                        },
                        "progress": {
                            "completed_units": completed_units,
                            "total_units": total_units,
                            "elapsed_seconds": elapsed,
                            "eta_seconds": eta if math.isfinite(eta) else None,
                        },
                        "summary_so_far": summary_rows,
                    }
                    checkpoint_path.write_text(json.dumps(checkpoint, indent=2))
                    rows_since_checkpoint = 0
            fx_agg = _aggregate(runs, bootstrap_count=bootstrap_count)
            per_fixture_depth[depth][fx.name] = fx_agg
            per_fixture_runs[depth][fx.name] = runs
            all_runs.extend(runs)

        row = {"depth": depth, **_aggregate(all_runs, bootstrap_count=bootstrap_count)}
        summary_rows.append(row)
        elapsed = time.perf_counter() - depth_start
        print(
            f"[status] depth {depth} complete in {elapsed:.1f}s "
            f"(avg_score={row['avg_score']:.1f}, avg_think_ms={row['avg_think_ms']:.2f})"
        )

    total_elapsed = time.perf_counter() - total_start
    print(f"[status] evaluation complete in {total_elapsed:.1f}s")
    if checkpoint_path:
        final_checkpoint = {
            "meta": {
                "git_sha": _git_sha(),
                "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "incomplete": False,
            },
            "progress": {
                "completed_units": completed_units,
                "total_units": total_units,
                "elapsed_seconds": total_elapsed,
                "eta_seconds": 0.0,
            },
            "summary": summary_rows,
            "per_fixture": per_fixture_depth,
        }
        checkpoint_path.write_text(json.dumps(final_checkpoint, indent=2))
    return summary_rows, per_fixture_depth, per_fixture_runs


def _print_metric_glossary() -> None:
    print("Metric Glossary:")
    print("  avg_score    : average final in-game score across runs")
    print("  avg_max      : average highest tile reached by end of run")
    print("  survive%     : % of runs that reached the move cap")
    print("  reach2048%   : % of runs with max tile >= 2048")
    print("  reach4096%   : % of runs with max tile >= 4096")
    print("  reach8192%   : % of runs with max tile >= 8192")
    print("  avg_moves    : average executed actions per run")
    print("  avg_eval     : average final score_board(board, powers)")
    print("  avg_think_ms : average best_action compute time per executed action")
    print("  think_p50/90/99_ms : percentile best_action latency per action")
    print("  cache_hit_rate% : transposition table hit rate during evaluation")


def _print_summary_table(rows: list[dict], title: str) -> None:
    print(title)
    print(
        "| depth | avg_score | score_ci95 | avg_max | max_ci95 | survive% | reach2048% | "
        "reach4096% | reach8192% | avg_moves | avg_eval | eval_ci95 | avg_think_ms | "
        "think_p50 | think_p90 | think_p99 | cache_hit_rate% |"
    )
    print("|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        score_ci = f"[{r['avg_score_ci95'][0]:.1f}, {r['avg_score_ci95'][1]:.1f}]"
        max_ci = f"[{r['avg_max_ci95'][0]:.1f}, {r['avg_max_ci95'][1]:.1f}]"
        eval_ci = f"[{r['avg_eval_ci95'][0]:.1f}, {r['avg_eval_ci95'][1]:.1f}]"
        print(
            f"| {r['depth']} | {r['avg_score']:.1f} | {score_ci} | {r['avg_max']:.1f} | {max_ci} | "
            f"{r['survive_pct']:.1f} | {r['reach2048_pct']:.1f} | {r['reach4096_pct']:.1f} | "
            f"{r['reach8192_pct']:.1f} | {r['avg_moves']:.1f} | {r['avg_eval']:.1f} | {eval_ci} | "
            f"{r['avg_think_ms']:.2f} | {r['think_p50_ms']:.2f} | {r['think_p90_ms']:.2f} | "
            f"{r['think_p99_ms']:.2f} | {r['cache_hit_rate_pct']:.1f} |"
        )


def _print_per_fixture(depth: int, stats: dict[str, dict]) -> None:
    print(f"\nPer-fixture metrics (depth={depth}):")
    print(
        "| fixture | avg_score | avg_max | survive% | reach2048% | "
        "reach4096% | reach8192% | avg_moves | avg_eval | think_p90 | cache_hit_rate% | "
        "action_mix(move/swap/delete) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in sorted(stats.keys()):
        s = stats[name]
        mix = f"{s['move_pct']:.1f}/{s['swap_pct']:.1f}/{s['delete_pct']:.1f}"
        print(
            f"| {name} | {s['avg_score']:.1f} | {s['avg_max']:.1f} | {s['survive_pct']:.1f} | "
            f"{s['reach2048_pct']:.1f} | {s['reach4096_pct']:.1f} | {s['reach8192_pct']:.1f} | "
            f"{s['avg_moves']:.1f} | {s['avg_eval']:.1f} | {s['think_p90_ms']:.2f} | "
            f"{s['cache_hit_rate_pct']:.1f} | {mix} |"
        )


def run_depth_calibration_legacy(n_moves: int, n_seeds: int, depths: list[int]) -> None:
    """Compatibility entrypoint used by benchmark_depth.py."""
    fixtures = _load_fixture_suite("depth_calibration")
    rows, per_fx, _ = evaluate_suite(
        fixtures=fixtures,
        depths=depths,
        n_moves=n_moves,
        n_seeds=n_seeds,
        seed_start=0,
        no_random=False,
    )

    _print_metric_glossary()
    print()
    _print_summary_table(rows, title="Depth Calibration Summary")

    # Keep the legacy style insight: incremental gain per depth.
    if len(depths) >= 2:
        print("\nDepth-to-depth delta (% change in avg_score):")
        print("| from_depth | to_depth | delta_avg_score% |")
        print("|---:|---:|---:|")
        by_depth = {r["depth"]: r for r in rows}
        for lo, hi in zip(depths, depths[1:]):
            base = by_depth[lo]["avg_score"]
            nxt = by_depth[hi]["avg_score"]
            pct = (nxt - base) / abs(base) * 100.0 if base else 0.0
            print(f"| {lo} | {hi} | {pct:+.2f} |")

    # Per-fixture output for the deepest depth by default.
    _print_per_fixture(depth=depths[-1], stats=per_fx[depths[-1]])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical evaluator for depth/heuristic experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  .venv/bin/python tests/evaluator.py\n"
            "  .venv/bin/python tests/evaluator.py --suite fixtures --depths 3,4,5 --seeds 30 --moves 80\n"
            "  .venv/bin/python tests/evaluator.py --suite depth_calibration --depths 2,3,4,5 --seeds 5 --moves 25\n"
            "  .venv/bin/python tests/evaluator.py --suite fixtures --boards late_game,jammed --depths 4 --per-fixture\n"
        ),
    )
    parser.add_argument("--suite", choices=["fixtures", "depth_calibration"], default="fixtures")
    parser.add_argument("--boards", default=None, help="Comma-separated fixture names to include (default: all)")
    parser.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS))
    parser.add_argument("--moves", type=int, default=DEFAULT_MOVES)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--per-fixture", action="store_true", help="Print per-fixture metrics for each depth")
    parser.add_argument("--json-out", default=None, help="Optional path to write machine-readable JSON results")
    parser.add_argument(
        "--jsonl-out",
        default=None,
        help="Optional path to write one JSON object per run (JSONL)",
    )
    parser.add_argument("--progress-every", type=int, default=5, help="Print status every N seeds (default: 5)")
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS, help="Bootstrap samples for CI (default: 400)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --jsonl-out runs (skips completed depth/fixture/seed triplets)",
    )
    parser.add_argument(
        "--checkpoint-out",
        default=None,
        help="Optional path to write periodic checkpoint snapshots",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write checkpoint after every N newly-written JSONL rows (default: 25)",
    )
    parser.add_argument(
        "--ab-depths",
        default=None,
        help="Optional A/B compare mode as 'baseline,candidate' depths (paired by fixture+seed)",
    )
    parser.add_argument(
        "--ab-metric",
        choices=["score", "max_tile", "final_eval"],
        default="score",
        help="Metric for A/B paired delta (default: score)",
    )
    parser.add_argument(
        "--ab-permutations",
        type=int,
        default=DEFAULT_AB_PERMUTATIONS,
        help=f"Permutation count for A/B significance test (default: {DEFAULT_AB_PERMUTATIONS})",
    )
    args = parser.parse_args()

    if args.moves <= 0:
        parser.error("--moves must be > 0")
    if args.seeds <= 0:
        parser.error("--seeds must be > 0")
    if args.progress_every <= 0:
        parser.error("--progress-every must be > 0")
    if args.bootstraps <= 0:
        parser.error("--bootstraps must be > 0")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be > 0")
    if args.ab_permutations <= 0:
        parser.error("--ab-permutations must be > 0")
    if args.resume and not args.jsonl_out:
        parser.error("--resume requires --jsonl-out")

    try:
        depths = _parse_depths(args.depths)
    except ValueError as e:
        parser.error(str(e))

    selected_names = _parse_names(args.boards)

    try:
        fixtures = _load_fixture_suite(args.suite, selected_names=selected_names)
    except ValueError as e:
        parser.error(str(e))

    _print_metric_glossary()
    print()

    print(
        f"[status] starting evaluator: suite={args.suite} fixtures={len(fixtures)} "
        f"depths={depths} moves={args.moves} seeds={args.seeds} seed_start={args.seed_start}"
    )

    rows, per_fx, per_runs = evaluate_suite(
        fixtures=fixtures,
        depths=depths,
        n_moves=args.moves,
        n_seeds=args.seeds,
        seed_start=args.seed_start,
        no_random=args.no_random,
        progress_every=args.progress_every,
        bootstrap_count=args.bootstraps,
        jsonl_out=args.jsonl_out,
        resume=args.resume,
        checkpoint_out=args.checkpoint_out,
        checkpoint_every=args.checkpoint_every,
    )

    _print_summary_table(rows, title=f"Evaluation Summary (suite={args.suite}, fixtures={len(fixtures)})")

    if args.per_fixture:
        for d in depths:
            _print_per_fixture(depth=d, stats=per_fx[d])

    ab_result = None
    if args.ab_depths:
        try:
            ab_depths = _parse_depths(args.ab_depths)
        except ValueError:
            parser.error("--ab-depths must contain exactly two depths like '3,4'")
        if len(ab_depths) != 2:
            parser.error("--ab-depths must contain exactly two depths like '3,4'")
        d_base, d_cand = ab_depths
        if d_base not in per_runs or d_cand not in per_runs:
            parser.error("--ab-depths must be included in --depths")

        paired_deltas: list[float] = []
        metric_key = args.ab_metric
        for fx in fixtures:
            base_runs = per_runs[d_base][fx.name]
            cand_runs = per_runs[d_cand][fx.name]
            for i in range(min(len(base_runs), len(cand_runs))):
                paired_deltas.append(cand_runs[i][metric_key] - base_runs[i][metric_key])

        mean_delta = sum(paired_deltas) / len(paired_deltas) if paired_deltas else 0.0
        ci_lo, ci_hi = _bootstrap_mean_ci(
            paired_deltas,
            n_bootstrap=args.bootstraps,
            seed=args.seed_start + 17,
        )
        p_value = _paired_permutation_test(
            paired_deltas,
            n_permutations=args.ab_permutations,
            seed=args.seed_start + 29,
        )
        win_rate = sum(1 for x in paired_deltas if x > 0) * 100.0 / max(1, len(paired_deltas))
        loss_rate = sum(1 for x in paired_deltas if x < 0) * 100.0 / max(1, len(paired_deltas))

        ab_result = {
            "baseline_depth": d_base,
            "candidate_depth": d_cand,
            "metric": metric_key,
            "paired_samples": len(paired_deltas),
            "mean_delta": mean_delta,
            "mean_delta_ci95": [ci_lo, ci_hi],
            "win_rate_pct": win_rate,
            "loss_rate_pct": loss_rate,
            "p_value": p_value,
            "permutations": args.ab_permutations,
        }
        print("\nA/B Paired Comparison")
        print(
            f"  baseline={d_base} candidate={d_cand} metric={metric_key} "
            f"samples={ab_result['paired_samples']}"
        )
        print(
            f"  mean_delta={mean_delta:+.2f} "
            f"ci95=[{ci_lo:+.2f}, {ci_hi:+.2f}] "
            f"win%={win_rate:.1f} loss%={loss_rate:.1f} p={p_value:.4f}"
        )

    if args.json_out:
        payload = {
            "meta": {
                "git_sha": _git_sha(),
                "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
            "config": {
                "suite": args.suite,
                "boards": sorted(selected_names) if selected_names else "all",
                "depths": depths,
                "moves": args.moves,
                "seeds": args.seeds,
                "seed_start": args.seed_start,
                "no_random": args.no_random,
                "bootstraps": args.bootstraps,
                "progress_every": args.progress_every,
                "ab_depths": args.ab_depths,
                "ab_metric": args.ab_metric,
                "ab_permutations": args.ab_permutations,
                "resume": args.resume,
                "checkpoint_out": args.checkpoint_out,
                "checkpoint_every": args.checkpoint_every,
            },
            "summary": rows,
            "per_fixture": per_fx,
            "ab_result": ab_result,
            "jsonl_out": args.jsonl_out,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON results: {args.json_out}")


if __name__ == "__main__":
    main()
