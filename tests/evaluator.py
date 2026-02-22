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
import concurrent.futures
import datetime as dt
import hashlib
import importlib
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

from sim_utils import place_random_tile


DEFAULT_MOVES = 80
DEFAULT_SEEDS = 30
DEFAULT_DEPTHS = [3, 4, 5]
DEFAULT_BOOTSTRAPS = 400
DEFAULT_AB_PERMUTATIONS = 2000
DEFAULT_MODULE = "strategy"

POWERUP_LATE_BOARDS = [
    "late_powerup_bank",
    "late_powerup_jammed",
    "post4096_powerup_delete",
]


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


@dataclass(frozen=True)
class StrategyFns:
    apply_delete: object
    apply_move: object
    apply_swap: object
    best_action: object
    get_trans_stats: object
    is_game_over: object
    reset_trans_cache: object | None
    reset_trans_stats: object
    score_board: object


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
    elif suite == "powerup_late":
        boards_dir = Path(__file__).parent / "boards"
        names = selected_names if selected_names else set(POWERUP_LATE_BOARDS)
        for name in sorted(names):
            p = boards_dir / f"{name}.json"
            if not p.exists():
                raise ValueError(f"Unknown powerup_late board: {name}")
            data = json.loads(p.read_text())
            fixtures.append(
                Fixture(
                    name=name,
                    board=[row[:] for row in data["board"]],
                    score=int(data.get("score", 0)),
                    powers=dict(data.get("powers", {})),
                    description=data.get("description", ""),
                )
            )
    else:
        raise ValueError(f"Unknown suite: {suite}")

    if not fixtures:
        raise ValueError("No fixtures selected")
    return fixtures


def _load_existing_jsonl(path: str | None) -> tuple[set[tuple[str, int, str, int]], list[dict]]:
    if not path:
        return set(), []
    p = Path(path)
    if not p.exists():
        return set(), []
    completed: set[tuple[str, int, str, int]] = set()
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
                completed.add(
                    (
                        str(row.get("module_name", DEFAULT_MODULE)),
                        int(row["depth"]),
                        str(row["fixture"]),
                        int(row["seed"]),
                    )
                )
            rows.append(row)
    return completed, rows


_STRATEGY_FN_CACHE: dict[str, StrategyFns] = {}


def _load_strategy_fns(module_name: str) -> StrategyFns:
    cached = _STRATEGY_FN_CACHE.get(module_name)
    if cached is not None:
        return cached
    module = importlib.import_module(module_name)
    required = [
        "apply_delete",
        "apply_move",
        "apply_swap",
        "best_action",
        "get_trans_stats",
        "is_game_over",
        "score_board",
        "reset_trans_stats",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"module {module_name!r} missing required symbols: {missing}")
    fns = StrategyFns(
        apply_delete=getattr(module, "apply_delete"),
        apply_move=getattr(module, "apply_move"),
        apply_swap=getattr(module, "apply_swap"),
        best_action=getattr(module, "best_action"),
        get_trans_stats=getattr(module, "get_trans_stats"),
        is_game_over=getattr(module, "is_game_over"),
        reset_trans_cache=getattr(module, "reset_trans_cache", None),
        reset_trans_stats=getattr(module, "reset_trans_stats"),
        score_board=getattr(module, "score_board"),
    )
    _STRATEGY_FN_CACHE[module_name] = fns
    return fns


def _fixtures_fingerprint(fixtures: list[Fixture]) -> str:
    payload = [
        {
            "name": fx.name,
            "board": fx.board,
            "score": fx.score,
            "powers": fx.powers,
            "description": fx.description,
        }
        for fx in fixtures
    ]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _fixture_tags(fx: Fixture) -> list[str]:
    tags: list[str] = []
    empties = sum(1 for r in range(4) for c in range(4) if fx.board[r][c] == 0)
    max_tile = max(v for row in fx.board for v in row)
    if empties >= 8:
        tags.append("open")
    if empties <= 3:
        tags.append("jammed")
    if max_tile >= 2048:
        tags.append("late")
    if any(fx.powers.get(k, 0) > 0 for k in ("swap", "delete", "undo")):
        tags.append("powerup")
    if not tags:
        tags.append("mid")
    return tags


def _manifest_path(jsonl_out: str) -> Path:
    p = Path(jsonl_out)
    return p.with_suffix(p.suffix + ".manifest.json")


def _build_manifest(
    *,
    suite: str,
    boards: set[str] | None,
    depths: list[int],
    moves: int,
    seeds: int,
    seed_start: int,
    no_random: bool,
    cache_mode: str,
    fixtures: list[Fixture],
    module_name: str,
) -> dict:
    return {
        "suite": suite,
        "boards": sorted(boards) if boards else "all",
        "depths": depths,
        "moves": moves,
        "seeds": seeds,
        "seed_start": seed_start,
        "no_random": no_random,
        "cache_mode": cache_mode,
        "module_name": module_name,
        "fixtures_fingerprint": _fixtures_fingerprint(fixtures),
    }


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
    module_name: str,
    depth: int,
    n_moves: int,
    seed: int,
    no_random: bool,
    cache_mode: str,
) -> dict:
    fns = _load_strategy_fns(module_name)
    rng = random.Random(seed)
    board = [row[:] for row in fixture.board]
    score = int(fixture.score)
    powers = dict(fixture.powers)

    moves = 0
    think_ms_total = 0.0
    action_counts = {"move": 0, "swap": 0, "delete": 0}
    think_samples: list[float] = []
    if cache_mode == "reset-per-run" and fns.reset_trans_cache is not None:
        fns.reset_trans_cache()
    fns.reset_trans_stats()

    for _ in range(n_moves):
        if fns.is_game_over(board):
            break

        t0 = time.perf_counter()
        action = fns.best_action(board, powers, depth=depth)
        think_ms = (time.perf_counter() - t0) * 1000.0
        think_ms_total += think_ms
        think_samples.append(think_ms)

        if action is None:
            break

        kind = action[0]
        action_counts[kind] += 1

        if kind == "move":
            _, direction = action
            nb, delta, changed = fns.apply_move(board, direction)
            if not changed:
                break
            board = nb
            score += delta
            if not no_random:
                board = place_random_tile(board, rng)
        elif kind == "swap":
            _, r1, c1, r2, c2 = action
            board = fns.apply_swap(board, r1, c1, r2, c2)
            powers = dict(powers)
            powers["swap"] = max(0, powers.get("swap", 0) - 1)
            if not no_random:
                board = place_random_tile(board, rng)
        elif kind == "delete":
            _, value, _, _ = action
            board = fns.apply_delete(board, value)
            powers = dict(powers)
            powers["delete"] = max(0, powers.get("delete", 0) - 1)
            if not no_random:
                board = place_random_tile(board, rng)
        else:
            break

        moves += 1

    max_tile = max(v for row in board for v in row)
    cache_stats = fns.get_trans_stats()
    cache_total = cache_stats["hits"] + cache_stats["misses"]
    cache_hit_rate = cache_stats["hits"] * 100.0 / cache_total if cache_total else 0.0
    return {
        "moves": moves,
        "score": score,
        "max_tile": max_tile,
        "game_over": fns.is_game_over(board),
        "survived_to_cap": moves >= n_moves,
        "reach2048": max_tile >= 2048,
        "reach4096": max_tile >= 4096,
        "reach8192": max_tile >= 8192,
        "final_eval": fns.score_board(board, powers),
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


def _simulate_task(task: dict) -> tuple[int, str, int, dict]:
    fx = Fixture(
        name=task["fixture"]["name"],
        board=task["fixture"]["board"],
        score=task["fixture"]["score"],
        powers=task["fixture"]["powers"],
        description=task["fixture"].get("description", ""),
    )
    run = _simulate_one(
        fixture=fx,
        module_name=task["module_name"],
        depth=task["depth"],
        n_moves=task["n_moves"],
        seed=task["seed"],
        no_random=task["no_random"],
        cache_mode=task["cache_mode"],
    )
    return task["depth"], fx.name, task["seed"], run


def evaluate_suite(
    fixtures: list[Fixture],
    depths: list[int],
    n_moves: int,
    n_seeds: int,
    seed_start: int,
    no_random: bool,
    module_name: str = DEFAULT_MODULE,
    cache_mode: str = "warm",
    jobs: int = 1,
    progress_every: int = 5,
    bootstrap_count: int = DEFAULT_BOOTSTRAPS,
    jsonl_out: str | None = None,
    resume: bool = False,
    checkpoint_out: str | None = None,
    checkpoint_every: int = 25,
) -> tuple[list[dict], dict[int, dict[str, dict]], dict[int, dict[str, list[dict]]], dict[int, dict[str, dict]]]:
    summary_rows: list[dict] = []
    per_fixture_depth: dict[int, dict[str, dict]] = {}
    per_fixture_runs: dict[int, dict[str, list[dict]]] = {}
    per_group_depth: dict[int, dict[str, dict]] = {}

    existing_completed: set[tuple[str, int, str, int]] = set()
    existing_rows: list[dict] = []
    existing_row_by_key: dict[tuple[str, int, str, int], dict] = {}
    if resume and jsonl_out:
        existing_completed, existing_rows = _load_existing_jsonl(jsonl_out)
        for r in existing_rows:
            if all(k in r for k in ("depth", "fixture", "seed")):
                existing_row_by_key[
                    (
                        str(r.get("module_name", DEFAULT_MODULE)),
                        int(r["depth"]),
                        str(r["fixture"]),
                        int(r["seed"]),
                    )
                ] = r
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

    # Cache mode hooks are only meaningful within a single process.
    if cache_mode == "cold":
        fns = _load_strategy_fns(module_name)
        if fns.reset_trans_cache is not None:
            fns.reset_trans_cache()

    total_units = len(depths) * len(fixtures) * n_seeds
    completed_units = 0
    total_start = time.perf_counter()
    rows_since_checkpoint = 0
    task_specs: list[dict] = []
    ordered_keys: list[tuple[int, str, int]] = []
    results_by_key: dict[tuple[int, str, int], dict] = {}

    for depth in depths:
        for fx in fixtures:
            for i in range(n_seeds):
                seed = seed_start + i
                key = (module_name, depth, fx.name, seed)
                public_key = (depth, fx.name, seed)
                ordered_keys.append(public_key)
                if key in existing_completed:
                    match = existing_row_by_key.get(key)
                    if match is None:
                        continue
                    results_by_key[public_key] = {
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
                    completed_units += 1
                else:
                    task_specs.append(
                        {
                            "module_name": module_name,
                            "depth": depth,
                            "fixture": {
                                "name": fx.name,
                                "board": [row[:] for row in fx.board],
                                "score": fx.score,
                                "powers": dict(fx.powers),
                                "description": fx.description,
                            },
                            "n_moves": n_moves,
                            "seed": seed,
                            "no_random": no_random,
                            "cache_mode": cache_mode,
                        }
                    )

    if completed_units:
        print(f"[status] preloaded {completed_units}/{total_units} completed runs from resume data")

    def _on_result(depth: int, fixture_name: str, seed: int, run: dict) -> None:
        nonlocal completed_units, rows_since_checkpoint
        pk = (depth, fixture_name, seed)
        results_by_key[pk] = run
        if jsonl_path:
            row = {
                "module_name": module_name,
                "depth": depth,
                "fixture": fixture_name,
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
        if completed_units % progress_every == 0 or completed_units == total_units:
            print(
                f"[status] completed {completed_units}/{total_units} "
                f"latest=({fixture_name},d{depth},seed={seed},score={run['score']}) "
                f"eta={_format_eta(eta)}"
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

    if task_specs:
        print(
            f"[status] executing {len(task_specs)} new runs with jobs={jobs} "
            f"(module={module_name})"
        )
        if jobs > 1:
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
                    futs = [ex.submit(_simulate_task, task) for task in task_specs]
                    for fut in concurrent.futures.as_completed(futs):
                        depth, fixture_name, seed, run = fut.result()
                        _on_result(depth, fixture_name, seed, run)
            except (PermissionError, OSError) as e:
                print(f"[status] multiprocessing unavailable ({e}); falling back to jobs=1")
                for task in task_specs:
                    depth, fixture_name, seed, run = _simulate_task(task)
                    _on_result(depth, fixture_name, seed, run)
        else:
            for task in task_specs:
                depth, fixture_name, seed, run = _simulate_task(task)
                _on_result(depth, fixture_name, seed, run)

    # Build aggregate structures in deterministic order.
    for depth in depths:
        per_fixture_depth[depth] = {}
        per_fixture_runs[depth] = {}
        per_group_depth[depth] = {}
        all_runs: list[dict] = []
        for fx in fixtures:
            runs: list[dict] = []
            for i in range(n_seeds):
                seed = seed_start + i
                pk = (depth, fx.name, seed)
                if pk in results_by_key:
                    runs.append(results_by_key[pk])
            fx_agg = _aggregate(runs, bootstrap_count=bootstrap_count)
            per_fixture_depth[depth][fx.name] = fx_agg
            per_fixture_runs[depth][fx.name] = runs
            all_runs.extend(runs)
        row = {"depth": depth, **_aggregate(all_runs, bootstrap_count=bootstrap_count)}
        summary_rows.append(row)
        group_runs: dict[str, list[dict]] = {}
        for fx in fixtures:
            runs = per_fixture_runs[depth][fx.name]
            for tag in _fixture_tags(fx):
                group_runs.setdefault(tag, []).extend(runs)
        per_group_depth[depth] = {
            tag: _aggregate(runs, bootstrap_count=bootstrap_count)
            for tag, runs in group_runs.items()
        }
        print(
            f"[status] depth {depth} summary "
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
            "per_group": per_group_depth,
        }
        checkpoint_path.write_text(json.dumps(final_checkpoint, indent=2))
    return summary_rows, per_fixture_depth, per_fixture_runs, per_group_depth


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


def _print_per_group(depth: int, stats: dict[str, dict]) -> None:
    print(f"\nPer-group metrics (depth={depth}):")
    print(
        "| group | avg_score | avg_max | survive% | reach2048% | "
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


def _collect_paired_deltas(
    *,
    baseline_runs: dict[int, dict[str, list[dict]]],
    candidate_runs: dict[int, dict[str, list[dict]]],
    fixtures: list[Fixture],
    depths: list[int],
    metric_key: str,
) -> list[float]:
    deltas: list[float] = []
    for d in depths:
        for fx in fixtures:
            b_runs = baseline_runs[d][fx.name]
            c_runs = candidate_runs[d][fx.name]
            for i in range(min(len(b_runs), len(c_runs))):
                deltas.append(c_runs[i][metric_key] - b_runs[i][metric_key])
    return deltas


def run_depth_calibration_legacy(n_moves: int, n_seeds: int, depths: list[int]) -> None:
    """Compatibility entrypoint used by benchmark_depth.py."""
    fixtures = _load_fixture_suite("depth_calibration")
    rows, per_fx, _, _ = evaluate_suite(
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
            "  .venv/bin/python tests/evaluator.py --suite powerup_late --depths 6,7 --seeds 12 --moves 80\n"
            "  .venv/bin/python tests/evaluator.py --suite fixtures --boards late_game,jammed --depths 4 --per-fixture\n"
        ),
    )
    parser.add_argument("--suite", choices=["fixtures", "depth_calibration", "powerup_late"], default="fixtures")
    parser.add_argument("--boards", default=None, help="Comma-separated fixture names to include (default: all)")
    parser.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS))
    parser.add_argument("--module", default=DEFAULT_MODULE, help=f"Strategy module to evaluate (default: {DEFAULT_MODULE})")
    parser.add_argument("--baseline-module", default=None, help="Baseline strategy module for module-vs-module comparison")
    parser.add_argument("--candidate-module", default=None, help="Candidate strategy module for module-vs-module comparison")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes for simulation runs (default: 1)")
    parser.add_argument(
        "--cache-mode",
        choices=["warm", "cold", "reset-per-run"],
        default="warm",
        help="Transposition cache mode: warm(default), cold(per-suite reset), reset-per-run",
    )
    parser.add_argument("--moves", type=int, default=DEFAULT_MOVES)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--per-fixture", action="store_true", help="Print per-fixture metrics for each depth")
    parser.add_argument("--per-group", action="store_true", help="Print per-group (fixture tag) metrics for each depth")
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
    parser.add_argument(
        "--fail-if-score-drop",
        type=float,
        default=None,
        help="Fail run if candidate mean paired score delta is below negative threshold (e.g. 10 => fail if delta < -10)",
    )
    parser.add_argument(
        "--fail-if-think-increase",
        type=float,
        default=None,
        help="Fail run if candidate mean paired think-ms delta exceeds threshold",
    )
    args = parser.parse_args()

    if args.moves <= 0:
        parser.error("--moves must be > 0")
    if args.jobs <= 0:
        parser.error("--jobs must be > 0")
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
    if args.fail_if_score_drop is not None and args.fail_if_score_drop < 0:
        parser.error("--fail-if-score-drop must be >= 0")
    if args.fail_if_think_increase is not None and args.fail_if_think_increase < 0:
        parser.error("--fail-if-think-increase must be >= 0")
    if args.resume and not args.jsonl_out:
        parser.error("--resume requires --jsonl-out")
    if args.candidate_module and args.resume:
        parser.error("--resume is not supported with --candidate-module compare mode")
    if (
        args.fail_if_score_drop is not None or args.fail_if_think_increase is not None
    ) and not (args.ab_depths or args.candidate_module):
        parser.error("guardrail flags require --ab-depths or --candidate-module compare mode")

    try:
        depths = _parse_depths(args.depths)
    except ValueError as e:
        parser.error(str(e))

    selected_names = _parse_names(args.boards)

    try:
        fixtures = _load_fixture_suite(args.suite, selected_names=selected_names)
    except ValueError as e:
        parser.error(str(e))

    # Validate strategy module(s) up front.
    try:
        _load_strategy_fns(args.module)
        if args.baseline_module:
            _load_strategy_fns(args.baseline_module)
        if args.candidate_module:
            _load_strategy_fns(args.candidate_module)
    except Exception as e:
        parser.error(f"failed to import strategy module(s): {e}")

    _print_metric_glossary()
    print()

    print(
        f"[status] starting evaluator: suite={args.suite} fixtures={len(fixtures)} "
        f"depths={depths} moves={args.moves} seeds={args.seeds} seed_start={args.seed_start} "
        f"module={args.module} jobs={args.jobs} cache_mode={args.cache_mode}"
    )

    # Manifest validation / setup for resume-safe runs.
    if args.jsonl_out:
        manifest = _build_manifest(
            suite=args.suite,
            boards=selected_names,
            depths=depths,
            moves=args.moves,
            seeds=args.seeds,
            seed_start=args.seed_start,
            no_random=args.no_random,
            cache_mode=args.cache_mode,
            fixtures=fixtures,
            module_name=args.module,
        )
        mpath = _manifest_path(args.jsonl_out)
        if args.resume:
            if not mpath.exists():
                parser.error(f"--resume requested but manifest missing: {mpath}")
            prior = json.loads(mpath.read_text())
            if prior != manifest:
                parser.error(
                    "--resume manifest mismatch; run config changed. "
                    f"expected={prior} current={manifest}"
                )
        else:
            mpath.parent.mkdir(parents=True, exist_ok=True)
            mpath.write_text(json.dumps(manifest, indent=2))

    rows, per_fx, per_runs, per_group = evaluate_suite(
        fixtures=fixtures,
        depths=depths,
        n_moves=args.moves,
        n_seeds=args.seeds,
        seed_start=args.seed_start,
        no_random=args.no_random,
        module_name=args.module,
        cache_mode=args.cache_mode,
        jobs=args.jobs,
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
    if args.per_group:
        for d in depths:
            _print_per_group(depth=d, stats=per_group[d])

    ab_result = None
    guardrail_context = None
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

        metric_key = args.ab_metric
        paired_deltas: list[float] = []
        score_deltas: list[float] = []
        think_deltas: list[float] = []
        for fx in fixtures:
            base_runs = per_runs[d_base][fx.name]
            cand_runs = per_runs[d_cand][fx.name]
            for i in range(min(len(base_runs), len(cand_runs))):
                paired_deltas.append(cand_runs[i][metric_key] - base_runs[i][metric_key])
                score_deltas.append(cand_runs[i]["score"] - base_runs[i]["score"])
                think_deltas.append(cand_runs[i]["think_ms_mean"] - base_runs[i]["think_ms_mean"])

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
            "mean_score_delta": sum(score_deltas) / len(score_deltas) if score_deltas else 0.0,
            "mean_think_ms_delta": sum(think_deltas) / len(think_deltas) if think_deltas else 0.0,
        }
        guardrail_context = {
            "label": f"depth {d_base} -> {d_cand}",
            "mean_score_delta": ab_result["mean_score_delta"],
            "mean_think_ms_delta": ab_result["mean_think_ms_delta"],
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
        print(
            f"  guardrail_deltas: score={ab_result['mean_score_delta']:+.2f} "
            f"think_ms={ab_result['mean_think_ms_delta']:+.3f}"
        )

    module_compare_result = None
    if args.candidate_module:
        baseline_module = args.baseline_module or args.module
        print(
            f"\n[status] module compare: baseline={baseline_module} "
            f"candidate={args.candidate_module}"
        )
        if baseline_module == args.module:
            base_rows = rows
            base_runs = per_runs
        else:
            base_rows, _, base_runs, _ = evaluate_suite(
                fixtures=fixtures,
                depths=depths,
                n_moves=args.moves,
                n_seeds=args.seeds,
                seed_start=args.seed_start,
                no_random=args.no_random,
                module_name=baseline_module,
                cache_mode=args.cache_mode,
                jobs=args.jobs,
                progress_every=args.progress_every,
                bootstrap_count=args.bootstraps,
                jsonl_out=None,
                resume=False,
                checkpoint_out=None,
                checkpoint_every=args.checkpoint_every,
            )
        cand_rows, _, cand_runs, _ = evaluate_suite(
            fixtures=fixtures,
            depths=depths,
            n_moves=args.moves,
            n_seeds=args.seeds,
            seed_start=args.seed_start,
            no_random=args.no_random,
            module_name=args.candidate_module,
            cache_mode=args.cache_mode,
            jobs=args.jobs,
            progress_every=args.progress_every,
            bootstrap_count=args.bootstraps,
            jsonl_out=None,
            resume=False,
            checkpoint_out=None,
            checkpoint_every=args.checkpoint_every,
        )
        print()
        _print_summary_table(base_rows, title=f"Baseline Module Summary ({baseline_module})")
        print()
        _print_summary_table(cand_rows, title=f"Candidate Module Summary ({args.candidate_module})")

        metric_key = args.ab_metric
        deltas = _collect_paired_deltas(
            baseline_runs=base_runs,
            candidate_runs=cand_runs,
            fixtures=fixtures,
            depths=depths,
            metric_key=metric_key,
        )
        score_deltas = _collect_paired_deltas(
            baseline_runs=base_runs,
            candidate_runs=cand_runs,
            fixtures=fixtures,
            depths=depths,
            metric_key="score",
        )
        think_deltas = _collect_paired_deltas(
            baseline_runs=base_runs,
            candidate_runs=cand_runs,
            fixtures=fixtures,
            depths=depths,
            metric_key="think_ms_mean",
        )
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        ci_lo, ci_hi = _bootstrap_mean_ci(deltas, n_bootstrap=args.bootstraps, seed=args.seed_start + 41)
        p_value = _paired_permutation_test(
            deltas,
            n_permutations=args.ab_permutations,
            seed=args.seed_start + 53,
        )
        win_rate = sum(1 for x in deltas if x > 0) * 100.0 / max(1, len(deltas))
        loss_rate = sum(1 for x in deltas if x < 0) * 100.0 / max(1, len(deltas))
        module_compare_result = {
            "baseline_module": baseline_module,
            "candidate_module": args.candidate_module,
            "metric": metric_key,
            "paired_samples": len(deltas),
            "mean_delta": mean_delta,
            "mean_delta_ci95": [ci_lo, ci_hi],
            "win_rate_pct": win_rate,
            "loss_rate_pct": loss_rate,
            "p_value": p_value,
            "permutations": args.ab_permutations,
            "mean_score_delta": sum(score_deltas) / len(score_deltas) if score_deltas else 0.0,
            "mean_think_ms_delta": sum(think_deltas) / len(think_deltas) if think_deltas else 0.0,
        }
        guardrail_context = {
            "label": f"module {baseline_module} -> {args.candidate_module}",
            "mean_score_delta": module_compare_result["mean_score_delta"],
            "mean_think_ms_delta": module_compare_result["mean_think_ms_delta"],
        }
        print("\nModule A/B Comparison")
        print(
            f"  baseline={baseline_module} candidate={args.candidate_module} "
            f"metric={metric_key} samples={len(deltas)}"
        )
        print(
            f"  mean_delta={mean_delta:+.2f} "
            f"ci95=[{ci_lo:+.2f}, {ci_hi:+.2f}] "
            f"win%={win_rate:.1f} loss%={loss_rate:.1f} p={p_value:.4f}"
        )
        print(
            f"  guardrail_deltas: score={module_compare_result['mean_score_delta']:+.2f} "
            f"think_ms={module_compare_result['mean_think_ms_delta']:+.3f}"
        )

    guardrail_failures: list[str] = []
    if guardrail_context is not None:
        if args.fail_if_score_drop is not None:
            if guardrail_context["mean_score_delta"] < -args.fail_if_score_drop:
                guardrail_failures.append(
                    f"score delta {guardrail_context['mean_score_delta']:+.2f} < -{args.fail_if_score_drop:.2f}"
                )
        if args.fail_if_think_increase is not None:
            if guardrail_context["mean_think_ms_delta"] > args.fail_if_think_increase:
                guardrail_failures.append(
                    f"think_ms delta {guardrail_context['mean_think_ms_delta']:+.3f} > {args.fail_if_think_increase:.3f}"
                )
        if guardrail_failures:
            print(f"\n[guardrail] FAILED ({guardrail_context['label']})")
            for msg in guardrail_failures:
                print(f"[guardrail] {msg}")
        elif args.fail_if_score_drop is not None or args.fail_if_think_increase is not None:
            print(f"\n[guardrail] PASSED ({guardrail_context['label']})")

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
                "module": args.module,
                "jobs": args.jobs,
                "cache_mode": args.cache_mode,
                "baseline_module": args.baseline_module,
                "candidate_module": args.candidate_module,
                "bootstraps": args.bootstraps,
                "progress_every": args.progress_every,
                "ab_depths": args.ab_depths,
                "ab_metric": args.ab_metric,
                "ab_permutations": args.ab_permutations,
                "fail_if_score_drop": args.fail_if_score_drop,
                "fail_if_think_increase": args.fail_if_think_increase,
                "resume": args.resume,
                "checkpoint_out": args.checkpoint_out,
                "checkpoint_every": args.checkpoint_every,
            },
            "summary": rows,
            "per_fixture": per_fx,
            "per_group": per_group,
            "ab_result": ab_result,
            "module_compare_result": module_compare_result,
            "guardrail": {
                "context": guardrail_context,
                "failures": guardrail_failures,
                "passed": len(guardrail_failures) == 0 if guardrail_context is not None else None,
            },
            "jsonl_out": args.jsonl_out,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON results: {args.json_out}")

    if guardrail_failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
