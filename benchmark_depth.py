"""
benchmark_depth.py — Backwards-compatible wrapper around tests/evaluator.py.

This script is kept for compatibility with existing workflows.
It delegates to the canonical evaluator so depth studies and heuristic
studies use the same metric definitions.

DEPRECATION NOTE:
  Use tests/evaluator.py directly for all new experiments.
  Keep this wrapper only for backward-compatible command usage.
"""

from __future__ import annotations

import argparse

from tests.evaluator import run_depth_calibration_legacy

DEFAULT_N_MOVES = 25
DEFAULT_N_SEEDS = 5
DEFAULT_DEPTHS = [2, 3, 4, 5]


def _parse_depths(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) < 2:
        raise ValueError("--depths must contain at least two depths")
    if any(v <= 0 for v in vals):
        raise ValueError("--depths values must be positive integers")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark expectimax depth tradeoffs.")
    parser.add_argument("--moves", type=int, default=DEFAULT_N_MOVES, help=f"Moves per simulation (default: {DEFAULT_N_MOVES})")
    parser.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS, help=f"Random seeds per board/depth (default: {DEFAULT_N_SEEDS})")
    parser.add_argument(
        "--depths",
        default=",".join(str(d) for d in DEFAULT_DEPTHS),
        help=f"Comma-separated depth list (default: {','.join(str(d) for d in DEFAULT_DEPTHS)})",
    )
    args = parser.parse_args()

    if args.moves <= 0:
        parser.error("--moves must be > 0")
    if args.seeds <= 0:
        parser.error("--seeds must be > 0")

    try:
        depths = _parse_depths(args.depths)
    except ValueError as e:
        parser.error(str(e))

    run_depth_calibration_legacy(n_moves=args.moves, n_seeds=args.seeds, depths=depths)


if __name__ == "__main__":
    main()
