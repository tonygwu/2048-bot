#!/usr/bin/env python3
"""
Benchmark tests/run.py across interpreters and report speedups.

This measures:
1) summed per-move "think time" reported by run.py (best_action timing)
2) total wall-clock runtime per harness run
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import time
from pathlib import Path


THINK_MS_PATTERN = re.compile(r"\[(\d+) ms\]")


def run_once(
    root: Path,
    run_py: Path,
    interpreter: str,
    board: str,
    depth: int,
    moves: int,
    seed: int,
) -> tuple[int, float, int]:
    cmd = [
        interpreter,
        str(run_py),
        board,
        "--depth",
        str(depth),
        "--moves",
        str(moves),
        "--seed",
        str(seed),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    think_values = [int(v) for v in THINK_MS_PATTERN.findall(proc.stdout)]
    return sum(think_values), wall_s, len(think_values)


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPython vs PyPy on tests/run.py.")
    parser.add_argument("--board", default="mid_game", help="Fixture board name/path (default: mid_game)")
    parser.add_argument("--moves", type=int, default=80, help="Moves per run (default: 80)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--repeats", type=int, default=5, help="Runs per depth/interpreter (default: 5)")
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[4, 5],
        help="Depths to benchmark (default: 4 5)",
    )
    parser.add_argument("--cpython", default="python3", help="CPython executable (default: python3)")
    parser.add_argument("--pypy", default="pypy3.10", help="PyPy executable (default: pypy3.10)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    run_py = root / "tests" / "run.py"

    interpreters = [
        ("CPython", args.cpython),
        ("PyPy", args.pypy),
    ]

    data: dict[str, dict[int, dict[str, float]]] = {}
    for label, exe in interpreters:
        data[label] = {}
        for depth in args.depths:
            think_sums: list[float] = []
            walls: list[float] = []
            moves_seen: list[float] = []
            for _ in range(args.repeats):
                think_ms, wall_s, move_count = run_once(
                    root=root,
                    run_py=run_py,
                    interpreter=exe,
                    board=args.board,
                    depth=depth,
                    moves=args.moves,
                    seed=args.seed,
                )
                think_sums.append(float(think_ms))
                walls.append(wall_s)
                moves_seen.append(float(move_count))
            data[label][depth] = {
                "moves_mean": mean(moves_seen),
                "think_ms_mean": mean(think_sums),
                "think_ms_std": pstdev(think_sums),
                "wall_s_mean": mean(walls),
                "wall_s_std": pstdev(walls),
            }

    print(
        f"Board={args.board} Moves={args.moves} Seed={args.seed} "
        f"Repeats={args.repeats}"
    )
    print()
    print(
        "| Depth | CPython think ms | PyPy think ms | Think speedup | "
        "CPython wall s | PyPy wall s | Wall speedup |"
    )
    print(
        "|---:|---:|---:|---:|---:|---:|---:|"
    )
    for depth in args.depths:
        cp = data["CPython"][depth]
        pp = data["PyPy"][depth]
        think_speedup = cp["think_ms_mean"] / pp["think_ms_mean"] if pp["think_ms_mean"] else 0.0
        wall_speedup = cp["wall_s_mean"] / pp["wall_s_mean"] if pp["wall_s_mean"] else 0.0
        print(
            f"| {depth} | {cp['think_ms_mean']:.1f} ±{cp['think_ms_std']:.1f} | "
            f"{pp['think_ms_mean']:.1f} ±{pp['think_ms_std']:.1f} | "
            f"{think_speedup:.3f}x | "
            f"{cp['wall_s_mean']:.3f} ±{cp['wall_s_std']:.3f} | "
            f"{pp['wall_s_mean']:.3f} ±{pp['wall_s_std']:.3f} | "
            f"{wall_speedup:.3f}x |"
        )


if __name__ == "__main__":
    main()
