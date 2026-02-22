#!/usr/bin/env python3
"""Sweep powerup-value weight on late-game powerup fixtures.

Runs:
1) Baseline summary (module=strategy)
2) Candidate summary for each multiplier (module=strategy_powerup_weighted)
3) Evaluator module-vs-module A/B compare for each multiplier

Focus metrics:
- reach4096_pct
- reach8192_pct
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
EVALUATOR = ROOT / "tests" / "evaluator.py"
PYTHON = ROOT / ".venv" / "bin" / "python"

PRESETS: dict[str, list[float]] = {
    "conservative": [0.6, 0.8, 1.0],
    "aggressive": [1.0, 1.5, 2.0],
    "full": [0.5, 0.8, 1.0, 1.25, 1.5, 2.0],
}


def _run(cmd: list[str], env: dict[str, str]) -> None:
    proc = subprocess.run(cmd, env=env, cwd=ROOT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _summary_for(
    *,
    module: str,
    weight_mult: float | None,
    suite: str,
    depths: str,
    seeds: int,
    moves: int,
) -> dict:
    with tempfile.NamedTemporaryFile(prefix="powerup_summary_", suffix=".json", delete=False) as tf:
        json_out = tf.name
    try:
        env = os.environ.copy()
        if weight_mult is not None:
            env["POWERUP_WEIGHT_MULT"] = str(weight_mult)
        cmd = [
            str(PYTHON),
            str(EVALUATOR),
            "--suite",
            suite,
            "--depths",
            depths,
            "--seeds",
            str(seeds),
            "--moves",
            str(moves),
            "--module",
            module,
            "--json-out",
            json_out,
        ]
        _run(cmd, env)
        payload = json.loads(Path(json_out).read_text())
        rows = payload.get("summary", [])
        if not rows:
            raise RuntimeError("missing summary rows")
        # Use deepest depth row.
        row = sorted(rows, key=lambda r: r["depth"])[-1]
        return row
    finally:
        try:
            Path(json_out).unlink(missing_ok=True)
        except Exception:
            pass


def _ab_compare_max_tile(
    *,
    weight_mult: float,
    suite: str,
    depths: str,
    seeds: int,
    moves: int,
) -> dict:
    with tempfile.NamedTemporaryFile(prefix="powerup_ab_", suffix=".json", delete=False) as tf:
        json_out = tf.name
    try:
        env = os.environ.copy()
        env["POWERUP_WEIGHT_MULT"] = str(weight_mult)
        cmd = [
            str(PYTHON),
            str(EVALUATOR),
            "--suite",
            suite,
            "--depths",
            depths,
            "--seeds",
            str(seeds),
            "--moves",
            str(moves),
            "--module",
            "strategy",
            "--baseline-module",
            "strategy",
            "--candidate-module",
            "strategy_powerup_weighted",
            "--ab-metric",
            "max_tile",
            "--json-out",
            json_out,
        ]
        _run(cmd, env)
        payload = json.loads(Path(json_out).read_text())
        out = payload.get("module_compare_result", {}) or {}
        return {
            "mean_delta": float(out.get("mean_delta", 0.0)),
            "p_value": float(out.get("p_value", 1.0)),
        }
    finally:
        try:
            Path(json_out).unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Powerup-value sweep on late-game fixtures.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="aggressive")
    parser.add_argument("--suite", default="powerup_late")
    parser.add_argument("--depths", default="7")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--moves", type=int, default=60)
    args = parser.parse_args()

    mults = PRESETS[args.preset]
    print(
        f"[sweep] preset={args.preset} multipliers={mults} "
        f"suite={args.suite} depths={args.depths} seeds={args.seeds} moves={args.moves}"
    )
    base = _summary_for(
        module="strategy",
        weight_mult=None,
        suite=args.suite,
        depths=args.depths,
        seeds=args.seeds,
        moves=args.moves,
    )
    print(
        f"[baseline] reach4096={base['reach4096_pct']:.1f}% "
        f"reach8192={base['reach8192_pct']:.1f}% "
        f"avg_score={base['avg_score']:.1f} avg_think_ms={base['avg_think_ms']:.2f}"
    )
    print(
        "| mult | reach4096% | reach8192% | delta4096 | delta8192 | avg_score | avg_think_ms | "
        "ab_mean_delta(max_tile) | ab_p |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mult in mults:
        cand = _summary_for(
            module="strategy_powerup_weighted",
            weight_mult=mult,
            suite=args.suite,
            depths=args.depths,
            seeds=args.seeds,
            moves=args.moves,
        )
        ab = _ab_compare_max_tile(
            weight_mult=mult,
            suite=args.suite,
            depths=args.depths,
            seeds=args.seeds,
            moves=args.moves,
        )
        print(
            f"| {mult:.2f} | {cand['reach4096_pct']:.1f} | {cand['reach8192_pct']:.1f} | "
            f"{cand['reach4096_pct'] - base['reach4096_pct']:+.1f} | "
            f"{cand['reach8192_pct'] - base['reach8192_pct']:+.1f} | "
            f"{cand['avg_score']:.1f} | {cand['avg_think_ms']:.2f} | "
            f"{ab['mean_delta']:+.2f} | {ab['p_value']:.4f} |"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        raise
