#!/usr/bin/env python3
"""Unit tests for evaluator helpers."""

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parent.parent
EVALUATOR_PATH = ROOT / "tests" / "evaluator.py"


def _load_evaluator_module():
    spec = importlib.util.spec_from_file_location("eval_module_for_tests", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load_evaluator_module()


class TestEvaluatorHelpers(unittest.TestCase):
    def test_fixture_tags_cover_expected_groups(self) -> None:
        fixtures = [
            evaluator.Fixture(name="open", board=[[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], score=0, powers={}),
            evaluator.Fixture(name="jammed", board=[[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2, 4], [8, 16, 0, 0]], score=0, powers={}),
            evaluator.Fixture(name="late", board=[[2048, 64, 32, 16], [8, 4, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]], score=0, powers={}),
            evaluator.Fixture(name="powerup", board=[[2, 4, 8, 16], [32, 64, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], score=0, powers={"swap": 1}),
            evaluator.Fixture(name="mid", board=[[2, 4, 8, 16], [32, 64, 0, 0], [128, 256, 0, 0], [512, 1024, 0, 0]], score=0, powers={}),
        ]
        tags = {fx.name: evaluator._fixture_tags(fx) for fx in fixtures}
        self.assertIn("open", tags["open"])
        self.assertIn("jammed", tags["jammed"])
        self.assertIn("late", tags["late"])
        self.assertIn("powerup", tags["powerup"])
        self.assertEqual(tags["mid"], ["mid"])

    def test_manifest_includes_cache_mode(self) -> None:
        fx = evaluator.Fixture(
            name="f1",
            board=[[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            score=0,
            powers={},
        )
        manifest = evaluator._build_manifest(
            suite="fixtures",
            boards=None,
            depths=[3, 4],
            moves=10,
            seeds=2,
            seed_start=0,
            no_random=False,
            cache_mode="reset-per-run",
            fixtures=[fx],
            module_name="strategy",
        )
        self.assertEqual(manifest["cache_mode"], "reset-per-run")

    def test_collect_paired_deltas(self) -> None:
        fixture = evaluator.Fixture(
            name="fx",
            board=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            score=0,
            powers={},
        )
        baseline = {3: {"fx": [{"score": 100, "think_ms_mean": 1.0}, {"score": 120, "think_ms_mean": 1.5}]}}
        candidate = {3: {"fx": [{"score": 110, "think_ms_mean": 1.2}, {"score": 90, "think_ms_mean": 1.6}]}}
        score_deltas = evaluator._collect_paired_deltas(
            baseline_runs=baseline,
            candidate_runs=candidate,
            fixtures=[fixture],
            depths=[3],
            metric_key="score",
        )
        think_deltas = evaluator._collect_paired_deltas(
            baseline_runs=baseline,
            candidate_runs=candidate,
            fixtures=[fixture],
            depths=[3],
            metric_key="think_ms_mean",
        )
        self.assertEqual(score_deltas, [10, -30])
        self.assertAlmostEqual(think_deltas[0], 0.2, places=9)
        self.assertAlmostEqual(think_deltas[1], 0.1, places=9)

    def test_aggregate_includes_promotion_and_undo_fp_metrics(self) -> None:
        runs = [
            {
                "moves": 10,
                "actions": {"move": 8, "swap": 0, "delete": 0, "undo": 2},
                "think_ms_samples": [1.0, 2.0],
                "cache_hits": 2,
                "cache_misses": 1,
                "undo_used": 2,
                "undo_early_used": 1,
                "undo_plan_gap_only_used": 1,
                "undo_plan_gap_false_positive_used": 1,
                "undo_successes": 1,
                "undo_avg_immediate_recovery": 50.0,
                "score": 1000,
                "max_tile": 8192,
                "second_max_tile": 4096,
                "initial_second_max_tile": 2048,
                "peak_second_max_tile": 4096,
                "second_max_gain": 2048,
                "second_log_gain": 1.0,
                "promotion_stalled": False,
                "reach2048": True,
                "reach4096": True,
                "reach8192": True,
                "reach16384": False,
                "second_ge4096": True,
                "second_ge8192": False,
                "peak_second_ge4096": True,
                "peak_second_ge8192": False,
                "survived_to_cap": False,
                "final_eval": 123.0,
                "think_ms_total": 20.0,
            }
        ]
        agg = evaluator._aggregate(runs, bootstrap_count=0)
        self.assertAlmostEqual(agg["avg_second_gain"], 2048.0, places=9)
        self.assertAlmostEqual(agg["promotion_stall_pct"], 0.0, places=9)
        self.assertAlmostEqual(agg["peak_second_ge4096_pct"], 100.0, places=9)
        self.assertAlmostEqual(agg["undo_plan_gap_false_positive_rate_pct"], 50.0, places=9)


if __name__ == "__main__":
    unittest.main()
