#!/usr/bin/env python3
"""Unit tests for evaluator helpers."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


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
    @staticmethod
    def _make_run(
        *,
        max_tile: int,
        initial_second_max_tile: int,
        second_max_tile: int,
        peak_second_max_tile: int | None = None,
    ) -> dict:
        peak = second_max_tile if peak_second_max_tile is None else peak_second_max_tile
        second_gain = second_max_tile - initial_second_max_tile
        return {
            "moves": 10,
            "actions": {"move": 10, "swap": 0, "delete": 0, "undo": 0},
            "think_ms_samples": [1.0],
            "cache_hits": 0,
            "cache_misses": 0,
            "undo_used": 0,
            "undo_early_used": 0,
            "undo_plan_gap_only_used": 0,
            "undo_plan_gap_false_positive_used": 0,
            "undo_successes": 0,
            "undo_avg_immediate_recovery": 0.0,
            "score": 1000,
            "max_tile": max_tile,
            "second_max_tile": second_max_tile,
            "initial_second_max_tile": initial_second_max_tile,
            "peak_second_max_tile": peak,
            "second_max_gain": second_gain,
            "second_log_gain": 0.0,
            "promotion_stalled": second_gain <= 0,
            "reach2048": max_tile >= 2048,
            "reach4096": max_tile >= 4096,
            "reach8192": max_tile >= 8192,
            "reach16384": max_tile >= 16384,
            "second_ge4096": second_max_tile >= 4096,
            "second_ge8192": second_max_tile >= 8192,
            "peak_second_ge4096": peak >= 4096,
            "peak_second_ge8192": peak >= 8192,
            "survived_to_cap": True,
            "final_eval": 123.0,
            "think_ms_total": 10.0,
        }

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
        self.assertAlmostEqual(agg["promote1024_pct"], 0.0, places=9)
        self.assertAlmostEqual(agg["promote2048_pct"], 0.0, places=9)
        self.assertAlmostEqual(agg["promote4096_pct"], 100.0, places=9)
        self.assertAlmostEqual(agg["peak_second_ge4096_pct"], 100.0, places=9)
        self.assertAlmostEqual(agg["undo_plan_gap_false_positive_rate_pct"], 50.0, places=9)

    def test_promote_metrics_no_credit_for_static_threshold_tile(self) -> None:
        # A threshold tile exists (as max tile), but second-highest never crosses
        # the threshold during the run, so no promotion credit should be given.
        cases = [
            (1024, 512, "promote1024_pct"),
            (2048, 1024, "promote2048_pct"),
            (4096, 2048, "promote4096_pct"),
        ]
        for max_tile, second_tile, metric_key in cases:
            with self.subTest(metric=metric_key):
                run = self._make_run(
                    max_tile=max_tile,
                    initial_second_max_tile=second_tile,
                    second_max_tile=second_tile,
                    peak_second_max_tile=second_tile,
                )
                agg = evaluator._aggregate([run], bootstrap_count=0)
                self.assertAlmostEqual(agg[metric_key], 0.0, places=9)

    def test_promote_metrics_credit_active_crossing_with_existing_tile(self) -> None:
        # A threshold tile already exists (as max tile), and the policy creates
        # another one so second-highest crosses the threshold during gameplay.
        cases = [
            (1024, 512, "promote1024_pct"),
            (2048, 1024, "promote2048_pct"),
            (4096, 2048, "promote4096_pct"),
        ]
        for threshold, initial_second, metric_key in cases:
            with self.subTest(metric=metric_key):
                run = self._make_run(
                    max_tile=threshold,
                    initial_second_max_tile=initial_second,
                    second_max_tile=threshold,
                    peak_second_max_tile=threshold,
                )
                agg = evaluator._aggregate([run], bootstrap_count=0)
                self.assertAlmostEqual(agg[metric_key], 100.0, places=9)

    def test_simulate_move_planned_eval_uses_expectimax_projection(self) -> None:
        fixture = evaluator.Fixture(
            name="plan_eval_fixture",
            board=[[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            score=0,
            powers={"undo": 0, "swap": 0, "delete": 0},
        )
        seen_planned: list[float | None] = []

        fake_fns = evaluator.StrategyFns(
            apply_delete=lambda board, _value: [row[:] for row in board],
            apply_move=lambda board, direction: (
                ([[4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 4, True)
                if direction == "left"
                else ([row[:] for row in board], 0, False)
            ),
            apply_swap=lambda board, _r1, _c1, _r2, _c2: [row[:] for row in board],
            best_action=lambda _board, _powers, depth=0: ("move", "left"),
            expectimax=lambda _board, _depth, _is_max, _powers=None: 50.0,
            get_trans_stats=lambda: {"hits": 0, "misses": 0},
            is_game_over=lambda _board: False,
            reset_trans_cache=lambda: None,
            reset_trans_stats=lambda: None,
            score_board=lambda board, _powers=None: float(sum(v for row in board for v in row)),
        )

        def fake_analyze_undo(*, planned_eval=None, **_kwargs):
            seen_planned.append(planned_eval)
            return SimpleNamespace(
                should_undo=False,
                reasons=(),
                eval_drop=0.0,
                eval_drop_ratio=0.0,
                plan_gap=0.0,
                plan_gap_ratio=0.0,
                drop_trigger=0.0,
                gap_trigger=0.0,
                pressure=0.0,
                eval_before=0.0,
                eval_after=0.0,
            )

        with mock.patch.object(evaluator, "_load_strategy_fns", return_value=fake_fns):
            with mock.patch.object(evaluator, "analyze_undo", side_effect=fake_analyze_undo):
                evaluator._simulate_one(
                    fixture=fixture,
                    module_name="strategy",
                    depth=3,
                    n_moves=1,
                    seed=0,
                    no_random=True,
                    cache_mode="warm",
                )

        self.assertTrue(seen_planned)
        # planned_eval = expectimax(50) + score_delta(4)
        self.assertAlmostEqual(float(seen_planned[0]), 54.0, places=6)

    def test_simulate_fallback_after_undo_uses_expectimax_when_available(self) -> None:
        fixture = evaluator.Fixture(
            name="fallback_fixture",
            board=[[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            score=0,
            powers={"undo": 1, "swap": 0, "delete": 0},
        )
        def sentinel_expectimax(_board, _depth, _is_max, _powers=None):
            return 0.0
        seen_expectimax_arg: list[object | None] = []
        call_count = {"undo": 0}

        fake_fns = evaluator.StrategyFns(
            apply_delete=lambda board, _value: [row[:] for row in board],
            apply_move=lambda board, direction: (
                ([[4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 4, True)
                if direction == "left"
                else ([row[:] for row in board], 0, False)
            ),
            apply_swap=lambda board, _r1, _c1, _r2, _c2: [row[:] for row in board],
            best_action=lambda _board, _powers, depth=0: ("move", "left"),
            expectimax=sentinel_expectimax,
            get_trans_stats=lambda: {"hits": 0, "misses": 0},
            is_game_over=lambda _board: False,
            reset_trans_cache=lambda: None,
            reset_trans_stats=lambda: None,
            score_board=lambda board, _powers=None: float(sum(v for row in board for v in row)),
        )

        def fake_analyze_undo(**_kwargs):
            call_count["undo"] += 1
            if call_count["undo"] == 1:
                return SimpleNamespace(
                    should_undo=True,
                    reasons=("eval_drop",),
                    eval_drop=500.0,
                    eval_drop_ratio=0.5,
                    plan_gap=0.0,
                    plan_gap_ratio=0.0,
                    drop_trigger=100.0,
                    gap_trigger=100.0,
                    pressure=0.0,
                    eval_before=10.0,
                    eval_after=0.0,
                )
            return SimpleNamespace(
                should_undo=False,
                reasons=(),
                eval_drop=0.0,
                eval_drop_ratio=0.0,
                plan_gap=0.0,
                plan_gap_ratio=0.0,
                drop_trigger=0.0,
                gap_trigger=0.0,
                pressure=0.0,
                eval_before=0.0,
                eval_after=0.0,
            )

        def fake_best_fallback_move(*, expectimax_fn=None, **_kwargs):
            seen_expectimax_arg.append(expectimax_fn)
            return ("move", "left")

        with mock.patch.object(evaluator, "_load_strategy_fns", return_value=fake_fns):
            with mock.patch.object(evaluator, "analyze_undo", side_effect=fake_analyze_undo):
                with mock.patch.object(evaluator, "best_fallback_move", side_effect=fake_best_fallback_move):
                    evaluator._simulate_one(
                        fixture=fixture,
                        module_name="strategy",
                        depth=3,
                        n_moves=3,
                        seed=0,
                        no_random=True,
                        cache_mode="warm",
                    )

        self.assertTrue(seen_expectimax_arg)
        self.assertIs(seen_expectimax_arg[0], sentinel_expectimax)


if __name__ == "__main__":
    unittest.main()
