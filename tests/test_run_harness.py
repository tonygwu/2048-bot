#!/usr/bin/env python3
"""Unit tests for tests/run.py harness behavior."""

import importlib.util
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).parent.parent
RUN_PATH = ROOT / "tests" / "run.py"


def _load_run_module():
    spec = importlib.util.spec_from_file_location("run_module_for_tests", RUN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_mod = _load_run_module()


class TestRunHarness(unittest.TestCase):
    def test_direction_scores_passes_powers_to_expectimax(self) -> None:
        board = [
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        powers = {"undo": 1, "swap": 2, "delete": 1}
        seen: list[dict | None] = []

        def fake_expectimax(_board, _depth, _is_max, powers=None):
            seen.append(powers)
            return 42.0

        with mock.patch.object(run_mod, "_expectimax", side_effect=fake_expectimax):
            scores = run_mod.direction_scores(board, depth=3, powers=powers)

        self.assertTrue(any(val is not None for _, val in scores))
        self.assertTrue(seen)
        self.assertTrue(all(s == powers for s in seen))

    def test_apply_action_rejects_unknown_action_type(self) -> None:
        board = [[0, 0, 0, 0] for _ in range(4)]
        with self.assertRaises(ValueError):
            run_mod.apply_action(
                board,
                0,
                {"undo": 0, "swap": 0, "delete": 0},
                ("undo",),
                rng=random.Random(0),
                no_random=True,
            )

    def test_apply_action_rejects_invalid_move(self) -> None:
        board = [[2, 4, 8, 16], [32, 64, 128, 256], [2, 4, 8, 16], [32, 64, 128, 256]]
        with self.assertRaises(ValueError):
            run_mod.apply_action(
                board,
                0,
                {"undo": 0, "swap": 0, "delete": 0},
                ("move", "left"),
                rng=random.Random(0),
                no_random=True,
            )

    def test_move_recharges_delete_on_new_512(self) -> None:
        board = [
            [256, 256, 256, 256],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        _, _, powers = run_mod.apply_action(
            board,
            0,
            {"undo": 0, "swap": 0, "delete": 0},
            ("move", "left"),
            rng=random.Random(0),
            no_random=True,
        )
        self.assertEqual(powers["delete"], 2)

    def test_move_delete_recharge_respects_bank_cap(self) -> None:
        board = [
            [256, 256, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        _, _, powers = run_mod.apply_action(
            board,
            0,
            {"undo": 0, "swap": 0, "delete": 2},
            ("move", "left"),
            rng=random.Random(0),
            no_random=True,
        )
        self.assertEqual(powers["delete"], 2)

    def test_swap_does_not_spawn_random_tile(self) -> None:
        board = [
            [4, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        next_board, _, powers = run_mod.apply_action(
            board,
            0,
            {"undo": 0, "swap": 1, "delete": 0},
            ("swap", 0, 0, 0, 1),
            rng=random.Random(0),
            no_random=False,
        )
        self.assertEqual(next_board[0][0], 2)
        self.assertEqual(next_board[0][1], 4)
        self.assertEqual(sum(1 for r in range(4) for c in range(4) if next_board[r][c] > 0), 2)
        self.assertEqual(powers["swap"], 0)

    def test_delete_does_not_spawn_random_tile(self) -> None:
        board = [
            [2, 4, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        next_board, _, powers = run_mod.apply_action(
            board,
            0,
            {"undo": 0, "swap": 0, "delete": 1},
            ("delete", 2, 0, 0),
            rng=random.Random(0),
            no_random=False,
        )
        self.assertEqual(next_board[0][0], 0)
        self.assertEqual(next_board[0][2], 0)
        self.assertEqual(sum(1 for r in range(4) for c in range(4) if next_board[r][c] > 0), 1)
        self.assertEqual(powers["delete"], 0)

    def test_move_planned_eval_uses_expectimax_projection(self) -> None:
        fixture = {
            "name": "plan_eval_fixture",
            "description": "move planned eval alignment",
            "board": [
                [2, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            "score": 0,
            "powers": {"undo": 0, "swap": 0, "delete": 0},
        }
        seen_planned: list[float | None] = []

        def fake_analyze_undo(*, planned_eval=None, **_kwargs):
            seen_planned.append(planned_eval)
            return SimpleNamespace(should_undo=False)

        with mock.patch.object(run_mod, "best_action", return_value=("move", "left")):
            with mock.patch.object(run_mod, "_expectimax", return_value=123.0):
                with mock.patch.object(run_mod, "analyze_undo", side_effect=fake_analyze_undo):
                    run_mod.run(
                        fixture=fixture,
                        num_moves=1,
                        fixed_depth=3,
                        rng=random.Random(0),
                        show_scores=False,
                        peek=False,
                        no_random=True,
                    )

        self.assertTrue(seen_planned)
        # 2+2 merge yields +4 score delta, so planned_eval should be 123 + 4.
        self.assertAlmostEqual(float(seen_planned[0]), 127.0, places=6)

    def test_run_uses_power_aware_static_eval(self) -> None:
        fixture = {
            "name": "unit_fixture",
            "description": "test",
            "board": [
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            "score": 0,
            "powers": {"undo": 1, "swap": 1, "delete": 1},
        }
        seen_powers: list[dict | None] = []

        def fake_score(_board, powers=None):
            seen_powers.append(powers)
            return 0.0

        with mock.patch.object(run_mod, "score_board", side_effect=fake_score):
            with mock.patch.object(run_mod, "best_action", return_value=None):
                with mock.patch.object(run_mod, "is_game_over", return_value=False):
                    run_mod.run(
                        fixture=fixture,
                        num_moves=1,
                        fixed_depth=4,
                        rng=random.Random(0),
                        show_scores=False,
                        peek=False,
                        no_random=True,
                    )

        self.assertTrue(seen_powers)
        self.assertEqual(seen_powers[0], fixture["powers"])


if __name__ == "__main__":
    unittest.main()
