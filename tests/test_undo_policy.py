#!/usr/bin/env python3
"""Unit tests for undo policy triggers and fallback move selection."""

import json
import unittest
from pathlib import Path

from strategy import apply_move, score_board
from undo_policy import analyze_undo, best_fallback_move


BOARDS_DIR = Path(__file__).parent / "boards"


def _key(board: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in board)


def _load_board(name: str) -> list[list[int]]:
    data = json.loads((BOARDS_DIR / f"{name}.json").read_text())
    return data["board"]


class TestUndoPolicy(unittest.TestCase):
    def test_triggers_undo_on_large_eval_drop(self) -> None:
        board_before = _load_board("mid_game")
        board_after = _load_board("corner_trap")
        mapping = {
            _key(board_before): 5200.0,
            _key(board_after): 4550.0,
        }

        def score_fn(board, _powers):
            return mapping[_key(board)]

        decision = analyze_undo(
            board_before=board_before,
            powers_before={"undo": 1, "swap": 0, "delete": 0},
            board_after=board_after,
            powers_after={"undo": 1, "swap": 0, "delete": 0},
            planned_eval=5100.0,
            score_board_fn=score_fn,
            apply_move_fn=apply_move,
        )
        self.assertTrue(decision.should_undo)
        self.assertIn("eval_drop", decision.reasons)

    def test_triggers_undo_on_large_plan_gap(self) -> None:
        board_before = _load_board("mid_game")
        board_after = _load_board("corner_trap")
        mapping = {
            _key(board_before): 5000.0,
            _key(board_after): 4970.0,
        }

        def score_fn(board, _powers):
            return mapping[_key(board)]

        decision = analyze_undo(
            board_before=board_before,
            powers_before={"undo": 1, "swap": 0, "delete": 0},
            board_after=board_after,
            powers_after={"undo": 1, "swap": 0, "delete": 0},
            planned_eval=5400.0,
            score_board_fn=score_fn,
            apply_move_fn=apply_move,
        )
        self.assertTrue(decision.should_undo)
        self.assertEqual(decision.reasons, ("plan_gap",))

    def test_plan_gap_only_undo_is_suppressed_when_undo_was_just_unlocked(self) -> None:
        board_before = _load_board("mid_game")
        board_after = _load_board("corner_trap")
        mapping = {
            _key(board_before): 5000.0,
            _key(board_after): 4970.0,
        }

        def score_fn(board, _powers):
            return mapping[_key(board)]

        decision = analyze_undo(
            board_before=board_before,
            powers_before={"undo": 0, "swap": 0, "delete": 0},
            board_after=board_after,
            powers_after={"undo": 1, "swap": 0, "delete": 0},
            planned_eval=5400.0,
            score_board_fn=score_fn,
            apply_move_fn=apply_move,
        )
        self.assertFalse(decision.should_undo)
        self.assertEqual(decision.reasons, ())

    def test_eval_drop_still_triggers_when_undo_was_just_unlocked(self) -> None:
        board_before = _load_board("mid_game")
        board_after = _load_board("corner_trap")
        mapping = {
            _key(board_before): 5000.0,
            _key(board_after): 4200.0,
        }

        def score_fn(board, _powers):
            return mapping[_key(board)]

        decision = analyze_undo(
            board_before=board_before,
            powers_before={"undo": 0, "swap": 0, "delete": 0},
            board_after=board_after,
            powers_after={"undo": 1, "swap": 0, "delete": 0},
            planned_eval=5200.0,
            score_board_fn=score_fn,
            apply_move_fn=apply_move,
        )
        self.assertTrue(decision.should_undo)
        self.assertIn("eval_drop", decision.reasons)

    def test_plan_gap_only_undo_is_suppressed_when_eval_improved(self) -> None:
        board_before = _load_board("mid_game")
        board_after = _load_board("corner_trap")
        mapping = {
            _key(board_before): 5000.0,
            _key(board_after): 5300.0,
        }

        def score_fn(board, _powers):
            return mapping[_key(board)]

        decision = analyze_undo(
            board_before=board_before,
            powers_before={"undo": 1, "swap": 0, "delete": 0},
            board_after=board_after,
            powers_after={"undo": 1, "swap": 0, "delete": 0},
            planned_eval=5600.0,
            score_board_fn=score_fn,
            apply_move_fn=apply_move,
        )
        self.assertFalse(decision.should_undo)
        self.assertEqual(decision.reasons, ())

    def test_banked_undo_lowers_trigger(self) -> None:
        board_before = _load_board("mid_game")
        board_after = _load_board("corner_trap")
        mapping = {
            _key(board_before): 5000.0,
            _key(board_after): 4900.0,
        }

        def score_fn(board, _powers):
            return mapping[_key(board)]

        one_undo = analyze_undo(
            board_before=board_before,
            powers_before={"undo": 1, "swap": 0, "delete": 0},
            board_after=board_after,
            powers_after={"undo": 1, "swap": 0, "delete": 0},
            planned_eval=5000.0,
            score_board_fn=score_fn,
            apply_move_fn=apply_move,
        )
        two_undo = analyze_undo(
            board_before=board_before,
            powers_before={"undo": 2, "swap": 0, "delete": 0},
            board_after=board_after,
            powers_after={"undo": 2, "swap": 0, "delete": 0},
            planned_eval=5000.0,
            score_board_fn=score_fn,
            apply_move_fn=apply_move,
        )
        self.assertLess(two_undo.drop_trigger, one_undo.drop_trigger)

    def test_fallback_move_avoids_blocked_direction(self) -> None:
        board = _load_board("early_game")
        fallback = best_fallback_move(
            board=board,
            powers={"undo": 0, "swap": 0, "delete": 0},
            depth=4,
            blocked_direction="right",
            apply_move_fn=apply_move,
            score_board_fn=score_board,
            expectimax_fn=None,
        )
        self.assertIsNotNone(fallback)
        assert fallback is not None
        self.assertEqual(fallback[0], "move")
        self.assertNotEqual(fallback[1], "right")

    def test_undo_proximity_relief_depends_on_merge_proximity_not_max_tile(self) -> None:
        board_with_prox = [
            [2048, 1024, 512, 256],
            [64, 64, 32, 16],
            [8, 4, 2, 0],
            [2, 8, 16, 0],
        ]
        board_without_prox = [
            [2048, 1024, 512, 256],
            [64, 32, 16, 8],
            [8, 4, 2, 0],
            [2, 8, 16, 0],
        ]
        mapping = {
            _key(board_with_prox): 5000.0,
            _key(board_without_prox): 5000.0,
        }

        def score_fn(board, _powers):
            return mapping[_key(board)]

        with_prox = analyze_undo(
            board_before=board_with_prox,
            powers_before={"undo": 1, "swap": 0, "delete": 0},
            board_after=board_with_prox,
            powers_after={"undo": 1, "swap": 0, "delete": 0},
            planned_eval=5000.0,
            score_board_fn=score_fn,
            apply_move_fn=apply_move,
        )
        without_prox = analyze_undo(
            board_before=board_without_prox,
            powers_before={"undo": 1, "swap": 0, "delete": 0},
            board_after=board_without_prox,
            powers_after={"undo": 1, "swap": 0, "delete": 0},
            planned_eval=5000.0,
            score_board_fn=score_fn,
            apply_move_fn=apply_move,
        )

        self.assertLess(with_prox.drop_trigger, without_prox.drop_trigger)


if __name__ == "__main__":
    unittest.main()
