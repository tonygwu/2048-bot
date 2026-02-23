#!/usr/bin/env python3
"""Tests for strategy evaluation decomposition and cache behavior."""

import json
import unittest
from pathlib import Path

import strategy_eval
from strategy import (
    EvalWeights,
    extract_eval_features,
    get_trans_stats,
    load_trans_table,
    normalize_powers,
    reset_trans_stats,
    score_board,
    score_from_features,
)
from transposition_cache import TranspositionCache


BOARDS_DIR = Path(__file__).parent / "boards"


def _load_board(name: str) -> tuple[list[list[int]], dict]:
    data = json.loads((BOARDS_DIR / f"{name}.json").read_text())
    return data["board"], data.get("powers", {})


class TestStrategyEval(unittest.TestCase):
    def test_score_from_features_matches_score_board(self) -> None:
        board, powers = _load_board("mid_game")
        features = extract_eval_features(board, powers)
        self.assertAlmostEqual(score_from_features(features), score_board(board, powers), places=9)

    def test_zero_weights_zero_out_eval(self) -> None:
        board, powers = _load_board("late_game")
        features = extract_eval_features(board, powers)
        zero = EvalWeights(
            empty=0.0,
            gradient=0.0,
            monotonicity=0.0,
            roughness=0.0,
            merge_potential=0.0,
            max_tile_log=0.0,
            sum_log=0.0,
            mobility=0.0,
            corner=0.0,
            powerup=0.0,
            promotion=0.0,
            high_merge=0.0,
        )
        self.assertEqual(score_from_features(features, zero), 0.0)

    def test_score_cache_hit_miss_accounting(self) -> None:
        board, powers = _load_board("corner_trap")
        reset_trans_stats()
        _ = score_board(board, powers)
        _ = score_board(board, powers)
        stats = get_trans_stats()
        self.assertEqual(stats["misses"], 1)
        self.assertGreaterEqual(stats["hits"], 1)

    def test_oversized_preload_preserves_table(self) -> None:
        board, powers = _load_board("mid_game")
        original_cache = strategy_eval._TRANS_CACHE
        try:
            test_cache = TranspositionCache(cap=2)
            strategy_eval._TRANS_CACHE = test_cache
            load_trans_table({(1, 0, 0): 1.0, (2, 0, 0): 2.0, (3, 0, 0): 3.0})
            before = len(test_cache.table)
            _ = score_board(board, powers)
            self.assertEqual(len(test_cache.table), before)
            self.assertGreaterEqual(len(test_cache.new_entries), 1)
        finally:
            strategy_eval._TRANS_CACHE = original_cache

    def test_normalize_powers_caps_uses_to_two(self) -> None:
        self.assertEqual(
            normalize_powers({"undo": 7, "swap": 5, "delete": 3}),
            {"undo": 2, "swap": 2, "delete": 2},
        )
        self.assertEqual(
            normalize_powers({"undo": -1, "swap": -9, "delete": 1}),
            {"undo": 0, "swap": 0, "delete": 1},
        )

    def test_score_board_cache_key_clamps_power_counts(self) -> None:
        board, _ = _load_board("mid_game")
        original_cache = strategy_eval._TRANS_CACHE
        try:
            test_cache = TranspositionCache(cap=100)
            strategy_eval._TRANS_CACHE = test_cache
            v1 = score_board(board, {"undo": 7, "swap": 5, "delete": 9})
            v2 = score_board(board, {"undo": 2, "swap": 2, "delete": 2})
            self.assertAlmostEqual(v1, v2, places=9)
            self.assertEqual(len(test_cache.table), 1)
            stats = get_trans_stats()
            self.assertEqual(stats["misses"], 1)
            self.assertGreaterEqual(stats["hits"], 1)
        finally:
            strategy_eval._TRANS_CACHE = original_cache

    def test_score_board_distinguishes_undo_uses(self) -> None:
        board, _ = _load_board("mid_game")
        original_cache = strategy_eval._TRANS_CACHE
        try:
            test_cache = TranspositionCache(cap=100)
            strategy_eval._TRANS_CACHE = test_cache
            reset_trans_stats()
            v0 = score_board(board, {"undo": 0, "swap": 0, "delete": 0})
            v2 = score_board(board, {"undo": 2, "swap": 0, "delete": 0})
            self.assertGreater(v2, v0)
            # Base board score is now cached board-only; power terms are added dynamically.
            self.assertEqual(len(test_cache.table), 1)
            stats = get_trans_stats()
            self.assertEqual(stats["misses"], 1)
            self.assertGreaterEqual(stats["hits"], 1)
        finally:
            strategy_eval._TRANS_CACHE = original_cache

    def test_powerup_value_uses_piecewise_stage_values(self) -> None:
        board = [
            [32, 16, 8, 4],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        base = extract_eval_features(board, {"undo": 0, "swap": 0, "delete": 0}).powerup_value
        self.assertEqual(base, 0.0)
        # stage = log2(32) = 5; no proximity applies at this stage.
        self.assertAlmostEqual(
            extract_eval_features(board, {"undo": 1, "swap": 0, "delete": 0}).powerup_value,
            40.0 * 5.0,
            places=9,
        )
        self.assertAlmostEqual(
            extract_eval_features(board, {"undo": 2, "swap": 0, "delete": 0}).powerup_value,
            70.0 * 5.0,
            places=9,
        )
        self.assertAlmostEqual(
            extract_eval_features(board, {"undo": 0, "swap": 1, "delete": 0}).powerup_value,
            80.0 * 5.0,
            places=9,
        )
        self.assertAlmostEqual(
            extract_eval_features(board, {"undo": 0, "swap": 2, "delete": 0}).powerup_value,
            140.0 * 5.0,
            places=9,
        )
        self.assertAlmostEqual(
            extract_eval_features(board, {"undo": 0, "swap": 0, "delete": 1}).powerup_value,
            160.0 * 5.0,
            places=9,
        )
        self.assertAlmostEqual(
            extract_eval_features(board, {"undo": 0, "swap": 0, "delete": 2}).powerup_value,
            280.0 * 5.0,
            places=9,
        )

    def test_second_charge_is_sublinear_vs_two_first_charges(self) -> None:
        board = [
            [32, 16, 8, 4],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        undo1 = extract_eval_features(board, {"undo": 1, "swap": 0, "delete": 0}).powerup_value
        undo2 = extract_eval_features(board, {"undo": 2, "swap": 0, "delete": 0}).powerup_value
        swap1 = extract_eval_features(board, {"undo": 0, "swap": 1, "delete": 0}).powerup_value
        swap2 = extract_eval_features(board, {"undo": 0, "swap": 2, "delete": 0}).powerup_value
        delete1 = extract_eval_features(board, {"undo": 0, "swap": 0, "delete": 1}).powerup_value
        delete2 = extract_eval_features(board, {"undo": 0, "swap": 0, "delete": 2}).powerup_value
        self.assertLess(undo2, 2.0 * undo1)
        self.assertLess(swap2, 2.0 * swap1)
        self.assertLess(delete2, 2.0 * delete1)

    def test_swap_proximity_uses_board_structure_even_in_late_game(self) -> None:
        board_close = [
            [8192, 1024, 512, 256],
            [128, 128, 32, 16],
            [8, 4, 2, 0],
            [2, 8, 16, 0],
        ]
        board_far = [
            [8192, 1024, 512, 256],
            [128, 64, 32, 16],
            [8, 4, 2, 0],
            [2, 8, 16, 0],
        ]
        # Keep delete at cap so only swap proximity changes here.
        powers = {"undo": 0, "swap": 0, "delete": 2}
        close_val = extract_eval_features(board_close, powers).powerup_value
        far_val = extract_eval_features(board_far, powers).powerup_value
        self.assertGreater(close_val, far_val)

    def test_undo_proximity_uses_board_structure_even_in_late_game(self) -> None:
        board_close = [
            [8192, 4096, 1024, 512],
            [256, 128, 64, 64],
            [16, 8, 4, 2],
            [2, 4, 8, 0],
        ]
        board_far = [
            [8192, 4096, 1024, 512],
            [256, 128, 64, 32],
            [16, 8, 4, 2],
            [2, 4, 8, 0],
        ]
        # Keep swap/delete at cap so only undo proximity changes here.
        powers = {"undo": 0, "swap": 2, "delete": 2}
        close_val = extract_eval_features(board_close, powers).powerup_value
        far_val = extract_eval_features(board_far, powers).powerup_value
        self.assertGreater(close_val, far_val)

    def test_delete_proximity_uses_board_structure_even_in_late_game(self) -> None:
        board_close = [
            [8192, 4096, 1024, 512],
            [256, 256, 64, 32],
            [16, 8, 4, 2],
            [2, 4, 8, 0],
        ]
        board_far = [
            [8192, 4096, 1024, 512],
            [256, 128, 64, 32],
            [16, 8, 4, 2],
            [2, 4, 8, 0],
        ]
        # Keep swap at cap so only delete proximity changes here.
        powers = {"undo": 0, "swap": 2, "delete": 0}
        close_val = extract_eval_features(board_close, powers).powerup_value
        far_val = extract_eval_features(board_far, powers).powerup_value
        self.assertGreater(close_val, far_val)


if __name__ == "__main__":
    unittest.main()
