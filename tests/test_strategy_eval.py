#!/usr/bin/env python3
"""Tests for strategy evaluation decomposition and cache behavior."""

import json
import unittest
from pathlib import Path

from strategy import (
    EvalWeights,
    extract_eval_features,
    get_trans_stats,
    reset_trans_stats,
    score_board,
    score_from_features,
)


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


if __name__ == "__main__":
    unittest.main()
