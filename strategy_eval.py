"""Evaluation model and transposition-cache plumbing."""

import math
from dataclasses import dataclass

from strategy_config import (
    DEFAULT_EVAL_WEIGHTS,
    DEFAULT_POWERUP_POLICY,
    EvalWeights,
)
from strategy_core import DIRECTIONS, apply_move, board_to_bb
from transposition_cache import TranspositionCache

# Bump this string whenever heuristic weights or eval logic changes.
# The SQLite cache stores scores per-version; stale entries are ignored.
SCORE_BOARD_VERSION = "1.1"

_TRANS_CAP = 500_000
_TRANS_CACHE = TranspositionCache(cap=_TRANS_CAP)


def load_trans_table(entries: dict) -> None:
    """Bulk-load pre-computed entries into the transposition cache."""
    _TRANS_CACHE.load(entries)


def drain_new_entries() -> dict:
    """Return and clear entries added since the last drain."""
    return _TRANS_CACHE.drain_new_entries()


def get_trans_stats() -> dict:
    """Return current hit/miss counters."""
    return _TRANS_CACHE.stats()


def reset_trans_stats() -> None:
    """Zero transposition-cache hit/miss counters."""
    _TRANS_CACHE.reset_stats()


_SNAKE = [
    [16, 15, 14, 13],
    [9, 10, 11, 12],
    [8, 7, 6, 5],
    [1, 2, 3, 4],
]

_SNAKE_ROW_SHOULD_DECREASE = [True, False, True, False]


@dataclass(frozen=True)
class EvalFeatures:
    empties: int
    gradient: float
    monotonicity: float
    roughness: float
    merge_potential: float
    max_log: float
    sum_log: float
    legal_moves: int
    corner_max_log: float
    powerup_value: float


def _gradient(board: list[list[int]]) -> float:
    s = 0.0
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            if v > 0:
                s += _SNAKE[r][c] * math.log2(v)
    return s


def _monotonicity(board: list[list[int]]) -> float:
    total = 0.0
    for r in range(4):
        pen = 0.0
        should_decrease = _SNAKE_ROW_SHOULD_DECREASE[r]
        for c in range(3):
            la = math.log2(board[r][c]) if board[r][c] > 0 else 0.0
            lb = math.log2(board[r][c + 1]) if board[r][c + 1] > 0 else 0.0
            if should_decrease:
                pen += max(0.0, lb - la)
            else:
                pen += max(0.0, la - lb)
        total += pen

    for c in range(4):
        up_pen = down_pen = 0.0
        for r in range(3):
            la = math.log2(board[r][c]) if board[r][c] > 0 else 0.0
            lb = math.log2(board[r + 1][c]) if board[r + 1][c] > 0 else 0.0
            up_pen += max(0.0, la - lb)
            down_pen += max(0.0, lb - la)
        total += min(up_pen, down_pen)

    return -total


def _roughness(board: list[list[int]]) -> float:
    s = 0.0
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            if v <= 0:
                continue
            lv = math.log2(v)
            if c + 1 < 4 and board[r][c + 1] > 0:
                s += abs(lv - math.log2(board[r][c + 1]))
            if r + 1 < 4 and board[r + 1][c] > 0:
                s += abs(lv - math.log2(board[r + 1][c]))
    return s


def _merge_potential(board: list[list[int]]) -> float:
    s = 0.0
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            if v == 0:
                continue
            lv = math.log2(v)
            if c + 1 < 4 and board[r][c + 1] == v:
                s += lv
            if r + 1 < 4 and board[r + 1][c] == v:
                s += lv
            for c2 in range(c + 2, 4):
                if board[r][c2] != 0:
                    if board[r][c2] == v:
                        s += 0.3 * lv
                    break
            for r2 in range(r + 2, 4):
                if board[r2][c] != 0:
                    if board[r2][c] == v:
                        s += 0.3 * lv
                    break
    return s


def _legal_move_count(board: list[list[int]]) -> int:
    count = 0
    for d in DIRECTIONS:
        _, _, changed = apply_move(board, d)
        if changed:
            count += 1
    return count


def _proximity_to_unlock(max_tile: int, unlock_tile: int) -> float:
    if max_tile <= 0 or max_tile >= unlock_tile:
        return 0.0
    return math.log2(max_tile) / math.log2(unlock_tile)


def _powerup_value(max_tile: int, powers: dict) -> float:
    if not powers or max_tile <= 0:
        return 0.0

    stage = math.log2(max_tile)
    p = DEFAULT_POWERUP_POLICY
    swap_uses = min(powers.get("swap", 0), p.max_swap_uses)
    delete_uses = min(powers.get("delete", 0), p.max_delete_uses)

    prox_swap = 0.0
    prox_delete = 0.0
    if max_tile >= p.prox_swap_min_tile and swap_uses < p.max_swap_uses:
        prox_swap = p.prox_scale * _proximity_to_unlock(max_tile, p.swap_unlock_tile)
    if max_tile >= p.prox_delete_min_tile and delete_uses < p.max_delete_uses:
        prox_delete = p.prox_scale * _proximity_to_unlock(max_tile, p.delete_unlock_tile)

    effective_swaps = swap_uses + prox_swap
    effective_deletes = delete_uses + prox_delete
    value_per_swap = p.w_base * stage
    value_per_delete = p.w_base * stage * p.delete_scale
    return value_per_swap * effective_swaps + value_per_delete * effective_deletes


def extract_eval_features(board: list[list[int]], powers: dict | None = None) -> EvalFeatures:
    if powers is None:
        powers = {}
    empties = sum(1 for r in range(4) for c in range(4) if board[r][c] == 0)
    max_val = max(board[r][c] for r in range(4) for c in range(4))
    tile_sum = sum(board[r][c] for r in range(4) for c in range(4))
    max_log = math.log2(max_val) if max_val > 0 else 0.0
    corner_max_log = max_log if max_val > 0 and board[0][0] == max_val else 0.0
    return EvalFeatures(
        empties=empties,
        gradient=_gradient(board),
        monotonicity=_monotonicity(board),
        roughness=_roughness(board),
        merge_potential=_merge_potential(board),
        max_log=max_log,
        sum_log=math.log2(tile_sum + 1.0),
        legal_moves=_legal_move_count(board),
        corner_max_log=corner_max_log,
        powerup_value=_powerup_value(max_val, powers),
    )


def score_from_features(features: EvalFeatures, weights: EvalWeights = DEFAULT_EVAL_WEIGHTS) -> float:
    return (
        weights.empty * features.empties
        + weights.gradient * features.gradient
        + weights.monotonicity * features.monotonicity
        - weights.roughness * features.roughness
        + weights.merge_potential * features.merge_potential
        + weights.max_tile_log * features.max_log
        + weights.sum_log * features.sum_log
        + weights.mobility * features.legal_moves
        + weights.corner * features.corner_max_log
        + weights.powerup * features.powerup_value
    )


def score_board(board: list[list[int]], powers: dict | None = None) -> float:
    if powers is None:
        powers = {}
    key = (board_to_bb(board), powers.get("swap", 0), powers.get("delete", 0))
    cached = _TRANS_CACHE.get(key)
    if cached is not None:
        return cached
    result = score_from_features(extract_eval_features(board, powers))
    _TRANS_CACHE.store(key, result)
    return result
