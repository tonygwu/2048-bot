"""Search policy and action selection."""

import math

from strategy_actions import (
    Action,
    ActionTuple,
    DeleteAction,
    MoveAction,
    SwapAction,
    action_to_tuple,
)
from strategy_config import DEFAULT_DEPTH_POLICY
from strategy_core import DIRECTIONS, apply_move, empty_cells
from strategy_eval import _roughness, normalize_powers, score_board


def auto_depth(board: list[list[int]]) -> int:
    """Return adaptive expectimax depth from board-state features."""
    p = DEFAULT_DEPTH_POLICY
    empties = 0
    max_tile = 0
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            if v == 0:
                empties += 1
            elif v > max_tile:
                max_tile = v

    max_log = math.log2(max_tile) if max_tile > 0 else 0.0
    fullness = (16 - empties) / 16.0
    valid = 0
    for d in DIRECTIONS:
        _, _, changed = apply_move(board, d)
        if changed:
            valid += 1
    blocked = (4 - valid) / 3.0
    rough_n = min(1.0, _roughness(board) / p.rough_norm_divisor)
    score = p.base + p.w_max * max_log + p.w_full * fullness + p.w_blocked * blocked + p.w_rough * rough_n

    if empties >= p.open_empties_threshold:
        score -= p.open_penalty
    elif empties <= p.jammed_empties_threshold:
        score += p.jammed_bonus
    if valid <= p.low_valid_bonus_threshold:
        score += p.low_valid_bonus
        if (
            empties >= p.midgame_empties_threshold
            and rough_n >= p.midgame_rough_damp_threshold
        ):
            score -= p.midgame_low_valid_damp
    if (
        empties <= p.surgery_empties_threshold
        and valid <= p.surgery_valid_threshold
        and rough_n >= p.surgery_rough_threshold
    ):
        score += p.surgery_bonus

    depth = max(p.min_depth, min(p.soft_max, int(round(score))))
    if (
        empties <= p.near_death_empties_threshold
        and valid <= p.near_death_valid_threshold
        and rough_n >= p.near_death_rough_threshold
        and max_log >= p.near_death_max_log_threshold
    ):
        return p.max_depth
    return depth


def _expectimax(board: list[list[int]], depth: int, is_max: bool, powers: dict | None = None) -> float:
    powers = normalize_powers(powers)
    if depth == 0:
        return score_board(board, powers)

    if is_max:
        best = float("-inf")
        any_valid = False
        for d in DIRECTIONS:
            nb, _, changed = apply_move(board, d)
            if not changed:
                continue
            any_valid = True
            val = _expectimax(nb, depth - 1, False, powers)
            if val > best:
                best = val
        return best if any_valid else score_board(board, powers)

    empties = empty_cells(board)
    if not empties:
        return score_board(board, powers)
    sample = empties if len(empties) <= 6 else empties[::len(empties) // 6 + 1][:6]

    total = 0.0
    for r, c in sample:
        board[r][c] = 2
        total += 0.9 * _expectimax(board, depth - 1, True, powers)
        board[r][c] = 4
        total += 0.1 * _expectimax(board, depth - 1, True, powers)
        board[r][c] = 0
    return total / len(sample)


def apply_swap(board: list[list[int]], r1: int, c1: int, r2: int, c2: int) -> list[list[int]]:
    nb = [row[:] for row in board]
    nb[r1][c1], nb[r2][c2] = nb[r2][c2], nb[r1][c1]
    return nb


def apply_delete(board: list[list[int]], value: int) -> list[list[int]]:
    return [[0 if v == value else v for v in row] for row in board]


def _top_positions(board: list[list[int]], k: int = 6) -> list[tuple[int, int]]:
    tiles = sorted(((board[r][c], r, c) for r in range(4) for c in range(4) if board[r][c] > 0), reverse=True)
    return [(r, c) for _, r, c in tiles[:k]]


def best_action_obj(board: list[list[int]], powers: dict | None = None, depth: int = 4) -> Action | None:
    powers = normalize_powers(powers)
    best_val = float("-inf")
    best_act: Action | None = None

    for d in DIRECTIONS:
        nb, score_delta, changed = apply_move(board, d)
        if not changed:
            continue
        val = _expectimax(nb, depth - 1, False, powers) + score_delta
        if val > best_val:
            best_val = val
            best_act = MoveAction(direction=d)

    if powers["swap"] > 0:
        powers_after = {**powers, "swap": powers["swap"] - 1}
        top = _top_positions(board, k=6)
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                r1, c1 = top[i]
                r2, c2 = top[j]
                if board[r1][c1] == board[r2][c2]:
                    continue
                nb = apply_swap(board, r1, c1, r2, c2)
                val = _expectimax(nb, depth - 1, False, powers_after)
                if val > best_val:
                    best_val = val
                    best_act = SwapAction(r1=r1, c1=c1, r2=r2, c2=c2)

    if powers["delete"] > 0:
        powers_after = {**powers, "delete": powers["delete"] - 1}
        empties = sum(1 for r in range(4) for c in range(4) if board[r][c] == 0)
        candidates = [2, 4, 8]
        if empties <= 3:
            candidates += [16, 32]
        present = {board[r][c] for r in range(4) for c in range(4) if board[r][c] > 0}
        for v in candidates:
            if v not in present:
                continue
            nb = apply_delete(board, v)
            val = _expectimax(nb, depth - 1, False, powers_after)
            if val > best_val:
                best_val = val
                pos = next((r, c) for r in range(4) for c in range(4) if board[r][c] == v)
                best_act = DeleteAction(value=v, row=pos[0], col=pos[1])

    return best_act


def best_action(board: list[list[int]], powers: dict | None = None, depth: int = 4) -> ActionTuple | None:
    return action_to_tuple(best_action_obj(board, powers=powers, depth=depth))


def best_move(board: list[list[int]], depth: int = 4) -> str | None:
    action = best_action_obj(board, powers={}, depth=depth)
    if not isinstance(action, MoveAction):
        return None
    return action.direction
