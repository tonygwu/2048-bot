"""Undo decision helpers shared by live bot and evaluator harness.

The strategy module plans one step ahead, while actual outcomes can drift due to
random spawn and tactical cliffs. This policy decides when to spend Undo and how
to avoid immediately repeating the same failed move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from strategy_config import DEFAULT_POWERUP_POLICY, PowerUpPolicy
from strategy_core import DIRECTIONS
from strategy_eval import _roughness, normalize_powers

Board = list[list[int]]
ActionTuple = tuple


@dataclass(frozen=True)
class UndoDecision:
    should_undo: bool
    reasons: tuple[str, ...]
    eval_before: float
    eval_after: float
    eval_drop: float
    planned_eval: float
    plan_gap: float
    drop_trigger: float
    gap_trigger: float
    pressure: float


def _legal_move_count(
    board: Board,
    apply_move_fn: Callable[[Board, str], tuple[Board, int, bool]],
) -> int:
    valid = 0
    for d in DIRECTIONS:
        _, _, changed = apply_move_fn(board, d)
        if changed:
            valid += 1
    return valid


def board_pressure(
    board: Board,
    *,
    apply_move_fn: Callable[[Board, str], tuple[Board, int, bool]],
    policy: PowerUpPolicy = DEFAULT_POWERUP_POLICY,
) -> float:
    """Compute board pressure in [0, 1] using fullness, mobility, roughness."""
    empties = sum(1 for r in range(4) for c in range(4) if board[r][c] == 0)
    fullness = (16 - empties) / 16.0
    valid_moves = _legal_move_count(board, apply_move_fn)
    mobility_pressure = (4 - valid_moves) / 3.0
    rough_pressure = min(1.0, _roughness(board) / policy.pressure_roughness_norm_divisor)
    w_sum = policy.pressure_fullness_w + policy.pressure_mobility_w + policy.pressure_roughness_w
    if w_sum <= 0:
        return 0.0
    pressure = (
        policy.pressure_fullness_w * fullness
        + policy.pressure_mobility_w * mobility_pressure
        + policy.pressure_roughness_w * rough_pressure
    ) / w_sum
    return max(0.0, min(1.0, pressure))


def _effective_trigger(
    *,
    base_trigger: float,
    board_before: Board,
    powers_before: dict | None,
    apply_move_fn: Callable[[Board, str], tuple[Board, int, bool]],
    policy: PowerUpPolicy,
) -> tuple[float, float]:
    pressure = board_pressure(board_before, apply_move_fn=apply_move_fn, policy=policy)
    threshold = base_trigger - policy.undo_pressure_relief * pressure

    powers_n = normalize_powers(powers_before)
    if powers_n.get("undo", 0) >= policy.undo_bank_uses_threshold:
        threshold -= policy.undo_bank_relief

    max_tile = max(v for row in board_before for v in row)
    if max_tile >= policy.undo_prox_min_tile:
        threshold -= policy.undo_prox_relief

    threshold = max(policy.undo_trigger_floor, threshold)
    return threshold, pressure


def project_action(
    board: Board,
    powers: dict | None,
    action: ActionTuple | None,
    *,
    apply_move_fn: Callable[[Board, str], tuple[Board, int, bool]],
    apply_swap_fn: Callable[[Board, int, int, int, int], Board],
    apply_delete_fn: Callable[[Board, int], Board],
) -> tuple[Board, dict, int, bool] | None:
    """Project immediate board/power state after an action (no random spawn)."""
    if action is None:
        return None

    kind = action[0]
    powers_after = normalize_powers(powers)
    score_delta = 0

    if kind == "move":
        _, direction = action
        board_after, score_delta, changed = apply_move_fn(board, direction)
        if not changed:
            return None
    elif kind == "swap":
        _, r1, c1, r2, c2 = action
        board_after = apply_swap_fn(board, r1, c1, r2, c2)
        powers_after = dict(powers_after)
        powers_after["swap"] = max(0, powers_after.get("swap", 0) - 1)
        changed = board_after != board
    elif kind == "delete":
        _, value, _, _ = action
        board_after = apply_delete_fn(board, value)
        powers_after = dict(powers_after)
        powers_after["delete"] = max(0, powers_after.get("delete", 0) - 1)
        changed = board_after != board
    else:
        return None

    return board_after, powers_after, score_delta, changed


def projected_action_eval(
    board: Board,
    powers: dict | None,
    action: ActionTuple | None,
    *,
    score_board_fn: Callable[[Board, dict | None], float],
    apply_move_fn: Callable[[Board, str], tuple[Board, int, bool]],
    apply_swap_fn: Callable[[Board, int, int, int, int], Board],
    apply_delete_fn: Callable[[Board, int], Board],
) -> float | None:
    projection = project_action(
        board,
        powers,
        action,
        apply_move_fn=apply_move_fn,
        apply_swap_fn=apply_swap_fn,
        apply_delete_fn=apply_delete_fn,
    )
    if projection is None:
        return None
    board_after, powers_after, _, _ = projection
    return float(score_board_fn(board_after, powers_after))


def analyze_undo(
    *,
    board_before: Board,
    powers_before: dict | None,
    board_after: Board,
    powers_after: dict | None,
    planned_eval: float | None,
    score_board_fn: Callable[[Board, dict | None], float],
    apply_move_fn: Callable[[Board, str], tuple[Board, int, bool]],
    policy: PowerUpPolicy = DEFAULT_POWERUP_POLICY,
) -> UndoDecision:
    """Evaluate whether Undo should be spent after a move resolved."""
    powers_before_n = normalize_powers(powers_before)
    powers_after_n = normalize_powers(powers_after)
    eval_before = float(score_board_fn(board_before, powers_before_n))
    eval_after = float(score_board_fn(board_after, powers_after_n))
    planned = eval_before if planned_eval is None else float(planned_eval)

    eval_drop = eval_before - eval_after
    plan_gap = planned - eval_after
    drop_trigger, pressure = _effective_trigger(
        base_trigger=policy.undo_drop_trigger,
        board_before=board_before,
        powers_before=powers_before_n,
        apply_move_fn=apply_move_fn,
        policy=policy,
    )
    gap_trigger, _ = _effective_trigger(
        base_trigger=policy.undo_plan_gap_trigger,
        board_before=board_before,
        powers_before=powers_before_n,
        apply_move_fn=apply_move_fn,
        policy=policy,
    )

    reasons: list[str] = []
    if eval_drop >= drop_trigger:
        reasons.append("eval_drop")
    if plan_gap >= gap_trigger:
        reasons.append("plan_gap")

    should_undo = powers_after_n.get("undo", 0) > 0 and bool(reasons)
    return UndoDecision(
        should_undo=should_undo,
        reasons=tuple(reasons),
        eval_before=eval_before,
        eval_after=eval_after,
        eval_drop=eval_drop,
        planned_eval=planned,
        plan_gap=plan_gap,
        drop_trigger=drop_trigger,
        gap_trigger=gap_trigger,
        pressure=pressure,
    )


def best_fallback_move(
    *,
    board: Board,
    powers: dict | None,
    depth: int,
    blocked_direction: str | None,
    apply_move_fn: Callable[[Board, str], tuple[Board, int, bool]],
    score_board_fn: Callable[[Board, dict | None], float],
    expectimax_fn: Callable[[Board, int, bool, dict | None], float] | None = None,
) -> ActionTuple | None:
    """Pick best legal move, optionally excluding one direction."""
    powers_n = normalize_powers(powers)
    best_val = float("-inf")
    best_action: ActionTuple | None = None

    for d in DIRECTIONS:
        if blocked_direction is not None and d == blocked_direction:
            continue
        nb, score_delta, changed = apply_move_fn(board, d)
        if not changed:
            continue
        if expectimax_fn is not None and depth > 1:
            val = float(expectimax_fn([row[:] for row in nb], depth - 1, False, powers_n)) + float(score_delta)
        else:
            val = float(score_board_fn(nb, powers_n)) + float(score_delta)
        if val > best_val:
            best_val = val
            best_action = ("move", d)

    return best_action
