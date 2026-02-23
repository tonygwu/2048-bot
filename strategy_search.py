"""Search policy and action selection."""

import math
from collections import Counter

from strategy_actions import (
    Action,
    ActionTuple,
    DeleteAction,
    MoveAction,
    SwapAction,
    action_to_tuple,
)
from strategy_config import DEFAULT_DEPTH_POLICY, DEFAULT_POWERUP_POLICY
from strategy_core import DIRECTIONS, apply_move, board_to_bb, empty_cells, is_game_over
from strategy_eval import MIN_TRANS_CACHE_TILE, _roughness, normalize_powers, score_board
from transposition_cache import TranspositionCache

_SEARCH_CACHE_CAP = 250_000
SEARCH_CACHE_VERSION = "s1"
_SEARCH_CACHE = TranspositionCache(cap=_SEARCH_CACHE_CAP)
_TERMINAL_LOSS_PENALTY = 25_000.0
_MIN_SEARCH_CACHE_EXP = MIN_TRANS_CACHE_TILE.bit_length() - 1


def reset_search_trans_cache() -> None:
    _SEARCH_CACHE.table.clear()
    _SEARCH_CACHE.new_entries.clear()
    _SEARCH_CACHE.keep_oversized_preload = False
    _SEARCH_CACHE.reset_stats()


def reset_search_trans_stats() -> None:
    _SEARCH_CACHE.reset_stats()


def get_search_trans_stats() -> dict:
    stats = _SEARCH_CACHE.stats()
    return {"hits": stats["hits"], "misses": stats["misses"], "size": len(_SEARCH_CACHE.table)}


def get_search_trans_table_size() -> int:
    return len(_SEARCH_CACHE.table)


def _max_exp_from_bb(bb: int) -> int:
    v = int(bb)
    out = 0
    for _ in range(16):
        nib = v & 0xF
        if nib > out:
            out = nib
        v >>= 4
    return out


def _normalize_search_key(key) -> tuple[int, int, int, int, int, int]:
    if isinstance(key, tuple) and len(key) >= 6:
        try:
            is_max = 1 if int(key[5]) != 0 else 0
        except Exception:
            is_max = 1 if bool(key[5]) else 0
        return (
            int(key[0]),
            max(0, min(2, int(key[1]))),
            max(0, min(2, int(key[2]))),
            max(0, min(2, int(key[3]))),
            max(0, int(key[4])),
            is_max,
        )
    raise ValueError(f"unexpected search transposition key shape: {key!r}")


def load_search_trans_table(entries: dict) -> None:
    normalized = {_normalize_search_key(k): float(v) for k, v in entries.items()}
    _SEARCH_CACHE.load(normalized)


def drain_search_new_entries() -> dict:
    return _SEARCH_CACHE.drain_new_entries()


def evict_search_trans_below_max_tile(min_tile: int) -> int:
    threshold = max(0, int(min_tile))
    if threshold <= 1:
        return 0
    min_exp = threshold.bit_length() - 1
    drop_keys = [k for k in _SEARCH_CACHE.table.keys() if _max_exp_from_bb(int(k[0])) < min_exp]
    for key in drop_keys:
        _SEARCH_CACHE.table.pop(key, None)
        _SEARCH_CACHE.new_entries.pop(key, None)
    if len(_SEARCH_CACHE.table) <= _SEARCH_CACHE.cap:
        _SEARCH_CACHE.keep_oversized_preload = False
    return len(drop_keys)


def _search_cache_get(key):
    return _SEARCH_CACHE.get(key)


def _search_cache_store(key, value: float) -> None:
    _SEARCH_CACHE.store(key, float(value))


def _search_key(board: list[list[int]], powers: dict, depth: int, is_max: bool) -> tuple:
    return (
        board_to_bb(board),
        powers["undo"],
        powers["swap"],
        powers["delete"],
        depth,
        1 if is_max else 0,
    )


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
    if empties <= p.tight_empties_threshold and valid <= p.tight_valid_threshold:
        score += p.tight_bonus

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
    board_bb = board_to_bb(board)
    use_cache = _max_exp_from_bb(board_bb) >= _MIN_SEARCH_CACHE_EXP
    key = None
    if use_cache:
        key = (
            board_bb,
            powers["undo"],
            powers["swap"],
            powers["delete"],
            depth,
            1 if is_max else 0,
        )
        cached = _search_cache_get(key)
        if cached is not None:
            return cached

    if depth == 0:
        out = score_board(board, powers)
        if is_game_over(board):
            out -= _TERMINAL_LOSS_PENALTY
        if key is not None:
            _search_cache_store(key, out)
        return out

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
        if any_valid:
            out = best
        else:
            out = score_board(board, powers) - _TERMINAL_LOSS_PENALTY
        if key is not None:
            _search_cache_store(key, out)
        return out

    empties = empty_cells(board)
    if not empties:
        out = score_board(board, powers)
        if is_game_over(board):
            out -= _TERMINAL_LOSS_PENALTY
        if key is not None:
            _search_cache_store(key, out)
        return out
    sample = empties if len(empties) <= 6 else empties[::len(empties) // 6 + 1][:6]

    total = 0.0
    for r, c in sample:
        board[r][c] = 2
        total += 0.9 * _expectimax(board, depth - 1, True, powers)
        board[r][c] = 4
        total += 0.1 * _expectimax(board, depth - 1, True, powers)
        board[r][c] = 0
    out = total / len(sample)
    if key is not None:
        _search_cache_store(key, out)
    return out


def apply_swap(board: list[list[int]], r1: int, c1: int, r2: int, c2: int) -> list[list[int]]:
    nb = [row[:] for row in board]
    nb[r1][c1], nb[r2][c2] = nb[r2][c2], nb[r1][c1]
    return nb


def apply_delete(board: list[list[int]], value: int) -> list[list[int]]:
    return [[0 if v == value else v for v in row] for row in board]


def _swap_candidate_pairs(board: list[list[int]], powers_after: dict) -> list[tuple[int, int, int, int]]:
    """Return a pressure-bounded shortlist of swap pairs.

    We rank all distinct non-empty tile pairs by static evaluator score after
    swap, then run deeper expectimax only on the highest-ranked subset.
    """
    positions = [(r, c) for r in range(4) for c in range(4) if board[r][c] > 0]
    if len(positions) < 2:
        return []

    scored_pairs: list[tuple[float, tuple[int, int, int, int]]] = []
    for i in range(len(positions)):
        r1, c1 = positions[i]
        for j in range(i + 1, len(positions)):
            r2, c2 = positions[j]
            if board[r1][c1] == board[r2][c2]:
                continue
            nb = apply_swap(board, r1, c1, r2, c2)
            scored_pairs.append((score_board(nb, powers_after), (r1, c1, r2, c2)))

    if not scored_pairs:
        return []

    p = DEFAULT_POWERUP_POLICY
    pressure = _board_pressure(board)
    cap = int(round(p.swap_pair_cap_calm * (1.0 - pressure) + p.swap_pair_cap_pressure * pressure))
    cap = max(1, min(len(scored_pairs), cap))

    scored_pairs.sort(key=lambda item: item[0], reverse=True)
    return [pair for _, pair in scored_pairs[:cap]]


def _delete_candidates(board: list[list[int]], empties: int, max_tile: int) -> list[int]:
    counts = Counter(board[r][c] for r in range(4) for c in range(4) if board[r][c] > 0)
    candidates: list[int] = [2, 4, 8]
    if empties <= 5:
        candidates += [16, 32]
    if max_tile >= 1024 and empties <= 5:
        candidates += [64]
    if max_tile >= 2048 and empties <= 4:
        candidates += [128]
    if max_tile >= 4096 and empties <= 3:
        candidates += [256]

    # Add duplicated values that are "small enough" relative to late-board scale.
    late_cap = max(64, max_tile // 4) if max_tile > 0 else 64
    dup_values = sorted(v for v, cnt in counts.items() if cnt >= 2 and v <= late_cap)
    candidates.extend(dup_values)

    # In jammed late boards, also consider singleton clutter values, but keep
    # this conservative so we don't burn delete on core high-chain anchors.
    singleton_cap = 0
    if max_tile >= 1024 and empties <= 4:
        singleton_cap = max(128, max_tile // 16)
    if singleton_cap > 0:
        singleton_values = sorted(v for v, cnt in counts.items() if cnt == 1 and v <= singleton_cap)
        candidates.extend(singleton_values)

    out: list[int] = []
    seen: set[int] = set()
    for v in candidates:
        if v not in counts or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _legal_move_count(board: list[list[int]]) -> int:
    valid = 0
    for d in DIRECTIONS:
        _, _, changed = apply_move(board, d)
        if changed:
            valid += 1
    return valid


def _board_pressure(board: list[list[int]]) -> float:
    p = DEFAULT_POWERUP_POLICY
    empties = sum(1 for r in range(4) for c in range(4) if board[r][c] == 0)
    fullness = (16 - empties) / 16.0
    valid_moves = _legal_move_count(board)
    mobility_pressure = (4 - valid_moves) / 3.0
    rough_pressure = min(1.0, _roughness(board) / p.pressure_roughness_norm_divisor)
    w_sum = p.pressure_fullness_w + p.pressure_mobility_w + p.pressure_roughness_w
    if w_sum <= 0:
        return 0.0
    pressure = (
        p.pressure_fullness_w * fullness
        + p.pressure_mobility_w * mobility_pressure
        + p.pressure_roughness_w * rough_pressure
    ) / w_sum
    return max(0.0, min(1.0, pressure))


def _required_powerup_advantage(board: list[list[int]], uses_left: int, kind: str) -> float:
    p = DEFAULT_POWERUP_POLICY
    pressure = _board_pressure(board)
    # Calm boards get a higher spend bar; jammed boards relax it.
    base_margin = p.spend_margin_calm * (1.0 - pressure) + p.spend_margin_pressure * pressure
    reserve_margin = p.reserve_margin_per_extra_use * max(0, uses_left - 1)
    mult = p.swap_margin_mult if kind == "swap" else p.delete_margin_mult
    values = sorted((board[r][c] for r in range(4) for c in range(4)), reverse=True)
    max_tile = values[0] if values else 0
    second_tile = values[1] if len(values) > 1 else 0
    empties = sum(1 for r in range(4) for c in range(4) if board[r][c] == 0)
    stage_factor = 1.0
    if second_tile < p.late_stage_second_tile_threshold:
        if max_tile >= p.ultra_late_tile_threshold:
            stage_factor = p.ultra_late_margin_mult
        elif max_tile >= p.late_stage_tile_threshold:
            stage_factor = p.late_stage_margin_mult
    required = (base_margin + reserve_margin) * mult * stage_factor
    # Preserve the final charge unless the tactical gain is clear. The reserve
    # relaxes on jammed boards where immediate intervention is more valuable.
    if uses_left <= 1:
        last_use_margin = p.last_use_margin_calm * (1.0 - pressure) + p.last_use_margin_pressure * pressure
        last_use_mult = p.last_use_swap_mult if kind == "swap" else p.last_use_delete_mult
        required += last_use_margin * last_use_mult
    # Delete-specific patience: in sparse ultra-late boards, avoid spending a
    # delete on small clutter unless it strongly outperforms normal moves.
    if kind == "delete" and max_tile >= p.delete_patience_min_tile and empties >= p.delete_patience_min_empties:
        required += (
            p.delete_patience_base_margin
            + p.delete_patience_extra_per_empty * (empties - p.delete_patience_min_empties)
        )
    return max(0.0, required)


def best_action_obj(board: list[list[int]], powers: dict | None = None, depth: int = 4) -> Action | None:
    powers = normalize_powers(powers)
    best_move_val = float("-inf")
    best_move_act: MoveAction | None = None
    best_swap_val = float("-inf")
    best_swap_act: SwapAction | None = None
    best_delete_val = float("-inf")
    best_delete_act: DeleteAction | None = None
    empties = sum(1 for r in range(4) for c in range(4) if board[r][c] == 0)
    max_tile = max(board[r][c] for r in range(4) for c in range(4))

    for d in DIRECTIONS:
        nb, score_delta, changed = apply_move(board, d)
        if not changed:
            continue
        val = _expectimax(nb, depth - 1, False, powers) + score_delta
        if val > best_move_val:
            best_move_val = val
            best_move_act = MoveAction(direction=d)

    if powers["swap"] > 0:
        powers_after = {**powers, "swap": powers["swap"] - 1}
        for r1, c1, r2, c2 in _swap_candidate_pairs(board, powers_after):
            nb = apply_swap(board, r1, c1, r2, c2)
            val = _expectimax(nb, depth - 1, False, powers_after)
            if val > best_swap_val:
                best_swap_val = val
                best_swap_act = SwapAction(r1=r1, c1=c1, r2=r2, c2=c2)

    if powers["delete"] > 0:
        powers_after = {**powers, "delete": powers["delete"] - 1}
        for v in _delete_candidates(board, empties=empties, max_tile=max_tile):
            nb = apply_delete(board, v)
            val = _expectimax(nb, depth - 1, False, powers_after)
            if val > best_delete_val:
                best_delete_val = val
                pos = next((r, c) for r in range(4) for c in range(4) if board[r][c] == v)
                best_delete_act = DeleteAction(value=v, row=pos[0], col=pos[1])

    # If no legal move exists, fall back to the strongest power-up action.
    if best_move_act is None:
        if best_swap_act is None and best_delete_act is None:
            return None
        if best_swap_act is None:
            return best_delete_act
        if best_delete_act is None:
            return best_swap_act
        return best_swap_act if best_swap_val >= best_delete_val else best_delete_act

    best_val = best_move_val
    best_act: Action = best_move_act

    # Spend a power-up only when its expected value clears the dynamic reserve bar.
    if best_swap_act is not None:
        req = _required_powerup_advantage(board, uses_left=powers["swap"], kind="swap")
        if best_swap_val >= best_move_val + req and best_swap_val > best_val:
            best_val = best_swap_val
            best_act = best_swap_act

    if best_delete_act is not None:
        req = _required_powerup_advantage(board, uses_left=powers["delete"], kind="delete")
        if best_delete_val >= best_move_val + req and best_delete_val > best_val:
            best_val = best_delete_val
            best_act = best_delete_act

    return best_act


def best_action(board: list[list[int]], powers: dict | None = None, depth: int = 4) -> ActionTuple | None:
    return action_to_tuple(best_action_obj(board, powers=powers, depth=depth))


def best_move(board: list[list[int]], depth: int = 4) -> str | None:
    action = best_action_obj(board, powers={}, depth=depth)
    if not isinstance(action, MoveAction):
        return None
    return action.direction
