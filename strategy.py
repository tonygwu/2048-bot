"""
strategy.py — Pure-Python 2048 game logic + Expectimax AI

Board representation: list[list[int]], 4×4, 0 = empty cell.

Key functions:
  apply_move(board, direction) → (new_board, score_delta, changed)
  best_move(board, depth=4)   → direction string or None (game over)
"""

import math
from dataclasses import dataclass
from transposition_cache import TranspositionCache
from strategy_config import (
    DEFAULT_DEPTH_POLICY,
    DEFAULT_EVAL_WEIGHTS,
    DEFAULT_POWERUP_POLICY,
    EvalWeights,
)

DIRECTIONS = ["up", "down", "left", "right"]


@dataclass(frozen=True)
class MoveAction:
    direction: str


@dataclass(frozen=True)
class SwapAction:
    r1: int
    c1: int
    r2: int
    c2: int


@dataclass(frozen=True)
class DeleteAction:
    value: int
    row: int
    col: int


Action = MoveAction | SwapAction | DeleteAction
ActionTuple = tuple


def action_to_tuple(action: Action | None) -> ActionTuple | None:
    """Convert typed action objects to the legacy tuple action format."""
    if action is None:
        return None
    if isinstance(action, MoveAction):
        return ("move", action.direction)
    if isinstance(action, SwapAction):
        return ("swap", action.r1, action.c1, action.r2, action.c2)
    if isinstance(action, DeleteAction):
        return ("delete", action.value, action.row, action.col)
    raise TypeError(f"Unknown action type: {type(action)!r}")


def action_from_tuple(action: ActionTuple | None) -> Action | None:
    """Convert legacy tuple actions to typed action objects."""
    if action is None:
        return None
    kind = action[0]
    if kind == "move":
        return MoveAction(direction=action[1])
    if kind == "swap":
        _, r1, c1, r2, c2 = action
        return SwapAction(r1=r1, c1=c1, r2=r2, c2=c2)
    if kind == "delete":
        _, value, row, col = action
        return DeleteAction(value=value, row=row, col=col)
    raise ValueError(f"Unknown action tuple kind: {kind!r}")

# ── Transposition table ────────────────────────────────────────────────────────
# Bump this string whenever heuristic weights or eval logic changes.
# The SQLite cache stores scores per-version; stale entries are ignored.
SCORE_BOARD_VERSION = "1.1"

_TRANS_CAP   = 500_000       # evict (clear all) when table grows beyond this
_TRANS_CACHE = TranspositionCache(cap=_TRANS_CAP)


def board_to_bb(board: list[list[int]]) -> int:
    """Encode a 4x4 board as a 64-bit int: 4 bits per cell = log2(tile), row-major."""
    bb = 0
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            if v > 0:
                bb |= (v.bit_length() - 1) << (4 * (r * 4 + c))
    return bb


def load_trans_table(entries: dict) -> None:
    """Bulk-load pre-computed entries into _TRANS_TABLE (called at bot startup)."""
    _TRANS_CACHE.load(entries)


def drain_new_entries() -> dict:
    """Return and clear the set of entries added since the last drain (for DB flush)."""
    return _TRANS_CACHE.drain_new_entries()


def get_trans_stats() -> dict:
    """Return a copy of the current hit/miss counters."""
    return _TRANS_CACHE.stats()


def reset_trans_stats() -> None:
    """Zero the hit/miss counters (call between games)."""
    _TRANS_CACHE.reset_stats()

# ── Game simulation ────────────────────────────────────────────────────────────

def _slide_row(row: list[int]) -> tuple[list[int], int]:
    """Slide and merge one row to the left. Returns (new_row, score_delta)."""
    tiles = [x for x in row if x != 0]
    merged: list[int] = []
    score = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            v = tiles[i] * 2
            merged.append(v)
            score += v
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged += [0] * (4 - len(merged))
    return merged, score


def apply_move(board: list[list[int]], direction: str) -> tuple[list[list[int]], int, bool]:
    """
    Apply a move. Returns (new_board, score_delta, changed).
    Does NOT place a new tile — that is handled by the chance node.
    """
    score = 0

    if direction == "left":
        rows = []
        for row in board:
            r, s = _slide_row(row)
            rows.append(r)
            score += s
        new_board = rows

    elif direction == "right":
        rows = []
        for row in board:
            r, s = _slide_row(row[::-1])
            rows.append(r[::-1])
            score += s
        new_board = rows

    elif direction == "up":
        cols = []
        for c in range(4):
            col = [board[r][c] for r in range(4)]
            new_col, s = _slide_row(col)
            cols.append(new_col)
            score += s
        new_board = [[cols[c][r] for c in range(4)] for r in range(4)]

    elif direction == "down":
        cols = []
        for c in range(4):
            col = [board[r][c] for r in range(4)]
            new_col, s = _slide_row(col[::-1])
            cols.append(new_col[::-1])
            score += s
        new_board = [[cols[c][r] for c in range(4)] for r in range(4)]

    else:
        raise ValueError(f"Unknown direction: {direction!r}")

    changed = new_board != board
    return new_board, score, changed


def empty_cells(board: list[list[int]]) -> list[tuple[int, int]]:
    return [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]


def is_game_over(board: list[list[int]]) -> bool:
    """True when no move changes the board."""
    for d in DIRECTIONS:
        _, _, changed = apply_move(board, d)
        if changed:
            return False
    return True


# ── Adaptive search depth ──────────────────────────────────────────────────────
# Depth is chosen from multiple board features (not max tile alone):
# - max tile stage (late-game value concentration)
# - fullness / empty-cell count (chance branching + tactical tightness)
# - mobility (how many legal directions remain)
# - roughness (local value discontinuity; higher = messier board)
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

    score = (
        p.base
        + p.w_max * max_log
        + p.w_full * fullness
        + p.w_blocked * blocked
        + p.w_rough * rough_n
    )

    # Open boards: reduce depth because future is less constrained.
    if empties >= p.open_empties_threshold:
        score -= p.open_penalty
    # Jammed boards: raise depth to look further for escapes.
    elif empties <= p.jammed_empties_threshold:
        score += p.jammed_bonus
    if valid <= p.low_valid_bonus_threshold:
        score += p.low_valid_bonus

    # Extra tactical bump for low-empty, high-roughness boards that often
    # require "surgery" despite modest max-tile values.
    if (
        empties <= p.surgery_empties_threshold
        and valid <= p.surgery_valid_threshold
        and rough_n >= p.surgery_rough_threshold
    ):
        score += p.surgery_bonus

    depth = max(p.min_depth, min(p.soft_max, int(round(score))))

    # Near-death board: allow depth 6 only when the board is both extremely
    # constrained and rough with a late-game max tile.
    if (
        empties <= p.near_death_empties_threshold
        and valid <= p.near_death_valid_threshold
        and rough_n >= p.near_death_rough_threshold
        and max_log >= p.near_death_max_log_threshold
    ):
        return p.max_depth

    return depth


# ── Heuristic evaluation ───────────────────────────────────────────────────────
#
# Eval: V(s,p) = w_E·E + w_G·G(s) + w_mono·Mono(s) − w_rough·Rough(s)
#             + w_merge·MergePot(s) + w_max·M + w_sum·log2(ΣTile+1)
#             + w_mob·LegalMoves(s)
#             + PowerupValue(s, p)
#
# All tile quantities are in log-space: L(x) = log2(x), L(0) = 0.
# p = powers dict {"swap": N, "delete": N}; incorporated so the tree
# naturally prices the cost of spending a power-up.

# Snake-pattern gradient weights — upper-left (0,0) is home corner.
# G(s) = Σ _SNAKE[r][c] × L(tile[r][c])
_SNAKE = [
    [16, 15, 14, 13],
    [ 9, 10, 11, 12],
    [ 8,  7,  6,  5],
    [ 1,  2,  3,  4],
]

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

# Row directions required by the snake pattern:
#   rows 0, 2 → tiles should DECREASE left→right (weights 16↘13 and 8↘5)
#   rows 1, 3 → tiles should INCREASE left→right (weights 9↗12 and 1↗4)
_SNAKE_ROW_SHOULD_DECREASE = [True, False, True, False]

# ── Power-up valuation ─────────────────────────────────────────────────────────
#
# Each power-up use is worth _W_PU_BASE × log2(max_tile) heuristic units,
# so value scales with game stage (later = scarcer board, higher rescue value).
#
# "Effective uses" = actual_uses + proximity_bonus, where proximity_bonus ∈ [0, 1)
# models how close the board is to earning the next use (swap unlocks at 256,
# delete at 512).  Proximity is only counted while max_tile < unlock_tile;
# once the threshold is passed the tree itself will see the earned use in
# future boards — carrying a phantom 0.95 "inventory" forever was stale signal.
def _gradient(board: list[list[int]]) -> float:
    """G(s) = Σ _SNAKE[r][c] × L(tile). Anchors large tiles to the corner."""
    s = 0.0
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            if v > 0:
                s += _SNAKE[r][c] * math.log2(v)
    return s


def _monotonicity(board: list[list[int]]) -> float:
    """Penalty-based monotonicity.

    Rows: direction-aware — each row has a required monotone direction dictated
    by the snake pattern (_SNAKE_ROW_SHOULD_DECREASE).  Only penalise violations
    of that specific direction; never reward going the wrong way.

    Columns: agnostic — take the smaller penalty across both up and down, since
    the snake winds vertically and the preferred column direction is less fixed.

    Return value: 0 for a perfectly monotone board, increasingly negative for
    non-monotone layouts.
    """
    total = 0.0
    for r in range(4):
        pen = 0.0
        should_decrease = _SNAKE_ROW_SHOULD_DECREASE[r]
        for c in range(3):
            la = math.log2(board[r][c])     if board[r][c]     > 0 else 0.0
            lb = math.log2(board[r][c + 1]) if board[r][c + 1] > 0 else 0.0
            if should_decrease:
                pen += max(0.0, lb - la)   # penalise increases left→right
            else:
                pen += max(0.0, la - lb)   # penalise decreases left→right
        total += pen

    for c in range(4):
        up_pen = down_pen = 0.0
        for r in range(3):
            la = math.log2(board[r][c])     if board[r][c]     > 0 else 0.0
            lb = math.log2(board[r + 1][c]) if board[r + 1][c] > 0 else 0.0
            up_pen   += max(0.0, la - lb)
            down_pen += max(0.0, lb - la)
        total += min(up_pen, down_pen)

    return -total   # negate: 0 = perfectly monotone, negative = rough


def _roughness(board: list[list[int]]) -> float:
    """Σ |L(a) − L(b)| over all adjacent nonzero pairs.
    Returned as a positive number; the caller subtracts it."""
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
    """Reward positions where merges are immediately available or one slide away.

    +L(tile)   for each adjacent equal pair (direct merge)
    +0.3·L(tile) for equal tiles in the same row/col separated only by zeros
    """
    s = 0.0
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            if v == 0:
                continue
            lv = math.log2(v)
            # Direct horizontal merge
            if c + 1 < 4 and board[r][c + 1] == v:
                s += lv
            # Direct vertical merge
            if r + 1 < 4 and board[r + 1][c] == v:
                s += lv
            # Line-of-sight horizontal (only zeros between c and c2)
            for c2 in range(c + 2, 4):
                if board[r][c2] != 0:
                    if board[r][c2] == v:
                        s += 0.3 * lv
                    break
            # Line-of-sight vertical
            for r2 in range(r + 2, 4):
                if board[r2][c] != 0:
                    if board[r2][c] == v:
                        s += 0.3 * lv
                    break
    return s


def _legal_move_count(board: list[list[int]]) -> int:
    """Number of directions that change the board (0..4)."""
    count = 0
    for d in DIRECTIONS:
        _, _, changed = apply_move(board, d)
        if changed:
            count += 1
    return count


def _proximity_to_unlock(max_tile: int, unlock_tile: int) -> float:
    """Fraction of the way to the next power-up unlock, in log2-space.

    Returns a value in [0, 0.95) while max_tile < unlock_tile, then 0 once
    the threshold is passed.  Once max_tile >= unlock_tile the game has already
    awarded the use; future re-awards happen as new tiles are created and will
    be reflected in the powers dict directly.  Carrying a phantom 0.95 bonus
    forever was stale, direction-less signal.

    Examples (swap unlock = 256):
      max_tile=128 → log2(128)/log2(256) = 7/8 = 0.875
      max_tile= 64 → 6/8 = 0.75
      max_tile≥256 → 0.0  (threshold passed; tree handles future awards)
    """
    if max_tile <= 0 or max_tile >= unlock_tile:
        return 0.0
    return math.log2(max_tile) / math.log2(unlock_tile)


def _powerup_value(max_tile: int, powers: dict) -> float:
    """Heuristic value of the current power-up inventory.

    effective_swaps   = min(actual_swaps,   MAX) + proximity_swap   (if below cap)
    effective_deletes = min(actual_deletes, MAX) + proximity_delete (if below cap)

    Value per effective use scales with log2(max_tile) so power-ups are
    worth more later in the game.
    """
    if not powers or max_tile <= 0:
        return 0.0

    stage = math.log2(max_tile)   # ≈ 1 early, ≈ 14 very late

    p = DEFAULT_POWERUP_POLICY
    swap_uses = min(powers.get("swap", 0), p.max_swap_uses)
    delete_uses = min(powers.get("delete", 0), p.max_delete_uses)

    # Dampen and gate proximity so early-game boards do not overvalue
    # hypothetical future unlocks over immediate survival/shape quality.
    prox_swap = 0.0
    prox_delete = 0.0
    if max_tile >= p.prox_swap_min_tile and swap_uses < p.max_swap_uses:
        prox_swap = p.prox_scale * _proximity_to_unlock(max_tile, p.swap_unlock_tile)
    if max_tile >= p.prox_delete_min_tile and delete_uses < p.max_delete_uses:
        prox_delete = p.prox_scale * _proximity_to_unlock(max_tile, p.delete_unlock_tile)

    effective_swaps   = swap_uses   + prox_swap
    effective_deletes = delete_uses + prox_delete

    value_per_swap = p.w_base * stage
    value_per_delete = p.w_base * stage * p.delete_scale

    return value_per_swap * effective_swaps + value_per_delete * effective_deletes


def extract_eval_features(board: list[list[int]], powers: dict | None = None) -> EvalFeatures:
    """Extract raw evaluation features from a board state."""
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
    """Compute scalar board score from raw features and a weight set."""
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
    """Static evaluation of a board position (higher = better for the player).

    powers (optional) is the current {"swap": N, "delete": N} inventory.
    When supplied, the power-up value is folded into the score so the search
    tree naturally prices spending a use.

    Results are cached by (board_bb, swap_uses, delete_uses).
    """
    if powers is None:
        powers = {}
    swap_uses   = powers.get("swap",   0)
    delete_uses = powers.get("delete", 0)
    bb  = board_to_bb(board)
    key = (bb, swap_uses, delete_uses)

    cached = _TRANS_CACHE.get(key)
    if cached is not None:
        return cached

    features = extract_eval_features(board, powers)
    result = score_from_features(features)
    _TRANS_CACHE.store(key, result)
    return result


# ── Expectimax ────────────────────────────────────────────────────────────────

def _expectimax(board: list[list[int]], depth: int, is_max: bool,
                powers: dict | None = None) -> float:
    if powers is None:
        powers = {}
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

    else:  # chance node
        empties = empty_cells(board)
        if not empties:
            return score_board(board, powers)

        # Cap the number of empty cells we branch on when the board is open.
        # Empirically, sampling 8 cells instead of 15 gives ~3× speedup with
        # negligible quality loss because the position heuristic already
        # accounts for empty-cell count.
        sample = empties if len(empties) <= 6 else empties[::len(empties) // 6 + 1][:6]

        total = 0.0
        for r, c in sample:
            board[r][c] = 2
            total += 0.9 * _expectimax(board, depth - 1, True, powers)
            board[r][c] = 4
            total += 0.1 * _expectimax(board, depth - 1, True, powers)
            board[r][c] = 0   # restore

        return total / len(sample)


# ── Power-up simulation ────────────────────────────────────────────────────────

def apply_swap(board: list[list[int]], r1: int, c1: int, r2: int, c2: int) -> list[list[int]]:
    """Return a new board with the tiles at (r1,c1) and (r2,c2) swapped."""
    nb = [row[:] for row in board]
    nb[r1][c1], nb[r2][c2] = nb[r2][c2], nb[r1][c1]
    return nb


def apply_delete(board: list[list[int]], value: int) -> list[list[int]]:
    """Return a new board with all tiles equal to 'value' removed (set to 0)."""
    return [[0 if v == value else v for v in row] for row in board]


def _top_positions(board: list[list[int]], k: int = 6) -> list[tuple[int, int]]:
    """Return (row, col) of the top-k tiles by value, for swap candidate generation."""
    tiles = sorted(
        ((board[r][c], r, c) for r in range(4) for c in range(4) if board[r][c] > 0),
        reverse=True,
    )
    return [(r, c) for _, r, c in tiles[:k]]


def best_action_obj(
    board: list[list[int]],
    powers: dict | None = None,
    depth: int = 4,
) -> Action | None:
    """Evaluate all available actions (moves + power-ups) and return the best.

    Returns one of:
      MoveAction(direction)
      SwapAction(r1, c1, r2, c2)
      DeleteAction(value, row, col)

    Power-up cost is priced implicitly: when a power-up is used the search
    tree receives a decremented powers dict, so score_board at every leaf
    reflects one fewer use.  No explicit shadow cost needed.

    Returns None if no action is possible (game over).
    """
    if powers is None:
        powers = {}
    best_val = float("-inf")
    best_act: Action | None = None

    # ── Regular moves (powers unchanged through the tree) ─────────────────────
    for d in DIRECTIONS:
        nb, score_delta, changed = apply_move(board, d)
        if not changed:
            continue
        val = _expectimax(nb, depth - 1, False, powers) + score_delta
        if val > best_val:
            best_val = val
            best_act = MoveAction(direction=d)

    # ── Swap two tiles (tree sees swap count decremented by 1) ────────────────
    if powers.get("swap", 0) > 0:
        powers_after = {**powers, "swap": powers["swap"] - 1}
        top = _top_positions(board, k=6)
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                r1, c1 = top[i]
                r2, c2 = top[j]
                if board[r1][c1] == board[r2][c2]:
                    continue   # swapping equal tiles does nothing useful
                nb = apply_swap(board, r1, c1, r2, c2)
                val = _expectimax(nb, depth - 1, False, powers_after)
                if val > best_val:
                    best_val = val
                    best_act = SwapAction(r1=r1, c1=c1, r2=r2, c2=c2)

    # ── Delete tiles by value (tree sees delete count decremented by 1) ───────
    if powers.get("delete", 0) > 0:
        powers_after = {**powers, "delete": powers["delete"] - 1}
        empties = sum(1 for r in range(4) for c in range(4) if board[r][c] == 0)
        # Prefer deleting small clutter; expand candidates when board is jammed
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


def best_action(
    board: list[list[int]],
    powers: dict | None = None,
    depth: int = 4,
) -> ActionTuple | None:
    """Back-compat wrapper returning tuple actions.

    Prefer `best_action_obj` for new code.
    """
    return action_to_tuple(best_action_obj(board, powers=powers, depth=depth))


def best_move(board: list[list[int]], depth: int = 4) -> str | None:
    """Backwards-compat wrapper: return only the direction string (no powers)."""
    action = best_action_obj(board, powers={}, depth=depth)
    if not isinstance(action, MoveAction):
        return None
    return action.direction
