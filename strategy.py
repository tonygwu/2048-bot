"""
strategy.py — Pure-Python 2048 game logic + Expectimax AI

Board representation: list[list[int]], 4×4, 0 = empty cell.

Key functions:
  apply_move(board, direction) → (new_board, score_delta, changed)
  best_move(board, depth=4)   → direction string or None (game over)
"""

import math

DIRECTIONS = ["up", "down", "left", "right"]

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


# ── Heuristic evaluation ───────────────────────────────────────────────────────
#
# Eval: V(s,p) = w_E·E + w_G·G(s) + w_mono·Mono(s) − w_rough·Rough(s)
#             + w_merge·MergePot(s) + w_max·M + w_score·ΣTile
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

_W_EMPTY = 270.0   # raw empty-cell count
_W_GRAD  =   2.2   # snake-gradient coefficient
_W_MONO  =  47.0   # monotonicity (penalty-based; higher = more monotone)
_W_ROUGH =  35.0   # roughness penalty (subtracted)
_W_MERGE =  65.0   # near-term merge potential
_W_MAX   = 120.0   # explicit max-tile push (log2 of max tile)
_W_SCORE =   1.0   # score proxy (sum of tile values; keep small)

# ── Power-up valuation ─────────────────────────────────────────────────────────
#
# Each power-up use is worth _W_PU_BASE × log2(max_tile) heuristic units,
# so value scales with game stage (later = scarcer board, higher rescue value).
#
# "Effective uses" = actual_uses + proximity_bonus, where proximity_bonus ∈ [0, 1)
# represents how close the board is to earning the next use from the game
# (swap unlocks at 256, delete at 512).  The bonus is zeroed out when uses
# are already at the cap, which creates a natural incentive to spend a power-up
# right before earning a replacement — you go from (cap, near-cap-ignored) to
# (cap-1, near-cap-counts ≈ 0.9) and the board improvement only needs to cover
# the small residual gap.
_W_PU_BASE        = 100.0   # heuristic units per effective use, per log2(max_tile)
_PU_DELETE_SCALE  =   1.2   # delete is slightly more valuable than swap
_MAX_SWAP_USES    =   2     # game caps storable swap uses at 2
_MAX_DELETE_USES  =   2     # game caps storable delete uses at 2
_SWAP_UNLOCK_TILE =  256    # creating a 256 tile earns a swap use
_DELETE_UNLOCK_TILE = 512   # creating a 512 tile earns a delete use


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

    For each row/column we compute the sum of log-drops in both possible
    directions, then keep the *smaller* penalty (the direction the row is
    closer to being monotone in).  Summed over all rows + cols and negated,
    so the return value is 0 for a perfectly monotone board and increasingly
    negative for non-monotone layouts.
    """
    total = 0.0
    for r in range(4):
        left_pen = right_pen = 0.0
        for c in range(3):
            la = math.log2(board[r][c])     if board[r][c]     > 0 else 0.0
            lb = math.log2(board[r][c + 1]) if board[r][c + 1] > 0 else 0.0
            left_pen  += max(0.0, la - lb)   # drop going left→right
            right_pen += max(0.0, lb - la)   # drop going right→left
        total += min(left_pen, right_pen)

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


def _proximity_to_unlock(max_tile: int, unlock_tile: int) -> float:
    """Fraction of the way to the next power-up unlock, in log2-space.

    Returns a value in [0, 0.95).  Capped below 1 so it never fully equals
    an earned use — only the actual powers dict can do that.

    Examples (swap unlock = 256):
      max_tile=128 → log2(128)/log2(256) = 7/8 = 0.875
      max_tile= 64 → 6/8 = 0.75
      max_tile=256 → capped at 0.95  (you've reached/passed the threshold)
    """
    if max_tile <= 0:
        return 0.0
    return min(0.95, math.log2(max_tile) / math.log2(unlock_tile))


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

    swap_uses   = min(powers.get("swap",   0), _MAX_SWAP_USES)
    delete_uses = min(powers.get("delete", 0), _MAX_DELETE_USES)

    prox_swap   = _proximity_to_unlock(max_tile, _SWAP_UNLOCK_TILE)   if swap_uses   < _MAX_SWAP_USES   else 0.0
    prox_delete = _proximity_to_unlock(max_tile, _DELETE_UNLOCK_TILE) if delete_uses < _MAX_DELETE_USES else 0.0

    effective_swaps   = swap_uses   + prox_swap
    effective_deletes = delete_uses + prox_delete

    value_per_swap   = _W_PU_BASE * stage
    value_per_delete = _W_PU_BASE * stage * _PU_DELETE_SCALE

    return value_per_swap * effective_swaps + value_per_delete * effective_deletes


def score_board(board: list[list[int]], powers: dict | None = None) -> float:
    """Static evaluation of a board position (higher = better for the player).

    powers (optional) is the current {"swap": N, "delete": N} inventory.
    When supplied, the power-up value is folded into the score so the search
    tree naturally prices spending a use.
    """
    if powers is None:
        powers = {}
    empties = sum(1 for r in range(4) for c in range(4) if board[r][c] == 0)
    max_val = max(board[r][c] for r in range(4) for c in range(4))
    tile_sum = sum(board[r][c] for r in range(4) for c in range(4))

    return (
          _W_EMPTY *  empties
        + _W_GRAD  * _gradient(board)
        + _W_MONO  * _monotonicity(board)
        - _W_ROUGH * _roughness(board)
        + _W_MERGE * _merge_potential(board)
        + _W_MAX   * (math.log2(max_val) if max_val > 0 else 0.0)
        + _W_SCORE * tile_sum
        + _powerup_value(max_val, powers)
    )


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


def best_action(
    board: list[list[int]],
    powers: dict | None = None,
    depth: int = 4,
) -> tuple | None:
    """Evaluate all available actions (moves + power-ups) and return the best.

    Return value is a tuple whose first element identifies the action:
      ("move",   direction)
      ("swap",   r1, c1, r2, c2)
      ("delete", value, row, col)   # (row,col) = position of one tile with that value

    Power-up cost is priced implicitly: when a power-up is used the search
    tree receives a decremented powers dict, so score_board at every leaf
    reflects one fewer use.  No explicit shadow cost needed.

    Returns None if no action is possible (game over).
    """
    if powers is None:
        powers = {}
    best_val = float("-inf")
    best_act: tuple | None = None

    # ── Regular moves (powers unchanged through the tree) ─────────────────────
    for d in DIRECTIONS:
        nb, score_delta, changed = apply_move(board, d)
        if not changed:
            continue
        val = _expectimax(nb, depth - 1, False, powers) + score_delta
        if val > best_val:
            best_val = val
            best_act = ("move", d)

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
                    best_act = ("swap", r1, c1, r2, c2)

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
                best_act = ("delete", v, pos[0], pos[1])

    return best_act


def best_move(board: list[list[int]], depth: int = 4) -> str | None:
    """Backwards-compat wrapper: return only the direction string (no powers)."""
    action = best_action(board, powers={}, depth=depth)
    if action is None or action[0] != "move":
        return None
    return action[1]
