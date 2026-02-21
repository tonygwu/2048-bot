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
# Eval: V(s) = w_E·E + w_G·G(s) + w_mono·Mono(s) − w_rough·Rough(s)
#            + w_merge·MergePot(s) + w_max·M + w_score·ΣTile
#
# All tile quantities are in log-space: L(x) = log2(x), L(0) = 0.

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


def score_board(board: list[list[int]]) -> float:
    """Static evaluation of a board position (higher = better for the player)."""
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
    )


# ── Expectimax ────────────────────────────────────────────────────────────────

def _expectimax(board: list[list[int]], depth: int, is_max: bool) -> float:
    if depth == 0:
        return score_board(board)

    if is_max:
        best = float("-inf")
        any_valid = False
        for d in DIRECTIONS:
            nb, _, changed = apply_move(board, d)
            if not changed:
                continue
            any_valid = True
            val = _expectimax(nb, depth - 1, False)
            if val > best:
                best = val
        return best if any_valid else score_board(board)

    else:  # chance node
        empties = empty_cells(board)
        if not empties:
            return score_board(board)

        # Cap the number of empty cells we branch on when the board is open.
        # Empirically, sampling 8 cells instead of 15 gives ~3× speedup with
        # negligible quality loss because the position heuristic already
        # accounts for empty-cell count.
        sample = empties if len(empties) <= 6 else empties[::len(empties) // 6 + 1][:6]

        total = 0.0
        for r, c in sample:
            board[r][c] = 2
            total += 0.9 * _expectimax(board, depth - 1, True)
            board[r][c] = 4
            total += 0.1 * _expectimax(board, depth - 1, True)
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


# Shadow costs prevent power-ups from being used except when they give a
# significantly better outcome than any regular move.
_SWAP_COST   = 380.0
_DELETE_COST = 420.0


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

    Power-ups are only considered when powers[key] > 0.  A shadow cost is
    subtracted from each power-up's expectimax value so they are used only
    when they provide a meaningful advantage over the best regular move.

    Returns None if no action is possible (game over).
    """
    if powers is None:
        powers = {}
    best_val = float("-inf")
    best_act: tuple | None = None

    # ── Regular moves ─────────────────────────────────────────────────────────
    for d in DIRECTIONS:
        nb, score_delta, changed = apply_move(board, d)
        if not changed:
            continue
        val = _expectimax(nb, depth - 1, False) + score_delta
        if val > best_val:
            best_val = val
            best_act = ("move", d)

    # ── Swap two tiles ─────────────────────────────────────────────────────────
    if powers.get("swap", 0) > 0:
        top = _top_positions(board, k=6)
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                r1, c1 = top[i]
                r2, c2 = top[j]
                if board[r1][c1] == board[r2][c2]:
                    continue   # swapping equal tiles does nothing useful
                nb = apply_swap(board, r1, c1, r2, c2)
                val = _expectimax(nb, depth - 1, False) - _SWAP_COST
                if val > best_val:
                    best_val = val
                    best_act = ("swap", r1, c1, r2, c2)

    # ── Delete tiles by value ──────────────────────────────────────────────────
    if powers.get("delete", 0) > 0:
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
            val = _expectimax(nb, depth - 1, False) - _DELETE_COST
            if val > best_val:
                best_val = val
                # Return position of any tile with that value (for browser clicking)
                pos = next((r, c) for r in range(4) for c in range(4) if board[r][c] == v)
                best_act = ("delete", v, pos[0], pos[1])

    return best_act


def best_move(board: list[list[int]], depth: int = 4) -> str | None:
    """Backwards-compat wrapper: return only the direction string (no powers)."""
    action = best_action(board, powers={}, depth=depth)
    if action is None or action[0] != "move":
        return None
    return action[1]
