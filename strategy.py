"""
strategy.py — Pure-Python 2048 game logic + Expectimax AI

Board representation: list[list[int]], 4×4, 0 = empty cell.

Key functions:
  apply_move(board, direction) → (new_board, score_delta, changed)
  best_move(board, depth=4)   → direction string or None (game over)
"""

import math
from dataclasses import dataclass

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

_TRANS_TABLE: dict = {}      # (board_bb, swap_uses, delete_uses) → score
_NEW_ENTRIES: dict = {}      # same keys — entries added this run (for DB flush)
_TRANS_STATS = {"hits": 0, "misses": 0}
_TRANS_CAP   = 500_000       # evict (clear all) when table grows beyond this
_KEEP_OVERSIZED_PRELOAD = False  # preserve huge DB preloads instead of nuking on first miss


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
    _TRANS_TABLE.update(entries)
    global _KEEP_OVERSIZED_PRELOAD
    _KEEP_OVERSIZED_PRELOAD = len(_TRANS_TABLE) > _TRANS_CAP


def drain_new_entries() -> dict:
    """Return and clear the set of entries added since the last drain (for DB flush)."""
    out = dict(_NEW_ENTRIES)
    _NEW_ENTRIES.clear()
    return out


def get_trans_stats() -> dict:
    """Return a copy of the current hit/miss counters."""
    return dict(_TRANS_STATS)


def reset_trans_stats() -> None:
    """Zero the hit/miss counters (call between games)."""
    _TRANS_STATS["hits"] = 0
    _TRANS_STATS["misses"] = 0

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
_DEPTH_MIN = 2
_DEPTH_MAX = 6
_D_SOFT_MAX = 5
_D_BASE = 0.55
_D_W_MAX = 0.18
_D_W_FULL = 1.10
_D_W_BLOCKED = 0.75
_D_W_ROUGH = 0.45


def auto_depth(board: list[list[int]]) -> int:
    """Return adaptive expectimax depth from board-state features."""
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

    rough_n = min(1.0, _roughness(board) / 36.0)

    score = (
        _D_BASE
        + _D_W_MAX * max_log
        + _D_W_FULL * fullness
        + _D_W_BLOCKED * blocked
        + _D_W_ROUGH * rough_n
    )

    # Open boards: reduce depth because future is less constrained.
    if empties >= 8:
        score -= 0.90
    # Jammed boards: raise depth to look further for escapes.
    elif empties <= 2:
        score += 0.35
    if valid <= 2:
        score += 0.25

    # Extra tactical bump for low-empty, high-roughness boards that often
    # require "surgery" despite modest max-tile values.
    if empties <= 3 and valid <= 3 and rough_n >= 0.9:
        score += 0.40

    depth = max(_DEPTH_MIN, min(_D_SOFT_MAX, int(round(score))))

    # Near-death board: allow depth 6 only when the board is both extremely
    # constrained and rough with a late-game max tile.
    if empties <= 1 and valid <= 2 and rough_n >= 0.95 and max_log >= 11.0:
        return _DEPTH_MAX

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

_W_EMPTY  = 270.0   # raw empty-cell count
_W_GRAD   =   2.2   # snake-gradient coefficient
_W_MONO   =  47.0   # monotonicity (penalty-based; higher = more monotone)
_W_ROUGH  =  35.0   # roughness penalty (subtracted)
_W_MERGE  =  65.0   # near-term merge potential
_W_MAX    = 120.0   # explicit max-tile push (log2 of max tile)
_W_SUMLOG = 140.0   # compressed tile-mass signal: log2(sum tiles + 1)
_W_MOB    =  95.0   # legal move count (0..4), favors robust/maneuverable states
_W_CORNER =  80.0   # bonus per log2(max_tile) when max tile sits at (0,0)

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
_W_PU_BASE        =  40.0   # heuristic units per effective use, per log2(max_tile)
_PU_DELETE_SCALE  =   2.0   # delete costs more to spend (removes ALL tiles of value)
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

    swap_uses   = min(powers.get("swap",   0), _MAX_SWAP_USES)
    delete_uses = min(powers.get("delete", 0), _MAX_DELETE_USES)

    # Dampen and gate proximity so early-game boards do not overvalue
    # hypothetical future unlocks over immediate survival/shape quality.
    prox_swap = 0.0
    prox_delete = 0.0
    if max_tile >= 128 and swap_uses < _MAX_SWAP_USES:
        prox_swap = 0.25 * _proximity_to_unlock(max_tile, _SWAP_UNLOCK_TILE)
    if max_tile >= 256 and delete_uses < _MAX_DELETE_USES:
        prox_delete = 0.25 * _proximity_to_unlock(max_tile, _DELETE_UNLOCK_TILE)

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

    Results are cached in _TRANS_TABLE keyed by (board_bb, swap_uses, delete_uses).
    """
    if powers is None:
        powers = {}
    swap_uses   = powers.get("swap",   0)
    delete_uses = powers.get("delete", 0)
    bb  = board_to_bb(board)
    key = (bb, swap_uses, delete_uses)

    cached = _TRANS_TABLE.get(key)
    if cached is not None:
        _TRANS_STATS["hits"] += 1
        return cached

    _TRANS_STATS["misses"] += 1

    empties = sum(1 for r in range(4) for c in range(4) if board[r][c] == 0)
    max_val = max(board[r][c] for r in range(4) for c in range(4))
    tile_sum = sum(board[r][c] for r in range(4) for c in range(4))
    legal_moves = _legal_move_count(board)

    corner_bonus = (_W_CORNER * math.log2(max_val)
                    if max_val > 0 and board[0][0] == max_val else 0.0)

    result = (
          _W_EMPTY *  empties
        + _W_GRAD  * _gradient(board)
        + _W_MONO  * _monotonicity(board)
        - _W_ROUGH * _roughness(board)
        + _W_MERGE * _merge_potential(board)
        + _W_MAX   * (math.log2(max_val) if max_val > 0 else 0.0)
        + _W_SUMLOG * math.log2(tile_sum + 1.0)
        + _W_MOB   * legal_moves
        + corner_bonus
        + _powerup_value(max_val, powers)
    )

    global _KEEP_OVERSIZED_PRELOAD
    if len(_TRANS_TABLE) >= _TRANS_CAP:
        if _KEEP_OVERSIZED_PRELOAD:
            # When a large table is preloaded from SQLite, keep it intact and
            # avoid inserting new in-memory keys once at capacity.
            _NEW_ENTRIES[key] = result
            return result
        _TRANS_TABLE.clear()
        _KEEP_OVERSIZED_PRELOAD = False
    _TRANS_TABLE[key] = result
    _NEW_ENTRIES[key] = result
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
