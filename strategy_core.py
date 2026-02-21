"""Core deterministic board mechanics for 2048."""

DIRECTIONS = ["up", "down", "left", "right"]


def board_to_bb(board: list[list[int]]) -> int:
    """Encode a 4x4 board as a 64-bit int: 4 bits per cell = log2(tile), row-major."""
    bb = 0
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            if v > 0:
                bb |= (v.bit_length() - 1) << (4 * (r * 4 + c))
    return bb


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
    """Apply a move. Returns (new_board, score_delta, changed)."""
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
