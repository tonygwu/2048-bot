"""Shared simulation helpers used by tests and benchmarks."""

import random


def place_random_tile(board: list[list[int]], rng: random.Random) -> list[list[int]]:
    """Spawn a 2 (90%) or 4 (10%) in a random empty cell. Returns a new board."""
    empties = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]
    if not empties:
        return board
    r, c = rng.choice(empties)
    nb = [row[:] for row in board]
    nb[r][c] = 2 if rng.random() < 0.9 else 4
    return nb


def _count_created_in_line(line: list[int], target_tile: int) -> int:
    """Count how many times target_tile is created by left-slide merges on one line."""
    tiles = [x for x in line if x != 0]
    created = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] * 2
            if merged == target_tile:
                created += 1
            i += 2
        else:
            i += 1
    return created


def count_created_tile(board: list[list[int]], direction: str, target_tile: int) -> int:
    """Count new target_tile merges created by a move direction on board."""
    if target_tile <= 0:
        return 0
    created = 0
    if direction == "left":
        for row in board:
            created += _count_created_in_line(row, target_tile)
    elif direction == "right":
        for row in board:
            created += _count_created_in_line(row[::-1], target_tile)
    elif direction == "up":
        for c in range(4):
            col = [board[r][c] for r in range(4)]
            created += _count_created_in_line(col, target_tile)
    elif direction == "down":
        for c in range(4):
            col = [board[r][c] for r in range(4)]
            created += _count_created_in_line(col[::-1], target_tile)
    else:
        raise ValueError(f"Unknown direction: {direction!r}")
    return created


def recharge_delete_uses(
    powers: dict,
    created_512_tiles: int,
    *,
    max_uses: int = 2,
) -> dict:
    """Apply delete recharge from newly-created 512 tiles, capped at max_uses."""
    if created_512_tiles <= 0:
        return powers
    out = dict(powers)
    out["delete"] = min(max_uses, max(0, int(out.get("delete", 0))) + int(created_512_tiles))
    return out
