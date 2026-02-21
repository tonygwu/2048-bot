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
