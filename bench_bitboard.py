#!/usr/bin/env python3
"""
bench_bitboard.py — Benchmark list-based vs bitboard apply_move.

Bitboard encoding
─────────────────
Each cell stores log2(tile) in 4 bits  (0 = empty, 1 = tile-2, 2 = tile-4 …)
All 16 cells are packed into one 64-bit Python int, row-major:

  bits  0-15  → row 0   (cell 0,0 in bits 0-3;  cell 0,3 in bits 12-15)
  bits 16-31  → row 1
  bits 32-47  → row 2
  bits 48-63  → row 3

Key idea
────────
There are only 2^16 = 65 536 possible row values.  We precompute the result
of sliding every row left and right once at startup (~3 ms).  Then each call
to apply_move is just 4 table lookups + a few bit shifts — no list allocation.
"""

import sys, time
sys.path.insert(0, '.')
from strategy import apply_move as list_apply_move, score_board

# ── Build move tables ─────────────────────────────────────────────────────────
# _LEFT[row]  → (new_row, score_delta)   slide row toward index 0
# _RIGHT[row] → (new_row, score_delta)   slide row toward index 3

_LEFT  = [None] * (1 << 16)
_RIGHT = [None] * (1 << 16)

def _build_tables():
    def slide_left(cells):
        """Slide 4 exponent values left; return (new_cells, score_delta)."""
        tiles = [x for x in cells if x]          # drop zeros
        out, score, i = [], 0, 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                merged = tiles[i] + 1             # add exponents = double tile
                out.append(merged)
                score += 1 << merged              # actual tile value gained
                i += 2
            else:
                out.append(tiles[i]); i += 1
        out += [0] * (4 - len(out))
        return out, score

    def encode(cells):
        return cells[0] | (cells[1] << 4) | (cells[2] << 8) | (cells[3] << 12)

    for row_val in range(1 << 16):
        cells = [(row_val >> (4 * i)) & 0xF for i in range(4)]

        lc, ls = slide_left(cells)
        _LEFT[row_val] = (encode(lc), ls)

        rc, rs = slide_left(cells[::-1])     # reverse → slide → reverse
        _RIGHT[row_val] = (encode(rc[::-1]), rs)

_build_tables()


# ── Public helpers ────────────────────────────────────────────────────────────

def pack(board):
    """list[list[int]] → 64-bit bitboard."""
    bb = 0
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            bb |= (v.bit_length() - 1 if v else 0) << (4 * (r * 4 + c))
    return bb


def unpack(bb):
    """64-bit bitboard → list[list[int]]."""
    board = []
    for r in range(4):
        row = []
        for c in range(4):
            exp = (bb >> (4 * (r * 4 + c))) & 0xF
            row.append((1 << exp) if exp else 0)
        board.append(row)
    return board


def bb_apply_move(bb, direction):
    """Apply a move to a bitboard.  Returns (new_bb, score_delta, changed)."""
    ROW_MASK = 0xFFFF
    new_bb = score = 0

    if direction in ('left', 'right'):
        table = _LEFT if direction == 'left' else _RIGHT
        for r in range(4):
            row = (bb >> (r * 16)) & ROW_MASK
            nr, s = table[row]
            new_bb |= nr << (r * 16)
            score += s

    else:  # 'up' or 'down' — treat each column as a row
        table = _LEFT if direction == 'up' else _RIGHT
        for c in range(4):
            # Pack column c top-to-bottom into a 16-bit row value
            col = sum(((bb >> (4 * (r * 4 + c))) & 0xF) << (4 * r)
                      for r in range(4))
            nc, s = table[col]
            # Scatter result back into column c
            for r in range(4):
                new_bb |= ((nc >> (4 * r)) & 0xF) << (4 * (r * 4 + c))
            score += s

    return new_bb, score, new_bb != bb


# ── Benchmark ─────────────────────────────────────────────────────────────────

BOARD = [
    [512, 256, 128,  64],
    [  8,  16,  32,  32],
    [  4,   8,   4,   2],
    [  0,   0,   2,   0],
]
DIRS = ['left', 'right', 'up', 'down']
N = 400_000      # calls per implementation
bb = pack(BOARD)

# ── Correctness check ─────────────────────────────────────────────────────────
print("Correctness check (list result == bitboard result):")
all_ok = True
for d in DIRS:
    lr, ls, lc = list_apply_move(BOARD, d)
    br, bs, bc = bb_apply_move(bb, d)
    match = (unpack(br) == lr and bs == ls and bc == lc)
    if not match:
        all_ok = False
    print(f"  {d:5s}: {'OK' if match else 'MISMATCH  ← bug!'}")
if not all_ok:
    sys.exit(1)
print()

# ── Speed: apply_move ─────────────────────────────────────────────────────────
print(f"apply_move benchmark  ({N:,} iterations × 4 directions = {N*4:,} calls)")

# warm-up
for _ in range(2000):
    for d in DIRS:
        list_apply_move(BOARD, d)
        bb_apply_move(bb, d)

t0 = time.perf_counter()
for _ in range(N):
    for d in DIRS:
        list_apply_move(BOARD, d)
t_list = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(N):
    for d in DIRS:
        bb_apply_move(bb, d)
t_bb = time.perf_counter() - t0

print(f"  list-based : {t_list:.3f}s  ({N*4/t_list/1e6:.2f}M ops/s)")
print(f"  bitboard   : {t_bb:.3f}s  ({N*4/t_bb/1e6:.2f}M ops/s)")
print(f"  speedup    : {t_list/t_bb:.2f}×")
print()

# ── Speed: score_board (for context) ─────────────────────────────────────────
print(f"score_board benchmark  ({N:,} iterations — not yet bitboard-ized)")
t0 = time.perf_counter()
for _ in range(N):
    score_board(BOARD)
t_score = time.perf_counter() - t0
print(f"  score_board: {t_score:.3f}s  ({N/t_score/1e6:.2f}M ops/s)")
print()

# ── Estimate end-to-end impact ────────────────────────────────────────────────
# At depth 4: ~2,549 nodes total, ~2,304 leaf evaluations.
# Each max node calls apply_move up to 4 times → ~245 × 4 ≈ 980 apply_move calls.
# Each leaf calls score_board once → 2,304 score_board calls.
apply_calls   = 980
score_calls   = 2304
t_per_apply   = t_list  / (N * 4)
t_per_bb      = t_bb    / (N * 4)
t_per_score   = t_score / N

tree_list = apply_calls * t_per_apply + score_calls * t_per_score
tree_bb   = apply_calls * t_per_bb   + score_calls * t_per_score

print("Estimated depth-4 tree cost (per action):")
print(f"  apply_move share  : {apply_calls * t_per_apply * 1000:.2f}ms"
      f"  ({apply_calls * t_per_apply / tree_list * 100:.0f}% of total)")
print(f"  score_board share : {score_calls * t_per_score * 1000:.2f}ms"
      f"  ({score_calls * t_per_score / tree_list * 100:.0f}% of total)")
print(f"  total (list)      : {tree_list * 1000:.2f}ms")
print(f"  total (bb)        : {tree_bb   * 1000:.2f}ms")
print(f"  end-to-end speedup: {tree_list / tree_bb:.2f}×")
print()
print("Note: end-to-end speedup is limited by score_board's share.")
print("Bitboard-izing score_board too would compound both gains.")
