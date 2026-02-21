"""
cache.py — SQLite persistence for the score_board transposition table.

Database: cache/transposition.db
Schema:
  entries(board_bb INTEGER, swap_uses INTEGER, delete_uses INTEGER,
          version TEXT, score REAL,
          PRIMARY KEY (board_bb, swap_uses, delete_uses, version))

board_bb is a 64-bit value that can exceed Python's sqlite3 signed-int limit
(2^63 − 1) for late-game boards.  We store it as a signed int64 using
_to_signed / _from_signed helpers and reconstruct the unsigned value on read.
"""

import os
import sqlite3

_DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "cache", "transposition.db")
_ENV_DB_PATH = os.environ.get("TRANS_DB_PATH")
DB_PATH = os.path.abspath(os.path.expanduser(_ENV_DB_PATH)) if _ENV_DB_PATH else _DEFAULT_DB_PATH


# ── Signed-int helpers (SQLite stores INTEGER as signed 64-bit) ───────────────

def _to_signed(bb: int) -> int:
    """Unsigned 64-bit → signed 64-bit (for SQLite storage)."""
    return bb if bb < (1 << 63) else bb - (1 << 64)


def _from_signed(val: int) -> int:
    """Signed 64-bit → unsigned 64-bit (after reading from SQLite)."""
    return val if val >= 0 else val + (1 << 64)


# ── DB init ───────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            board_bb    INTEGER NOT NULL,
            swap_uses   INTEGER NOT NULL,
            delete_uses INTEGER NOT NULL,
            version     TEXT    NOT NULL,
            score       REAL    NOT NULL,
            PRIMARY KEY (board_bb, swap_uses, delete_uses, version)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_version ON entries(version)")
    conn.commit()
    return conn


# ── Public API ────────────────────────────────────────────────────────────────

def load_version(version: str) -> dict:
    """Load all cached entries for *version* into a dict keyed by
    (board_bb, swap_uses, delete_uses) → score.  Returns {} if DB is absent."""
    if not os.path.exists(DB_PATH):
        return {}
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT board_bb, swap_uses, delete_uses, score FROM entries WHERE version = ?",
            (version,),
        ).fetchall()
    finally:
        conn.close()
    return {
        (_from_signed(bb), su, du): score
        for bb, su, du, score in rows
    }


def save_entries(entries: dict, version: str) -> int:
    """Insert-or-replace *entries* into the DB under *version*.
    entries: {(board_bb, swap_uses, delete_uses): score}
    Returns the number of rows written."""
    if not entries:
        return 0
    conn = _connect()
    try:
        rows = [
            (_to_signed(bb), su, du, version, score)
            for (bb, su, du), score in entries.items()
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO entries "
            "(board_bb, swap_uses, delete_uses, version, score) VALUES (?,?,?,?,?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def list_versions() -> dict:
    """Return {version: row_count} for all versions stored in the DB."""
    if not os.path.exists(DB_PATH):
        return {}
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT version, COUNT(*) FROM entries GROUP BY version"
        ).fetchall()
    finally:
        conn.close()
    return {v: cnt for v, cnt in rows}


def get_all_states(version: str | None = None) -> list[tuple]:
    """Return raw rows as (board_bb_unsigned, swap_uses, delete_uses, version, score).
    If version is None, return all versions."""
    if not os.path.exists(DB_PATH):
        return []
    conn = _connect()
    try:
        if version is not None:
            rows = conn.execute(
                "SELECT board_bb, swap_uses, delete_uses, version, score "
                "FROM entries WHERE version = ?",
                (version,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT board_bb, swap_uses, delete_uses, version, score FROM entries"
            ).fetchall()
    finally:
        conn.close()
    return [(_from_signed(bb), su, du, v, sc) for bb, su, du, v, sc in rows]


def decode_board(bb: int) -> list[list[int]]:
    """Decode a bitboard back to a 4x4 list[list[int]] (tile values, not exponents)."""
    board = []
    for r in range(4):
        row = []
        for c in range(4):
            exp = (bb >> (4 * (r * 4 + c))) & 0xF
            row.append((1 << exp) if exp else 0)
        board.append(row)
    return board
