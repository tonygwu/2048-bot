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
_SHARED_DB_PATH = os.path.expanduser("~/Code/transposition.db")
_ENV_DB_PATH = os.environ.get("TRANS_DB_PATH")


def _ensure_default_symlink_to_shared(default_path: str, shared_path: str) -> str:
    """Best-effort bootstrap for fresh clones:
    create cache/transposition.db -> ~/Code/transposition.db when absent.
    """
    db_dir = os.path.dirname(default_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # Keep existing files/symlinks unchanged.
    if os.path.lexists(default_path):
        return default_path

    try:
        os.symlink(shared_path, default_path)
    except OSError:
        # If symlink creation is unavailable, fallback to the local path.
        pass
    return default_path


def _resolve_db_path() -> str:
    if _ENV_DB_PATH:
        return os.path.abspath(os.path.expanduser(_ENV_DB_PATH))
    return _ensure_default_symlink_to_shared(_DEFAULT_DB_PATH, _SHARED_DB_PATH)


DB_PATH = _resolve_db_path()

_SQLITE_TIMEOUT_S = 30.0
_SQLITE_BUSY_TIMEOUT_MS = 30_000
_SCHEMA_READY = False

_CREATE_ENTRIES_SQL = """
    CREATE TABLE IF NOT EXISTS entries (
        board_bb    INTEGER NOT NULL,
        swap_uses   INTEGER NOT NULL,
        delete_uses INTEGER NOT NULL,
        version     TEXT    NOT NULL,
        score       REAL    NOT NULL,
        PRIMARY KEY (board_bb, swap_uses, delete_uses, version)
    )
"""
_CREATE_VERSION_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_version ON entries(version)"


# ── Signed-int helpers (SQLite stores INTEGER as signed 64-bit) ───────────────

def _to_signed(bb: int) -> int:
    """Unsigned 64-bit → signed 64-bit (for SQLite storage)."""
    return bb if bb < (1 << 63) else bb - (1 << 64)


def _from_signed(val: int) -> int:
    """Signed 64-bit → unsigned 64-bit (after reading from SQLite)."""
    return val if val >= 0 else val + (1 << 64)


# ── DB init ───────────────────────────────────────────────────────────────────

def _set_common_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")


def _set_rw_pragmas(conn: sqlite3.Connection) -> None:
    _set_common_pragmas(conn)
    # WAL improves concurrent read/write behavior for multi-process workloads.
    conn.execute("PRAGMA journal_mode = WAL").fetchone()
    conn.execute("PRAGMA synchronous = NORMAL")


def _connect_rw() -> sqlite3.Connection:
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    return sqlite3.connect(DB_PATH, timeout=_SQLITE_TIMEOUT_S)


def _connect_ro() -> sqlite3.Connection:
    return sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True,
        timeout=_SQLITE_TIMEOUT_S,
    )


def init_db() -> None:
    """Initialize schema and connection mode once per process."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    conn = _connect_rw()
    try:
        _set_rw_pragmas(conn)
        conn.execute(_CREATE_ENTRIES_SQL)
        conn.execute(_CREATE_VERSION_INDEX_SQL)
        conn.commit()
    finally:
        conn.close()
    _SCHEMA_READY = True


# ── Public API ────────────────────────────────────────────────────────────────

def load_version(version: str) -> dict:
    """Load all cached entries for *version* into a dict keyed by
    (board_bb, swap_uses, delete_uses) → score.  Returns {} if DB is absent."""
    if not os.path.exists(DB_PATH):
        return {}
    conn = _connect_ro()
    try:
        _set_common_pragmas(conn)
        try:
            rows = conn.execute(
                "SELECT board_bb, swap_uses, delete_uses, score FROM entries WHERE version = ?",
                (version,),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return {}
            raise
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

    def _clamp_uses(v: int) -> int:
        # Game semantics cap each power-up to [0, 2].
        return max(0, min(2, int(v)))

    init_db()
    conn = _connect_rw()
    try:
        _set_rw_pragmas(conn)
        rows = [
            (_to_signed(bb), _clamp_uses(su), _clamp_uses(du), version, score)
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
    conn = _connect_ro()
    try:
        _set_common_pragmas(conn)
        try:
            rows = conn.execute(
                "SELECT version, COUNT(*) FROM entries GROUP BY version"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return {}
            raise
    finally:
        conn.close()
    return {v: cnt for v, cnt in rows}


def get_all_states(version: str | None = None) -> list[tuple]:
    """Return raw rows as (board_bb_unsigned, swap_uses, delete_uses, version, score).
    If version is None, return all versions."""
    if not os.path.exists(DB_PATH):
        return []
    conn = _connect_ro()
    try:
        _set_common_pragmas(conn)
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
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return []
            raise
    finally:
        conn.close()
    return [(_from_signed(bb), su, du, v, sc) for bb, su, du, v, sc in rows]


def get_recompute_states(
    current_version: str,
    only_missing_current: bool = False,
    limit: int | None = None,
    offset: int = 0,
) -> list[tuple[int, int, int]]:
    """Return unique (board_bb_unsigned, swap_uses, delete_uses) states for recompute.

    If only_missing_current=True, return only states that do not yet have a row in
    current_version. Results are ordered deterministically by key and support
    optional LIMIT/OFFSET for batch processing.
    """
    if not os.path.exists(DB_PATH):
        return []

    lim = None if limit is None else max(0, int(limit))
    off = max(0, int(offset))
    conn = _connect()
    try:
        if only_missing_current:
            sql = """
                SELECT e.board_bb, e.swap_uses, e.delete_uses
                FROM entries e
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM entries cur
                    WHERE cur.version = ?
                      AND cur.board_bb = e.board_bb
                      AND cur.swap_uses = e.swap_uses
                      AND cur.delete_uses = e.delete_uses
                )
                GROUP BY e.board_bb, e.swap_uses, e.delete_uses
                ORDER BY e.board_bb, e.swap_uses, e.delete_uses
            """
            params: list = [current_version]
        else:
            sql = """
                SELECT board_bb, swap_uses, delete_uses
                FROM entries
                GROUP BY board_bb, swap_uses, delete_uses
                ORDER BY board_bb, swap_uses, delete_uses
            """
            params = []

        if lim is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([lim, off])
        elif off > 0:
            sql += " LIMIT -1 OFFSET ?"
            params.append(off)

        rows = conn.execute(sql, tuple(params)).fetchall()
    finally:
        conn.close()

    return [(_from_signed(bb), su, du) for bb, su, du in rows]


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
