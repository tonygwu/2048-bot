"""
cache.py — SQLite persistence for the score_board transposition table.

Database: cache/transposition.db
Schema:
  entries(board_bb INTEGER, undo_uses INTEGER, swap_uses INTEGER,
          delete_uses INTEGER, version TEXT, score REAL,
          PRIMARY KEY (board_bb, undo_uses, swap_uses, delete_uses, version))

board_bb is a 64-bit value that can exceed Python's sqlite3 signed-int limit
(2^63 − 1) for late-game boards.  We store it as a signed int64 using
_to_signed / _from_signed helpers and reconstruct the unsigned value on read.
"""

import os
import sqlite3
from collections.abc import Callable

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
        undo_uses   INTEGER NOT NULL,
        swap_uses   INTEGER NOT NULL,
        delete_uses INTEGER NOT NULL,
        version     TEXT    NOT NULL,
        score       REAL    NOT NULL,
        PRIMARY KEY (board_bb, undo_uses, swap_uses, delete_uses, version)
    )
"""
_CREATE_VERSION_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_version ON entries(version)"
_CREATE_VERSION_KEY_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_version_board_power_undo "
    "ON entries(version, board_bb, undo_uses, swap_uses, delete_uses)"
)


# ── Signed-int helpers (SQLite stores INTEGER as signed 64-bit) ───────────────

def _to_signed(bb: int) -> int:
    """Unsigned 64-bit → signed 64-bit (for SQLite storage)."""
    return bb if bb < (1 << 63) else bb - (1 << 64)


def _from_signed(val: int) -> int:
    """Signed 64-bit → unsigned 64-bit (after reading from SQLite)."""
    return val if val >= 0 else val + (1 << 64)


def _max_exp_from_bb_unsigned(bb: int) -> int:
    """Return max tile exponent across 16 nibbles in a packed bitboard."""
    v = int(bb)
    out = 0
    for _ in range(16):
        nib = v & 0xF
        if nib > out:
            out = nib
        v >>= 4
    return out


def _bb_max_exp_from_signed_sql(val: int) -> int:
    """SQLite UDF: signed int64 board_bb -> max tile exponent (0 for empty)."""
    try:
        return _max_exp_from_bb_unsigned(_from_signed(int(val)))
    except Exception:
        return 0


def _tile_to_min_exp(tile: int) -> int:
    """Inclusive lower bound tile -> minimum exponent."""
    t = max(0, int(tile))
    if t <= 1:
        return 0
    # ceil(log2(t)) for non-powers-of-two, exact for powers-of-two.
    if (t & (t - 1)) == 0:
        return t.bit_length() - 1
    return t.bit_length()


def _tile_to_max_exp(tile: int) -> int:
    """Inclusive upper bound tile -> maximum exponent."""
    t = max(0, int(tile))
    if t <= 1:
        return 0
    # floor(log2(t)) for non-powers-of-two, exact for powers-of-two.
    return t.bit_length() - 1


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


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return bool(row)


def _entries_table_info(conn: sqlite3.Connection) -> tuple[list[str], list[str]]:
    rows = conn.execute("PRAGMA table_info(entries)").fetchall()
    cols = [str(r[1]) for r in rows]
    pk_cols = [str(r[1]) for r in sorted((r for r in rows if int(r[5]) > 0), key=lambda r: int(r[5]))]
    return cols, pk_cols


def _migrate_entries_schema_if_needed(conn: sqlite3.Connection) -> None:
    """Best-effort in-place migration for legacy `entries` schemas.

    For board-only caching we only persist canonical zero-power rows
    (undo_uses=swap_uses=delete_uses=0), so we avoid expensive table rewrites.
    Legacy DBs that predate undo support only need a new `undo_uses` column.
    """
    if not _table_exists(conn, "entries"):
        return

    cols, _ = _entries_table_info(conn)
    col_set = set(cols)

    if "undo_uses" not in col_set:
        conn.execute("ALTER TABLE entries ADD COLUMN undo_uses INTEGER NOT NULL DEFAULT 0")
        cols, _ = _entries_table_info(conn)
        col_set = set(cols)

    required_cols = {"board_bb", "undo_uses", "swap_uses", "delete_uses", "version", "score"}
    if required_cols.issubset(col_set):
        return

    # Unexpected/partial shape fallback: rebuild table into canonical schema.
    conn.execute("ALTER TABLE entries RENAME TO entries_legacy")
    conn.execute(_CREATE_ENTRIES_SQL)

    legacy_rows = conn.execute("PRAGMA table_info(entries_legacy)").fetchall()
    legacy_set = {str(r[1]) for r in legacy_rows}
    if "undo_uses" in legacy_set:
        conn.execute(
            "INSERT INTO entries (board_bb, undo_uses, swap_uses, delete_uses, version, score) "
            "SELECT board_bb, undo_uses, swap_uses, delete_uses, version, score "
            "FROM entries_legacy"
        )
    else:
        conn.execute(
            "INSERT INTO entries (board_bb, undo_uses, swap_uses, delete_uses, version, score) "
            "SELECT board_bb, 0, swap_uses, delete_uses, version, score "
            "FROM entries_legacy"
        )
    conn.execute("DROP TABLE entries_legacy")


def init_db() -> None:
    """Initialize schema and connection mode once per process."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    conn = _connect_rw()
    try:
        _set_rw_pragmas(conn)
        conn.execute(_CREATE_ENTRIES_SQL)
        _migrate_entries_schema_if_needed(conn)
        conn.execute(_CREATE_VERSION_INDEX_SQL)
        conn.execute(_CREATE_VERSION_KEY_INDEX_SQL)
        conn.commit()
    finally:
        conn.close()
    _SCHEMA_READY = True


# ── Public API ────────────────────────────────────────────────────────────────

def load_version(
    version: str,
    progress_cb: Callable[[int, int], None] | None = None,
    batch_size: int = 200_000,
    total_rows: int | None = None,
) -> dict:
    """Load all cached entries for *version* into a dict keyed by
    board_bb → score.
    Returns {} if DB is absent.

    progress_cb receives (loaded_rows, total_rows) after each batch.
    """
    if not os.path.exists(DB_PATH):
        return {}
    batch = max(1, int(batch_size))
    conn = _connect_ro()
    try:
        _set_common_pragmas(conn)
        use_legacy_shape = False
        try:
            if total_rows is None:
                total_rows = conn.execute(
                    "SELECT COUNT(*) FROM entries "
                    "WHERE version = ? AND undo_uses = 0 AND swap_uses = 0 AND delete_uses = 0",
                    (version,),
                ).fetchone()[0]
            cur = conn.execute(
                "SELECT board_bb, undo_uses, swap_uses, delete_uses, score "
                "FROM entries WHERE version = ?",
                (version,),
            )
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return {}
            if "no such column" in str(exc).lower() and "undo_uses" in str(exc).lower():
                use_legacy_shape = True
                if total_rows is None:
                    total_rows = conn.execute(
                        "SELECT COUNT(*) FROM entries "
                        "WHERE version = ? AND swap_uses = 0 AND delete_uses = 0",
                        (version,),
                    ).fetchone()[0]
                cur = conn.execute(
                    "SELECT board_bb, swap_uses, delete_uses, score FROM entries WHERE version = ?",
                    (version,),
                )
            else:
                raise
        loaded = 0
        out: dict[int, float] = {}
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                break
            if use_legacy_shape:
                for bb, su, du, score in rows:
                    if su == 0 and du == 0:
                        out[_from_signed(bb)] = score
            else:
                for bb, uu, su, du, score in rows:
                    if uu == 0 and su == 0 and du == 0:
                        out[_from_signed(bb)] = score
            loaded += len(rows)
            if progress_cb is not None:
                try:
                    progress_cb(loaded, total_rows)
                except Exception:
                    pass
        if progress_cb is not None and loaded == 0:
            try:
                progress_cb(0, total_rows or 0)
            except Exception:
                pass
        return out
    finally:
        conn.close()


def load_version_by_max_tile_range(
    version: str,
    min_max_tile: int | None = None,
    max_max_tile: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    batch_size: int = 200_000,
    total_rows: int | None = None,
) -> dict:
    """Load cached entries for *version* filtered by board max-tile range.

    Filter semantics:
    - min_max_tile is inclusive lower bound on board max tile
    - max_max_tile is inclusive upper bound on board max tile
    """
    if not os.path.exists(DB_PATH):
        return {}
    batch = max(1, int(batch_size))
    conn = _connect_ro()
    try:
        _set_common_pragmas(conn)
        conn.create_function("bb_max_exp", 1, _bb_max_exp_from_signed_sql)

        where = ["version = ?"]
        params: list = [version]
        if min_max_tile is not None:
            where.append("bb_max_exp(board_bb) >= ?")
            params.append(_tile_to_min_exp(min_max_tile))
        if max_max_tile is not None:
            where.append("bb_max_exp(board_bb) <= ?")
            params.append(_tile_to_max_exp(max_max_tile))
        where_sql = " AND ".join(where)

        try:
            if total_rows is None:
                total_rows = conn.execute(
                    f"SELECT COUNT(*) FROM entries WHERE {where_sql}",
                    tuple(params),
                ).fetchone()[0]
            cur = conn.execute(
                "SELECT board_bb, swap_uses, delete_uses, score "
                f"FROM entries WHERE {where_sql} "
                "ORDER BY board_bb, swap_uses, delete_uses",
                tuple(params),
            )
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return {}
            raise

        loaded = 0
        out: dict[tuple[int, int, int], float] = {}
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                break
            for bb, su, du, score in rows:
                out[(_from_signed(bb), su, du)] = score
            loaded += len(rows)
            if progress_cb is not None:
                try:
                    progress_cb(loaded, total_rows)
                except Exception:
                    pass
        if progress_cb is not None and loaded == 0:
            try:
                progress_cb(0, total_rows or 0)
            except Exception:
                pass
        return out
    finally:
        conn.close()


def save_entries(
    entries: dict,
    version: str,
    progress_cb: Callable[[int, int], None] | None = None,
    batch_size: int = 100_000,
) -> int:
    """Insert-or-replace *entries* into the DB under *version*.
    entries: {board_bb: score} (tuple keys are accepted and reduced to board_bb).
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
        board_scores: dict[int, float] = {}
        for key, score in entries.items():
            if isinstance(key, int):
                bb = key
            elif isinstance(key, tuple) and len(key) >= 1:
                bb = key[0]
            else:
                raise ValueError(f"unexpected cache entry key shape: {key!r}")
            board_scores[int(bb)] = float(score)

        rows = [
            (_to_signed(bb), _clamp_uses(0), _clamp_uses(0), _clamp_uses(0), version, score)
            for bb, score in board_scores.items()
        ]
        total = len(rows)
        step = max(1, int(batch_size))
        written = 0
        for i in range(0, total, step):
            chunk = rows[i:i + step]
            conn.executemany(
                "INSERT OR REPLACE INTO entries "
                "(board_bb, undo_uses, swap_uses, delete_uses, version, score) VALUES (?,?,?,?,?,?)",
                chunk,
            )
            written += len(chunk)
            if progress_cb is not None:
                try:
                    progress_cb(written, total)
                except Exception:
                    pass
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
                "SELECT version, COUNT(*) FROM entries "
                "WHERE undo_uses = 0 AND swap_uses = 0 AND delete_uses = 0 "
                "GROUP BY version"
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return {}
            if "no such column" in str(exc).lower() and "undo_uses" in str(exc).lower():
                rows = conn.execute(
                    "SELECT version, COUNT(*) FROM entries "
                    "WHERE swap_uses = 0 AND delete_uses = 0 "
                    "GROUP BY version"
                ).fetchall()
            else:
                raise
    finally:
        conn.close()
    return {v: cnt for v, cnt in rows}


def get_all_states(version: str | None = None) -> list[tuple]:
    """Return raw rows as
    (board_bb_unsigned, version, score) for canonical zero-power cache rows.
    If version is None, return all versions."""
    if not os.path.exists(DB_PATH):
        return []
    conn = _connect_ro()
    try:
        _set_common_pragmas(conn)
        use_legacy_shape = False
        try:
            if version is not None:
                rows = conn.execute(
                    "SELECT board_bb, undo_uses, swap_uses, delete_uses, version, score "
                    "FROM entries WHERE version = ?",
                    (version,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT board_bb, undo_uses, swap_uses, delete_uses, version, score "
                    "FROM entries"
                ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return []
            if "no such column" in str(exc).lower() and "undo_uses" in str(exc).lower():
                use_legacy_shape = True
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
            else:
                raise
    finally:
        conn.close()
    if use_legacy_shape:
        return [(_from_signed(bb), v, sc) for bb, su, du, v, sc in rows if su == 0 and du == 0]
    return [(_from_signed(bb), v, sc) for bb, uu, su, du, v, sc in rows if uu == 0 and su == 0 and du == 0]


def get_recompute_states(
    current_version: str,
    only_missing_current: bool = False,
    limit: int | None = None,
    offset: int = 0,
    order_by_max_tile: str | None = None,
) -> list[int]:
    """Return unique board_bb_unsigned states.

    If only_missing_current=True, return only states that do not yet have a row in
    current_version. Results are ordered deterministically by key unless
    order_by_max_tile is "asc"/"desc", in which case ordering is by max tile
    exponent (low/high first), then key. Supports optional LIMIT/OFFSET.
    """
    if not os.path.exists(DB_PATH):
        return []

    lim = None if limit is None else max(0, int(limit))
    off = max(0, int(offset))
    order_norm = None if order_by_max_tile is None else str(order_by_max_tile).strip().lower()
    if order_norm not in {None, "asc", "desc"}:
        raise ValueError(f"order_by_max_tile must be None/'asc'/'desc', got: {order_by_max_tile!r}")
    conn = _connect_ro()
    try:
        _set_common_pragmas(conn)
        if order_norm is not None:
            conn.create_function("bb_max_exp", 1, _bb_max_exp_from_signed_sql)
        try:
            if only_missing_current:
                sql = """
                    SELECT e.board_bb
                    FROM entries e
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM entries cur
                        WHERE cur.version = ?
                          AND cur.board_bb = e.board_bb
                          AND cur.undo_uses = 0
                          AND cur.swap_uses = 0
                          AND cur.delete_uses = 0
                    )
                    GROUP BY e.board_bb
                """
                params: list = [current_version]
            else:
                sql = """
                    SELECT e.board_bb
                    FROM entries e
                    GROUP BY e.board_bb
                """
                params = []
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return []
            raise

        if order_norm is None:
            sql += "\n ORDER BY e.board_bb"
        else:
            sql += f"\n ORDER BY bb_max_exp(e.board_bb) {order_norm.upper()}, e.board_bb"

        if lim is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([lim, off])
        elif off > 0:
            sql += " LIMIT -1 OFFSET ?"
            params.append(off)

        try:
            rows = conn.execute(sql, tuple(params)).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return []
            if "no such column" in str(exc).lower() and "undo_uses" in str(exc).lower():
                if only_missing_current:
                    sql = """
                        SELECT e.board_bb
                        FROM entries e
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM entries cur
                            WHERE cur.version = ?
                              AND cur.board_bb = e.board_bb
                              AND cur.swap_uses = 0
                              AND cur.delete_uses = 0
                        )
                        GROUP BY e.board_bb
                    """
                    params = [current_version]
                else:
                    sql = "SELECT e.board_bb FROM entries e GROUP BY e.board_bb"

                if order_norm is None:
                    sql += "\n ORDER BY e.board_bb"
                else:
                    sql += f"\n ORDER BY bb_max_exp(e.board_bb) {order_norm.upper()}, e.board_bb"
                if lim is not None:
                    sql += " LIMIT ? OFFSET ?"
                    params.extend([lim, off])
                elif off > 0:
                    sql += " LIMIT -1 OFFSET ?"
                    params.append(off)
                rows = conn.execute(sql, tuple(params)).fetchall()
            else:
                raise
    finally:
        conn.close()
    return [_from_signed(int(row[0])) for row in rows]


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
