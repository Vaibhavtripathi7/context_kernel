"""SQLite + FTS5 persistence layer.

Runs in WAL mode so the orchestrator can write while the TUI reads stats. The
full-text index uses an FTS5 content table kept in sync by a trigger trio, so
the log text is stored once.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_DB: Path = Path.home() / ".local" / "share" / "ack" / "kernel.db"


@dataclass(frozen=True, slots=True)
class LogEntry:
    """One captured output chunk from the agent."""

    session_id: str
    raw_content: str
    entry_type: str
    timestamp: float = field(default_factory=time.time)
    compressed_summary: str = ""
    token_estimate: int = 0
    was_pruned: bool = False
    id: int | None = None


@dataclass(slots=True)
class SessionRecord:
    """Header row for one agent invocation."""

    agent_command: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    total_tokens_saved: int = 0


class StorageEngine:
    """SQLite + FTS5 store. Use directly with open()/close() or as a context
    manager. db_path defaults to ~/.local/share/ack/kernel.db and its
    parent directories are created on open.
    """

    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._write_lock = threading.Lock()

    def open(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        self._migrate()

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self._conn.close()
            self._conn = None

    def __enter__(self) -> StorageEngine:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("StorageEngine is not open — call .open() first.")
        return self._conn

    def _apply_pragmas(self) -> None:
        for pragma in (
            "PRAGMA journal_mode = WAL",
            "PRAGMA synchronous  = NORMAL",
            "PRAGMA foreign_keys = ON",
            "PRAGMA cache_size   = -8000",
            "PRAGMA temp_store   = MEMORY",
            "PRAGMA mmap_size    = 268435456",
        ):
            self._db.execute(pragma)

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        with self._write_lock:
            db = self._db
            db.execute("BEGIN")
            try:
                yield db
                db.execute("COMMIT")
            except Exception:
                db.execute("ROLLBACK")
                raise

    def _migrate(self) -> None:
        with self._transaction() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id         TEXT PRIMARY KEY,
                    agent_command      TEXT NOT NULL,
                    started_at         REAL NOT NULL,
                    ended_at           REAL,
                    total_tokens_saved INTEGER NOT NULL DEFAULT 0
                )
            """)

            db.execute("""
                CREATE TABLE IF NOT EXISTS log_entries (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id         TEXT    NOT NULL
                                           REFERENCES sessions(session_id)
                                           ON DELETE CASCADE,
                    timestamp          REAL    NOT NULL,
                    entry_type         TEXT    NOT NULL DEFAULT 'stdout',
                    raw_content        TEXT    NOT NULL,
                    compressed_summary TEXT    NOT NULL DEFAULT '',
                    token_estimate     INTEGER NOT NULL DEFAULT 0,
                    was_pruned         INTEGER NOT NULL DEFAULT 0
                )
            """)
            db.execute("""
                CREATE INDEX IF NOT EXISTS idx_log_session_time
                    ON log_entries(session_id, timestamp DESC)
            """)

            db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS log_fts USING fts5(
                    raw_content,
                    compressed_summary,
                    entry_type,
                    content     = 'log_entries',
                    content_rowid = 'id',
                    tokenize    = 'porter ascii'
                )
            """)

            db.execute("""
                CREATE TRIGGER IF NOT EXISTS log_fts_ai
                AFTER INSERT ON log_entries BEGIN
                    INSERT INTO log_fts(rowid, raw_content, compressed_summary, entry_type)
                    VALUES (new.id, new.raw_content, new.compressed_summary, new.entry_type);
                END
            """)

            db.execute("""
                CREATE TRIGGER IF NOT EXISTS log_fts_ad
                AFTER DELETE ON log_entries BEGIN
                    INSERT INTO log_fts(log_fts, rowid, raw_content, compressed_summary, entry_type)
                    VALUES ('delete', old.id, old.raw_content,
                            old.compressed_summary, old.entry_type);
                END
            """)

            db.execute("""
                CREATE TRIGGER IF NOT EXISTS log_fts_au
                AFTER UPDATE ON log_entries BEGIN
                    INSERT INTO log_fts(log_fts, rowid, raw_content, compressed_summary, entry_type)
                    VALUES ('delete', old.id, old.raw_content,
                            old.compressed_summary, old.entry_type);
                    INSERT INTO log_fts(rowid, raw_content, compressed_summary, entry_type)
                    VALUES (new.id, new.raw_content, new.compressed_summary, new.entry_type);
                END
            """)

    def create_session(self, agent_command: str) -> SessionRecord:
        rec = SessionRecord(agent_command=agent_command)
        with self._transaction() as db:
            db.execute(
                "INSERT INTO sessions(session_id, agent_command, started_at) VALUES (?,?,?)",
                (rec.session_id, rec.agent_command, rec.started_at),
            )
        return rec

    def close_session(self, session_id: str, tokens_saved: int = 0) -> None:
        with self._transaction() as db:
            db.execute(
                """UPDATE sessions
                   SET ended_at = ?, total_tokens_saved = ?
                   WHERE session_id = ?""",
                (time.time(), tokens_saved, session_id),
            )

    def get_session(self, session_id: str) -> SessionRecord | None:
        row = self._db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return SessionRecord(
            session_id=row["session_id"],
            agent_command=row["agent_command"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            total_tokens_saved=row["total_tokens_saved"],
        )

    def list_sessions(self, limit: int = 20) -> list[sqlite3.Row]:
        return self._db.execute(
            "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def insert_entry(self, entry: LogEntry) -> int:
        with self._transaction() as db:
            cur = db.execute(
                """
                INSERT INTO log_entries
                    (session_id, timestamp, entry_type, raw_content,
                     compressed_summary, token_estimate, was_pruned)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.session_id,
                    entry.timestamp,
                    entry.entry_type,
                    entry.raw_content,
                    entry.compressed_summary,
                    entry.token_estimate,
                    int(entry.was_pruned),
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def bulk_insert_entries(self, entries: Sequence[LogEntry]) -> None:
        with self._transaction() as db:
            db.executemany(
                """
                INSERT INTO log_entries
                    (session_id, timestamp, entry_type, raw_content,
                     compressed_summary, token_estimate, was_pruned)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.session_id,
                        e.timestamp,
                        e.entry_type,
                        e.raw_content,
                        e.compressed_summary,
                        e.token_estimate,
                        int(e.was_pruned),
                    )
                    for e in entries
                ],
            )

    def search(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[sqlite3.Row]:
        """FTS5 MATCH over raw content and summaries, ordered by BM25 rank."""
        if session_id is not None:
            return self._db.execute(
                """
                SELECT le.*, bm25(log_fts) AS rank
                FROM   log_fts
                JOIN   log_entries le ON log_fts.rowid = le.id
                WHERE  log_fts    MATCH ?
                  AND  le.session_id = ?
                ORDER  BY rank
                LIMIT  ?
                """,
                (query, session_id, limit),
            ).fetchall()

        return self._db.execute(
            """
            SELECT le.*, bm25(log_fts) AS rank
            FROM   log_fts
            JOIN   log_entries le ON log_fts.rowid = le.id
            WHERE  log_fts MATCH ?
            ORDER  BY rank
            LIMIT  ?
            """,
            (query, limit),
        ).fetchall()

    def get_entry(self, entry_id: int) -> sqlite3.Row | None:
        """Fetch a single log entry by its primary-key id, or None if absent.

        This is the direct-lookup path behind `ack recall <id>`: the recall
        handle stamped on a pruned injection is exactly this id.
        """
        row: sqlite3.Row | None = self._db.execute(
            "SELECT * FROM log_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        return row

    def get_recent_entries(self, session_id: str, limit: int = 50) -> list[sqlite3.Row]:
        return self._db.execute(
            """
            SELECT * FROM log_entries
            WHERE  session_id = ?
            ORDER  BY timestamp DESC
            LIMIT  ?
            """,
            (session_id, limit),
        ).fetchall()

    def stats(self, session_id: str) -> dict[str, int]:
        row = self._db.execute(
            """
            SELECT
                COUNT(*)                                         AS total_entries,
                SUM(was_pruned)                                  AS pruned_entries,
                SUM(token_estimate)                              AS raw_tokens,
                SUM(CASE WHEN was_pruned THEN token_estimate
                         ELSE 0 END)                             AS tokens_saved
            FROM log_entries
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        return {
            "total_entries":  int(row["total_entries"]  or 0),
            "pruned_entries": int(row["pruned_entries"] or 0),
            "raw_tokens":     int(row["raw_tokens"]     or 0),
            "tokens_saved":   int(row["tokens_saved"]   or 0),
        }
