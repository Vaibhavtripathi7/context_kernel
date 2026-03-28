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
