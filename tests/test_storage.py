"""StorageEngine tests: schema/CRUD, FTS5 search, WAL concurrency, stats."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from context_kernel.memory.storage import LogEntry, SessionRecord, StorageEngine


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_ack.db"


@pytest.fixture
def engine(db_path: Path) -> StorageEngine:
    """Open StorageEngine, yield, then close."""
    eng = StorageEngine(db_path=db_path)
    eng.open()
    yield eng
    eng.close()


@pytest.fixture
def session(engine: StorageEngine) -> SessionRecord:
    """A valid session row already in the database."""
    return engine.create_session("pytest-agent --test")


class TestSchemaSetup:
    """Schema migration must be idempotent and configure WAL mode."""

    def test_open_creates_db_file(self, db_path: Path) -> None:
        eng = StorageEngine(db_path=db_path)
        assert not db_path.exists(), "DB should not exist before open()"
        eng.open()
        assert db_path.exists(), "DB file must be created by open()"
        eng.close()

    def test_migration_is_idempotent(self, db_path: Path) -> None:
        """Calling open() twice on the same file must not raise."""
        eng = StorageEngine(db_path=db_path)
        eng.open()
        eng.close()
        eng2 = StorageEngine(db_path=db_path)
        eng2.open()
        eng2.close()

    def test_wal_mode_is_active(self, engine: StorageEngine) -> None:
        """
        SQLite must report journal_mode = wal after open().
        WAL is the concurrency guarantee required by the orchestrator loop.
        """
        row = engine._db.execute("PRAGMA journal_mode").fetchone()  # type: ignore[attr-defined]
        assert row[0].lower() == "wal", (
            f"Expected WAL journal mode, got: {row[0]!r}.  "
            "Check that 'PRAGMA journal_mode = WAL' fires in _apply_pragmas()."
        )

    def test_foreign_keys_are_enforced(self, engine: StorageEngine) -> None:
        """Inserting a log_entry with a nonexistent session_id must raise."""
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            engine.insert_entry(
                LogEntry(
                    session_id="does-not-exist",
                    raw_content="orphan entry",
                    entry_type="stdout",
                )
            )
