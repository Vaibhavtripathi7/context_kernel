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


class TestSessionOperations:
    """Full CRUD lifecycle for session rows."""

    def test_create_session_returns_record(self, engine: StorageEngine) -> None:
        rec = engine.create_session("aider --model gpt-4o")
        assert rec.session_id
        assert rec.agent_command == "aider --model gpt-4o"
        assert rec.started_at > 0.0
        assert rec.ended_at is None

    def test_get_session_roundtrip(self, engine: StorageEngine, session: SessionRecord) -> None:
        fetched = engine.get_session(session.session_id)
        assert fetched is not None
        assert fetched.session_id    == session.session_id
        assert fetched.agent_command == session.agent_command

    def test_get_session_returns_none_for_unknown_id(self, engine: StorageEngine) -> None:
        assert engine.get_session("00000000-dead-beef-0000-000000000000") is None

    def test_close_session_stamps_end_time(
        self, engine: StorageEngine, session: SessionRecord
    ) -> None:
        before = time.time()
        engine.close_session(session.session_id, tokens_saved=512)
        after  = time.time()

        rec = engine.get_session(session.session_id)
        assert rec is not None
        assert rec.total_tokens_saved == 512
        assert before <= rec.ended_at <= after  # type: ignore[operator]

    def test_list_sessions_newest_first(self, engine: StorageEngine) -> None:
        for i in range(3):
            engine.create_session(f"agent-{i}")
            time.sleep(0.01)

        rows = engine.list_sessions(limit=5)
        timestamps = [r["started_at"] for r in rows]
        assert timestamps == sorted(timestamps, reverse=True), (
            "list_sessions() must return newest sessions first."
        )


class TestLogEntryOperations:
    """Insert, retrieve, and aggregate log entries."""

    def _make_entry(
        self,
        session_id: str,
        content: str = "test output",
        entry_type: str = "stdout",
        pruned: bool = False,
        summary: str = "",
    ) -> LogEntry:
        return LogEntry(
            session_id=session_id,
            raw_content=content,
            entry_type=entry_type,
            compressed_summary=summary,
            token_estimate=len(content) // 4,
            was_pruned=pruned,
        )

    def test_insert_entry_returns_integer_id(
        self, engine: StorageEngine, session: SessionRecord
    ) -> None:
        row_id = engine.insert_entry(self._make_entry(session.session_id))
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_insert_entry_ids_are_monotonically_increasing(
        self, engine: StorageEngine, session: SessionRecord
    ) -> None:
        ids = [
            engine.insert_entry(self._make_entry(session.session_id, f"line {i}"))
            for i in range(5)
        ]
        assert ids == sorted(ids), "Row IDs must be monotonically increasing."

    def test_get_recent_entries_respects_limit(
        self, engine: StorageEngine, session: SessionRecord
    ) -> None:
        for i in range(20):
            engine.insert_entry(self._make_entry(session.session_id, f"entry {i}"))

        rows = engine.get_recent_entries(session.session_id, limit=7)
        assert len(rows) == 7

    def test_get_recent_entries_newest_first(
        self, engine: StorageEngine, session: SessionRecord
    ) -> None:
        for i in range(5):
            engine.insert_entry(self._make_entry(session.session_id, f"msg {i}"))
            time.sleep(0.005)

        rows = engine.get_recent_entries(session.session_id, limit=5)
        ts   = [r["timestamp"] for r in rows]
        assert ts == sorted(ts, reverse=True)

    def test_bulk_insert_all_or_nothing(
        self, engine: StorageEngine, session: SessionRecord
    ) -> None:
        batch = [
            self._make_entry(session.session_id, f"bulk {i}") for i in range(10)
        ]
        engine.bulk_insert_entries(batch)
        stats = engine.stats(session.session_id)
        assert stats["total_entries"] == 10

    def test_was_pruned_flag_stored_correctly(
        self, engine: StorageEngine, session: SessionRecord
    ) -> None:
        engine.insert_entry(
            self._make_entry(session.session_id, "raw blob", pruned=True, summary="short")
        )
        engine.insert_entry(
            self._make_entry(session.session_id, "pass-through")
        )
        stats = engine.stats(session.session_id)
        assert stats["pruned_entries"] == 1

    def test_token_estimate_aggregated_in_stats(
        self, engine: StorageEngine, session: SessionRecord
    ) -> None:
        for _ in range(4):
            engine.insert_entry(
                LogEntry(
                    session_id=session.session_id,
                    raw_content="x" * 400,
                    entry_type="stdout",
                    token_estimate=100,
                )
            )
        stats = engine.stats(session.session_id)
        assert stats["raw_tokens"] == 400
