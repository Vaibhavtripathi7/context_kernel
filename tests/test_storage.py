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
