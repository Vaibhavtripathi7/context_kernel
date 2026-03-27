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
