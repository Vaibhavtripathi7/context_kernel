"""Orchestrator tests: prompt detection, buffer flush, stats, PTY integration."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

import pytest

from context_kernel.core.orchestrator import Orchestrator, OrchestratorConfig, OrchestratorStats
from context_kernel.memory.storage import StorageEngine
from context_kernel.pruners.base import BasePruner, PrunerMetadata


PROJECT_ROOT = Path(__file__).parent.parent


def _make_storage(tmp_path: Path) -> StorageEngine:
    eng = StorageEngine(db_path=tmp_path / "orch_test.db")
    eng.open()
    return eng


def _make_orchestrator(
    storage: StorageEngine,
    pruners: Optional[list[BasePruner]] = None,
    threshold: int = 5,
    annotate: bool = True,
) -> Orchestrator:
    session = storage.create_session("test-agent")
    return Orchestrator(
        command=["echo", "placeholder"],
        session_id=session.session_id,
        storage=storage,
        pruners=pruners or [],
        config=OrchestratorConfig(
            pruning_threshold_lines=threshold,
            buffer_flush_timeout=0.05,
            annotate_injections=annotate,
        ),
    )


@pytest.fixture
def storage(tmp_path: Path) -> StorageEngine:
    eng = _make_storage(tmp_path)
    yield eng
    eng.close()


@pytest.fixture
def orch(storage: StorageEngine) -> Orchestrator:
    return _make_orchestrator(storage)


@pytest.fixture
def pipe_pair():
    """An (r_fd, w_fd) pair for capturing os.write output in unit tests."""
    r_fd, w_fd = os.pipe()
    yield r_fd, w_fd
    for fd in (r_fd, w_fd):
        try:
            os.close(fd)
        except OSError:
            pass


def _read_pipe(r_fd: int, timeout: float = 0.5) -> bytes:
    """Drain all available data from the read end of a pipe."""
    import select
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([r_fd], [], [], 0.05)
        if not ready:
            break
        chunk = os.read(r_fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)
