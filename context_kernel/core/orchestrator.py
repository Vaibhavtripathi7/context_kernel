"""PTY spawn loop, output buffering and stream injection.

The agent runs inside a real pseudo-terminal so interactive prompts and colour
output behave as if it were launched directly. A single select() loop forwards
keystrokes to the child and routes the child's output through the pruners
before it reaches the user's terminal.
"""
from __future__ import annotations

import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import time
import tty
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..memory.storage import LogEntry, StorageEngine
from ..pruners.base import BasePruner

_DIM   = "\033[2m"
_CYAN  = "\033[36m"
_BOLD  = "\033[1m"
_RESET = "\033[0m"

_SELECT_TIMEOUT = 0.08


@dataclass
class OrchestratorConfig:
    pruning_threshold_lines: int   = 30
    buffer_flush_timeout:    float = 0.15
    read_chunk_bytes:        int   = 8192
    annotate_injections:     bool  = True


@dataclass
class OrchestratorStats:
    total_bytes_read:     int   = 0
    total_bytes_injected: int   = 0
    total_pruner_hits:    int   = 0
    tokens_saved:         int   = 0
    session_start:        float = field(default_factory=time.monotonic)


_PROMPT_BYTES: tuple[bytes, ...] = (
    b"[Y/n]",
    b"[y/N]",
    b"[Y/N]",
    b"[yes/no]",
    b"(y/n)",
    b"(yes/no)",
    b"? ",
)
_PROMPT_TAIL_CHARS = frozenset("?:>")
_PROMPT_MAX_LINE_LEN = 120


class Orchestrator:
    """Intercepts an agent's PTY output and runs it through the pruners.

    pruners are tried in order and the first match wins; an empty list
    means everything passes through untouched. storage must already be
    open. stats_callback and text_callback are optional hooks used by
    the TUI to mirror live metrics and output.
    """

    def __init__(
        self,
        command:        list[str],
        session_id:     str,
        storage:        StorageEngine,
        pruners:        list[BasePruner] | None = None,
        config:         OrchestratorConfig | None = None,
        stats_callback: Callable[[OrchestratorStats], None] | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must be a non-empty list of strings")

        self.command        = command
        self.session_id     = session_id
        self.storage        = storage
        self.pruners:list[BasePruner] = pruners or []
        self.config         = config or OrchestratorConfig()
        self.stats_callback = stats_callback
        self.text_callback: Callable[[str], None] | None = None

        self._stats               = OrchestratorStats()
        self._child_pid:           int | None = None
        self._master_fd:           int | None = None
        self._buffer:              list[bytes]   = []
        self._last_data_monotonic: float         = 0.0
        self._saved_tty:           list[Any] | None = None

    @property
    def stats(self) -> OrchestratorStats:
        return self._stats

    def add_pruner(self, pruner: BasePruner) -> None:
        self.pruners.append(pruner)

    def run(self) -> int:
        """Spawn the agent, block until it exits, and return its exit code.

        The terminal is always restored, even if the child crashes.
        """
        self._master_fd, child_pid = self._spawn_in_pty()
        self._child_pid = child_pid

        self._enter_raw_mode()
        self._install_sigwinch_handler()

        try:
            return self._io_loop(child_pid)
        finally:
            self._restore_terminal()
            if self._master_fd is not None:
                try:
                    os.close(self._master_fd)
                except OSError:
                    pass
                self._master_fd = None
            self.storage.close_session(
                self.session_id,
                tokens_saved=self._stats.tokens_saved,
            )
