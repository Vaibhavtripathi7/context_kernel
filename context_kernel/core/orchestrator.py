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

    def _spawn_in_pty(self) -> tuple[int, int]:
        """Fork the agent into a fresh PTY; return (master_fd, child_pid)."""
        master_fd, slave_fd = pty.openpty()

        if sys.stdout.isatty():
            rows, cols = self._get_terminal_size()
            self._set_winsize(slave_fd, rows, cols)

        child_pid = os.fork()

        if child_pid == 0:
            os.setsid()
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)

            os.dup2(slave_fd, sys.stdin.fileno())
            os.dup2(slave_fd, sys.stdout.fileno())
            os.dup2(slave_fd, sys.stderr.fileno())

            if slave_fd > 2:
                os.close(slave_fd)
            os.close(master_fd)

            os.execvp(self.command[0], self.command)
            os._exit(127)

        os.close(slave_fd)
        return master_fd, child_pid

    def _io_loop(self, child_pid: int) -> int:
        assert self._master_fd is not None
        master_fd = self._master_fd
        stdin_fd  = sys.stdin.fileno()
        stdout_fd = sys.stdout.fileno()
        exit_code = 0

        while True:
            try:
                rlist, _, _ = select.select(
                    [master_fd, stdin_fd],
                    [],
                    [],
                    _SELECT_TIMEOUT,
                )
            except InterruptedError:
                continue
            except (ValueError, OSError):
                break

            if master_fd in rlist:
                try:
                    chunk = os.read(master_fd, self.config.read_chunk_bytes)
                except OSError:
                    chunk = b""
                if not chunk:
                    break
                self._stats.total_bytes_read += len(chunk)
                self._last_data_monotonic = time.monotonic()
                self._accumulate(chunk)
                self._flush_buffer(stdout_fd, force=False)

            if stdin_fd in rlist:
                try:
                    keys = os.read(stdin_fd, 256)
                except OSError:
                    keys = b""
                if keys:
                    try:
                        os.write(master_fd, keys)
                    except OSError:
                        pass

            elapsed_since_data = time.monotonic() - self._last_data_monotonic
            if (
                self._buffer
                and elapsed_since_data >= self.config.buffer_flush_timeout
            ):
                self._flush_buffer(stdout_fd, force=True)

        if self._buffer:
            self._flush_buffer(stdout_fd, force=True)

        try:
            _, status = os.waitpid(child_pid, 0)
            exit_code = os.waitstatus_to_exitcode(status)
        except ChildProcessError:
            exit_code = 0

        return exit_code
