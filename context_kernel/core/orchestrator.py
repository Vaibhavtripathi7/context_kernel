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
import re
import select
import signal
import struct
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..memory.storage import LogEntry, StorageEngine
from ..pruners.base import BasePruner

_DIM   = "\033[2m"
_CYAN  = "\033[36m"
_RESET = "\033[0m"

_ANSI_ESC = re.compile(r"\x1b\[[0-9;]*[mGKHFJA-Z]")
_TRACEBACK_MARKER = "Traceback (most recent call last):"


@dataclass
class OrchestratorConfig:
    pruning_threshold_lines: int   = 30
    buffer_flush_timeout:    float = 0.15
    read_chunk_bytes:        int   = 8192
    annotate_injections:     bool  = True
    max_buffer_bytes:        int   = 262144


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
        self.pruners: list[BasePruner] = pruners or []
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

    def run(self) -> int:
        """Spawn the agent, block until it exits, and return its exit code.

        The terminal is always restored, even if the child crashes.
        """
        self._master_fd, child_pid = self._spawn_in_pty()
        self._child_pid = child_pid

        self._enter_raw_mode()
        self._install_signal_handlers()

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

            try:
                os.execvp(self.command[0], self.command)
            except OSError as exc:
                os.write(2, f"ack: cannot run {self.command[0]!r}: {exc.strerror}\n".encode())
            os._exit(127)

        os.close(slave_fd)
        return master_fd, child_pid

    def _io_loop(self, child_pid: int) -> int:
        """Pump I/O between the user and the child until the child exits.

        While nothing is buffered the loop blocks in select(); it only polls on
        the flush timeout while it is still holding data to emit, so it stays at
        ~0% CPU when idle. stdin is dropped from the watch set once it reaches
        EOF so a closed/piped input never spins the loop.
        """
        assert self._master_fd is not None
        master_fd = self._master_fd
        stdin_fd  = sys.stdin.fileno()
        stdout_fd = sys.stdout.fileno()
        exit_code = 0
        watched   = [master_fd, stdin_fd]

        while True:
            timeout = self.config.buffer_flush_timeout if self._buffer else None
            try:
                rlist, _, _ = select.select(watched, [], [], timeout)
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
                else:
                    watched = [master_fd]

            if self._buffer:
                elapsed_since_data = time.monotonic() - self._last_data_monotonic
                if elapsed_since_data >= self.config.buffer_flush_timeout:
                    self._flush_buffer(stdout_fd, force=True)

        if self._buffer:
            self._flush_buffer(stdout_fd, force=True)

        try:
            _, status = os.waitpid(child_pid, 0)
            exit_code = os.waitstatus_to_exitcode(status)
        except ChildProcessError:
            exit_code = 0

        return exit_code

    def _accumulate(self, chunk: bytes) -> None:
        self._buffer.append(chunk)

        if self._tail_is_prompt(chunk):
            self._flush_buffer(sys.stdout.fileno(), force=True)

    def _flush_buffer(self, stdout_fd: int, *, force: bool = False) -> None:
        """Emit the buffered output, pruning it first if it qualifies.

        force controls only whether to emit now (silence timeout / EOF), never
        whether to prune: output below pruning_threshold_lines and interactive
        prompts always pass through verbatim. An unfinished traceback is held
        (up to max_buffer_bytes) so it prunes as one unit across PTY reads.
        """
        if not self._buffer:
            return

        raw: bytes = b"".join(self._buffer)
        self._buffer.clear()

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")

        line_count = len(text.splitlines(keepends=True))

        below_threshold = line_count < self.config.pruning_threshold_lines
        if not force and below_threshold:
            self._buffer.append(raw)
            return

        if (
            not force
            and len(raw) < self.config.max_buffer_bytes
            and self._is_incomplete_traceback(text)
        ):
            self._buffer.append(raw)
            return

        if below_threshold or self._text_is_prompt(text):
            self._emit(stdout_fd, raw)
            self._persist(text, pruned=False)
            if self.text_callback is not None:
                self.text_callback(text)
            return

        summary: str | None = None
        for pruner in self.pruners:
            result = pruner.compress(text)
            if result is not None:
                summary = result
                saved = max(0, (len(text) - len(summary)) // 4)
                self._stats.tokens_saved      += saved
                self._stats.total_pruner_hits += 1
                break

        if summary is not None:
            self._persist(text, pruned=True, summary=summary)
            injection = self._format_injection(summary, line_count)
            injected  = self._terminal_newlines(injection).encode("utf-8")
            self._stats.total_bytes_injected += len(injected)
            self._emit(stdout_fd, injected)
            _display = injection
        else:
            self._emit(stdout_fd, raw)
            self._persist(text, pruned=False)
            _display = text

        if self.text_callback is not None:
            self.text_callback(_display)

        if self.stats_callback is not None:
            self.stats_callback(self._stats)

    def _terminal_newlines(self, text: str) -> str:
        """Convert text ACK generates itself to CRLF while the terminal is raw.

        Raw mode disables the terminal's NL->CRLF output mapping, so a bare
        ``\\n`` would leave the cursor in the same column (the "staircase"
        effect). Child passthrough already carries CRLF from its own PTY.
        """
        if self._saved_tty is None:
            return text
        return text.replace("\r\n", "\n").replace("\n", "\r\n")

    def _emit(self, fd: int, data: bytes) -> None:
        offset = 0
        while offset < len(data):
            try:
                offset += os.write(fd, data[offset:])
            except OSError:
                break

    def _persist(self, text: str, *, pruned: bool, summary: str = "") -> None:
        entry = LogEntry(
            session_id=self.session_id,
            raw_content=text,
            entry_type="stdout",
            compressed_summary=summary,
            token_estimate=max(0, len(text) // 4),
            was_pruned=pruned,
        )
        try:
            self.storage.insert_entry(entry)
        except Exception:  # noqa: BLE001
            pass

    def _format_injection(self, summary: str, original_lines: int) -> str:
        body = f"{_CYAN}{summary}{_RESET}\n"
        if not self.config.annotate_injections:
            return body

        summary_lines = len(summary.splitlines())
        banner = (
            f"{_DIM}[ACK] Compressed {original_lines} lines → "
            f"{summary_lines} lines  (full log stored in DB){_RESET}\n"
        )
        return banner + body

    def _is_incomplete_traceback(self, text: str) -> bool:
        """True if text holds a traceback still streaming its frames.

        A finished traceback ends in a non-indented exception line; while frames
        are still arriving the last non-blank line is an indented frame line, or
        the header itself.
        """
        if _TRACEBACK_MARKER not in text:
            return False
        clean = _ANSI_ESC.sub("", text)
        nonblank = [ln for ln in clean.splitlines() if ln.strip()]
        if not nonblank:
            return False
        last = nonblank[-1]
        if _TRACEBACK_MARKER in last:
            return True
        return last[:1].isspace()

    def _tail_is_prompt(self, chunk: bytes) -> bool:
        tail = chunk[-200:]
        return any(p in tail for p in _PROMPT_BYTES)

    def _text_is_prompt(self, text: str) -> bool:
        stripped = text.rstrip()
        if stripped and stripped[-1] in _PROMPT_TAIL_CHARS:
            last_line = stripped.rsplit("\n", 1)[-1]
            if len(last_line) <= _PROMPT_MAX_LINE_LEN:
                return True
        tail_bytes = text[-200:].encode("utf-8", errors="replace")
        return any(p in tail_bytes for p in _PROMPT_BYTES)

    def _enter_raw_mode(self) -> None:
        if not sys.stdin.isatty():
            return
        self._saved_tty = termios.tcgetattr(sys.stdin.fileno())
        tty.setraw(sys.stdin.fileno())

    def _restore_terminal(self) -> None:
        if self._saved_tty is not None and sys.stdin.isatty():
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, self._saved_tty)
            self._saved_tty = None

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        signal.signal(signal.SIGWINCH, self._on_sigwinch)
        signal.signal(signal.SIGTERM, self._on_terminate)
        signal.signal(signal.SIGHUP, self._on_terminate)

    def _on_sigwinch(self, signum: int, frame: object) -> None:  # noqa: ARG002
        if self._master_fd is not None and sys.stdout.isatty():
            rows, cols = self._get_terminal_size()
            self._set_winsize(self._master_fd, rows, cols)

    def _on_terminate(self, signum: int, frame: object) -> None:  # noqa: ARG002
        """Restore the terminal and forward the signal to the child before exit."""
        self._restore_terminal()
        if self._child_pid is not None:
            try:
                os.killpg(self._child_pid, signum)
            except OSError:
                pass
        os._exit(128 + signum)

    @staticmethod
    def _get_terminal_size() -> tuple[int, int]:
        try:
            cols, rows = os.get_terminal_size(sys.stdout.fileno())
            return rows, cols
        except OSError:
            return 24, 80

    @staticmethod
    def _set_winsize(fd: int, rows: int, cols: int) -> None:
        packed = struct.pack("HHHH", rows, cols, 0, 0)
        try:
            fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)
        except OSError:
            pass
