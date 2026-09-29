"""Run one command on a pipe and prune its output before the agent reads it.

Coding agents run shell commands in their own subprocesses, so ACK has to sit
on that path. Claude Code hands every Bash command to CLAUDE_CODE_SHELL_PREFIX
as one script; `ack exec --claude` runs that script here. Output is flushed on
a short silence or a max hold time rather than on exit, because the agent may
be polling a long-running process such as a dev server.
"""
from __future__ import annotations

import os
import re
import select
import shlex
import shutil
import signal
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from .orchestrator import Orchestrator

_BASH_TOOL_MARKERS = (" eval ", "pwd -P >|")
_ACK_READER        = re.compile(r"""(?:^|[\s/'"(;&|])ack['"]?\s+(?:recall|search)(?=[\s'"]|$)""")
_SNAPSHOT_SHELL    = re.compile(r"snapshot-(bash|zsh)-")
_FORWARDED_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def is_bash_tool_script(script: str) -> bool:
    """True for scripts Claude's Bash tool builds; its hooks look different
    and must run untouched because Claude parses their output."""
    return all(marker in script for marker in _BASH_TOOL_MARKERS)


def invokes_ack_reader(script: str) -> bool:
    """True if the script runs `ack recall` or `ack search`, whose output must
    never be pruned again."""
    return bool(_ACK_READER.search(script))


def shell_for(script: str) -> str:
    """The shell Claude built the script for (named in its snapshot file)."""
    match = _SNAPSHOT_SHELL.search(script)
    if match:
        found = shutil.which(match.group(1))
        if found:
            return found
    return os.environ.get("SHELL") or "/bin/sh"


def writable_archive(default: Path) -> tuple[Path, bool]:
    """Return (db_path, is_fallback). Inside Claude's sandbox the home directory
    is read-only, so fall back to the writable TMPDIR it provides."""
    if _can_write(default.parent):
        return default, False
    base = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    return base / "ack" / "kernel.db", True


def _can_write(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".ack-write-{os.getpid()}"
        probe.touch()
        probe.unlink()
    except OSError:
        return False
    return True


def recall_hint(ack_path: Path, db: Path, *, explicit_db: bool) -> str:
    """The recall command to print in banners, runnable from the agent's shell."""
    on_path = shutil.which("ack")
    same = on_path is not None and Path(on_path).resolve() == ack_path.resolve()
    hint = f"{'ack' if same else shlex.quote(str(ack_path))} recall"
    if explicit_db:
        hint += f" --db {shlex.quote(str(db))}"
    return hint


@dataclass(frozen=True)
class ExecTiming:
    silence:          float = 2.0
    max_hold:         float = 5.0
    drain_after_exit: float = 0.2


def run_piped(
    argv: list[str], orch: Orchestrator, out_fd: int, timing: ExecTiming | None = None
) -> int:
    """Run argv with stdout and stderr on one pipe, streaming its output through
    orch to out_fd. Returns the command's exit code (128 + signal if killed)."""
    timing = timing or ExecTiming()
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.close(r)
            os.dup2(w, 1)
            os.dup2(w, 2)
            os.close(w)
            # Python ignores these at startup and exec keeps ignored signals.
            for name in ("SIGPIPE", "SIGXFSZ"):
                if hasattr(signal, name):
                    signal.signal(getattr(signal, name), signal.SIG_DFL)
            os.execvp(argv[0], argv)
        except OSError as exc:
            os.write(2, f"ack: cannot run {argv[0]}: {exc.strerror}\n".encode())
        finally:
            os._exit(127)
    os.close(w)

    def forward(signum: int, _frame: FrameType | None) -> None:
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass

    previous = {}
    if threading.current_thread() is threading.main_thread():
        previous = {sig: signal.signal(sig, forward) for sig in _FORWARDED_SIGNALS}
    try:
        status = _pump(r, pid, orch, out_fd, timing)
    except BaseException:
        os.close(r)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    if not orch.output_closed:
        orch.flush(out_fd, final=True)
        if orch.output_closed:
            os.close(r)
        else:
            _hand_off(r, out_fd)
    code = os.waitstatus_to_exitcode(status)
    return 128 - code if code < 0 else code


def _pump(r: int, pid: int, orch: Orchestrator, out_fd: int, timing: ExecTiming) -> int:
    """Read until the command's shell exits or our output is closed, flushing
    on silence, on max hold, or when the buffer is full. Returns the raw wait
    status."""
    status: int | None = None
    exit_deadline = first = last = 0.0
    while True:
        now = time.monotonic()
        if status is None:
            done, wait_status = os.waitpid(pid, os.WNOHANG)
            if done:
                status, exit_deadline = wait_status, now + timing.drain_after_exit
        elif now >= exit_deadline:
            return status

        try:
            ready, _, _ = select.select([r], [], [], 0.05)
        except InterruptedError:
            continue
        if ready:
            chunk = os.read(r, 65536)
            if not chunk:
                if status is None:
                    _, status = os.waitpid(pid, 0)
                return status
            if not orch.buffered_bytes:
                first = time.monotonic()
            orch.feed(chunk)
            last = time.monotonic()

        if orch.buffered_bytes:
            now = time.monotonic()
            if (
                now - last >= timing.silence
                or now - first >= timing.max_hold
                or orch.buffered_bytes >= orch.config.max_buffer_bytes
            ):
                orch.flush(out_fd)
                first = last = now

        if orch.output_closed:
            # Like a shell pipeline: close our end so the command gets SIGPIPE
            # on its next write, and stop reading and archiving.
            os.close(r)
            if status is None:
                _, status = os.waitpid(pid, 0)
            return status


def _hand_off(r: int, out_fd: int) -> None:
    """Keep forwarding output from processes the command left running.

    The shell has exited, but something it backgrounded may still hold the
    pipe. Waiting would hang the agent, and closing the pipe would kill that
    process with SIGPIPE on its next write. A detached copier forwards it raw
    and exits when the last writer closes.
    """
    if os.fork() == 0:
        try:
            os.setsid()
            while chunk := os.read(r, 65536):
                os.write(out_fd, chunk)
        except OSError:
            pass
        os._exit(0)
    os.close(r)
