"""Executor tests: Claude script routing, archive location, the pipe loop."""
from __future__ import annotations

import os
import select
import shutil
import signal
import stat
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from context_kernel.core import executor
from context_kernel.core.executor import ExecTiming
from context_kernel.core.orchestrator import Orchestrator, OrchestratorConfig
from context_kernel.memory.storage import StorageEngine
from context_kernel.pruners.shell_pruner import ShellPruner

# Exactly as Claude Code 2.x hands it to CLAUDE_CODE_SHELL_PREFIX (observed).
CLAUDE_SCRIPT = (
    "source /home/u/.claude/shell-snapshots/snapshot-bash-1790-ab.sh 2>/dev/null || true "
    "&& { shopt -u extglob || setopt NO_EXTENDED_GLOB NO_BARE_GLOB_QUAL; } >/dev/null 2>&1 "
    "|| true && eval 'python3 examples/demo_agent.py' < /dev/null "
    "&& pwd -P >| /tmp/claude-a900-cwd"
)
HOOK_SCRIPT = '"${CLAUDE_PLUGIN_ROOT}/hooks/run-hook.cmd" session-start'


class TestScriptRouting:
    def test_real_bash_tool_script_is_recognised(self) -> None:
        assert executor.is_bash_tool_script(CLAUDE_SCRIPT)

    def test_hook_script_is_not(self) -> None:
        assert not executor.is_bash_tool_script(HOOK_SCRIPT)

    def test_changed_script_shape_is_not(self) -> None:
        assert not executor.is_bash_tool_script(CLAUDE_SCRIPT.replace("pwd -P >|", "pwd >"))

    @pytest.mark.parametrize("cmd", [
        "ack recall 3",
        "/home/u/.venv/bin/ack recall 3",
        "ack recall 'KeyError' | head -n 5",
        "poetry run ack search foo",
    ])
    def test_ack_readers_detected(self, cmd: str) -> None:
        assert executor.invokes_ack_reader(f"x && eval '{cmd}' < /dev/null")

    @pytest.mark.parametrize("cmd", ["echo stack recall", "python3 -m pack recall", "ls"])
    def test_other_commands_not_detected(self, cmd: str) -> None:
        assert not executor.invokes_ack_reader(f"x && eval '{cmd}' < /dev/null")

    def test_shell_from_snapshot_name(self) -> None:
        assert executor.shell_for(CLAUDE_SCRIPT).endswith("bash")

    def test_shell_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/sh")
        assert executor.shell_for("echo hi") == "/bin/sh"


class TestArchiveLocation:
    def test_default_when_writable(self, tmp_path: Path) -> None:
        default = tmp_path / "ack" / "kernel.db"
        assert executor.writable_archive(default) == (default, False)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    def test_tmpdir_fallback_when_read_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
        monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
        try:
            path, fallback = executor.writable_archive(locked / "ack" / "kernel.db")
        finally:
            locked.chmod(stat.S_IRWXU)
        assert fallback
        assert path == tmp_path / "tmp" / "ack" / "kernel.db"


class TestRecallHint:
    def test_bare_ack_when_on_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ack = tmp_path / "ack"
        ack.write_text("#!/bin/sh\n")
        ack.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        assert executor.recall_hint(ack, tmp_path / "k.db", explicit_db=False) == "ack recall"

    def test_absolute_path_and_db_otherwise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATH", "/nonexistent")
        ack = tmp_path / "my env" / "ack"
        hint = executor.recall_hint(ack, tmp_path / "k.db", explicit_db=True)
        assert hint == f"'{ack}' recall --db {tmp_path / 'k.db'}"


FAST = ExecTiming(silence=0.3, max_hold=0.8, drain_after_exit=0.1)


@pytest.fixture
def orch(tmp_path: Path) -> Orchestrator:
    storage = StorageEngine(db_path=tmp_path / "exec.db")
    storage.open()
    session = storage.create_session("exec-test")
    return Orchestrator(
        command=["exec-test"], session_id=session.session_id, storage=storage,
        pruners=[ShellPruner()], config=OrchestratorConfig(color=False),
    )


def _collect(r: int, seconds: float) -> list[tuple[float, bytes]]:
    """Read from r for up to `seconds`, stamping each chunk with its arrival time."""
    got: list[tuple[float, bytes]] = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        ready, _, _ = select.select([r], [], [], 0.05)
        if ready:
            chunk = os.read(r, 65536)
            if not chunk:
                break
            got.append((time.monotonic(), chunk))
    return got


@pytest.mark.integration
class TestRunPiped:
    def test_exit_code_passthrough(self, orch: Orchestrator) -> None:
        r, w = os.pipe()
        assert executor.run_piped(["sh", "-c", "echo hi; exit 7"], orch, w, FAST) == 7
        os.close(w)
        assert b"".join(c for _, c in _collect(r, 1)) == b"hi\n"

    def test_signal_death_maps_to_128_plus_signal(self, orch: Orchestrator) -> None:
        r, w = os.pipe()
        try:
            assert executor.run_piped(["sh", "-c", "kill -9 $$"], orch, w, FAST) == 137
        finally:
            os.close(w)
            os.close(r)

    def test_streamed_flood_is_one_banner(self, orch: Orchestrator) -> None:
        r, w = os.pipe()
        cmd = "for i in $(seq 300); do echo 'WARNING retrying'; done"
        executor.run_piped(["sh", "-c", cmd], orch, w, FAST)
        os.close(w)
        out = b"".join(c for _, c in _collect(r, 1)).decode()
        assert out.count("[ACK] Compressed") == 1

    @pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
    def test_heartbeat_visible_while_running(self, orch: Orchestrator) -> None:
        r, w = os.pipe()
        cmd = "for i in 1 2 3 4 5 6 7 8 9 10; do echo beat $i; sleep 0.2; done"
        start = time.monotonic()
        t = threading.Thread(target=executor.run_piped, args=(["sh", "-c", cmd], orch, w, FAST))
        t.start()
        got = _collect(r, 3)
        t.join(timeout=10)
        assert not t.is_alive(), "run_piped did not return"
        assert got, "no output at all"
        assert got[0][0] - start < 1.5, "first line held until exit"

    def test_background_child_does_not_block(self, orch: Orchestrator) -> None:
        r, w = os.pipe()
        start = time.monotonic()
        code = executor.run_piped(
            ["sh", "-c", "(sleep 1.5; echo late) & echo now"], orch, w, FAST
        )
        assert code == 0
        assert time.monotonic() - start < 1.0
        os.close(w)
        out = b"".join(c for _, c in _collect(r, 3))
        assert b"now" in out and b"late" in out

    def test_command_gets_default_sigpipe(self, orch: Orchestrator) -> None:
        r, w = os.pipe()
        cmd = "set -o pipefail; yes | head -1; echo pipe=$?"
        executor.run_piped(["bash", "-c", cmd], orch, w, FAST)
        os.close(w)
        out = b"".join(c for _, c in _collect(r, 1))
        os.close(r)
        # 141 is what a plain shell reports: yes dies of SIGPIPE, silently.
        assert out == b"y\npipe=141\n"


def _kill_stray_sleeps() -> None:
    """Kill `sleep 5` processes left in this session by the signal tests."""
    ps_bin = shutil.which("ps")
    assert ps_bin
    ps = subprocess.run(
        [ps_bin, "-eo", "pid=,sid=,args="], capture_output=True, text=True, check=True
    )
    sid = os.getsid(0)
    for line in ps.stdout.splitlines():
        pid, psid, args = line.split(None, 2)
        if int(psid) == sid and args.strip() == "sleep 5":
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.fixture
def self_term() -> Iterator[None]:
    """Send SIGTERM to this process after 0.5s, as a user stopping ACK would."""
    timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM))
    timer.start()
    yield
    timer.cancel()
    timer.join()
    _kill_stray_sleeps()


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
class TestSignalForwarding:
    def _run(self, orch: Orchestrator, cmd: str) -> int:
        r, w = os.pipe()
        try:
            return executor.run_piped(["sh", "-c", cmd], orch, w, FAST)
        finally:
            os.close(w)
            os.close(r)

    def test_child_status_wins_when_it_traps(
        self, orch: Orchestrator, self_term: None
    ) -> None:
        assert self._run(orch, 'trap "exit 3" TERM; sleep 5 & wait') == 3

    def test_forwarded_term_kills_child(self, orch: Orchestrator, self_term: None) -> None:
        before = signal.getsignal(signal.SIGTERM)
        assert self._run(orch, "sleep 5") == 143
        assert signal.getsignal(signal.SIGTERM) is before
