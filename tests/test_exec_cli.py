"""End-to-end tests for `ack exec`, run as a real process."""
from __future__ import annotations

import contextlib
import os
import shlex
import signal
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
ACK = [sys.executable, "-c", "from context_kernel.cli import main; main()"]
TRACEBACK = (
    "import sys\n"
    "print('Traceback (most recent call last):')\n"
    "print('  File \"/app/src/views/checkout.py\", line 201, in post')\n"
    "print('    order = cart.checkout()')\n"
    "for i in range(40):\n"
    "    print(f'  File \"/venv/lib/python3.11/site-packages/django/base.py\", line {200 + i}, in f')\n"  # noqa: E501
    "    print('    response = cb()')\n"
    "print(\"KeyError: 'card_token'\")\n"
)

pytestmark = pytest.mark.integration


def claude_script(cmd: str, tmp: Path) -> str:
    """A script shaped like the one Claude Code hands its shell prefix."""
    return (
        f"source {tmp}/snapshot-bash-1.sh 2>/dev/null || true && "
        f"eval {shlex.quote(cmd)} < /dev/null && pwd -P >| {tmp}/cwd"
    )


def run_ack(args: list[str], tmp: Path, **kw: object) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "CLAUDE_CODE_SESSION_ID": "sess-1", **kw.pop("env", {})}  # type: ignore[dict-item]
    return subprocess.run(
        [*ACK, *args], capture_output=True, text=True, cwd=tmp, env=env, timeout=60, **kw
    )  # type: ignore[call-overload]


def rows(db: Path) -> list[sqlite3.Row]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    return con.execute("SELECT * FROM log_entries ORDER BY id").fetchall()


@pytest.fixture
def tb_file(tmp_path: Path) -> Path:
    f = tmp_path / "tb.py"
    f.write_text(TRACEBACK)
    return f


class TestClaudeForm:
    def test_prunes_and_archives(self, tmp_path: Path, tb_file: Path) -> None:
        db = tmp_path / "k.db"
        res = run_ack(["exec", "--claude", "--db", str(db),
                       claude_script(f"python3 {tb_file}; echo 2 failed", tmp_path)], tmp_path)
        assert res.returncode == 0
        assert "[ACK] Compressed" in res.stdout
        assert "full log: " in res.stdout and "recall --db" in res.stdout
        assert "\x1b" not in res.stdout
        assert "line 210" not in res.stdout
        assert "2 failed" in res.stdout
        stored = rows(db)
        assert len(stored) == 1 and stored[0]["was_pruned"] == 1
        assert "line 210" in stored[0]["raw_content"]
        assert stored[0]["session_id"] == "sess-1"

    def test_hook_script_untouched(self, tmp_path: Path) -> None:
        db = tmp_path / "k.db"
        payload = "python3 -c \"import json; print(json.dumps({'a': list(range(60))}, indent=1))\""
        res = run_ack(["exec", "--claude", "--db", str(db), payload], tmp_path)
        expected = subprocess.run(payload, shell=True, capture_output=True, text=True).stdout  # noqa: S602
        assert res.stdout == expected
        assert not db.exists()

    def test_ack_recall_output_not_pruned(self, tmp_path: Path) -> None:
        db = tmp_path / "k.db"
        cmd = "for i in $(seq 60); do echo same line; done  # ack recall 1"
        res = run_ack(["exec", "--claude", "--db", str(db), claude_script(cmd, tmp_path)], tmp_path)
        assert res.stdout.count("same line") == 60

    def test_ack_reader_exec_failure_reports_and_exits_127(self, tmp_path: Path) -> None:
        # No snapshot-bash marker, so shell_for() falls back to $SHELL, which we
        # point at a path that does not exist, forcing the passthrough exec to fail.
        script = f"true && eval 'ack recall 1' < /dev/null && pwd -P >| {tmp_path}/cwd"
        res = run_ack(["exec", "--claude", "--db", str(tmp_path / "k.db"), script],
                      tmp_path, env={"SHELL": "/nonexistent/ack-test-shell"})
        assert res.returncode == 127
        assert "ack: cannot run /nonexistent/ack-test-shell" in res.stderr
        assert "Traceback" not in res.stdout + res.stderr

    def test_exit_codes(self, tmp_path: Path) -> None:
        db = str(tmp_path / "k.db")
        assert run_ack(["exec", "--claude", "--db", db, claude_script("exit 7", tmp_path)],
                       tmp_path).returncode == 7
        assert run_ack(["exec", "--claude", "--db", db, claude_script("kill -9 $$", tmp_path)],
                       tmp_path).returncode == 137

    def test_cwd_file_written(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        run_ack(["exec", "--claude", "--db", str(tmp_path / "k.db"),
                 claude_script("cd sub", tmp_path)], tmp_path)
        assert (tmp_path / "cwd").read_text().strip().endswith("/sub")

    def test_fail_open_when_archive_unusable(self, tmp_path: Path) -> None:
        res = run_ack(["exec", "--claude", "--db", str(tmp_path),  # a directory: cannot open
                       claude_script("echo still works; exit 4", tmp_path)], tmp_path)
        assert res.stdout == "still works\n"
        assert res.returncode == 4

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    def test_sandbox_fallback_archive(self, tmp_path: Path, tb_file: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        home.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            res = run_ack(["exec", "--claude", claude_script(f"python3 {tb_file}", tmp_path)],
                          tmp_path, env={"HOME": str(home), "TMPDIR": str(tmp_path / "t")})
        finally:
            home.chmod(stat.S_IRWXU)
        assert f"--db {tmp_path / 't' / 'ack' / 'kernel.db'}" in res.stdout
        assert len(rows(tmp_path / "t" / "ack" / "kernel.db")) == 1

    def test_sigterm_flushes_archives_and_exits_143(self, tmp_path: Path) -> None:
        db = tmp_path / "k.db"
        proc = subprocess.Popen(
            [*ACK, "exec", "--claude", "--db", str(db),
             claude_script("echo started; sleep 30", tmp_path)],
            stdout=subprocess.PIPE, text=True, cwd=tmp_path,
            env={**os.environ, "CLAUDE_CODE_SESSION_ID": "sess-1"},
            start_new_session=True,  # ack leads its own group, the way Claude spawns it
        )
        time.sleep(1.0)
        # Claude signals the whole group on a Bash-tool timeout, not just ack's pid.
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            out, _ = proc.communicate(timeout=10)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        assert proc.returncode == 143
        assert "started" in out
        assert any("started" in r["raw_content"] for r in rows(db))


class TestGenericForm:
    def test_prunes_a_plain_command(self, tmp_path: Path, tb_file: Path) -> None:
        res = run_ack(["exec", "--db", str(tmp_path / "k.db"), "--", "python3", str(tb_file)],
                      tmp_path)
        assert "[ACK] Compressed" in res.stdout

    def test_stdin_reaches_the_command(self, tmp_path: Path) -> None:
        res = run_ack(["exec", "--db", str(tmp_path / "k.db"), "--", "sh", "-c",
                       'read x; echo "got:$x"'], tmp_path, input="hello\n")
        assert res.stdout == "got:hello\n"

    def test_missing_command_is_127_without_traceback(self, tmp_path: Path) -> None:
        res = run_ack(["exec", "--db", str(tmp_path / "k.db"), "--", "no-such-cmd-xyz"], tmp_path)
        assert res.returncode == 127
        assert "Traceback" not in res.stdout + res.stderr
        assert "no-such-cmd-xyz" in res.stdout + res.stderr

    def test_closed_pipe_is_quiet(self, tmp_path: Path) -> None:
        cmd = " ".join(shlex.quote(a) for a in ACK)
        res = subprocess.run(
            f"{cmd} exec --db {tmp_path / 'k.db'} -- seq 1 200000 | head -1",
            shell=True, capture_output=True, text=True, timeout=60,  # noqa: S602
        )
        assert res.stdout == "1\n"
        assert "Traceback" not in res.stderr
