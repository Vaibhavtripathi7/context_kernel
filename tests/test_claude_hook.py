"""Tests for wiring ACK into Claude Code as its shell prefix."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from context_kernel import claude_hook
from context_kernel.claude_hook import PREFIX_KEY, HookError

# The shim only checks the script exists and passes its path on as argv[0];
# it imports ACK itself, so the test venv's real context_kernel runs.
ACK_SCRIPT = "import sys\nfrom context_kernel.cli import main\nsys.exit(main())\n"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home dir"          # a space on purpose
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    return h


@pytest.fixture
def project(tmp_path: Path) -> Path:
    p = tmp_path / "proj"
    p.mkdir()
    return p


@pytest.fixture
def fake_ack(tmp_path: Path) -> Path:
    f = tmp_path / "venv bin" / "ack's"   # a space and a quote on purpose
    f.parent.mkdir()
    f.write_text(ACK_SCRIPT)
    return f


def install(project: Path, fake_ack: Path, env: dict[str, str] | None = None,
            user: bool = False, python: Path = Path(sys.executable)) -> Path:
    return claude_hook.install(project, user=user, python=python,
                               ack_script=fake_ack, env=env or {})


def archive(home: Path) -> Path:
    return home / ".local" / "share" / "ack" / "kernel.db"


def run_shim(shim: Path, script: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(shim), script], capture_output=True, text=True, timeout=60,
                          env={**os.environ, **env})


def read(path: Path) -> dict:
    return json.loads(path.read_text())


CLAUDE_SCRIPT = "true && eval 'echo hi; exit 3' < /dev/null && pwd -P >| /dev/null"
# Only bash sets BASH_VERSION, so the output shows which shell ran the script.
BASH_SCRIPT = (
    "source /nowhere/snapshot-bash-1.sh 2>/dev/null || true && "
    "eval 'echo ${BASH_VERSION:+bash}; exit 3' < /dev/null && pwd -P >| /dev/null"
)


class TestInstall:
    def test_writes_shim_and_local_settings(
        self, home: Path, project: Path, fake_ack: Path
    ) -> None:
        target = install(project, fake_ack)
        assert target == project / ".claude" / "settings.local.json"
        shim = claude_hook.shim_path()
        assert read(target)["env"][PREFIX_KEY] == str(shim)
        res = run_shim(shim, CLAUDE_SCRIPT)
        assert (res.stdout, res.returncode) == ("hi\n", 3)
        assert archive(home).exists()

    def test_keeps_other_settings(self, home: Path, project: Path, fake_ack: Path) -> None:
        target = project / ".claude" / "settings.local.json"
        target.parent.mkdir()
        target.write_text(json.dumps({"model": "x", "env": {"FOO": "1"}}))
        install(project, fake_ack)
        data = read(target)
        assert data["model"] == "x" and data["env"]["FOO"] == "1"

    def test_idempotent(self, home: Path, project: Path, fake_ack: Path) -> None:
        first = install(project, fake_ack).read_text()
        assert install(project, fake_ack).read_text() == first

    def test_user_scope(self, home: Path, project: Path, fake_ack: Path) -> None:
        assert install(project, fake_ack, user=True) == home / ".claude" / "settings.json"

    def test_refuses_env_conflict(self, home: Path, project: Path, fake_ack: Path) -> None:
        with pytest.raises(HookError, match=PREFIX_KEY):
            install(project, fake_ack, env={PREFIX_KEY: "/other/prefix"})

    def test_refuses_settings_conflict(self, home: Path, project: Path, fake_ack: Path) -> None:
        shared = project / ".claude" / "settings.json"
        shared.parent.mkdir()
        shared.write_text(json.dumps({"env": {PREFIX_KEY: "/other/prefix"}}))
        with pytest.raises(HookError, match="settings.json"):
            install(project, fake_ack)

    def test_refuses_malformed_settings_and_leaves_it(
        self, home: Path, project: Path, fake_ack: Path
    ) -> None:
        target = project / ".claude" / "settings.local.json"
        target.parent.mkdir()
        target.write_text("{not json")
        with pytest.raises(HookError):
            install(project, fake_ack)
        assert target.read_text() == "{not json"

    def test_failed_self_test_writes_no_settings(
        self, home: Path, project: Path, fake_ack: Path
    ) -> None:
        with pytest.raises(HookError, match="self-test"):
            install(project, fake_ack, python=Path("/bin/false"))
        assert not (project / ".claude" / "settings.local.json").exists()

    def test_self_test_fails_when_ack_cannot_import(
        self, home: Path, project: Path, fake_ack: Path, tmp_path: Path
    ) -> None:
        # -S hides site-packages, so context_kernel is gone but python still
        # runs, like an editable install whose clone was moved. The shim's
        # fallback then runs the command with plain sh, so ACK never sees it.
        no_site = tmp_path / "python"
        no_site.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -S "$@"\n')
        no_site.chmod(0o755)
        with pytest.raises(HookError, match="self-test"):
            install(project, fake_ack, python=no_site)
        assert not (project / ".claude" / "settings.local.json").exists()

    def test_reinstall_with_broken_python_leaves_shim_working(
        self, home: Path, project: Path, fake_ack: Path
    ) -> None:
        install(project, fake_ack)
        with pytest.raises(HookError, match="self-test"):
            install(project, fake_ack, python=Path("/bin/false"))
        res = run_shim(claude_hook.shim_path(), CLAUDE_SCRIPT)
        assert (res.stdout, res.returncode) == ("hi\n", 3)

    def test_non_object_env_is_an_error(self, home: Path, project: Path, fake_ack: Path) -> None:
        shared = project / ".claude" / "settings.json"
        shared.parent.mkdir()
        shared.write_text(json.dumps({"env": None}))
        with pytest.raises(HookError, match="settings.json"):
            install(project, fake_ack)
        assert not (project / ".claude" / "settings.local.json").exists()


def write_shim(tmp_path: Path, python: Path, ack_script: Path) -> Path:
    shim = tmp_path / "shim.sh"
    shim.write_text(claude_hook.render_shim(python, ack_script))
    shim.chmod(0o755)
    return shim


class TestShimFallback:
    def test_runs_command_directly_when_ack_is_gone(self, home: Path, tmp_path: Path) -> None:
        shim = write_shim(tmp_path, Path("/gone/python"), Path("/gone/ack"))
        res = run_shim(shim, CLAUDE_SCRIPT)
        assert (res.stdout, res.returncode) == ("hi\n", 3)

    def test_hook_script_falls_back_to_bin_sh(self, home: Path, tmp_path: Path) -> None:
        shim = write_shim(tmp_path, Path("/gone/python"), Path("/gone/ack"))
        res = run_shim(shim, "echo hook ran", SHELL="/nowhere/shell")
        assert (res.stdout, res.returncode) == ("hook ran\n", 0)

    @pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
    def test_missing_ack_uses_the_snapshot_shell(self, home: Path, tmp_path: Path) -> None:
        shim = write_shim(tmp_path, Path("/gone/python"), Path("/gone/ack"))
        res = run_shim(shim, BASH_SCRIPT, SHELL="/nowhere/shell")
        assert (res.stdout, res.returncode) == ("bash\n", 3)

    def test_project_pythonpath_cannot_break_ack(
        self, home: Path, tmp_path: Path, fake_ack: Path
    ) -> None:
        shadow = tmp_path / "shadow"
        shadow.mkdir()
        (shadow / "click.py").write_text("raise ImportError('shadowed')\n")
        shim = write_shim(tmp_path, Path(sys.executable), fake_ack)
        res = run_shim(shim, CLAUDE_SCRIPT, PYTHONPATH=str(shadow))
        assert (res.stdout, res.returncode) == ("hi\n", 3)
        assert archive(home).exists(), "ACK itself should have run, not the fallback"

    @pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
    def test_runs_command_when_ack_cannot_import(
        self, home: Path, tmp_path: Path, fake_ack: Path
    ) -> None:
        # -S hides site-packages, so context_kernel is gone but python still runs,
        # like an editable install whose clone was moved.
        no_site = tmp_path / "python"
        no_site.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -S "$@"\n')
        no_site.chmod(0o755)
        shim = write_shim(tmp_path, no_site, fake_ack)
        res = run_shim(shim, BASH_SCRIPT, SHELL="/nowhere/shell")
        assert (res.stdout, res.returncode) == ("bash\n", 3)
        assert "Traceback" not in res.stderr
        assert not archive(home).exists()


class TestUninstall:
    def test_removes_only_ack_key(self, home: Path, project: Path, fake_ack: Path) -> None:
        target = project / ".claude" / "settings.local.json"
        target.parent.mkdir()
        target.write_text(json.dumps({"env": {"FOO": "1"}}))
        install(project, fake_ack)
        assert claude_hook.uninstall(project, user=False) == target
        assert read(target) == {"env": {"FOO": "1"}}

    def test_noop_when_not_installed(self, home: Path, project: Path) -> None:
        assert claude_hook.uninstall(project, user=False) is None

    def test_deletes_settings_file_left_empty(
        self, home: Path, project: Path, fake_ack: Path
    ) -> None:
        target = install(project, fake_ack)
        claude_hook.uninstall(project, user=False)
        assert not target.exists()

    def test_keeps_symlinked_settings_file_left_empty(
        self, home: Path, project: Path, fake_ack: Path
    ) -> None:
        real = project / "real-settings.json"
        link = project / ".claude" / "settings.local.json"
        link.parent.mkdir()
        real.write_text("{}")
        link.symlink_to(real)
        install(project, fake_ack)
        assert claude_hook.uninstall(project, user=False) == link
        assert link.is_symlink()
        assert PREFIX_KEY not in read(real).get("env", {})
