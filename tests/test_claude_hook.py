"""Tests for wiring ACK into Claude Code as its shell prefix."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from context_kernel import claude_hook
from context_kernel.claude_hook import PREFIX_KEY, HookError

FAKE_ACK = (
    "import os, sys\n"
    "if sys.argv[1:4] != ['exec', '--claude', '--']:\n"
    "    sys.exit(90)\n"
    "os.execv('/bin/sh', ['sh', '-c', sys.argv[-1]])\n"
)


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
    f = tmp_path / "venv bin" / "ack"   # a space on purpose
    f.parent.mkdir()
    f.write_text(FAKE_ACK)
    return f


def install(project: Path, fake_ack: Path, env: dict[str, str] | None = None,
            user: bool = False) -> Path:
    return claude_hook.install(project, user=user, python=Path(sys.executable),
                               ack_script=fake_ack, env=env or {})


def read(path: Path) -> dict:
    return json.loads(path.read_text())


CLAUDE_SCRIPT = "true && eval 'echo hi; exit 3' < /dev/null && pwd -P >| /dev/null"


class TestInstall:
    def test_writes_shim_and_local_settings(
        self, home: Path, project: Path, fake_ack: Path
    ) -> None:
        target = install(project, fake_ack)
        assert target == project / ".claude" / "settings.local.json"
        shim = claude_hook.shim_path()
        assert read(target)["env"][PREFIX_KEY] == str(shim)
        res = subprocess.run([str(shim), CLAUDE_SCRIPT], capture_output=True, text=True)
        assert (res.stdout, res.returncode) == ("hi\n", 3)

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
        self, home: Path, project: Path, tmp_path: Path
    ) -> None:
        broken = tmp_path / "broken_ack"
        broken.write_text("import sys\nsys.exit(1)\n")
        with pytest.raises(HookError, match="self-test"):
            install(project, broken)
        assert not (project / ".claude" / "settings.local.json").exists()

    def test_reinstall_with_broken_ack_leaves_shim_working(
        self, home: Path, project: Path, fake_ack: Path, tmp_path: Path
    ) -> None:
        install(project, fake_ack)
        broken = tmp_path / "broken_ack"
        broken.write_text("import sys\nsys.exit(1)\n")
        with pytest.raises(HookError, match="self-test"):
            install(project, broken)
        shim = claude_hook.shim_path()
        res = subprocess.run([str(shim), CLAUDE_SCRIPT], capture_output=True, text=True)
        assert (res.stdout, res.returncode) == ("hi\n", 3)

    def test_non_object_env_is_an_error(self, home: Path, project: Path, fake_ack: Path) -> None:
        shared = project / ".claude" / "settings.json"
        shared.parent.mkdir()
        shared.write_text(json.dumps({"env": None}))
        with pytest.raises(HookError, match="settings.json"):
            install(project, fake_ack)
        assert not (project / ".claude" / "settings.local.json").exists()


class TestShimFallback:
    def test_runs_command_directly_when_ack_is_gone(self, home: Path, tmp_path: Path) -> None:
        shim = tmp_path / "shim.sh"
        shim.write_text(claude_hook.render_shim(Path("/gone/python"), Path("/gone/ack")))
        shim.chmod(0o755)
        res = subprocess.run([str(shim), CLAUDE_SCRIPT], capture_output=True, text=True)
        assert (res.stdout, res.returncode) == ("hi\n", 3)


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
