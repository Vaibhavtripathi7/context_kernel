"""Connect ACK to Claude Code through CLAUDE_CODE_SHELL_PREFIX.

Claude runs every Bash command through the prefix after its permission checks,
so a prefix is the one place ACK can see command output before the model does.
The prefix is a tiny shell shim: if ACK is removed or can no longer import,
the shim runs the command directly with the shell Claude built it for, so
Claude keeps working.
"""
from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PREFIX_KEY = "CLAUDE_CODE_SHELL_PREFIX"


class HookError(Exception):
    """Install or uninstall could not proceed safely."""


def shim_path() -> Path:
    return Path.home() / ".local" / "share" / "ack" / "claude-prefix.sh"


def settings_paths(project_dir: Path) -> dict[str, Path]:
    return {
        "local":   project_dir / ".claude" / "settings.local.json",
        "project": project_dir / ".claude" / "settings.json",
        "user":    Path.home() / ".claude" / "settings.json",
    }


def render_shim(python: Path, ack_script: Path) -> str:
    # Runs under `python -E -c`. With -c, sys.path[0] is the current directory,
    # where a project's own modules could shadow ACK's imports, so drop it. If
    # ACK cannot import, run the script with the shell Claude built it for.
    bootstrap = (
        "import os, re, shutil, sys\n"
        "del sys.path[0]\n"
        "ack, script = sys.argv[1], sys.argv[-1]\n"
        "try:\n"
        "    from context_kernel.cli import main\n"
        "except BaseException:\n"
        '    os.environ["ACK_SHIM_FALLBACK"] = "1"\n'
        '    m = re.search(r"snapshot-(bash|zsh)-", script)\n'
        '    sh = m and shutil.which(m.group(1)) or "/bin/sh"\n'
        '    os.execv(sh, [sh, "-c", script])\n'
        'sys.argv = [ack, "exec", "--claude", "--", script]\n'
        "main()\n"
    )
    py, ack, code = (shlex.quote(str(x)) for x in (python, ack_script, bootstrap))
    return (
        "#!/bin/sh\n"
        "# Claude Code shell prefix installed by `ack hook install`.\n"
        "# -E keeps a project's PYTHONPATH from breaking ACK; the command itself\n"
        "# still gets the full environment. If ACK is gone or cannot import, run\n"
        "# the command with the shell Claude built it for.\n"
        f"if [ -x {py} ] && [ -f {ack} ]; then\n"
        f'    exec {py} -E -c {code} {ack} "$@"\n'
        "fi\n"
        "for last; do :; done\n"
        'case "$last" in\n'
        "    *snapshot-zsh-*)  sh=zsh ;;\n"
        "    *snapshot-bash-*) sh=bash ;;\n"
        "    *)                sh=/bin/sh ;;\n"
        "esac\n"
        'command -v "$sh" >/dev/null 2>&1 || sh=/bin/sh\n'
        "export ACK_SHIM_FALLBACK=1\n"
        'exec "$sh" -c "$last"\n'
    )


def find_conflicts(project_dir: Path, env: Mapping[str, str]) -> list[str]:
    """Places where a different shell prefix is already set."""
    ours = str(shim_path())
    found: list[str] = []
    if env.get(PREFIX_KEY, ours) != ours:
        found.append(f"the {PREFIX_KEY} environment variable")
    for path in settings_paths(project_dir).values():
        data  = _read(path)
        value = _env_of(path, data).get(PREFIX_KEY)
        if value is not None and value != ours:
            found.append(str(path))
    return found


def install(
    project_dir: Path, *, user: bool, python: Path, ack_script: Path, env: Mapping[str, str]
) -> Path:
    """Write the shim, check it works, and point Claude's settings at it."""
    conflicts = find_conflicts(project_dir, env)
    if conflicts:
        raise HookError(
            f"{PREFIX_KEY} is already set to something else in: {', '.join(conflicts)}. "
            "Remove it there first."
        )
    shim = _install_shim(python, ack_script)

    target = settings_paths(project_dir)["user" if user else "local"]
    data = _read(target)
    settings_env = _env_of(target, data)
    settings_env[PREFIX_KEY] = str(shim)
    data["env"] = settings_env
    _write(target, data)
    return target


def uninstall(project_dir: Path, *, user: bool) -> Path | None:
    """Remove ACK's prefix from the settings file; return it, or None if absent."""
    target = settings_paths(project_dir)["user" if user else "local"]
    data = _read(target)
    env = _env_of(target, data)
    if env.get(PREFIX_KEY) != str(shim_path()):
        return None
    del env[PREFIX_KEY]
    if env:
        data["env"] = env
    else:
        data.pop("env", None)
    if data or target.is_symlink():
        # Dotfile managers often symlink settings files. Deleting the link would
        # leave the real file behind with our key in it, so write the empty
        # object through the link.
        _write(target, data)
    else:
        target.unlink(missing_ok=True)
    return target


def _install_shim(python: Path, ack_script: Path) -> Path:
    """Render the shim into a temp file next to the real one, self-test that
    file, then swap it in atomically. A failed render or self-test leaves the
    existing shim (if any) untouched, so a bad reinstall can't break projects
    that already point at it.
    """
    shim = shim_path()
    shim.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=shim.parent, prefix=".claude-prefix-", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(render_shim(python, ack_script))
        tmp.chmod(0o755)
        _self_test(tmp, python)
        os.replace(tmp, shim)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return shim


def _self_test(shim: Path, python: Path) -> None:
    """Run a script through the shim and check ACK itself handled it.

    Both fallback paths (a missing python/ack file, or ack_script that
    cannot import) set ACK_SHIM_FALLBACK before running the script with a
    plain shell, so a marker of "1" means ACK never started.
    """
    script = "true && eval 'echo ack-ok:${ACK_SHIM_FALLBACK:-0}' < /dev/null && pwd -P >| /dev/null"
    try:
        result = subprocess.run(  # noqa: S603
            [str(shim), script], capture_output=True, text=True, timeout=30,
            env={**os.environ, "CLAUDE_CODE_SESSION_ID": "ack-hook-self-test"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HookError(f"self-test could not run the shim: {exc}") from exc
    if "ack-ok:1" in result.stdout:
        raise HookError(
            f"self-test failed: the shim ran the command without ACK "
            f"(ACK could not start with {python})"
        )
    if result.returncode != 0 or "ack-ok:0" not in result.stdout:
        raise HookError(f"self-test failed:\n{result.stdout}{result.stderr}".rstrip())


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HookError(f"{path} is not valid JSON ({exc}); fix it and retry") from exc
    if not isinstance(data, dict):
        raise HookError(f"{path} does not hold a JSON object")
    return data


def _env_of(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """The 'env' object inside loaded settings. A missing key is {}; any
    other non-object value is a HookError naming the file."""
    if "env" not in data:
        return {}
    env = data["env"]
    if not isinstance(env, dict):
        raise HookError(f'{path} has a non-object "env"')
    return env


def _write(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically. Writes through a symlink so the link survives,
    and keeps the previous file's permissions when there was one."""
    target = path.resolve() if path.is_symlink() else path
    target.parent.mkdir(parents=True, exist_ok=True)
    old_mode = target.stat().st_mode if target.exists() else None
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        if old_mode is not None:
            os.chmod(tmp, stat.S_IMODE(old_mode))
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
