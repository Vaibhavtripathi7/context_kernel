"""CLI tests: terminal-escape sanitisation and the `recall` command."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from context_kernel.cli import _sanitize, main
from context_kernel.memory.storage import LogEntry, StorageEngine


class TestSanitize:
    """Stored agent output carries raw escape sequences; displaying them must
    never be able to reprogram the user's terminal."""

    def test_strips_color_csi(self) -> None:
        assert _sanitize("\x1b[31mred\x1b[0m text") == "red text"

    def test_strips_truecolor_csi(self) -> None:
        assert _sanitize("\x1b[38;2;255;0;0mx\x1b[0m") == "x"

    def test_strips_mouse_and_app_mode_toggles(self) -> None:
        evil = "\x1b[?1000h\x1b[?1006hclick\x1b[?2004h"
        assert _sanitize(evil) == "click"

    def test_strips_osc_title_sequence(self) -> None:
        assert _sanitize("\x1b]0;malicious title\x07ok") == "ok"

    def test_strips_lone_control_chars(self) -> None:
        assert _sanitize("a\x00b\x07c\x7f") == "abc"

    def test_preserves_newlines_and_tabs(self) -> None:
        assert _sanitize("line1\nline2\tend") == "line1\nline2\tend"

    def test_plain_text_unchanged(self) -> None:
        assert _sanitize("KeyError: 'card_token'") == "KeyError: 'card_token'"


class TestRecall:
    """`ack recall` pages a stored log back by numeric handle or FTS query."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "recall.db"

    def _seed(self, db_path: Path) -> dict[str, int]:
        """Create a session with two entries; return their recall handles."""
        with StorageEngine(db_path=db_path) as eng:
            sid = eng.create_session("pytest-agent").session_id
            flood = eng.insert_entry(
                LogEntry(session_id=sid, raw_content="Traceback: ValueError deep in the log",
                         entry_type="stderr", was_pruned=True)
            )
            note = eng.insert_entry(
                LogEntry(session_id=sid, raw_content="build finished ok", entry_type="stdout")
            )
        return {"flood": flood, "note": note}

    def test_recall_by_id_returns_full_content(self, db_path: Path) -> None:
        ids = self._seed(db_path)
        result = CliRunner().invoke(main, ["recall", str(ids["flood"]), "--db", str(db_path)])
        assert result.exit_code == 0
        assert "ValueError deep in the log" in result.output
        assert f"ack #{ids['flood']}" in result.output

    def test_recall_by_query_finds_the_entry(self, db_path: Path) -> None:
        self._seed(db_path)
        result = CliRunner().invoke(main, ["recall", "ValueError", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "ValueError deep in the log" in result.output

    def test_recall_unknown_id_exits_nonzero(self, db_path: Path) -> None:
        self._seed(db_path)
        result = CliRunner().invoke(main, ["recall", "999999", "--db", str(db_path)])
        assert result.exit_code == 1
        assert "No entry" in result.output

    def test_recall_sanitizes_escape_sequences_by_default(self, db_path: Path) -> None:
        with StorageEngine(db_path=db_path) as eng:
            sid = eng.create_session("pytest-agent").session_id
            evil = eng.insert_entry(
                LogEntry(session_id=sid, raw_content="\x1b[?1000hgotcha", entry_type="stdout")
            )
        result = CliRunner().invoke(main, ["recall", str(evil), "--db", str(db_path)])
        assert "\x1b[?1000h" not in result.output
        assert "gotcha" in result.output

    def test_recall_raw_flag_preserves_bytes(self, db_path: Path) -> None:
        with StorageEngine(db_path=db_path) as eng:
            sid = eng.create_session("pytest-agent").session_id
            evil = eng.insert_entry(
                LogEntry(session_id=sid, raw_content="\x1b[31mred", entry_type="stdout")
            )
        # color=True stops click.echo from stripping ANSI on a non-tty, so we
        # test the flag itself rather than click's terminal-detection.
        result = CliRunner().invoke(
            main, ["recall", str(evil), "--raw", "--db", str(db_path)], color=True
        )
        assert "\x1b[31m" in result.output


class TestStartupImports:
    def test_cli_import_skips_tui_and_tree_sitter(self) -> None:
        code = (
            "import sys, context_kernel.cli; "
            "print('textual' in sys.modules, 'tree_sitter' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        assert out == "False False"
