"""CLI helper tests: terminal-escape sanitisation of stored output on display."""
from __future__ import annotations

from context_kernel.cli import _sanitize


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
