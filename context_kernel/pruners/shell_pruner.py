"""Pruner for shell / compiler output.

Compresses three common high-token patterns: Python tracebacks (keep the
exception and the deepest user frames), Rust/GCC error walls (dedupe by error
code), and repetitive log floods (frequency table). The raw text is untouched;
only the live stream is summarised.
"""
from __future__ import annotations

import re
from collections import Counter

from .base import BasePruner, PrunerMetadata

_PY_TRACEBACK_HDR = re.compile(r"Traceback \(most recent call last\):", re.MULTILINE)

_PY_FRAME_LINE = re.compile(
    r'^\s+File "(?P<file>[^"]+)", line (?P<lineno>\d+), in (?P<func>.+)',
)

_RUST_ERROR_HDR = re.compile(r"^error\[E\d+\]:", re.MULTILINE)

_GCC_DIAG = re.compile(
    r"^\S+\.(?:c|cpp|cc|cxx|h|hpp):\d+:\d+: (?:error|warning|note):",
    re.MULTILINE,
)

_ANSI_ESC = re.compile(r"\x1b\[[0-9;]*[mGKHFJA-Z]")

_STDLIB_INDICATORS = frozenset(
    ["site-packages", "/lib/python", "<frozen", "/lib64/python", "distlib"]
)


def _strip_ansi(text: str) -> str:
    return _ANSI_ESC.sub("", text)


class ShellPruner(BasePruner):
    """Collapses stack traces and repetitive build/log output.

    dedup_threshold is the share of duplicate lines needed to treat output
    as a flood; max_rust_errors caps how many unique Rust errors are listed.
    """

    metadata = PrunerMetadata(
        name="shell_pruner",
        description="Collapses stack traces and repetitive compiler/build output",
        version="0.1.0",
    )

    def __init__(self, dedup_threshold: float = 0.6, max_rust_errors: int = 10) -> None:
        self.dedup_threshold = dedup_threshold
        self.max_rust_errors = max_rust_errors

    def matches(self, text: str) -> bool:
        clean = _strip_ansi(text)
        return bool(
            _PY_TRACEBACK_HDR.search(clean)
            or _RUST_ERROR_HDR.search(clean)
            or _GCC_DIAG.search(clean)
            or self._is_highly_repetitive(clean)
        )
