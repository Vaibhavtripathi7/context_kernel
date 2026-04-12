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

    def compress(self, text: str) -> str | None:
        if not self.matches(text):
            return None

        clean = _strip_ansi(text)

        if _PY_TRACEBACK_HDR.search(clean):
            return self._compress_python_tracebacks(clean)
        if _RUST_ERROR_HDR.search(clean):
            return self._compress_rust_errors(clean)
        if _GCC_DIAG.search(clean):
            return self._compress_gcc_errors(clean)
        if self._is_highly_repetitive(clean):
            return self._compress_repetitive(clean)

        return None

    def _compress_python_tracebacks(self, text: str) -> str:
        lines = text.splitlines()
        blocks: list[dict[str, list[str] | str]] = []
        current_block: dict[str, list[str] | str] | None = None

        for line in lines:
            if _PY_TRACEBACK_HDR.search(line):
                current_block = {"frames": [], "exception": ""}
                blocks.append(current_block)
                continue

            if current_block is None:
                continue

            frame_match = _PY_FRAME_LINE.match(line)
            if frame_match:
                filepath = frame_match.group("file")
                lineno   = frame_match.group("lineno")
                func     = frame_match.group("func")
                frames: list[str] = current_block["frames"]  # type: ignore[assignment]
                frames.append(f"{filepath}:{lineno} in {func}")
                continue

            if line and not line.startswith(" ") and not line.startswith("\t"):
                current_block["exception"] = line.strip()
                current_block = None

        if not blocks:
            return text[:800] + "\n… [truncated by ACK shell_pruner]"

        parts: list[str] = []
        for idx, block in enumerate(blocks, start=1):
            frames_raw: list[str] = block["frames"]  # type: ignore[assignment]
            exc: str = str(block["exception"])

            user_frames = [
                f for f in frames_raw
                if not any(indicator in f for indicator in _STDLIB_INDICATORS)
            ] or frames_raw
            relevant = user_frames[-3:]

            header = (
                f"── Traceback {idx}/{len(blocks)} ──"
                if len(blocks) > 1
                else "── Traceback ──"
            )
            frame_str = "\n".join(f"  at {f}" for f in relevant)
            parts.append(f"{header}\n{frame_str}\n  ↳ {exc}")

        first_tb = next(
            (i for i, ln in enumerate(lines) if _PY_TRACEBACK_HDR.search(ln)),
            0,
        )
        preamble = "\n".join(lines[:first_tb]).strip()

        result: list[str] = []
        if preamble:
            result.append(preamble)
        result.extend(parts)
        return "\n\n".join(result)
