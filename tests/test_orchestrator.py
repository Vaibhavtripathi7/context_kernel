"""Orchestrator tests: prompt detection, buffer flush, stats, PTY integration."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

import pytest

from context_kernel.core.orchestrator import Orchestrator, OrchestratorConfig, OrchestratorStats
from context_kernel.memory.storage import StorageEngine
from context_kernel.pruners.base import BasePruner, PrunerMetadata


PROJECT_ROOT = Path(__file__).parent.parent


def _make_storage(tmp_path: Path) -> StorageEngine:
    eng = StorageEngine(db_path=tmp_path / "orch_test.db")
    eng.open()
    return eng


def _make_orchestrator(
    storage: StorageEngine,
    pruners: Optional[list[BasePruner]] = None,
    threshold: int = 5,
    annotate: bool = True,
) -> Orchestrator:
    session = storage.create_session("test-agent")
    return Orchestrator(
        command=["echo", "placeholder"],
        session_id=session.session_id,
        storage=storage,
        pruners=pruners or [],
        config=OrchestratorConfig(
            pruning_threshold_lines=threshold,
            buffer_flush_timeout=0.05,
            annotate_injections=annotate,
        ),
    )


@pytest.fixture
def storage(tmp_path: Path) -> StorageEngine:
    eng = _make_storage(tmp_path)
    yield eng
    eng.close()


@pytest.fixture
def orch(storage: StorageEngine) -> Orchestrator:
    return _make_orchestrator(storage)


@pytest.fixture
def pipe_pair():
    """An (r_fd, w_fd) pair for capturing os.write output in unit tests."""
    r_fd, w_fd = os.pipe()
    yield r_fd, w_fd
    for fd in (r_fd, w_fd):
        try:
            os.close(fd)
        except OSError:
            pass


def _read_pipe(r_fd: int, timeout: float = 0.5) -> bytes:
    """Drain all available data from the read end of a pipe."""
    import select
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([r_fd], [], [], 0.05)
        if not ready:
            break
        chunk = os.read(r_fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


class TestPromptDetection:
    """
    Verify _text_is_prompt and _tail_is_prompt correctly identify
    interactive prompts vs. normal output.

    Prompts must NEVER be pruned — if an agent is waiting for Y/N and ACK
    swallows the question, the entire session hangs.
    """

    def test_yn_prompt_text(self, orch: Orchestrator) -> None:
        assert orch._text_is_prompt("Overwrite existing file? [Y/n] ")  # type: ignore[attr-defined]

    def test_yn_capital_prompt(self, orch: Orchestrator) -> None:
        assert orch._text_is_prompt("Confirm deletion? [Y/N] ")  # type: ignore[attr-defined]

    def test_yes_no_long_form_prompt(self, orch: Orchestrator) -> None:
        assert orch._text_is_prompt("Continue with this action? (yes/no) ")  # type: ignore[attr-defined]

    def test_question_mark_tail_detected(self, orch: Orchestrator) -> None:
        assert orch._text_is_prompt("Are you sure?")  # type: ignore[attr-defined]

    def test_colon_tail_detected(self, orch: Orchestrator) -> None:
        assert orch._text_is_prompt("Enter API key: ")  # type: ignore[attr-defined]

    def test_normal_multiline_output_not_prompt(self, orch: Orchestrator) -> None:
        text = "Building project...\nCompiling module A\nCompiling module B\nDone.\n"
        assert not orch._text_is_prompt(text)  # type: ignore[attr-defined]

    def test_stack_trace_not_prompt(self, orch: Orchestrator) -> None:
        text = (
            "Traceback (most recent call last):\n"
            '  File "/app/main.py", line 5, in main\n'
            "ValueError: bad input\n"
        )
        assert not orch._text_is_prompt(text)  # type: ignore[attr-defined]

    def test_tail_bytes_prompt_detection(self, orch: Orchestrator) -> None:
        chunk = b"Some preamble...\nAre you sure? [Y/n] "
        assert orch._tail_is_prompt(chunk)  # type: ignore[attr-defined]

    def test_tail_bytes_non_prompt(self, orch: Orchestrator) -> None:
        chunk = b"error: undefined reference to `main'\n"
        assert not orch._tail_is_prompt(chunk)  # type: ignore[attr-defined]


class TestInjectionFormatting:
    """Verify the annotation envelope wrapping pruner summaries."""

    def test_annotation_banner_present_when_enabled(self, storage: StorageEngine) -> None:
        orch = _make_orchestrator(storage, annotate=True)
        result = orch._format_injection("short summary", original_lines=80)  # type: ignore[attr-defined]
        assert "[ACK]" in result
        assert "80" in result

    def test_annotation_banner_absent_when_disabled(self, storage: StorageEngine) -> None:
        orch = _make_orchestrator(storage, annotate=False)
        result = orch._format_injection("short summary", original_lines=80)  # type: ignore[attr-defined]
        assert "[ACK]" not in result

    def test_summary_text_always_in_output(self, storage: StorageEngine) -> None:
        orch = _make_orchestrator(storage, annotate=True)
        result = orch._format_injection("ValueError: bad arg\n  at app/main.py:5", 30)  # type: ignore[attr-defined]
        assert "ValueError: bad arg" in result

    def test_no_annotate_output_is_just_summary(self, storage: StorageEngine) -> None:
        orch   = _make_orchestrator(storage, annotate=False)
        result = orch._format_injection("my summary", 10)  # type: ignore[attr-defined]
        assert "[ACK]" not in result
        assert "my summary" in result


class TestBufferFlush:
    """
    Unit tests for _flush_buffer using an os.pipe() as a fake stdout.

    The write-end of the pipe is passed to _flush_buffer(w_fd).
    After each call we drain the read-end to inspect what was emitted.
    """

    def test_below_threshold_not_flushed_without_force(
        self, storage: StorageEngine, pipe_pair: tuple[int, int]
    ) -> None:
        r_fd, w_fd = pipe_pair
        orch = _make_orchestrator(storage, threshold=10)
        orch._buffer = [b"line1\n", b"line2\n", b"line3\n"]

        orch._flush_buffer(w_fd, force=False)  # type: ignore[attr-defined]

        data = _read_pipe(r_fd, timeout=0.1)
        assert data == b"", (
            "Buffer below threshold must NOT be flushed when force=False."
        )
        assert len(orch._buffer) > 0

    def test_below_threshold_flushed_with_force(
        self, storage: StorageEngine, pipe_pair: tuple[int, int]
    ) -> None:
        r_fd, w_fd = pipe_pair
        orch = _make_orchestrator(storage, threshold=50)
        orch._buffer = [b"hello world\n"]

        orch._flush_buffer(w_fd, force=True)  # type: ignore[attr-defined]

        data = _read_pipe(r_fd)
        assert b"hello world" in data

    def test_below_threshold_not_pruned_even_when_forced(
        self, storage: StorageEngine, pipe_pair: tuple[int, int]
    ) -> None:
        """
        A forced flush of below-threshold output must pass through verbatim.

        force controls only *whether to emit now* (e.g. at the silence
        timeout or EOF), never *whether to prune*.  Output below
        pruning_threshold_lines is always emitted unchanged, even with a
        pruner that would otherwise match.
        """
        r_fd, w_fd = pipe_pair

        class AlwaysPrune(BasePruner):
            metadata = PrunerMetadata("always", "fires on everything")
            def matches(self, text: str) -> bool:
                return True
            def compress(self, text: str) -> Optional[str]:
                return "COMPRESSED"

        orch = _make_orchestrator(storage, pruners=[AlwaysPrune()], threshold=50)
        orch._buffer = [b"line one\nline two\nline three\n"]

        orch._flush_buffer(w_fd, force=True)  # type: ignore[attr-defined]

        data = _read_pipe(r_fd)
        assert b"line one" in data, "Below-threshold output must pass through verbatim."
        assert b"COMPRESSED" not in data, "Below-threshold output must not be pruned."
        assert orch.stats.total_pruner_hits == 0

    def test_prompt_always_passes_through_verbatim(
        self, storage: StorageEngine, pipe_pair: tuple[int, int]
    ) -> None:
        r_fd, w_fd = pipe_pair
        class AlwaysPrune(BasePruner):
            metadata = PrunerMetadata("always", "fires on everything")
            def matches(self, text: str) -> bool:
                return True
            def compress(self, text: str) -> Optional[str]:
                return "COMPRESSED"

        orch = _make_orchestrator(storage, pruners=[AlwaysPrune()], threshold=1)
        prompt_bytes = b"Continue? [Y/n] "
        orch._buffer = [prompt_bytes]

        orch._flush_buffer(w_fd, force=True)  # type: ignore[attr-defined]

        data = _read_pipe(r_fd)
        assert b"Continue? [Y/n]" in data, "Prompt must pass through verbatim."
        assert b"COMPRESSED" not in data, "Pruner must not fire on interactive prompts."

    def test_above_threshold_triggers_pruner(
        self, storage: StorageEngine, pipe_pair: tuple[int, int]
    ) -> None:
        r_fd, w_fd = pipe_pair

        class FixedPruner(BasePruner):
            metadata = PrunerMetadata("fixed", "always returns 'SUMMARY'")
            def matches(self, text: str) -> bool:
                return True
            def compress(self, text: str) -> Optional[str]:
                return "SUMMARY LINE"

        orch = _make_orchestrator(storage, pruners=[FixedPruner()], threshold=3)
        five_lines = b"\n".join(b"output line %d" % i for i in range(5)) + b"\n"
        orch._buffer = [five_lines]

        orch._flush_buffer(w_fd, force=True)  # type: ignore[attr-defined]

        data = _read_pipe(r_fd)
        assert b"SUMMARY LINE" in data
        assert b"output line 0" not in data

    def test_no_pruner_match_passes_through_raw(
        self, storage: StorageEngine, pipe_pair: tuple[int, int]
    ) -> None:
        r_fd, w_fd = pipe_pair

        class NeverPrune(BasePruner):
            metadata = PrunerMetadata("never", "never fires")
            def matches(self, text: str) -> bool:
                return False
            def compress(self, text: str) -> Optional[str]:
                return None

        orch = _make_orchestrator(storage, pruners=[NeverPrune()], threshold=3)
        raw = b"\n".join(b"line %d" % i for i in range(5)) + b"\n"
        orch._buffer = [raw]

        orch._flush_buffer(w_fd, force=True)  # type: ignore[attr-defined]

        data = _read_pipe(r_fd)
        assert b"line 0" in data

    def test_stats_incremented_on_prune(
        self, storage: StorageEngine, pipe_pair: tuple[int, int]
    ) -> None:
        r_fd, w_fd = pipe_pair
        original_text = "A" * 1000
        summary_text  = "B" * 100

        class MeasuredPruner(BasePruner):
            metadata = PrunerMetadata("measured", "for stats testing")
            def matches(self, text: str) -> bool:
                return True
            def compress(self, text: str) -> Optional[str]:
                return summary_text

        orch = _make_orchestrator(storage, pruners=[MeasuredPruner()], threshold=1)
        orch._buffer = [original_text.encode()]

        orch._flush_buffer(w_fd, force=True)  # type: ignore[attr-defined]

        assert orch.stats.total_pruner_hits == 1
        assert orch.stats.tokens_saved > 0
        assert orch.stats.tokens_saved == (len(original_text) - len(summary_text)) // 4
