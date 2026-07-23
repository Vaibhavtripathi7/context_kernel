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

    def test_banner_shows_recall_handle_when_id_present(self, storage: StorageEngine) -> None:
        orch = _make_orchestrator(storage, annotate=True)
        result = orch._format_injection("short summary", 80, entry_id=42)  # type: ignore[attr-defined]
        assert "recall: ack #42" in result

    def test_banner_falls_back_when_id_missing(self, storage: StorageEngine) -> None:
        orch = _make_orchestrator(storage, annotate=True)
        result = orch._format_injection("short summary", 80, entry_id=None)  # type: ignore[attr-defined]
        assert "ack #" not in result
        assert "full log stored in DB" in result

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


class TestIncompleteTracebackBuffering:
    """
    A traceback can span several PTY reads. ACK must hold an unfinished one
    (still streaming frames, no exception line yet) so it prunes as a single
    unit instead of a broken half.
    """

    @staticmethod
    def _frames(n: int) -> bytes:
        return b"".join(
            b'  File "/app/module_%d.py", line %d, in fn\n    do_something()\n' % (i, i)
            for i in range(n)
        )

    def test_incomplete_traceback_is_held(
        self, storage: StorageEngine, pipe_pair: tuple[int, int]
    ) -> None:
        r_fd, w_fd = pipe_pair
        orch = _make_orchestrator(storage, threshold=3)
        orch._buffer = [b"Traceback (most recent call last):\n" + self._frames(10)]

        orch._flush_buffer(w_fd, force=False)  # type: ignore[attr-defined]

        data = _read_pipe(r_fd, timeout=0.1)
        assert data == b"", "An unfinished traceback must not be emitted yet."
        assert orch._buffer, "The partial traceback must remain buffered."

    def test_complete_traceback_is_flushed(
        self, storage: StorageEngine, pipe_pair: tuple[int, int]
    ) -> None:
        r_fd, w_fd = pipe_pair
        from context_kernel.pruners.shell_pruner import ShellPruner

        orch = _make_orchestrator(storage, pruners=[ShellPruner()], threshold=3)
        full = (
            b"Traceback (most recent call last):\n"
            + self._frames(10)
            + b"ValueError: boom\n"
        )
        orch._buffer = [full]

        orch._flush_buffer(w_fd, force=False)  # type: ignore[attr-defined]

        data = _read_pipe(r_fd)
        assert b"ValueError" in data, "A finished traceback must be emitted."
        assert not orch._buffer

    def test_incomplete_traceback_flushed_when_forced(
        self, storage: StorageEngine, pipe_pair: tuple[int, int]
    ) -> None:
        """The silence timeout / EOF (force=True) must flush even a partial."""
        r_fd, w_fd = pipe_pair
        orch = _make_orchestrator(storage, threshold=3)
        orch._buffer = [b"Traceback (most recent call last):\n" + self._frames(10)]

        orch._flush_buffer(w_fd, force=True)  # type: ignore[attr-defined]

        data = _read_pipe(r_fd)
        assert b"Traceback" in data
        assert not orch._buffer


class TestStatsTracking:
    """Verify the OrchestratorStats counters accumulate correctly."""

    def test_stats_initial_state(self, orch: Orchestrator) -> None:
        s = orch.stats
        assert s.total_bytes_read     == 0
        assert s.total_bytes_injected == 0
        assert s.total_pruner_hits    == 0
        assert s.tokens_saved         == 0

    def test_bytes_read_accumulates(
        self, storage: StorageEngine, pipe_pair: tuple[int, int]
    ) -> None:
        r_fd, w_fd = pipe_pair
        orch = _make_orchestrator(storage, threshold=1)
        chunk = b"x" * 256
        orch._accumulate(chunk)  # type: ignore[attr-defined]
        orch._stats.total_bytes_read += len(chunk)
        assert orch.stats.total_bytes_read == 256

    def test_stats_callback_called_after_prune(
        self, storage: StorageEngine, pipe_pair: tuple[int, int]
    ) -> None:
        r_fd, w_fd = pipe_pair
        received: list[OrchestratorStats] = []

        class AnyPruner(BasePruner):
            metadata = PrunerMetadata("any", "always prunes")
            def matches(self, text: str) -> bool:
                return True
            def compress(self, text: str) -> Optional[str]:
                return "summary"

        session = storage.create_session("callback-test")
        orch = Orchestrator(
            command=["echo"],
            session_id=session.session_id,
            storage=storage,
            pruners=[AnyPruner()],
            config=OrchestratorConfig(pruning_threshold_lines=1),
            stats_callback=received.append,
        )
        orch._buffer = [b"line1\nline2\n"]

        orch._flush_buffer(w_fd, force=True)  # type: ignore[attr-defined]

        assert len(received) >= 1, "stats_callback must be called after a prune event."
        assert received[-1].total_pruner_hits >= 1


class TestOrchestratorIntegration:
    """
    End-to-end integration tests that launch a real child process.

    Each test writes a self-contained driver script and runs it via
    subprocess.run so that os.fork() + pty.openpty() happen in
    a clean process with no pytest state.  We assert on the DB stats that
    the driver script prints to stdout.
    """

    def _run_driver(self, script_content: str, tmp_path: Path, timeout: int = 20) -> str:
        """Write a driver script and run it, returning its stdout."""
        script_path = tmp_path / "_driver.py"
        script_path.write_text(textwrap.dedent(script_content))
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, (
            f"Driver script failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        return result.stdout

    @pytest.mark.integration
    def test_orchestrator_captures_echo_output(self, tmp_path: Path) -> None:
        """
        Spawn python3 -c "print(...)" via ACK and verify the output is
        persisted to the database.
        """
        db_path = tmp_path / "integration_echo.db"
        driver  = textwrap.dedent(f"""\
            import sys
            from pathlib import Path
            sys.path.insert(0, r"{PROJECT_ROOT}")
            from context_kernel.memory.storage import StorageEngine
            from context_kernel.core.orchestrator import Orchestrator, OrchestratorConfig

            db = StorageEngine(db_path=Path(r"{db_path}"))
            db.open()
            session = db.create_session("integration-echo")

            orch = Orchestrator(
                command=[r"{sys.executable}", "-c", "print('hello from ack integration')"],
                session_id=session.session_id,
                storage=db,
                config=OrchestratorConfig(
                    pruning_threshold_lines=200,
                    buffer_flush_timeout=0.05,
                ),
            )
            orch.run()

            stats   = db.stats(session.session_id)
            entries = db.get_recent_entries(session.session_id, limit=10)
            combined = " ".join(r["raw_content"] for r in entries)
            print(f"entries={{stats['total_entries']}}")
            print(f"contains_hello={{('hello from ack integration' in combined)}}")
            db.close()
        """)
        out = self._run_driver(driver, tmp_path)
        assert "entries=1"           in out, f"Expected 1 DB entry.\nGot: {out}"
        assert "contains_hello=True" in out, f"Expected 'hello from ack integration' in DB.\nGot: {out}"

    @pytest.mark.integration
    def test_orchestrator_pruner_fires_on_large_output(self, tmp_path: Path) -> None:
        """
        Spawn a script that emits a 60-line Python traceback.  The
        ShellPruner must fire and DB entries must show was_pruned=True.
        """
        db_path    = tmp_path / "integration_prune.db"
        agent_path = tmp_path / "big_traceback_agent.py"
        agent_path.write_text(textwrap.dedent("""\
            import sys
            lines = ["Traceback (most recent call last):"]
            for i in range(35):
                lines.append(f'  File "/app/module_{i}.py", line {i+1}, in fn')
                lines.append(f"    do_something({i})")
            lines.append("ValueError: injected integration error")
            sys.stdout.write("\\n".join(lines) + "\\n")
            sys.stdout.flush()
        """))

        driver = textwrap.dedent(f"""\
            import sys
            from pathlib import Path
            sys.path.insert(0, r"{PROJECT_ROOT}")
            from context_kernel.memory.storage import StorageEngine
            from context_kernel.core.orchestrator import Orchestrator, OrchestratorConfig
            from context_kernel.pruners.shell_pruner import ShellPruner

            db = StorageEngine(db_path=Path(r"{db_path}"))
            db.open()
            session = db.create_session("integration-prune")

            orch = Orchestrator(
                command=[r"{sys.executable}", r"{agent_path}"],
                session_id=session.session_id,
                storage=db,
                pruners=[ShellPruner()],
                config=OrchestratorConfig(
                    pruning_threshold_lines=20,
                    buffer_flush_timeout=0.05,
                    annotate_injections=False,
                ),
            )
            orch.run()

            stats = db.stats(session.session_id)
            print(f"pruner_hits={{orch.stats.total_pruner_hits}}")
            print(f"tokens_saved={{orch.stats.tokens_saved}}")
            print(f"pruned_entries={{stats['pruned_entries']}}")
            db.close()
        """)
        out = self._run_driver(driver, tmp_path)
        assert "pruner_hits=1" in out, f"Pruner did not fire.\nOutput:\n{out}"
        tokens_line = [ln for ln in out.splitlines() if ln.startswith("tokens_saved=")]
        assert tokens_line, f"tokens_saved line missing.\nOutput:\n{out}"
        saved = int(tokens_line[0].split("=")[1])
        assert saved > 0, f"Expected tokens_saved > 0, got {saved}."

    @pytest.mark.integration
    def test_orchestrator_exit_code_propagated(self, tmp_path: Path) -> None:
        """
        The orchestrator must propagate the child's exit code exactly.
        We run a script that exits with code 42 and verify the driver gets 42.
        """
        db_path = tmp_path / "integration_exit.db"
        driver  = textwrap.dedent(f"""\
            import sys
            from pathlib import Path
            sys.path.insert(0, r"{PROJECT_ROOT}")
            from context_kernel.memory.storage import StorageEngine
            from context_kernel.core.orchestrator import Orchestrator, OrchestratorConfig

            db = StorageEngine(db_path=Path(r"{db_path}"))
            db.open()
            session = db.create_session("exit-code-test")

            orch = Orchestrator(
                command=[r"{sys.executable}", "-c", "import sys; sys.exit(42)"],
                session_id=session.session_id,
                storage=db,
                config=OrchestratorConfig(buffer_flush_timeout=0.05),
            )
            code = orch.run()
            print(f"exit_code={{code}}")
            db.close()
        """)
        out = self._run_driver(driver, tmp_path)
        assert "exit_code=42" in out, (
            f"Expected exit_code=42 to be printed by the driver.\nGot: {out}"
        )

    @pytest.mark.integration
    def test_orchestrator_restores_terminal_after_run(self, tmp_path: Path) -> None:
        """
        After run() completes, the driver process must be able to print
        normally — proof that the terminal was restored from raw mode.
        """
        db_path = tmp_path / "integration_tty.db"
        driver  = textwrap.dedent(f"""\
            import sys
            from pathlib import Path
            sys.path.insert(0, r"{PROJECT_ROOT}")
            from context_kernel.memory.storage import StorageEngine
            from context_kernel.core.orchestrator import Orchestrator, OrchestratorConfig

            db = StorageEngine(db_path=Path(r"{db_path}"))
            db.open()
            session = db.create_session("tty-restore-test")

            orch = Orchestrator(
                command=[r"{sys.executable}", "-c", "print('done')"],
                session_id=session.session_id,
                storage=db,
                config=OrchestratorConfig(buffer_flush_timeout=0.05),
            )
            orch.run()
            print("terminal_ok=True")
            db.close()
        """)
        out = self._run_driver(driver, tmp_path)
        assert "terminal_ok=True" in out
