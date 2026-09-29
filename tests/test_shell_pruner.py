"""ShellPruner tests: detection, compression ratio, and signature fidelity."""
from __future__ import annotations

import re
import textwrap

import pytest

from context_kernel.pruners.shell_pruner import ShellPruner


def compression_ratio(original: str, summary: str) -> float:
    """
    Returns the fraction of tokens *removed* by compression.

    1.0 = perfect compression (zero output).
    0.0 = no compression (identical length).
    Negative values mean the summary is *longer* than the original (bad).

    Token estimate: 1 token ≈ 4 characters (GPT-4 rule of thumb).
    """
    original_tokens = max(1, len(original) // 4)
    summary_tokens  = max(0, len(summary)  // 4)
    return (original_tokens - summary_tokens) / original_tokens


def extract_python_exceptions(text: str) -> set[str]:
    """Pull every ExceptionType: message line from a blob."""
    return set(re.findall(r"(\w+Error|\w+Exception|\w+Warning):\s*.+", text))


def extract_rust_error_codes(text: str) -> set[str]:
    """Pull every error[E####] code from a Rust compiler blob."""
    return set(re.findall(r"error\[E\d+\]", text))


def extract_gcc_error_lines(text: str) -> list[str]:
    """Pull every GCC/Clang error: headline."""
    return re.findall(r"\S+\.(?:c|cpp):\d+:\d+: error: .+", text)


def _make_python_traceback_single() -> str:
    """
    One deep traceback: ~20 frames, heavy on stdlib, one user frame.

    Built line-by-line (not via textwrap.dedent + f-string embedding) to
    guarantee every File "..." line has exactly 2 spaces of indentation so
    the pruner's _PY_FRAME_LINE regex and exception-line heuristic both
    work correctly.
    """
    lines = ["Traceback (most recent call last):"]
    lines.append('  File "/app/src/api/router.py", line 88, in handle_request')
    lines.append("    return await self.dispatch(request)")
    lines.append(
        '  File "/home/user/.venv/lib/python3.13/site-packages/starlette/middleware.py",'
        " line 42, in __call__"
    )
    lines.append("    raise exc")
    for i in range(15):
        lines.append(f'  File "/usr/lib/python3.13/lib{i}.py", line {10 + i}, in _call')
        lines.append("    result = self._dispatch(event)")
    lines.append("ValueError: invalid literal for int() with base 10: 'abc'")
    return "\n".join(lines) + "\n"


def _make_python_traceback_chained() -> str:
    """Two chained exceptions — During handling of the above exception…"""
    return textwrap.dedent("""\
        Traceback (most recent call last):
          File "/app/src/db/pool.py", line 34, in _connect
            self._conn = sqlite3.connect(self.dsn)
          File "/usr/lib/python3.13/sqlite3/__init__.py", line 100, in connect
            return Connection(*args, **kwargs)
        sqlite3.OperationalError: unable to open database file

        During handling of the above exception, another exception occurred:

        Traceback (most recent call last):
          File "/app/src/services/user.py", line 12, in get_user
            conn = pool.acquire()
          File "/app/src/db/pool.py", line 41, in acquire
            raise ConnectionError(f"DB unavailable: {e}") from e
          File "/home/user/.venv/lib/python3.13/site-packages/sqlalchemy/pool/base.py", line 310, in checkout
            raise exc.TimeoutError("pool timeout")
        ConnectionError: DB unavailable: unable to open database file
    """)


def _make_python_traceback_massive() -> str:
    """
    50-frame traceback that simulates a deep async call chain (Django/FastAPI).
    This is the worst-case scenario for context-window pollution.
    """
    user_frames = "\n".join(
        [
            '  File "/app/src/views/checkout.py", line 201, in post',
            "    order = cart.checkout(user_id=user.pk, payment=payload)",
            '  File "/app/src/models/cart.py", line 88, in checkout',
            "    charge_result = payment_gateway.charge(amount, card_token)",
        ]
    )
    framework_frames = "\n".join(
        f'  File "/home/user/.venv/lib/python3.13/site-packages/django/core/handlers/base.py",'
        f" line {200 + i}, in _get_response\n"
        f"    response = wrapped_callback(request, *callback_args, **callback_kwargs)"
        for i in range(46)
    )
    return (
        "Traceback (most recent call last):\n"
        + user_frames
        + "\n"
        + framework_frames
        + "\nKeyError: 'card_token'\n"
    )


def _make_rust_errors(n_unique: int = 8, repeats_per: int = 4) -> str:
    """
    Simulate a Rust build with n_unique distinct errors, each appearing
    repeats_per times (once per crate that re-exports the broken type).
    """
    error_templates = [
        (f"error[E{3000 + i:04d}]", f"mismatched types in function `process_{i}`")
        for i in range(n_unique)
    ]
    lines: list[str] = [f"   Compiling myapp v0.1.0 (/workspace/myapp)"]
    for code, msg in error_templates:
        for rep in range(repeats_per):
            lines += [
                f"{code}: {msg}",
                f"  --> src/lib.rs:{10 + rep}:5",
                f"   |",
                f"   |   let x: u32 = some_string;",
                f"   |   ^^^^^^^^^ expected u32, found &str",
                f"   |",
            ]
    lines += [
        f"error: aborting due to {n_unique * repeats_per} previous errors",
        f"For more information about an error, try `rustc --explain E3000`.",
    ]
    return "\n".join(lines)


def _make_gcc_errors(n_files: int = 5, errors_per_file: int = 6) -> str:
    """Simulate a GCC build across multiple translation units."""
    lines: list[str] = ["make[1]: Entering directory '/workspace/build'"]
    for f in range(n_files):
        for e in range(errors_per_file):
            col = 10 + e * 3
            lines += [
                f"src/module_{f}.cpp:{20 + e}:{col}: error: "
                f"'undefined_sym_{e}' was not declared in this scope",
                f"   {20 + e} |     auto x = undefined_sym_{e}(val);",
                f"     |             {'~' * 12}",
            ]
    lines += [
        f"make[1]: *** [CMakeFiles/mylib.dir/all] Error 1",
        f"make: *** [all] Error 2",
    ]
    return "\n".join(lines)


def _make_log_flood(n_lines: int = 300, n_unique: int = 3) -> str:
    """
    A log stream where one dominant line fills ~75% of output, the rest share
    the remaining lines.  The dominant pattern must exceed ShellPruner's 60%
    repetition threshold to ensure _is_highly_repetitive fires.

    With a round-robin distribution (equal share per pattern) a corpus of
    n_unique >= 2 patterns would give at most 50% per pattern — below the 60%
    threshold.  We instead assign pattern[0] to every slot NOT divisible by 4,
    giving it ~75% presence regardless of n_unique.
    """
    patterns = [
        "WARNING:root:Connection timed out, retrying in 5s...",
        "DEBUG:root:Sending heartbeat to broker",
        "INFO:root:Queue depth: 14523 messages pending",
    ][:max(1, n_unique)]

    all_lines: list[str] = []
    for i in range(n_lines):
        if n_unique == 1 or i % 4 != 0:
            all_lines.append(patterns[0])
        else:
            all_lines.append(patterns[i % len(patterns)])
    return "\n".join(all_lines)


@pytest.fixture(scope="module")
def pruner() -> ShellPruner:
    return ShellPruner()


class TestMatches:
    """Verify ShellPruner.matches() fires on the right inputs."""

    def test_matches_python_traceback(self, pruner: ShellPruner) -> None:
        assert pruner.matches(_make_python_traceback_single())

    def test_matches_chained_exception(self, pruner: ShellPruner) -> None:
        assert pruner.matches(_make_python_traceback_chained())

    def test_matches_rust_errors(self, pruner: ShellPruner) -> None:
        assert pruner.matches(_make_rust_errors())

    def test_matches_gcc_errors(self, pruner: ShellPruner) -> None:
        assert pruner.matches(_make_gcc_errors())

    def test_matches_log_flood(self, pruner: ShellPruner) -> None:
        assert pruner.matches(_make_log_flood())

    def test_does_not_match_short_normal_output(self, pruner: ShellPruner) -> None:
        text = "Building project...\nCompiling foo.py\nDone in 0.3s\n"
        assert not pruner.matches(text)

    def test_does_not_match_sparse_unique_lines(self, pruner: ShellPruner) -> None:
        text = "\n".join(f"Step {i}: completed in {i * 0.1:.1f}s" for i in range(20))
        assert not pruner.matches(text)

    def test_matches_ansi_coloured_traceback(self, pruner: ShellPruner) -> None:
        coloured = "\033[31mTraceback (most recent call last):\033[0m\n"
        coloured += '  File "/app/main.py", line 5, in run\n'
        coloured += "    raise RuntimeError('boom')\n"
        coloured += "RuntimeError: boom\n"
        assert pruner.matches(coloured)

    def test_marker_inside_a_string_is_not_a_traceback(self, pruner: ShellPruner) -> None:
        code = [f"value_{i} = compute({i})" for i in range(40)]
        code.insert(20, '    if "Traceback (most recent call last):" in text:')
        code.insert(21, "        return True")
        summary = pruner.compress("\n".join(code))
        assert summary is None or "↳" not in summary

    def test_indented_traceback_in_a_log_still_matches(self, pruner: ShellPruner) -> None:
        text = (
            "2026-09-29 12:00:01 ERROR worker crashed\n"
            "    Traceback (most recent call last):\n"
            '      File "/app/worker.py", line 12, in run\n'
            "        job()\n"
            "    ValueError: bad job\n"
        )
        assert pruner.matches(text)


class TestPythonTracebackCompression:
    """Verify content quality and compression ratio for Python tracebacks."""

    def test_single_traceback_preserves_exception_line(self, pruner: ShellPruner) -> None:
        summary = pruner.compress(_make_python_traceback_single())
        assert summary is not None
        assert "ValueError" in summary
        assert "invalid literal" in summary

    def test_single_traceback_preserves_user_frame(self, pruner: ShellPruner) -> None:
        """The deepest user-code frame must appear in the summary."""
        summary = pruner.compress(_make_python_traceback_single())
        assert summary is not None
        assert "router.py" in summary

    def test_single_traceback_drops_stdlib_frames(self, pruner: ShellPruner) -> None:
        """stdlib frames (site-packages, /lib/python) must be filtered out."""
        summary = pruner.compress(_make_python_traceback_single())
        assert summary is not None
        assert "site-packages" not in summary
        assert "/usr/lib/python" not in summary

    def test_chained_exception_preserves_both_errors(self, pruner: ShellPruner) -> None:
        summary = pruner.compress(_make_python_traceback_chained())
        assert summary is not None
        assert "sqlite3.OperationalError" in summary or "OperationalError" in summary
        assert "ConnectionError" in summary

    def test_massive_traceback_summary_under_20_lines(self, pruner: ShellPruner) -> None:
        """A 50-frame traceback must compress to ≤ 20 summary lines."""
        text    = _make_python_traceback_massive()
        summary = pruner.compress(text)
        assert summary is not None
        assert len(summary.splitlines()) <= 20

    def test_single_traceback_compression_ratio_above_60pct(self, pruner: ShellPruner) -> None:
        text    = _make_python_traceback_single()
        summary = pruner.compress(text)
        assert summary is not None
        ratio   = compression_ratio(text, summary)
        assert ratio >= 0.60, (
            f"Expected ≥60% compression on single traceback, got {ratio:.1%}.\n"
            f"Original: {len(text)} chars → Summary: {len(summary)} chars."
        )

    def test_massive_traceback_compression_ratio_above_80pct(self, pruner: ShellPruner) -> None:
        text    = _make_python_traceback_massive()
        summary = pruner.compress(text)
        assert summary is not None
        ratio   = compression_ratio(text, summary)
        assert ratio >= 0.80, (
            f"Expected ≥80% compression on 50-frame traceback, got {ratio:.1%}.\n"
            f"Original: {len(text)} chars → Summary: {len(summary)} chars."
        )

    def test_signature_fidelity_100pct(self, pruner: ShellPruner) -> None:
        """Every exception type in the original must survive in the summary."""
        text         = _make_python_traceback_chained()
        summary      = pruner.compress(text)
        assert summary is not None
        raw_sigs     = extract_python_exceptions(text)
        pruned_sigs  = extract_python_exceptions(summary)
        missing      = raw_sigs - pruned_sigs
        assert not missing, (
            f"These exception signatures were lost in compression: {missing}\n"
            f"Raw sigs: {raw_sigs}\nPruned sigs: {pruned_sigs}"
        )


class TestRustErrorCompression:
    """Verify Rust compiler output is deduplicated and summarised correctly."""

    def test_headline_count_accurate(self, pruner: ShellPruner) -> None:
        text    = _make_rust_errors(n_unique=5, repeats_per=3)
        summary = pruner.compress(text)
        assert summary is not None
        assert "15" in summary or "error(s)" in summary

    def test_all_unique_error_codes_present(self, pruner: ShellPruner) -> None:
        n_unique     = 6
        text         = _make_rust_errors(n_unique=n_unique, repeats_per=4)
        summary      = pruner.compress(text)
        assert summary is not None
        raw_codes    = extract_rust_error_codes(text)
        pruned_codes = extract_rust_error_codes(summary)
        assert raw_codes == pruned_codes, (
            f"Missing codes in summary: {raw_codes - pruned_codes}"
        )

    def test_rust_compression_ratio_above_60pct(self, pruner: ShellPruner) -> None:
        text    = _make_rust_errors(n_unique=8, repeats_per=4)
        summary = pruner.compress(text)
        assert summary is not None
        ratio   = compression_ratio(text, summary)
        assert ratio >= 0.60, f"Rust errors: got {ratio:.1%} compression."

    def test_warning_count_mentioned(self, pruner: ShellPruner) -> None:
        base    = _make_rust_errors()
        text    = base + "\nwarning[W001]: unused variable `x`\n" * 5
        summary = pruner.compress(text)
        assert summary is not None
        assert "warning" in summary.lower()


class TestGCCErrorCompression:
    """Verify C/C++ diagnostic deduplication."""

    def test_error_count_in_summary(self, pruner: ShellPruner) -> None:
        n_files, per_file = 4, 5
        text    = _make_gcc_errors(n_files=n_files, errors_per_file=per_file)
        summary = pruner.compress(text)
        assert summary is not None
        assert "20" in summary or "error(s)" in summary

    def test_first_error_line_preserved(self, pruner: ShellPruner) -> None:
        text    = _make_gcc_errors(n_files=2, errors_per_file=3)
        summary = pruner.compress(text)
        assert summary is not None
        assert "undefined_sym_0" in summary

    def test_gcc_compression_ratio_above_60pct(self, pruner: ShellPruner) -> None:
        text    = _make_gcc_errors(n_files=5, errors_per_file=8)
        summary = pruner.compress(text)
        assert summary is not None
        ratio   = compression_ratio(text, summary)
        assert ratio >= 0.60, f"GCC errors: got {ratio:.1%} compression."


class TestRepetitiveOutputCompression:
    """Verify frequency-table pruning for repetitive log floods."""

    def test_frequency_table_shows_correct_count(self, pruner: ShellPruner) -> None:
        text    = _make_log_flood(n_lines=200, n_unique=1)
        summary = pruner.compress(text)
        assert summary is not None
        assert "200" in summary

    def test_unique_lines_listed_in_summary(self, pruner: ShellPruner) -> None:
        text    = _make_log_flood(n_lines=150, n_unique=3)
        summary = pruner.compress(text)
        assert summary is not None
        assert "Connection timed out" in summary

    def test_log_flood_compression_ratio_above_80pct(self, pruner: ShellPruner) -> None:
        text    = _make_log_flood(n_lines=300, n_unique=2)
        summary = pruner.compress(text)
        assert summary is not None
        ratio   = compression_ratio(text, summary)
        assert ratio >= 0.80, f"Log flood: got {ratio:.1%} compression."

    def test_below_repetition_threshold_not_compressed(self, pruner: ShellPruner) -> None:
        """9 lines (< minimum 10) should NOT be flagged as repetitive."""
        text = "\n".join(["same line"] * 9)
        assert not pruner._is_highly_repetitive(text)  # type: ignore[attr-defined]

    def test_diverse_output_not_flagged_repetitive(self, pruner: ShellPruner) -> None:
        """50 lines all unique → not repetitive."""
        text = "\n".join(f"Unique log message number {i}" for i in range(50))
        assert not pruner._is_highly_repetitive(text)  # type: ignore[attr-defined]


class TestEdgeCases:
    """Guard against crashes on degenerate inputs."""

    def test_empty_string_returns_none(self, pruner: ShellPruner) -> None:
        assert pruner.compress("") is None

    def test_single_line_no_match_returns_none(self, pruner: ShellPruner) -> None:
        assert pruner.compress("Build succeeded.\n") is None

    def test_traceback_with_no_user_frames_falls_back_gracefully(
        self, pruner: ShellPruner
    ) -> None:
        text = textwrap.dedent("""\
            Traceback (most recent call last):
              File "/home/user/.venv/lib/python3.13/site-packages/pkg/core.py", line 1, in run
                raise ValueError("oops")
            ValueError: oops
        """)
        summary = pruner.compress(text)
        assert summary is not None
        assert "ValueError" in summary

    def test_non_utf8_bytes_handled_via_ansi_stripping(self, pruner: ShellPruner) -> None:
        latin1_text = (
            "\x1b[31mTraceback (most recent call last):\x1b[0m\n"
            '  File "/app/main.py", line 5, in main\n'
            "RuntimeError: \xe9l\xe8ve (latin-1 encoded é)\n"
        )
        result = pruner.compress(latin1_text)
        assert result is None or isinstance(result, str)


_TB = (
    "Traceback (most recent call last):\n"
    '  File "/app/src/views/checkout.py", line 201, in post\n'
    "    order = cart.checkout()\n"
    + "".join(
        f'  File "/venv/lib/python3.11/site-packages/django/base.py", line {200 + i}, in f\n'
        "    response = cb()\n"
        for i in range(20)
    )
    + "KeyError: 'card_token'\n"
)


class TestLinesAfterTraceback:
    """With `ack exec` a command's whole output can arrive as one chunk, so text
    after the exception (test totals, exit markers) must survive the summary."""

    def test_short_trailing_lines_kept(self) -> None:
        text = _TB + "2 failed, 3 passed in 0.05s\nEXIT_CODE=1\n"
        out = ShellPruner().compress(text)
        assert out is not None
        assert "KeyError: 'card_token'" in out
        assert "2 failed, 3 passed in 0.05s" in out
        assert "EXIT_CODE=1" in out
        assert out.index("KeyError") < out.index("2 failed")

    def test_trailing_flood_condensed(self) -> None:
        text = _TB + "WARNING retrying\n" * 50
        out = ShellPruner().compress(text)
        assert out is not None
        assert "Repetitive output" in out
        assert out.count("WARNING retrying") == 1

    def test_nothing_after_traceback_adds_nothing(self) -> None:
        out = ShellPruner().compress(_TB)
        assert out is not None
        assert out.rstrip().endswith("KeyError: 'card_token'")

    def test_bare_exception_kept(self) -> None:
        """Bare exceptions without a message (like KeyboardInterrupt) must be captured."""
        text = (
            "Traceback (most recent call last):\n"
            '  File "/app/main.py", line 5, in main\n'
            "    wait_forever()\n"
            "KeyboardInterrupt\n"
        )
        out = ShellPruner().compress(text)
        assert out is not None
        assert "KeyboardInterrupt" in out

    def test_chained_bare_exception_with_trailing_line(self) -> None:
        """Chained traceback with bare exception must not leak raw traceback into postamble."""
        text = (
            "Traceback (most recent call last):\n"
            '  File "/app/db.py", line 10, in connect\n'
            "    self._conn = sqlite3.connect(self.dsn)\n"
            "ValueError: invalid database\n"
            "\n"
            "During handling of the above exception, another exception occurred:\n"
            "\n"
            "Traceback (most recent call last):\n"
            '  File "/app/service.py", line 20, in fetch\n'
            "    conn = db.connect()\n"
            "StopIteration\n"
            "CLEANUP_DONE\n"
        )
        out = ShellPruner().compress(text)
        assert out is not None
        assert "↳ StopIteration" in out
        assert "CLEANUP_DONE" in out
        assert "Traceback (most recent call last):" not in out
        assert 'File "' not in out
