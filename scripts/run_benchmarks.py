#!/usr/bin/env python3
"""A/B benchmark for the ShellPruner: raw vs. pruned agent output.

For each corpus item it measures token reduction (≈ chars/4) and signature
fidelity — the share of error signatures (exception class, Rust code, GCC
headline) that survive compression. No LLM is involved, so it's reproducible
and free to run.

    python scripts/run_benchmarks.py [--suite NAME] [--threshold R] [--json FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from context_kernel.pruners.shell_pruner import ShellPruner

ANSI_RESET  = "\033[0m"
ANSI_GREEN  = "\033[32m"
ANSI_RED    = "\033[31m"
ANSI_YELLOW = "\033[33m"
ANSI_CYAN   = "\033[36m"
ANSI_BOLD   = "\033[1m"
ANSI_DIM    = "\033[2m"

DEFAULT_COMPRESSION_THRESHOLD = 0.60
DEFAULT_FIDELITY_THRESHOLD    = 0.95


def gen_python_traceback(
    n_user_frames: int = 3,
    n_stdlib_frames: int = 15,
    exception_class: str = "ValueError",
    exception_msg: str = "invalid literal for int() with base 10: 'abc'",
) -> str:
    """Generate a realistic Python traceback with configurable depth."""
    user_frames = "\n".join(
        f'  File "/app/src/module_{i}.py", line {10 + i * 7}, in handler_{i}\n'
        f"    result = self._process(payload)"
        for i in range(n_user_frames)
    )
    stdlib_frames = "\n".join(
        f'  File "/usr/lib/python3.13/asyncio/base_events.py", line {200 + i}, in _run_once\n'
        f"    handle._run()"
        for i in range(n_stdlib_frames)
    )
    venv_frames = "\n".join(
        f'  File "/home/user/.venv/lib/python3.13/site-packages/pkg_{i}/core.py",'
        f" line {30 + i}, in __call__\n"
        f"    return self.handler(event)"
        for i in range(5)
    )
    return (
        "Traceback (most recent call last):\n"
        + user_frames + "\n"
        + venv_frames + "\n"
        + stdlib_frames + "\n"
        + f"{exception_class}: {exception_msg}\n"
    )


def gen_chained_tracebacks(n_chains: int = 3) -> str:
    """Generate n_chains chained exceptions (During handling of…)."""
    exceptions = [
        ("sqlite3.OperationalError", "unable to open database file"),
        ("ConnectionError",          "DB pool exhausted after 30s timeout"),
        ("HTTPException",            "503 Service Unavailable — upstream DB down"),
    ]
    parts: list[str] = []
    for i in range(min(n_chains, len(exceptions))):
        cls, msg = exceptions[i]
        parts.append(gen_python_traceback(
            n_user_frames=2, n_stdlib_frames=8,
            exception_class=cls, exception_msg=msg,
        ))
        if i < n_chains - 1:
            parts.append("During handling of the above exception, another exception occurred:\n")
    return "\n".join(parts)


def gen_rust_build_failure(
    n_unique_errors: int = 10,
    repetitions_per_error: int = 5,
) -> str:
    """
    Simulate a Rust build log where each unique error appears once per crate.
    This is the worst-case repetition pattern in real Rust monorepos.
    """
    lines: list[str] = [
        "   Compiling workspace_crate v0.1.0 (/workspace)",
        "   Compiling shared_types v0.2.1 (/workspace/shared)",
    ]
    for i in range(n_unique_errors):
        code = f"E{3000 + i:04d}"
        for rep in range(repetitions_per_error):
            lines += [
                f"error[{code}]: mismatched types in `process_batch_{i}` (crate replica {rep})",
                f"  --> src/processors/batch_{i}.rs:{20 + rep}:12",
                "   |",
                f"{20 + rep} |     let count: u64 = get_item_count();",
                "   |                   ^^^^  expected `u64`, found `i32`",
                "   |",
                "   = note: consider using `as u64`",
            ]
    lines += [
        f"error: aborting due to {n_unique_errors * repetitions_per_error} previous errors",
        "For more information, try `rustc --explain E3000`.",
        "error: could not compile `workspace_crate`",
    ]
    return "\n".join(lines)


def gen_gcc_build_failure(
    n_translation_units: int = 6,
    errors_per_tu: int = 7,
) -> str:
    """Simulate a C++ build with multiple translation units each emitting errors."""
    lines: list[str] = ["make[1]: Entering directory '/workspace/build'"]
    for tu in range(n_translation_units):
        for err in range(errors_per_tu):
            col = 8 + err * 4
            lines += [
                f"src/module_{tu}.cpp:{15 + err}:{col}: error: "
                f"'undefined_symbol_{err}' was not declared in this scope",
                f"   {15 + err} |     auto result = undefined_symbol_{err}(input);",
                f"       |                  {'~' * (len(str(err)) + 17)}^",
            ]
    lines += [
        "make[1]: *** [CMakeFiles/app.dir/all] Error 1",
        "make: *** [all] Error 2",
    ]
    return "\n".join(lines)


def gen_log_flood(n_total_lines: int = 400, n_unique_patterns: int = 3) -> str:
    """
    Simulate a runaway logger where one dominant message fills ~75% of output.

    The dominant-first distribution ensures the most-common line exceeds
    ShellPruner's 60% repetition threshold regardless of n_unique_patterns.
    (A round-robin distribution with n_unique >= 2 gives at most 50% per
    pattern, which would fall below the threshold.)
    """
    patterns = [
        "WARNING:app.db:Connection timed out, retrying in 5 seconds...",
        "DEBUG:app.cache:Cache miss — fetching from upstream DB",
        "INFO:app.worker:Queue depth: 14523 messages waiting",
    ][:max(1, n_unique_patterns)]

    lines: list[str] = []
    for idx in range(n_total_lines):
        if n_unique_patterns == 1 or idx % 4 != 0:
            lines.append(patterns[0])
        else:
            lines.append(patterns[idx % len(patterns)])
    return "\n".join(lines)


def gen_mixed_realistic(seed: int = 0) -> str:
    """Mix of normal output + a traceback mid-stream (the most common real pattern)."""
    preamble = (
        f"[agent] Reading file: src/services/payment.py\n"
        f"[agent] Running: pytest tests/test_payment.py -v\n"
        f"tests/test_payment.py::test_charge_user PASSED\n"
        f"tests/test_payment.py::test_refund_user FAILED\n"
        f"SHORT TEST SUMMARY INFO\n"
        f"FAILED tests/test_payment.py::test_refund_user - {seed} exception(s)\n"
    )
    tb = gen_python_traceback(
        n_user_frames=2, n_stdlib_frames=12,
        exception_class="AssertionError",
        exception_msg="assert charge_amount == refund_amount: 99.99 != 100.00",
    )
    return preamble + tb


def extract_signatures(text: str) -> set[str]:
    """
    Pull every actionable error signature from a text blob.

    An 'actionable signature' is anything an LLM agent needs to identify
    and fix the root cause:
      • Python: ExceptionType: message lines
      • Rust:   error[E####] codes
      • GCC:    filename.cpp:line:col: error: message headlines
    """
    sigs: set[str] = set()

    for m in re.finditer(r"(\w+(?:Error|Exception|Warning)):\s*(.{1,60})", text):
        sigs.add(f"{m.group(1)}:{m.group(2).strip()[:40]}")

    for m in re.finditer(r"(error\[E\d+\]):", text):
        sigs.add(m.group(1))

    for m in re.finditer(
        r"\S+\.(?:c|cpp):\d+:\d+: error: (.{1,60})", text
    ):
        sigs.add(f"gcc:{m.group(1).strip()[:40]}")

    return sigs


@dataclass
class ItemResult:
    """Per-corpus-item benchmark results."""

    name:              str
    naked_chars:       int
    proxied_chars:     int
    naked_tokens:      int
    proxied_tokens:    int
    compression_ratio: float
    naked_sigs:        int
    preserved_sigs:    int
    fidelity_ratio:    float
    pruner_fired:      bool
    duration_ms:       float


@dataclass
class SuiteResult:
    """Aggregate result for one benchmark suite."""

    suite_name:         str
    items:              list[ItemResult]    = field(default_factory=list)
    mean_compression:   float              = 0.0
    p10_compression:    float              = 0.0
    p50_compression:    float              = 0.0
    p90_compression:    float              = 0.0
    std_compression:    float              = 0.0
    mean_fidelity:      float              = 0.0
    min_fidelity:       float              = 1.0
    total_naked_tokens: int                = 0
    total_pruned_tokens:int                = 0
    overall_compression:float              = 0.0
    passed:             bool               = False


class BenchmarkRunner:
    """
    Orchestrates the A/B benchmark.

    Usage::

        runner = BenchmarkRunner(pruner=ShellPruner())
        suite  = runner.run_suite("traceback", traceback_corpus())
        runner.print_suite_report(suite)
        passed = runner.assert_suite(suite)
    """

    def __init__(
        self,
        pruner: ShellPruner,
        compression_threshold: float = DEFAULT_COMPRESSION_THRESHOLD,
        fidelity_threshold:    float = DEFAULT_FIDELITY_THRESHOLD,
    ) -> None:
        self.pruner                 = pruner
        self.compression_threshold  = compression_threshold
        self.fidelity_threshold     = fidelity_threshold

    def _benchmark_item(self, name: str, text: str) -> ItemResult:
        """Run one corpus item through the A (naked) and B (proxied) paths."""
        naked_chars  = len(text)
        naked_tokens = max(1, naked_chars // 4)
        naked_sigs   = extract_signatures(text)

        t0 = time.perf_counter()
        summary: str | None = None
        if self.pruner.matches(text):
            summary = self.pruner.compress(text)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        proxied_text   = summary if summary is not None else text
        proxied_chars  = len(proxied_text)
        proxied_tokens = max(1, proxied_chars // 4)
        proxied_sigs   = extract_signatures(proxied_text)

        compression = (naked_tokens - proxied_tokens) / naked_tokens
        preserved   = len(naked_sigs & proxied_sigs)
        fidelity    = preserved / len(naked_sigs) if naked_sigs else 1.0

        return ItemResult(
            name=name,
            naked_chars=naked_chars,
            proxied_chars=proxied_chars,
            naked_tokens=naked_tokens,
            proxied_tokens=proxied_tokens,
            compression_ratio=compression,
            naked_sigs=len(naked_sigs),
            preserved_sigs=preserved,
            fidelity_ratio=fidelity,
            pruner_fired=(summary is not None),
            duration_ms=elapsed_ms,
        )

    def run_suite(self, name: str, corpus: list[tuple[str, str]]) -> SuiteResult:
        """Run every (item_name, text) pair in corpus and aggregate."""
        suite = SuiteResult(suite_name=name)

        for item_name, text in corpus:
            result = self._benchmark_item(item_name, text)
            suite.items.append(result)

        if not suite.items:
            return suite

        ratios    = [r.compression_ratio for r in suite.items]
        fidelities = [r.fidelity_ratio   for r in suite.items]

        suite.mean_compression    = statistics.mean(ratios)
        suite.std_compression     = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
        suite.p10_compression     = _percentile(ratios, 10)
        suite.p50_compression     = _percentile(ratios, 50)
        suite.p90_compression     = _percentile(ratios, 90)
        suite.mean_fidelity       = statistics.mean(fidelities)
        suite.min_fidelity        = min(fidelities)
        suite.total_naked_tokens  = sum(r.naked_tokens  for r in suite.items)
        suite.total_pruned_tokens = sum(r.proxied_tokens for r in suite.items)
        suite.overall_compression = (
            (suite.total_naked_tokens - suite.total_pruned_tokens)
            / suite.total_naked_tokens
        )

        return suite

    def assert_suite(
        self,
        suite: SuiteResult,
        compression_threshold: float | None = None,
        fidelity_threshold:    float | None = None,
    ) -> bool:
        """
        Assert that the suite meets the pass thresholds.

        Returns True if all assertions pass; False otherwise.
        Prints coloured PASS/FAIL lines for each assertion.
        """
        ct = compression_threshold or self.compression_threshold
        ft = fidelity_threshold    or self.fidelity_threshold

        failures: list[str] = []

        def _check(label: str, actual: float, threshold: float, direction: str = ">=") -> None:
            ok = actual >= threshold if direction == ">=" else actual <= threshold

            icon   = f"{ANSI_GREEN}✓ PASS{ANSI_RESET}" if ok else f"{ANSI_RED}✗ FAIL{ANSI_RESET}"
            detail = f"{actual:.1%}  (threshold {direction} {threshold:.0%})"
            print(f"  {icon}  {label:<55} {detail}")

            if not ok:
                failures.append(f"{label}: {actual:.1%} {direction} {threshold:.0%} FAILED")

        print(f"\n{ANSI_BOLD}Assertions — {suite.suite_name}{ANSI_RESET}")

        _check(
            "Mean token compression",
            suite.mean_compression,
            ct,
        )
        _check(
            "P10 (worst-case) compression",
            suite.p10_compression,
            max(0.30, ct - 0.30),
        )
        _check(
            "Overall compression (aggregate tokens)",
            suite.overall_compression,
            ct,
        )
        _check(
            "Mean signature fidelity",
            suite.mean_fidelity,
            ft,
        )
        _check(
            "Min signature fidelity (worst item)",
            suite.min_fidelity,
            max(0.80, ft - 0.15),
        )

        passed = len(failures) == 0
        suite.passed = passed

        if passed:
            print(f"\n  {ANSI_GREEN}{ANSI_BOLD}All assertions passed.{ANSI_RESET}")
        else:
            print(f"\n  {ANSI_RED}{ANSI_BOLD}{len(failures)} assertion(s) failed.{ANSI_RESET}")

        return passed


    def print_suite_report(self, suite: SuiteResult) -> None:
        """Print a formatted per-item table and aggregate statistics."""
        _hline()
        print(f"{ANSI_BOLD}{ANSI_CYAN}Suite: {suite.suite_name}{ANSI_RESET}")
        _hline()

        hdr = (
            f"{'Item':<32}  {'Raw tok':>8}  {'Prnd tok':>8}  "
            f"{'Compress':>8}  {'Sigs':>5}  {'Fidelity':>8}  "
            f"{'Fired':>6}  {'ms':>6}"
        )
        print(f"{ANSI_DIM}{hdr}{ANSI_RESET}")
        print("-" * len(hdr))

        for r in suite.items:
            fired  = (
                f"{ANSI_GREEN}yes{ANSI_RESET}"
                if r.pruner_fired
                else f"{ANSI_YELLOW}no{ANSI_RESET}"
            )
            fidstr = (
                f"{ANSI_GREEN}{r.fidelity_ratio:.0%}{ANSI_RESET}"
                if r.fidelity_ratio >= DEFAULT_FIDELITY_THRESHOLD
                else f"{ANSI_RED}{r.fidelity_ratio:.0%}{ANSI_RESET}"
            )
            cmpstr = (
                f"{ANSI_GREEN}{r.compression_ratio:.0%}{ANSI_RESET}"
                if r.compression_ratio >= DEFAULT_COMPRESSION_THRESHOLD
                else f"{ANSI_YELLOW}{r.compression_ratio:.0%}{ANSI_RESET}"
            )
            print(
                f"{r.name:<32}  {r.naked_tokens:>8,}  {r.proxied_tokens:>8,}  "
                f"{cmpstr:>17}  {r.naked_sigs:>5}  {fidstr:>17}  "
                f"{fired:>15}  {r.duration_ms:>6.1f}"
            )

        print()
        print(f"  Total naked tokens   : {suite.total_naked_tokens:>10,}")
        print(f"  Total proxied tokens : {suite.total_pruned_tokens:>10,}")
        print(
            f"  Overall compression  : {ANSI_BOLD}"
            f"{suite.overall_compression:.1%}{ANSI_RESET}  "
            f"(saved ~{suite.total_naked_tokens - suite.total_pruned_tokens:,} tokens)"
        )
        print(
            f"  Mean compression     : {suite.mean_compression:.1%}  "
            f"± {suite.std_compression:.1%}  "
            f"[p10={suite.p10_compression:.0%}  p50={suite.p50_compression:.0%}  "
            f"p90={suite.p90_compression:.0%}]"
        )
        print(
            f"  Mean sig fidelity    : {suite.mean_fidelity:.1%}  "
            f"(worst={suite.min_fidelity:.0%})"
        )
