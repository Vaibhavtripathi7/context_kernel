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
