#!/usr/bin/env python3
"""Needle-in-pruned-log retrieval benchmark: does `ack recall` earn its keep?

The pruner benchmark (run_benchmarks.py) proves L1 compresses without dropping
the error *signature*. This asks the harder L2 question: when pruning drops a
buried *detail*, can the agent get it back, and is that cheaper than never
pruning at all? Three arms run over the same logs, using the real ShellPruner
and a real SQLite StorageEngine (no LLM, so it's reproducible and free):

    A  keep-everything   full raw log stays resident every turn
    B  prune-and-lose    only the pruner summary survives; detail is gone
    C  prune-and-recall  summary resident + `ack recall` pages the log back

Each corpus item hides a decision-relevant "needle" the pruner legitimately
drops. We measure which arm can still see it and the token cost over a
multi-turn session, where resident tokens are re-sent every turn.

    python scripts/run_recall_benchmark.py [--turns N] [--json FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from context_kernel.memory.storage import LogEntry, StorageEngine
from context_kernel.pruners.shell_pruner import ShellPruner

ANSI_RESET  = "\033[0m"
ANSI_GREEN  = "\033[32m"
ANSI_RED    = "\033[31m"
ANSI_YELLOW = "\033[33m"
ANSI_CYAN   = "\033[36m"
ANSI_BOLD   = "\033[1m"
ANSI_DIM    = "\033[2m"

DEFAULT_TURNS = 8


def _tokens(text: str) -> int:
    """Rough token estimate (≈ chars / 4), matching run_benchmarks.py."""
    return max(1, len(text) // 4)


def gen_flood_with_buried_alert() -> tuple[str, str]:
    """A log flood whose one buried CRITICAL line is the needle.

    The dominant line repeats past the 60% dedup threshold; several decoy lines
    repeat twice; the needle appears exactly once. The pruner's frequency table
    keeps the top 8, so the freq-1 needle falls off the bottom — lost by arm B.
    """
    needle = "CRITICAL:app.billing:cust_id=4821 double-charged tok_live_9c8f2a"
    lines = ["WARNING:app.db:Connection timed out, retrying in 5s..."] * 300
    # 12 distinct decoys, each thrice — enough to fill the top-8 table above a
    # freq-1 needle, so the needle is the line the frequency summary drops.
    for d in range(12):
        lines += [f"DEBUG:app.cache:cache miss on shard {d}"] * 3
    lines.insert(150, needle)
    return "\n".join(lines), needle


def gen_traceback_with_dropped_value() -> tuple[str, str]:
    """A deep traceback whose needle is the failing input in a source line.

    The pruner keeps `file:line in func` for the last user frames and the
    exception headline, but drops every source line. The exact token that blew
    up lives in a dropped source line — the summary says *what* failed, only
    recall says *with what value*.
    """
    needle = "token='tok_live_NEEDLE7h2k'"
    frames = "\n".join(
        f'  File "/app/src/service_{i}.py", line {40 + i}, in step_{i}\n'
        f"    charge(amount=99.99, {needle})"
        for i in range(6)
    )
    stdlib = "\n".join(
        f'  File "/usr/lib/python3.13/asyncio/base_events.py", line {200 + i}, in _run_once\n'
        f"    handle._run()"
        for i in range(12)
    )
    raw = (
        "Traceback (most recent call last):\n"
        + frames + "\n"
        + stdlib + "\n"
        + "PaymentError: gateway declined charge\n"
    )
    return raw, needle


def gen_rust_wall_with_span() -> tuple[str, str]:
    """A Rust error wall whose needle is one error's span note.

    The pruner dedupes to the `error[E####]:` header lines; the `--> path` span
    and `= note:` guidance below each error are dropped. The needle is the note
    that names the actual fix.
    """
    needle = "= note: `AccountId` is defined in crate `billing_core`, not `shared`"
    lines: list[str] = [
        "   Compiling billing_core v0.3.1 (/workspace/billing)",
    ]
    for i in range(12):
        code = f"E{3000 + i:04d}"
        lines += [
            f"error[{code}]: mismatched types in `settle_batch_{i}`",
            f"  --> src/settle/batch_{i}.rs:{20 + i}:14",
        ]
        if i == 7:
            lines.append(f"   {needle}")
    lines += [
        "error: aborting due to 12 previous errors",
        "error: could not compile `billing_core`",
    ]
    return "\n".join(lines), needle


def corpus() -> list[tuple[str, str, str]]:
    """Each item: (name, raw_log, needle) — a detail the pruner drops."""
    return [
        ("flood_buried_alert", *gen_flood_with_buried_alert()),
        ("traceback_dropped_value", *gen_traceback_with_dropped_value()),
        ("rust_span_note", *gen_rust_wall_with_span()),
    ]


@dataclass
class ArmResult:
    """One strategy's outcome for a single corpus item."""

    arm:              str
    label:            str
    needle_recovered: bool
    resident_tokens:  int   # carried in context on every turn
    recall_tokens:    int   # one-time cost to page the detail back (0 if none)

    def total_tokens(self, turns: int) -> int:
        """Tokens processed across a `turns`-turn session."""
        return self.resident_tokens * turns + self.recall_tokens


@dataclass
class ItemResult:
    """Per-corpus-item results across the three arms."""

    name:           str
    needle:         str
    raw_tokens:     int
    summary_tokens: int
    pruner_fired:   bool
    arms:           list[ArmResult] = field(default_factory=list)


@dataclass
class SuiteResult:
    """Aggregate across every corpus item at a fixed session length."""

    turns:              int
    items:              list[ItemResult] = field(default_factory=list)
    recovery_a:         float = 0.0
    recovery_b:         float = 0.0
    recovery_c:         float = 0.0
    total_a_tokens:     int   = 0
    total_b_tokens:     int   = 0
    total_c_tokens:     int   = 0
    token_saving_c_vs_a: float = 0.0
    passed:             bool  = False


class RecallBenchmark:
    """Runs the A/B/C retrieval benchmark against real storage + pruner."""

    def __init__(self, pruner: ShellPruner, storage: StorageEngine) -> None:
        self.pruner  = pruner
        self.storage = storage
        self._session_id = storage.create_session("recall-benchmark").session_id

    def _benchmark_item(self, name: str, raw: str, needle: str) -> ItemResult:
        summary = self.pruner.compress(raw)
        fired   = summary is not None
        resident_bc = summary if summary is not None else raw

        # Archive the full bytes exactly as the orchestrator would, then page
        # them back through the real recall path (get_entry by handle).
        entry_id = self.storage.insert_entry(
            LogEntry(
                session_id=self._session_id,
                raw_content=raw,
                entry_type="stdout",
                was_pruned=fired,
            )
        )
        recalled_row = self.storage.get_entry(entry_id)
        recalled     = recalled_row["raw_content"] if recalled_row is not None else ""

        arm_a = ArmResult(
            arm="A", label="keep-everything",
            needle_recovered=needle in raw,
            resident_tokens=_tokens(raw),
            recall_tokens=0,
        )
        arm_b = ArmResult(
            arm="B", label="prune-and-lose",
            needle_recovered=needle in resident_bc,
            resident_tokens=_tokens(resident_bc),
            recall_tokens=0,   # nothing to page back — the detail is gone
        )
        arm_c = ArmResult(
            arm="C", label="prune-and-recall",
            needle_recovered=needle in recalled,
            resident_tokens=_tokens(resident_bc),
            recall_tokens=_tokens(recalled),   # paid once, on demand
        )

        return ItemResult(
            name=name,
            needle=needle,
            raw_tokens=_tokens(raw),
            summary_tokens=_tokens(resident_bc),
            pruner_fired=fired,
            arms=[arm_a, arm_b, arm_c],
        )

    def run(self, items: list[tuple[str, str, str]], turns: int) -> SuiteResult:
        suite = SuiteResult(turns=turns)
        for name, raw, needle in items:
            suite.items.append(self._benchmark_item(name, raw, needle))

        if not suite.items:
            return suite

        n = len(suite.items)
        suite.recovery_a = sum(it.arms[0].needle_recovered for it in suite.items) / n
        suite.recovery_b = sum(it.arms[1].needle_recovered for it in suite.items) / n
        suite.recovery_c = sum(it.arms[2].needle_recovered for it in suite.items) / n
        suite.total_a_tokens = sum(it.arms[0].total_tokens(turns) for it in suite.items)
        suite.total_b_tokens = sum(it.arms[1].total_tokens(turns) for it in suite.items)
        suite.total_c_tokens = sum(it.arms[2].total_tokens(turns) for it in suite.items)
        suite.token_saving_c_vs_a = (
            (suite.total_a_tokens - suite.total_c_tokens) / suite.total_a_tokens
            if suite.total_a_tokens
            else 0.0
        )
        return suite


def print_report(suite: SuiteResult) -> None:
    """Per-item table + the session-level token comparison."""
    _hline()
    print(f"{ANSI_BOLD}{ANSI_CYAN}Needle-in-pruned-log retrieval "
          f"(session = {suite.turns} turns){ANSI_RESET}")
    _hline()

    hdr = (
        f"{'Item':<26}  {'Needle':<10}  "
        f"{'A keep':>8}  {'B prune':>8}  {'C recall':>9}"
    )
    print(f"{ANSI_DIM}{hdr}{ANSI_RESET}")
    print("-" * len(hdr))

    for it in suite.items:
        a, b, c = it.arms
        print(
            f"{it.name:<26}  {'detail':<10}  "
            f"{_mark(a.needle_recovered):>17}  "
            f"{_mark(b.needle_recovered):>17}  "
            f"{_mark(c.needle_recovered):>18}"
        )

    print()
    print(f"  Needle recovery   A={suite.recovery_a:.0%}   "
          f"{ANSI_RED}B={suite.recovery_b:.0%}{ANSI_RESET}   "
          f"{ANSI_GREEN}C={suite.recovery_c:.0%}{ANSI_RESET}")
    print(f"  Tokens over {suite.turns:>2} turns   "
          f"A={suite.total_a_tokens:>8,}   "
          f"B={suite.total_b_tokens:>8,}   "
          f"C={suite.total_c_tokens:>8,}")
    print(
        f"  {ANSI_BOLD}C vs A: {suite.token_saving_c_vs_a:.0%} fewer tokens "
        f"at the same needle recovery{ANSI_RESET}  "
        f"(≈ {suite.total_a_tokens - suite.total_c_tokens:,} saved)"
    )


def assert_suite(suite: SuiteResult) -> bool:
    """Pass iff C recovers every needle, beats B, and costs less than A."""
    failures: list[str] = []

    def _check(label: str, ok: bool, detail: str) -> None:
        icon = f"{ANSI_GREEN}✓ PASS{ANSI_RESET}" if ok else f"{ANSI_RED}✗ FAIL{ANSI_RESET}"
        print(f"  {icon}  {label:<48} {detail}")
        if not ok:
            failures.append(label)

    print(f"\n{ANSI_BOLD}Assertions{ANSI_RESET}")
    _check(
        "C recovers every needle",
        suite.recovery_c == 1.0,
        f"{suite.recovery_c:.0%}",
    )
    _check(
        "C recovers what B loses",
        suite.recovery_c > suite.recovery_b,
        f"C={suite.recovery_c:.0%} > B={suite.recovery_b:.0%}",
    )
    _check(
        "C matches A on recovery",
        suite.recovery_c >= suite.recovery_a,
        f"C={suite.recovery_c:.0%} ≥ A={suite.recovery_a:.0%}",
    )
    _check(
        "C processes fewer tokens than A",
        suite.total_c_tokens < suite.total_a_tokens,
        f"{suite.token_saving_c_vs_a:.0%} fewer",
    )

    passed = not failures
    suite.passed = passed
    if passed:
        print(f"\n  {ANSI_GREEN}{ANSI_BOLD}All assertions passed.{ANSI_RESET}")
    else:
        print(f"\n  {ANSI_RED}{ANSI_BOLD}{len(failures)} assertion(s) failed.{ANSI_RESET}")
    return passed


def _mark(ok: bool) -> str:
    return f"{ANSI_GREEN}✓{ANSI_RESET}" if ok else f"{ANSI_RED}✗ lost{ANSI_RESET}"


def _hline(width: int = 74) -> None:
    print("─" * width)


def _banner(text: str) -> None:
    _hline()
    print(f"  {ANSI_BOLD}{ANSI_CYAN}{text}{ANSI_RESET}")
    _hline()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ACK L2 benchmark: needle recovery after pruning via ack recall."
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=DEFAULT_TURNS,
        metavar="N",
        help=f"Turns the log stays resident in context (default: {DEFAULT_TURNS}).",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        help="Write machine-readable results to this JSON file.",
    )
    args = parser.parse_args(argv)

    _banner("ACK — L2 Recall  |  Needle-in-Pruned-Log Benchmark")
    print(f"  Session length : {args.turns} turns")
    print("  Arms           : A keep-everything · B prune-and-lose · C prune-and-recall")

    with tempfile.TemporaryDirectory() as tmp:
        storage = StorageEngine(db_path=Path(tmp) / "recall_bench.db")
        with storage:
            runner = RecallBenchmark(pruner=ShellPruner(), storage=storage)
            suite  = runner.run(corpus(), turns=args.turns)

    print_report(suite)
    passed = assert_suite(suite)

    print()
    if passed:
        print(f"{ANSI_GREEN}{ANSI_BOLD}✓  RECALL BENCHMARK PASSED{ANSI_RESET}")
    else:
        print(f"{ANSI_RED}{ANSI_BOLD}✗  RECALL BENCHMARK FAILED — see FAIL lines{ANSI_RESET}")

    if args.json:
        out_path = Path(args.json)
        payload = {
            "passed":              suite.passed,
            "turns":               suite.turns,
            "recovery_a":          suite.recovery_a,
            "recovery_b":          suite.recovery_b,
            "recovery_c":          suite.recovery_c,
            "total_a_tokens":      suite.total_a_tokens,
            "total_b_tokens":      suite.total_b_tokens,
            "total_c_tokens":      suite.total_c_tokens,
            "token_saving_c_vs_a": suite.token_saving_c_vs_a,
            "items": [
                {
                    "name":           it.name,
                    "needle":         it.needle,
                    "raw_tokens":     it.raw_tokens,
                    "summary_tokens": it.summary_tokens,
                    "pruner_fired":   it.pruner_fired,
                    "arms":           [asdict(a) for a in it.arms],
                }
                for it in suite.items
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nResults written to: {out_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
