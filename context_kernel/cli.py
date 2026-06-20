"""Command-line entrypoint and the optional Textual stats dashboard.

In --tui mode the orchestrator runs on a background thread and Textual renders
a stats/log overlay. Because both want the terminal, the agent runs
non-interactively there; plain ack run is the interactive path.
"""
from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path

import click
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, RichLog, Static

from .core.orchestrator import Orchestrator, OrchestratorConfig, OrchestratorStats
from .memory.pager import Pager
from .memory.storage import StorageEngine
from .pruners.shell_pruner import ShellPruner

# Rough input-token price used only for the end-of-session estimate. It is an
# order-of-magnitude figure (USD per million input tokens); override mentally
# for your own model. Kept conservative so the saving is never overstated.
_USD_PER_MILLION_INPUT_TOKENS = 3.0

# Stored output keeps the agent's raw bytes, which include terminal escape
# sequences (colours, but also cursor/mouse/app-mode toggles). Echoing those
# verbatim can reprogram the user's terminal, so any stored text printed back
# is stripped of escape sequences and other control chars first.
_ANSI_OSC   = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_ANSI_CSI   = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_ANSI_OTHER = re.compile(r"\x1b[@-Z\\-_]")
_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(text: str) -> str:
    """Strip escape sequences and control chars from stored output before echo."""
    text = _ANSI_OSC.sub("", text)
    text = _ANSI_CSI.sub("", text)
    text = _ANSI_OTHER.sub("", text)
    return _CTRL_CHARS.sub("", text)


def _print_session_summary(
    storage: StorageEngine,
    session_id: str,
    stats: OrchestratorStats,
) -> None:
    """Print a one-glance summary of what ACK saved this session (to stderr)."""
    db_stats          = storage.stats(session_id)
    chunks            = db_stats["total_entries"]
    pruned            = db_stats["pruned_entries"]
    raw_pruned_tokens = db_stats["tokens_saved"]  # raw token volume of pruned chunks
    saved             = stats.tokens_saved

    if chunks == 0:
        return

    elapsed = max(1, int(time.monotonic() - stats.session_start))
    pct     = (saved / raw_pruned_tokens * 100) if raw_pruned_tokens else 0.0
    cost    = saved / 1_000_000 * _USD_PER_MILLION_INPUT_TOKENS
    rate    = f"{_USD_PER_MILLION_INPUT_TOKENS:g}"

    lines = [
        click.style("[ACK] Session summary", fg="cyan", bold=True),
        f"  Chunks intercepted : {chunks:>8,}",
        f"  Chunks pruned      : {pruned:>8,}",
        f"  Tokens saved       : {saved:>8,}  (~{pct:.0f}% of pruned output)",
        f"  Est. cost saved    : ${cost:>7.2f}  (at ${rate}/M input tokens)",
        f"  Elapsed            : {elapsed:>7}s",
    ]
    try:
        click.echo("\n" + "\n".join(lines), err=True)
    except (BrokenPipeError, OSError):
        # Downstream (e.g. `... 2>&1 | head`) closed the pipe; never let the
        # summary crash the run or mask the agent's real exit code.
        pass


class StatsPanel(Static):
    stats: reactive[OrchestratorStats] = reactive(OrchestratorStats())

    def render(self) -> str:  # type: ignore[override]
        s: OrchestratorStats = self.stats
        elapsed = max(1, int(time.monotonic() - s.session_start))
        ratio = (
            f"{s.tokens_saved / max(1, s.total_bytes_read // 4):.0%}"
            if s.total_bytes_read
            else "0%"
        )
        return (
            "[bold cyan]ACK[/bold cyan] — Agent Context Kernel\n"
            "─────────────────────────────────\n"
            f"Bytes intercepted : {s.total_bytes_read:>10,}\n"
            f"Bytes injected    : {s.total_bytes_injected:>10,}\n"
            f"Pruner hits       : {s.total_pruner_hits:>10,}\n"
            f"Tokens saved      : {s.tokens_saved:>10,}\n"
            f"Compression ratio : {ratio:>10}\n"
            f"Elapsed           : {elapsed:>9}s\n"
        )


class AckDashboard(App[int]):
    """Live stats overlay; polls Orchestrator.stats once a second."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    #log-panel {
        width: 1fr;
        height: 100%;
        border: solid $primary;
        padding: 0 1;
    }

    #stats-panel {
        width: 36;
        height: 100%;
        border: solid $accent;
        padding: 1 2;
    }

    StatsPanel {
        height: auto;
    }

    RichLog {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("q",       "quit", "Quit"),
        Binding("ctrl+c",  "quit", "Quit"),
    ]

    def __init__(self, orchestrator: Orchestrator) -> None:
        super().__init__()
        self._orchestrator  = orchestrator
        self._orch_thread:   threading.Thread | None = None
        self._stats_panel:   StatsPanel | None       = None
        self._log_widget:    RichLog | None           = None
        self._log_buffer:    list[str]                   = []
        self._log_lock       = threading.Lock()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="log-panel"):
                self._log_widget = RichLog(
                    id="agent-log",
                    wrap=True,
                    highlight=True,
                    markup=True,
                )
                yield self._log_widget
            with Vertical(id="stats-panel"):
                self._stats_panel = StatsPanel()
                yield self._stats_panel
        yield Footer()

    def on_mount(self) -> None:
        self._orchestrator.text_callback = self._enqueue_log
        self._orch_thread = threading.Thread(
            target=self._orchestrator.run,
            daemon=True,
            name="ack-orchestrator",
        )
        self._orch_thread.start()
        self.set_interval(1.0, self._refresh_stats)
        self.set_interval(1 / 30, self._flush_log)

    def _refresh_stats(self) -> None:
        if self._stats_panel is not None:
            self._stats_panel.stats = self._orchestrator.stats
            self._stats_panel.refresh()

        if self._orch_thread is not None and not self._orch_thread.is_alive():
            self.exit(0)

    def _enqueue_log(self, text: str) -> None:
        with self._log_lock:
            self._log_buffer.append(text)

    def _flush_log(self) -> None:
        with self._log_lock:
            if not self._log_buffer:
                return
            chunks, self._log_buffer = self._log_buffer, []

        if self._log_widget is not None:
            for chunk in chunks:
                self._log_widget.write(chunk)

    async def action_quit(self) -> None:
        self.exit(0)


@click.group()
@click.version_option(version="0.1.0", prog_name="ack")
def main() -> None:
    """ACK — context-pruning proxy for terminal AI agents."""


@main.command(name="run")
@click.argument("agent_command", nargs=-1, required=True)
@click.option(
    "--db",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to the ACK SQLite database.",
)
@click.option(
    "--prune-threshold",
    "prune_threshold",
    default=30,
    show_default=True,
    type=int,
    help="Output lines before pruner activates.",
)
@click.option(
    "--no-annotate",
    "no_annotate",
    is_flag=True,
    default=False,
    help="Suppress [ACK] banners in the stream.",
)
@click.option(
    "--tui/--no-tui",
    default=False,
    show_default=True,
    help="Launch the Textual stats dashboard (experimental).",
)
def cmd_run(
    agent_command:   tuple[str, ...],
    db:              Path | None,
    prune_threshold: int,
    no_annotate:     bool,
    tui:             bool,
) -> None:
    """Wrap AGENT_COMMAND with ACK's PTY interceptor.

    Example: ack run -- aider --model gpt-4o
    """
    storage = StorageEngine(db_path=db) if db else StorageEngine()
    storage.open()

    session = storage.create_session(agent_command=" ".join(agent_command))

    config = OrchestratorConfig(
        pruning_threshold_lines=prune_threshold,
        annotate_injections=not no_annotate,
    )
    orch = Orchestrator(
        command=list(agent_command),
        session_id=session.session_id,
        storage=storage,
        pruners=[ShellPruner()],
        config=config,
    )

    if tui:
        exit_code = AckDashboard(orchestrator=orch).run() or 0
    else:
        exit_code = orch.run()
        if not no_annotate:
            _print_session_summary(storage, session.session_id, orch.stats)

    storage.close()
    sys.exit(exit_code)


@main.command(name="search")
@click.argument("query")
@click.option(
    "--session",
    "session_id",
    default=None,
    metavar="SESSION_ID",
    help="Restrict search to a specific session UUID.",
)
@click.option(
    "--limit",
    default=20,
    show_default=True,
    type=int,
    help="Maximum number of results.",
)
@click.option(
    "--db",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to the ACK SQLite database.",
)
def cmd_search(
    query:      str,
    session_id: str | None,
    limit:      int,
    db:         Path | None,
) -> None:
    """Full-text search over stored agent output (QUERY is an FTS5 expression)."""
    storage = StorageEngine(db_path=db) if db else StorageEngine()
    with storage:
        rows = storage.search(query, session_id=session_id, limit=limit)

    if not rows:
        click.echo(f"No results for: {query!r}")
        return

    for row in rows:
        sid_short = row["session_id"][:8]
        etype     = row["entry_type"]
        ts        = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["timestamp"]))
        pruned    = " [pruned]" if row["was_pruned"] else ""
        click.echo(click.style(f"[{ts}] session={sid_short}… type={etype}{pruned}", fg="cyan"))

        content = _sanitize(row["compressed_summary"] or row["raw_content"])
        preview = content[:300].strip()
        if len(content) > 300:
            preview += "\n  …"
        click.echo(f"  {preview}\n")


@main.command(name="toc")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
def cmd_toc(file: Path) -> None:
    """Print the symbol table-of-contents for a source FILE."""
    pager = Pager()
    try:
        fmap = pager.map_file(file)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Error parsing {file}: {exc}", err=True)
        sys.exit(1)

    click.echo(fmap.to_toc())


@main.command(name="sessions")
@click.option(
    "--limit",
    default=20,
    show_default=True,
    type=int,
    help="Number of sessions to list.",
)
@click.option(
    "--db",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to the ACK SQLite database.",
)
def cmd_sessions(limit: int, db: Path | None) -> None:
    """List recent ACK sessions."""
    storage = StorageEngine(db_path=db) if db else StorageEngine()
    with storage:
        rows = storage.list_sessions(limit=limit)

    if not rows:
        click.echo("No sessions found.")
        return

    click.echo(f"{'SESSION ID':<38}  {'STARTED':<20}  {'CMD'}")
    click.echo("─" * 90)
    for row in rows:
        sid   = row["session_id"]
        cmd   = _sanitize(row["agent_command"])[:40]
        ts    = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["started_at"]))
        ended = " ✓" if row["ended_at"] else " …"
        click.echo(f"{sid}  {ts}  {cmd}{ended}")


if __name__ == "__main__":
    main()
