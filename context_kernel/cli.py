"""Command-line entrypoint."""
from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import click

from .core.orchestrator import Orchestrator, OrchestratorConfig, OrchestratorStats
from .memory.storage import StorageEngine
from .pruners.shell_pruner import ShellPruner

_USD_PER_MILLION_INPUT_TOKENS = 3.0

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
    raw_pruned_tokens = db_stats["tokens_saved"]
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
        pass


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
        from .tui import AckDashboard

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


@main.command(name="recall")
@click.argument("target")
@click.option(
    "--session",
    "session_id",
    default=None,
    metavar="SESSION_ID",
    help="Session to search when TARGET is a query (default: most recent).",
)
@click.option(
    "--all",
    "all_sessions",
    is_flag=True,
    default=False,
    help="Search every session, not just the most recent.",
)
@click.option(
    "--limit",
    default=1,
    show_default=True,
    type=int,
    help="Max entries to return when TARGET is a query.",
)
@click.option(
    "--raw",
    is_flag=True,
    default=False,
    help="Print exact stored bytes without stripping escape sequences.",
)
@click.option(
    "--db",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to the ACK SQLite database.",
)
def cmd_recall(
    target:       str,
    session_id:   str | None,
    all_sessions: bool,
    limit:        int,
    raw:          bool,
    db:           Path | None,
) -> None:
    """Page a stored log back into view by handle or query.

    TARGET is either a numeric recall handle (the "ack #N" shown on a pruned
    banner -> `ack recall N`) or an FTS5 query, in which case the best match
    from the most recent session is returned. Use --all to widen the search.
    """
    render: Callable[[str], str] = (lambda t: t) if raw else _sanitize
    storage = StorageEngine(db_path=db) if db else StorageEngine()
    with storage:
        if target.isdigit():
            row  = storage.get_entry(int(target))
            rows = [row] if row is not None else []
        else:
            scope = session_id
            if scope is None and not all_sessions:
                recent = storage.list_sessions(limit=1)
                scope  = recent[0]["session_id"] if recent else None
            rows = storage.search(target, session_id=scope, limit=limit)

    if not rows:
        click.echo(f"No entry for: {target!r}", err=True)
        sys.exit(1)

    for row in rows:
        sid_short = row["session_id"][:8]
        ts        = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["timestamp"]))
        pruned    = " [pruned]" if row["was_pruned"] else ""
        click.echo(click.style(f"[ack #{row['id']}] session={sid_short}… {ts}{pruned}", fg="cyan"))
        click.echo(render(row["raw_content"]))


@main.command(name="toc")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
def cmd_toc(file: Path) -> None:
    """Print the symbol table-of-contents for a source FILE."""
    from .memory.pager import Pager

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
