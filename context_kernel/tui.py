"""Optional Textual stats dashboard for `ack run --tui`.

In --tui mode the orchestrator runs on a background thread and Textual renders
a stats/log overlay. Because both want the terminal, the agent runs
non-interactively there; plain ack run is the interactive path. Imported only
when --tui is used, so plain commands stay fast to start.
"""
from __future__ import annotations

import threading
import time

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, RichLog, Static

from .core.orchestrator import Orchestrator, OrchestratorStats


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
