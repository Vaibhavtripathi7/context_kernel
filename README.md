# ACK — Agent Context Kernel

**Intelligent memory middleware for AI coding agents.** ACK is a transparent
proxy that sits between you and any terminal-based AI agent (Aider, Claude
Code, etc.). It intercepts the agent's output in real time, **prunes the
noisy, repetitive, high-token blobs** (stack traces, build-error walls, log
floods) down to their actionable signal, and **archives the full, untouched
output** to a local, full-text-searchable database.

The agent — and your context window — only sees the summary. The full log is
always one `ack search` away.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)
![Platform](https://img.shields.io/badge/platform-POSIX-lightgrey.svg)

---

## The problem

Long AI coding sessions rot. A single failed test can dump a 60-frame
traceback; a broken build can repeat the same `error[E0308]` once per crate; a
runaway logger can flood 400 identical lines. Every one of those tokens:

- **pollutes the context window** the model reasons over, and
- **invalidates the prompt cache**, so the next turn is slower and costlier.

The signal in that wall of text is tiny — one exception type, a couple of user
frames, a unique error code. ACK keeps the signal and drops the noise, with no
new LLM in the loop.

## How it works

ACK spawns your agent inside a real **pseudo-terminal (PTY)**, so interactive
prompts, colors, and cursor movement behave exactly as if you'd run the agent
directly. It multiplexes I/O in a single `select()` loop:

```
                 ┌─────────────────────────── ACK ───────────────────────────┐
   you  ──stdin──┤                                                            │
                 │   keystrokes ─────────────────────────────────► master_fd │──► agent
                 │                                                  (PTY)      │     (in PTY)
   you  ◄─stdout─┤   agent output ─► buffer ─► pruners ─► stdout              │◄── agent
                 │                      │                                     │
                 │                      └─► full raw text ─► SQLite + FTS5    │
                 └────────────────────────────────────────────────────────────┘
```

- Output below a configurable line threshold passes through **verbatim**.
- Interactive prompts (`[Y/n]`, `Continue?`, `Enter key:`) are **never** pruned —
  the agent is waiting and you need to see the question.
- Larger blobs are routed through ordered **Pruners** (first match wins). A
  match injects a compact summary into the live stream; a miss passes through.
- Every chunk — pruned or not — is persisted to SQLite so nothing is lost.

## Install

ACK is POSIX-only (it uses `pty`, `fork`, and `termios`) and requires
**Python 3.13+**.

```bash
# with Poetry (recommended for development)
git clone https://github.com/vaibhavtripathi/context-kernel
cd context-kernel
poetry install
poetry run ack --help

# or with pip
pip install .
ack --help
```

`tree-sitter` is an optional dependency used by `ack toc` for accurate parsing;
if it isn't available, ACK falls back to a regex parser automatically.

## Quick start

```bash
# Wrap any agent — ACK is transparent
ack run -- aider --model gpt-4o
ack run -- claude --dangerously-skip-permissions

# Tune when pruning kicks in (output lines before pruners activate)
ack run --prune-threshold 50 -- aider

# Search the full archived output of every session (FTS5 syntax)
ack search "ImportError"
ack search "ModuleNotFoundError OR FileNotFoundError"
ack search "context window" --session abc123

# Print a symbol table-of-contents for a file instead of dumping the whole thing
ack toc context_kernel/core/orchestrator.py

# List recent sessions
ack sessions
```

### Try it in 10 seconds (no agent required)

```bash
# A bundled script that emits a 200-line log flood + a real traceback.
# Watch ACK collapse it live, then search the full archived output.
ack run -- python examples/crashing_agent.py
ack search "KeyError"
```

## What gets compressed

The built-in `ShellPruner` handles the three biggest context-window offenders:

| Input | Strategy | Result |
| --- | --- | --- |
| **Python tracebacks** | Keep the exception line + 3 deepest *user* frames; drop stdlib/venv noise. Chained exceptions each summarized. | `── Traceback ──` with the frames that matter |
| **Rust / GCC / Clang errors** | Deduplicate by error code / diagnostic line; surface counts + unique list. | `[Rust build: 32 errors → 8 unique]` |
| **Repetitive log floods** | Detect when one line dominates (≥60%); replace with a frequency table. | `× 400  WARNING:root:retrying...` |

The original bytes are never modified on disk — only the *live stream* the
agent sees is compressed.

## Benchmark results

ACK ships a reproducible A/B benchmark (no LLM calls, free to run) that
measures token reduction and **signature fidelity** — whether every actionable
error signature (exception class, error code, file:line) survives compression.

```bash
poetry run python scripts/run_benchmarks.py
```

Representative output across the bundled corpus (tracebacks, Rust/GCC builds,
log floods, mixed realistic output):

| Suite | Compression | Signature fidelity |
| --- | --- | --- |
| Python Traceback | 93% | 100% |
| Rust + GCC Builds | 91% | 100% |
| Log Flood | 99% | 100% |
| Mixed Realistic | 80% | 100% |
| **Global** | **~95%** | **100%** |

≈ 62,000 of 66,000 corpus tokens removed with zero loss of error signatures.

## Architecture

| Module | Responsibility |
| --- | --- |
| `core/orchestrator.py` | PTY spawn (`openpty` + `fork` + `setsid`), `select()` I/O loop, buffer management, stream injection, terminal raw/cooked mode, SIGWINCH forwarding. |
| `memory/storage.py` | SQLite in WAL mode + FTS5 content-table (trigger-synced, zero row duplication), BM25 search, per-session stats. |
| `memory/pager.py` | tree-sitter (with regex fallback) symbol mapper — gives an agent a file's table-of-contents so it can page in just one function. |
| `pruners/base.py` | `BasePruner` ABC: `matches()` fast gate + `compress()`. |
| `pruners/shell_pruner.py` | The built-in traceback / build-error / log-flood pruner. |
| `cli.py` | `click` CLI (`run`, `search`, `toc`, `sessions`) + experimental Textual stats dashboard. |

The full session archive lives at `~/.local/share/ack/kernel.db` by default
(override with `--db`).

## Writing your own pruner

```python
from typing import Optional
from context_kernel.pruners.base import BasePruner, PrunerMetadata

class MyPruner(BasePruner):
    metadata = PrunerMetadata(name="my_pruner", description="...")

    def matches(self, text: str) -> bool:
        # Cheap gate — called on every flush.
        return "my-pattern" in text

    def compress(self, text: str) -> Optional[str]:
        # Return a summary string, or None to pass through verbatim.
        return summarize(text)
```

Register it on the `Orchestrator` (`pruners=[MyPruner(), ShellPruner()]`) or at
runtime via `Orchestrator.add_pruner()`. Pruners run in order; the first to
return a non-`None` summary wins.

## Development

```bash
poetry install
poetry run pytest                 # full suite (unit + integration)
poetry run pytest -m "not integration"   # fast unit tests only
poetry run python scripts/run_benchmarks.py
poetry run ruff check context_kernel
poetry run mypy context_kernel
```

## Limitations & roadmap

- **POSIX only** — relies on `pty`/`fork`/`termios`. No Windows support (WSL works).
- **`ack toc` is Python-only** today; the tree-sitter integration is structured
  to add more languages.
- **`--tui` dashboard is experimental**: Textual and the PTY interceptor both
  want to own the terminal, so in `--tui` mode the agent runs non-interactively
  and output is captured but not mirrored live. Headless mode (`ack run`) is the
  recommended path. A future split-pane (tmux/Pilot) approach can lift this.
- **Heuristic pruning**: pruners use regex/structural heuristics, not an LLM, by
  design — they're fast, deterministic, and free.

## License

[MIT](LICENSE) © Vaibhav Tripathi
