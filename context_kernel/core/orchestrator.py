"""PTY spawn loop, output buffering and stream injection.

The agent runs inside a real pseudo-terminal so interactive prompts and colour
output behave as if it were launched directly. A single select() loop forwards
keystrokes to the child and routes the child's output through the pruners
before it reaches the user's terminal.
"""
from __future__ import annotations

import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import time
import tty
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..memory.storage import LogEntry, StorageEngine
from ..pruners.base import BasePruner

_DIM   = "\033[2m"
_CYAN  = "\033[36m"
_BOLD  = "\033[1m"
_RESET = "\033[0m"

_SELECT_TIMEOUT = 0.08


@dataclass
class OrchestratorConfig:
    pruning_threshold_lines: int   = 30
    buffer_flush_timeout:    float = 0.15
    read_chunk_bytes:        int   = 8192
    annotate_injections:     bool  = True


@dataclass
class OrchestratorStats:
    total_bytes_read:     int   = 0
    total_bytes_injected: int   = 0
    total_pruner_hits:    int   = 0
    tokens_saved:         int   = 0
    session_start:        float = field(default_factory=time.monotonic)


_PROMPT_BYTES: tuple[bytes, ...] = (
    b"[Y/n]",
    b"[y/N]",
    b"[Y/N]",
    b"[yes/no]",
    b"(y/n)",
    b"(yes/no)",
    b"? ",
)
_PROMPT_TAIL_CHARS = frozenset("?:>")
_PROMPT_MAX_LINE_LEN = 120
