"""Source-file table of contents.

Gives an agent a symbol map (classes, functions, methods + line ranges) so it
can page in a single definition instead of the whole file. Uses tree-sitter
when the wheel is installed and falls back to a regex parser otherwise.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

_TREE_SITTER_AVAILABLE = False
try:
    import tree_sitter_python as _tspython  # type: ignore[import-untyped]
    from tree_sitter import Language, Node, Parser  # type: ignore[import-untyped]

    _PY_LANGUAGE: Language | None = Language(_tspython.language())
    _TREE_SITTER_AVAILABLE = True
except (ImportError, AttributeError):
    _PY_LANGUAGE = None


@dataclass(frozen=True, slots=True)
class SymbolEntry:
    """A class, function or method. Line numbers are 1-indexed, inclusive."""

    name:       str
    kind:       str
    start_line: int
    end_line:   int
    docstring:  str = ""
    parent:     str = ""


@dataclass
class FileSymbolMap:
    path:       Path
    language:   str
    symbols:    list[SymbolEntry] = field(default_factory=list)
    file_hash:  str               = ""
    line_count: int               = 0

    def to_toc(self) -> str:
        """Render the symbol table as an indented text outline."""
        rows: list[str] = [f"# {self.path}  ({self.line_count} lines)"]

        methods_by_class: dict[str, list[SymbolEntry]] = {}
        for sym in self.symbols:
            if sym.parent:
                methods_by_class.setdefault(sym.parent, []).append(sym)

        for sym in self.symbols:
            if sym.parent:
                continue
            rows.append(
                f"{sym.kind:<10} {sym.name:<38} [{sym.start_line}–{sym.end_line}]"
            )
            if sym.kind == "class":
                for method in methods_by_class.get(sym.name, []):
                    rows.append(
                        f"  {'def':<8} {method.name:<36} [{method.start_line}–{method.end_line}]"
                    )

        return "\n".join(rows)
