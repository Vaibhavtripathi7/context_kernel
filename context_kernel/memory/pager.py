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


class Pager:
    """Builds and caches symbol maps keyed by file content hash."""

    def __init__(self) -> None:
        self._cache: dict[str, FileSymbolMap] = {}
        self._parser: Parser | None = (
            Parser(_PY_LANGUAGE)        # type: ignore[arg-type]
            if _TREE_SITTER_AVAILABLE and _PY_LANGUAGE is not None
            else None
        )

    def map_file(self, path: Path) -> FileSymbolMap:
        source_bytes = path.read_bytes()
        file_hash    = hashlib.sha256(source_bytes).hexdigest()[:16]
        cache_key    = f"{path.resolve()}:{file_hash}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        if _TREE_SITTER_AVAILABLE and self._parser is not None:
            fmap = self._parse_with_tree_sitter(path, source_bytes, file_hash)
        else:
            fmap = self._parse_with_regex(path, source_bytes, file_hash)

        self._cache[cache_key] = fmap
        return fmap

    def page_symbol(self, path: Path, symbol_name: str) -> str | None:
        """Return the source text of one named symbol, or None if not found."""
        fmap  = self.map_file(path)
        entry = next((s for s in fmap.symbols if s.name == symbol_name), None)
        if entry is None:
            return None

        source_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(source_lines[entry.start_line - 1 : entry.end_line])

    def toc(self, path: Path) -> str:
        return self.map_file(path).to_toc()

    def invalidate(self, path: Path) -> None:
        stale = [k for k in self._cache if k.startswith(str(path.resolve()))]
        for k in stale:
            del self._cache[k]

    def _parse_with_tree_sitter(
        self,
        path: Path,
        source: bytes,
        file_hash: str,
    ) -> FileSymbolMap:
        assert self._parser is not None
        tree = self._parser.parse(source)
        source_lines = source.decode("utf-8", errors="replace").splitlines()
        symbols: list[SymbolEntry] = []

        self._walk(tree.root_node, source_lines, symbols, parent_class="")

        return FileSymbolMap(
            path=path,
            language="python",
            symbols=symbols,
            file_hash=file_hash,
            line_count=len(source_lines),
        )

    def _walk(
        self,
        node:         Node,
        source_lines: list[str],
        out:          list[SymbolEntry],
        parent_class: str,
    ) -> None:
        if node.type in ("class_definition", "function_definition"):
            name_node = next(
                (c for c in node.children if c.type == "identifier"),
                None,
            )
            if name_node is None:
                return

            name = (name_node.text or b"<unknown>").decode("utf-8")
            kind = (
                "class"    if node.type == "class_definition" else
                "method"   if parent_class else
                "function"
            )
            out.append(SymbolEntry(
                name=name,
                kind=kind,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring=self._first_docstring(node),
                parent=parent_class,
            ))

            new_parent = name if node.type == "class_definition" else parent_class
            for child in node.children:
                self._walk(child, source_lines, out, new_parent)
            return

        for child in node.children:
            self._walk(child, source_lines, out, parent_class)

    @staticmethod
    def _first_docstring(node: Node) -> str:
        for child in node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        for expr in stmt.children:
                            if expr.type == "string" and expr.text:
                                raw = expr.text.decode("utf-8", errors="replace")
                                return str(raw).strip("\"'").strip()
        return ""

    _CLASS_RE = re.compile(r"^class\s+(\w+)")
    _DEF_RE   = re.compile(r"^(\s*)def\s+(\w+)")

    def _parse_with_regex(
        self,
        path: Path,
        source: bytes,
        file_hash: str,
    ) -> FileSymbolMap:
        text  = source.decode("utf-8", errors="replace")
        lines = text.splitlines()
        symbols: list[SymbolEntry] = []

        current_class: str | None = None

        for lineno, line in enumerate(lines, start=1):
            m_class = self._CLASS_RE.match(line)
            if m_class:
                if current_class and symbols:
                    for i in range(len(symbols) - 1, -1, -1):
                        if symbols[i].name == current_class and symbols[i].kind == "class":
                            symbols[i] = SymbolEntry(
                                name=symbols[i].name,
                                kind=symbols[i].kind,
                                start_line=symbols[i].start_line,
                                end_line=lineno - 1,
                                docstring=symbols[i].docstring,
                                parent=symbols[i].parent,
                            )
                            break

                current_class = m_class.group(1)
                symbols.append(SymbolEntry(
                    name=current_class,
                    kind="class",
                    start_line=lineno,
                    end_line=len(lines),
                    parent="",
                ))
                continue

            m_def = self._DEF_RE.match(line)
            if m_def:
                indent = len(m_def.group(1))
                name   = m_def.group(2)
                is_method = (indent > 0 and current_class is not None)
                symbols.append(SymbolEntry(
                    name=name,
                    kind="method" if is_method else "function",
                    start_line=lineno,
                    end_line=lineno,
                    parent=current_class if is_method and current_class else "",
                ))

        return FileSymbolMap(
            path=path,
            language="python",
            symbols=symbols,
            file_hash=file_hash,
            line_count=len(lines),
        )
