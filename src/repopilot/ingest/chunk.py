"""Split a repository into citable chunks.

Python files are split along top-level functions and classes (via `ast`), so a chunk is
usually one meaningful unit. Everything else falls back to overlapping line windows.
Each chunk keeps its file path and 1-based line range so answers can cite exact lines.
"""

from __future__ import annotations

import ast
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from repopilot import config

TEXT_EXTENSIONS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go", ".rs", ".java", ".kt",
    ".scala", ".rb", ".php", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".swift", ".m", ".sh",
    ".bash", ".zsh", ".sql", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json", ".ini",
    ".cfg", ".html", ".css", ".scss", ".vue", ".svelte", ".lua", ".r", ".jl", ".ex", ".exs",
}
TEXT_FILENAMES = {"Dockerfile", "Makefile", "README", "LICENSE", "CHANGELOG"}

# Generated, vendored or noisy paths that waste the token budget.
SKIP_DIRS = {
    ".git", "node_modules", "vendor", "third_party", "dist", "build", ".venv", "venv", "env",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".tox", "site-packages", ".next", "target",
    "coverage", ".idea", ".vscode", ".repopilot",
}
SKIP_SUFFIXES = (".min.js", ".min.css", ".lock", "-lock.json", ".map", ".snap")


@dataclass
class Chunk:
    id: str
    path: str          # repo-relative, forward slashes
    start_line: int    # 1-based, inclusive
    end_line: int      # 1-based, inclusive
    symbol: str        # e.g. "class Session", "def request", or "" for line windows
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


def iter_source_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if name.endswith(SKIP_SUFFIXES):
                continue
            if p.suffix.lower() not in TEXT_EXTENSIONS and name not in TEXT_FILENAMES:
                continue
            try:
                if p.stat().st_size > config.MAX_FILE_BYTES or p.is_symlink():
                    continue
            except OSError:
                continue
            yield p


def chunk_file(path: Path, rel: str) -> list[Chunk]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    if not text.strip():
        return []
    lines = text.splitlines()
    if path.suffix == ".py":
        chunks = _chunk_python(text, lines, rel)
        if chunks:
            return chunks
    return _chunk_lines(lines, rel)


def chunk_repo(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in iter_source_files(root):
        rel = p.relative_to(root).as_posix()
        chunks.extend(chunk_file(p, rel))
    return chunks


def _make(rel: str, lines: list[str], start: int, end: int, symbol: str = "") -> Chunk:
    """start/end are 1-based inclusive."""
    body = "\n".join(lines[start - 1 : end])
    return Chunk(id=f"{rel}:{start}-{end}", path=rel, start_line=start, end_line=end, symbol=symbol, text=body)


def _chunk_lines(lines: list[str], rel: str, offset: int = 1) -> list[Chunk]:
    """Overlapping windows over lines[...]; `offset` is the 1-based line number of lines[0]."""
    size, overlap = config.CHUNK_LINES, config.CHUNK_OVERLAP
    step = max(1, size - overlap)
    out = []
    for i in range(0, len(lines), step):
        window = lines[i : i + size]
        if any(l.strip() for l in window):
            start = offset + i
            out.append(_make_from(rel, window, start))
        if i + size >= len(lines):
            break
    return out


def _make_from(rel: str, window: list[str], start: int, symbol: str = "") -> Chunk:
    end = start + len(window) - 1
    return Chunk(id=f"{rel}:{start}-{end}", path=rel, start_line=start, end_line=end, symbol=symbol, text="\n".join(window))


def _chunk_python(text: str, lines: list[str], rel: str) -> list[Chunk]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []

    out: list[Chunk] = []
    covered_until = 0  # last line (1-based) already assigned to a chunk
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = min([d.lineno for d in node.decorator_list] + [node.lineno])
        end = node.end_lineno or node.lineno
        # Module-level code between definitions (imports, constants) gets its own chunks.
        if start - 1 > covered_until:
            out.extend(_chunk_lines(lines[covered_until : start - 1], rel, offset=covered_until + 1))
        if isinstance(node, ast.ClassDef):
            out.extend(_chunk_class(node, lines, rel, start, end))
        else:
            out.extend(_chunk_def(lines, rel, start, end, f"def {node.name}"))
        covered_until = end
    if covered_until < len(lines):
        out.extend(_chunk_lines(lines[covered_until:], rel, offset=covered_until + 1))
    return out


def _chunk_def(lines: list[str], rel: str, start: int, end: int, symbol: str) -> list[Chunk]:
    """One chunk per function; very long ones are windowed but keep their symbol."""
    if end - start + 1 <= config.CHUNK_LINES * 2:
        return [_make(rel, lines, start, end, symbol)]
    chunks = _chunk_lines(lines[start - 1 : end], rel, offset=start)
    for c in chunks:
        c.symbol = symbol
    return chunks


def _chunk_class(node: ast.ClassDef, lines: list[str], rel: str, start: int, end: int) -> list[Chunk]:
    """Small classes stay whole. Larger ones become a header chunk (signature, docstring,
    class attributes) plus one chunk per method, so a hit lands on the method that matters."""
    if end - start + 1 <= config.CHUNK_LINES:
        return [_make(rel, lines, start, end, f"class {node.name}")]
    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not methods:
        return _chunk_def(lines, rel, start, end, f"class {node.name}")

    out: list[Chunk] = []
    cursor = start  # first line not yet assigned
    for m in methods:
        m_start = min([d.lineno for d in m.decorator_list] + [m.lineno])
        m_end = m.end_lineno or m.lineno
        if m_start > cursor:  # header, or attributes/comments between methods
            gap = lines[cursor - 1 : m_start - 1]
            if any(l.strip() for l in gap):
                sym = f"class {node.name}"
                out.extend(_chunk_def(lines, rel, cursor, m_start - 1, sym))
        out.extend(_chunk_def(lines, rel, m_start, m_end, f"def {node.name}.{m.name}"))
        cursor = m_end + 1
    if cursor <= end and any(l.strip() for l in lines[cursor - 1 : end]):
        out.extend(_chunk_def(lines, rel, cursor, end, f"class {node.name}"))
    return out
