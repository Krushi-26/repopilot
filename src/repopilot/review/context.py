"""What the reviewer needs beyond the diff, and the checks that need no LLM at all.

- changed symbols: the functions/methods whose bodies the PR touches (Python, via ast)
- related code: where those symbols are used elsewhere, found with the same retriever as Q&A
- test gaps: changed public functions that no test file mentions (deterministic, free)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from repopilot.ingest.chunk import Chunk, chunk_repo, iter_source_files
from repopilot.ingest.index import Index
from repopilot.retrieval.hybrid import Retriever
from repopilot.retrieval.prior import path_prior
from repopilot.review.diff import FileDiff

TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*$|_test\.py$")


@dataclass
class Symbol:
    path: str
    name: str       # bare name, e.g. "send"
    qualname: str   # e.g. "Session.send"
    start: int
    end: int
    added: bool     # the whole definition is new in this PR
    first_changed: int  # first line inside it that the PR added (where comments can anchor)

    @property
    def is_public(self) -> bool:
        return not any(part.startswith("_") for part in self.qualname.split("."))


@dataclass
class Finding:
    """A deterministic review finding (no LLM)."""
    path: str
    line: int
    severity: str
    body: str


def changed_symbols(files: list[FileDiff], root: Path) -> list[Symbol]:
    out: list[Symbol] = []
    for f in files:
        if not f.path.endswith(".py") or TEST_PATH.search(f.path):
            continue
        try:
            tree = ast.parse((root / f.path).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        added = f.added_lines
        for node, qual in _walk_defs(tree):
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            end = node.end_lineno or node.lineno
            touched = {n for n in added if start <= n <= end}
            if not touched:
                continue
            # Report the innermost def only: a changed method, not also its whole class.
            if isinstance(node, ast.ClassDef) and any(
                isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
                and touched <= set(range(c.lineno, (c.end_lineno or c.lineno) + 1))
                for c in node.body
            ):
                continue
            whole_new = all(n in added for n in range(start, end + 1))
            out.append(Symbol(f.path, node.name, qual, start, end, whole_new, min(touched)))
    return out


def _walk_defs(tree: ast.Module, prefix: str = ""):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qual = f"{prefix}{node.name}"
            yield node, qual
            if isinstance(node, ast.ClassDef):
                yield from _walk_defs(node, prefix=f"{qual}.")


def find_test_gaps(symbols: list[Symbol], root: Path) -> list[Finding]:
    """Changed public functions whose name appears in no test file (including the PR's own)."""
    candidates = [s for s in symbols if s.is_public and s.name not in {"__init__", "main"}]
    if not candidates:
        return []
    test_text = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in iter_source_files(root)
        if TEST_PATH.search(p.relative_to(root).as_posix())
    )
    if not test_text:
        return []  # project has no tests at all; flagging every function would be noise
    findings = []
    for s in candidates:
        if not re.search(rf"\b{re.escape(s.name)}\b", test_text):
            what = "New" if s.added else "Changed"
            findings.append(
                Finding(s.path, s.first_changed, "suggestion", f"{what} `{s.qualname}` isn't referenced by any test. Consider adding one.")
            )
    return findings


def related_code(symbols: list[Symbol], retriever: Retriever, files: list[FileDiff], per_symbol: int = 2, limit: int = 8) -> list[Chunk]:
    """Chunks elsewhere in the repo that use the changed symbols (likely callers)."""
    changed_paths = {f.path for f in files}
    seen: set[str] = set()
    out: list[Chunk] = []
    for s in symbols:
        if s.name.startswith("_"):
            continue  # private helpers are called from their own module, and names collide across modules
        # A call site, not just any mention: "search(" not "search" in a docstring.
        word = re.compile(rf"\b{re.escape(s.name)}\s*\(")
        found = 0
        for h in retriever.search(s.name, k=15, mode="bm25"):
            c = h.chunk
            if c.path in changed_paths or c.id in seen or path_prior(c.path) < 1.0:
                continue
            if not word.search(c.text) or _defines(c, s.name):
                continue
            seen.add(c.id)
            out.append(c)
            found += 1
            if found >= per_symbol or len(out) >= limit:
                break
        if len(out) >= limit:
            break
    return out


def _defines(c: Chunk, name: str) -> bool:
    """The chunk is (another) definition of the name, not a caller."""
    return c.symbol.split(".")[-1].removeprefix("def ").removeprefix("class ") == name


def local_retriever(root: Path) -> Retriever:
    """A throwaway BM25 index of the PR's checkout. Fast (seconds), nothing written to disk."""
    chunks = chunk_repo(root)
    return Retriever(Index(name="pr", root=root, commit="", chunks=chunks, embeddings=None, embed_model=None))
