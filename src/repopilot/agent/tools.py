"""Read-only tools the agent can call to explore an indexed repository.

Safety: every tool is read-only and scoped to files that are in the index. There is no
shell, no network and no write access, and paths are matched against the index rather than
joined onto the filesystem, so "../../etc/passwd" simply isn't a known file.

Every span of code a tool returns is recorded in `seen`, which is what citations in the
final answer are checked against.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field

from repopilot.retrieval.hybrid import Retriever

MAX_READ_LINES = 150
MAX_SEARCH_RESULTS = 8
SNIPPET_LINES = 25

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the repository for code relevant to a natural-language query or identifier. "
                           "Returns the best-matching chunks with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, e.g. 'retry backoff' or 'HTTPAdapter.send'"},
                    "k": {"type": "integer", "description": f"Number of results (1-{MAX_SEARCH_RESULTS})", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_definition",
            "description": "Find where a function, method or class is defined, by exact name (e.g. 'send' or 'Session.send').",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": f"Read lines from a file in the repository (at most {MAX_READ_LINES} lines per call).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repo-relative path, as shown in search results"},
                    "start_line": {"type": "integer", "default": 1},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List indexed files whose path starts with a prefix (e.g. 'src/requests/'). Empty prefix lists all.",
            "parameters": {
                "type": "object",
                "properties": {"prefix": {"type": "string", "default": ""}},
            },
        },
    },
]


@dataclass
class Span:
    path: str
    start: int
    end: int


@dataclass
class RepoTools:
    retriever: Retriever
    search_mode: str = "hybrid"
    seen: list[Span] = field(default_factory=list)
    _files: dict[str, list[str]] = field(default_factory=dict)
    _symbol_table: list | None = None

    def __post_init__(self):
        if not self.retriever.has_vectors and self.search_mode != "bm25":
            self.search_mode = "bm25"
        self.paths = sorted({c.path for c in self.retriever.index.chunks})
        self._path_set = set(self.paths)

    # -- dispatch -------------------------------------------------------------------------

    def call(self, name: str, args: dict) -> str:
        """Run a tool by name. Errors are returned to the model as text, never raised."""
        fn = {
            "search_code": self.search_code,
            "find_definition": self.find_definition,
            "read_file": self.read_file,
            "list_files": self.list_files,
        }.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'. Available: search_code, find_definition, read_file, list_files."
        if "_invalid_json" in args:
            return "Error: tool arguments were not valid JSON."
        try:
            return fn(**{k: v for k, v in args.items() if not k.startswith("_")})
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"

    # -- tools ----------------------------------------------------------------------------

    def search_code(self, query: str, k: int = 5) -> str:
        k = max(1, min(int(k), MAX_SEARCH_RESULTS))
        hits = self.retriever.search(str(query), k=k, mode=self.search_mode)
        if not hits:
            return "No results."
        blocks = []
        for h in hits:
            c = h.chunk
            lines = c.text.splitlines()
            shown = lines[:SNIPPET_LINES]
            end = c.start_line + len(shown) - 1
            self.seen.append(Span(c.path, c.start_line, end))
            more = f"\n      ... ({len(lines) - len(shown)} more lines; use read_file)" if len(lines) > len(shown) else ""
            blocks.append(f"[{c.path}:{c.start_line}-{c.end_line}] {c.symbol}\n{_number(shown, c.start_line)}{more}")
        return "\n\n".join(blocks)

    def find_definition(self, name: str) -> str:
        want = str(name).strip().split(".")
        matches = [
            (path, qual, a, b) for path, qual, a, b in self._symbols()
            if qual.split(".")[-len(want):] == want
        ]
        if not matches:
            term = want[-1].lower()
            similar = [f"[{p}:{a}-{b}] {q}" for p, q, a, b in self._symbols() if term in q.split(".")[-1].lower()][:5]
            if similar:
                return f"No definition named exactly '{name}'. Names containing '{term}':\n" + "\n".join(similar)
            return f"No definition named '{name}' found. Try search_code."
        out = [f"[{path}:{a}-{b}] {qual}" for path, qual, a, b in matches[:5]]
        more = f"\n({len(matches) - 5} more)" if len(matches) > 5 else ""
        return "Definitions (use read_file to see the code):\n" + "\n".join(out) + more

    def _symbols(self) -> list[tuple[str, str, int, int]]:
        """Every def/class in the indexed Python files, including methods of small classes
        that were kept as a single chunk (so they have no chunk symbol of their own)."""
        if self._symbol_table is None:
            table = []
            for path in self.paths:
                if not path.endswith(".py"):
                    continue
                try:
                    tree = ast.parse("\n".join(self._lines(path)))
                except (SyntaxError, ValueError):
                    continue
                table.extend((path, qual, a, b) for qual, a, b in _walk_defs(tree.body, ""))
            self._symbol_table = table
        return self._symbol_table

    def read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        path = str(path).strip().removeprefix("./")
        if path not in self._path_set:
            close = [p for p in self.paths if p.endswith("/" + path.split("/")[-1])][:5]
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            return f"Error: '{path}' is not a file in this repository's index.{hint}"
        lines = self._lines(path)
        start = max(1, int(start_line or 1))
        end = min(len(lines), int(end_line) if end_line else start + MAX_READ_LINES - 1, start + MAX_READ_LINES - 1)
        if start > len(lines):
            return f"Error: {path} has only {len(lines)} lines."
        self.seen.append(Span(path, start, end))
        note = f"\n(file has {len(lines)} lines)" if end < len(lines) else ""
        return f"[{path}:{start}-{end}]\n{_number(lines[start - 1 : end], start)}{note}"

    def list_files(self, prefix: str = "") -> str:
        prefix = str(prefix or "").strip().removeprefix("./")
        found = [p for p in self.paths if p.startswith(prefix)]
        if not found:
            return f"No indexed files start with '{prefix}'."
        shown = found[:100]
        more = f"\n... and {len(found) - 100} more" if len(found) > 100 else ""
        return "\n".join(shown) + more

    # -- helpers --------------------------------------------------------------------------

    def _lines(self, path: str) -> list[str]:
        if path not in self._files:
            self._files[path] = (self.retriever.index.root / path).read_text(encoding="utf-8", errors="replace").splitlines()
        return self._files[path]


def _walk_defs(body: list, prefix: str):
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qual = f"{prefix}{node.name}"
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            yield qual, start, node.end_lineno or node.lineno
            if isinstance(node, ast.ClassDef):
                yield from _walk_defs(node.body, f"{qual}.")


def _number(lines: list[str], start: int) -> str:
    return "\n".join(f"{start + i:>5} | {line}" for i, line in enumerate(lines))


def tool_message(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def assistant_tool_message(text: str, tool_calls: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [
            {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
            for tc in tool_calls
        ],
    }
