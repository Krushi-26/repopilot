"""Parse a unified diff (git / GitHub format) into files, hunks and line numbers.

GitHub only accepts review comments on lines that appear in the diff, addressed by their
line number in the new file. So for every hunk we keep each line's new-file number, and the
reviewer is only allowed to comment on lines from this map.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from repopilot.ingest.chunk import SKIP_DIRS, SKIP_SUFFIXES

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


@dataclass
class DiffLine:
    kind: str            # '+', '-' or ' '
    new_no: int | None   # line number in the new file (None for removed lines)
    old_no: int | None
    text: str


@dataclass
class Hunk:
    header: str
    lines: list[DiffLine] = field(default_factory=list)


@dataclass
class FileDiff:
    path: str            # new path ('' when deleted)
    old_path: str
    status: str          # added | deleted | modified | renamed
    hunks: list[Hunk] = field(default_factory=list)
    binary: bool = False

    @property
    def added_lines(self) -> set[int]:
        return {l.new_no for h in self.hunks for l in h.lines if l.kind == "+" and l.new_no}

    @property
    def commentable_lines(self) -> set[int]:
        """Lines GitHub will accept a RIGHT-side comment on: added or context lines."""
        return {l.new_no for h in self.hunks for l in h.lines if l.new_no}

    @property
    def n_added(self) -> int:
        return len(self.added_lines)

    @property
    def n_removed(self) -> int:
        return sum(l.kind == "-" for h in self.hunks for l in h.lines)


def parse_diff(text: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    cur: FileDiff | None = None
    hunk: Hunk | None = None
    old_no = new_no = 0

    for raw in text.splitlines():
        if raw.startswith("diff --git "):
            m = re.match(r"diff --git a/(.*) b/(.*)$", raw)
            a, b = (m.group(1), m.group(2)) if m else ("", "")
            cur = FileDiff(path=b, old_path=a, status="modified")
            files.append(cur)
            hunk = None
            continue
        if cur is None:
            continue
        if hunk is not None and raw == "":
            raw = " "  # blank context line whose trailing space was stripped
        if hunk is None or raw[0] not in "+- \\":
            # File header area (or a line that can't be inside a hunk).
            if raw.startswith("new file mode"):
                cur.status = "added"
            elif raw.startswith("deleted file mode"):
                cur.status = "deleted"
            elif raw.startswith("rename from "):
                cur.status, cur.old_path = "renamed", raw[len("rename from "):]
            elif raw.startswith("rename to "):
                cur.path = raw[len("rename to "):]
            elif raw.startswith("Binary files") or raw.startswith("GIT binary patch"):
                cur.binary = True
            elif raw.startswith("--- "):
                cur.old_path = "" if raw[4:] == "/dev/null" else raw[4:].removeprefix("a/")
            elif raw.startswith("+++ "):
                cur.path = "" if raw[4:] == "/dev/null" else raw[4:].removeprefix("b/")
            elif m := HUNK_RE.match(raw):
                old_no, new_no = int(m.group(1)), int(m.group(3))
                hunk = Hunk(header=raw)
                cur.hunks.append(hunk)
            continue
        if m := HUNK_RE.match(raw):
            old_no, new_no = int(m.group(1)), int(m.group(3))
            hunk = Hunk(header=raw)
            cur.hunks.append(hunk)
        elif raw[0] == "+":  # inside a hunk, even "+++ x" is an added line, not a header
            hunk.lines.append(DiffLine("+", new_no, None, raw[1:]))
            new_no += 1
        elif raw[0] == "-":
            hunk.lines.append(DiffLine("-", None, old_no, raw[1:]))
            old_no += 1
        elif raw[0] == " ":
            hunk.lines.append(DiffLine(" ", new_no, old_no, raw[1:]))
            new_no += 1
            old_no += 1
        # "\ No newline at end of file" carries no line.
    for f in files:
        if f.status == "deleted":
            f.path = ""
    return files


DATA_SUFFIXES = (".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".xml")
MAX_DATA_LINES = 200  # a data file changing this much is almost always generated output


def is_reviewable(f: FileDiff) -> bool:
    """Skip deleted, binary, generated and vendored files: they waste the token budget."""
    if f.binary or f.status == "deleted" or not f.path:
        return False
    parts = f.path.split("/")
    if any(p in SKIP_DIRS for p in parts[:-1]):
        return False
    if f.path.endswith(SKIP_SUFFIXES + (".lock", ".svg", ".ipynb")):
        return False
    if f.path.endswith(DATA_SUFFIXES) and f.n_added + f.n_removed > MAX_DATA_LINES:
        return False
    return True
