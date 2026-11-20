"""Get a repository onto disk: shallow-clone a URL, or use a local path as-is."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from repopilot import config


def repo_slug(source: str) -> str:
    """'https://github.com/psf/requests.git' -> 'psf__requests'; './foo' -> 'foo'."""
    s = source.rstrip("/").removesuffix(".git")
    m = re.search(r"github\.com[/:]([^/]+)/([^/]+)$", s)
    if m:
        return f"{m.group(1)}__{m.group(2)}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", Path(s).resolve().name)


def fetch_repo(source: str, ref: str | None = None) -> tuple[Path, str]:
    """Return (local_path, commit_sha). Clones URLs shallowly into the clones dir.

    Pinning `ref` (a tag, branch or SHA) keeps eval results comparable across runs.
    """
    path = Path(source)
    if path.exists():
        return path.resolve(), _head_sha(path)

    dest = config.CLONES_DIR / repo_slug(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--quiet", source, str(dest)], check=True)
    if ref:
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "--quiet", "origin", ref], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", "FETCH_HEAD"], check=True)
    return dest, _head_sha(dest)


def _head_sha(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
