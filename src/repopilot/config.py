"""Central settings. Everything is overridable by environment variables."""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (no dependency). Existing env vars win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

HOME = Path(os.environ.get("REPOPILOT_HOME") or ".repopilot").resolve()
CLONES_DIR = HOME / "clones"
INDEXES_DIR = HOME / "indexes"
CACHE_PATH = HOME / "llm_cache.sqlite"
TRACE_PATH = HOME / "traces.jsonl"

# Chunking
CHUNK_LINES = int(os.environ.get("REPOPILOT_CHUNK_LINES", 60))
CHUNK_OVERLAP = int(os.environ.get("REPOPILOT_CHUNK_OVERLAP", 10))
MAX_FILE_BYTES = int(os.environ.get("REPOPILOT_MAX_FILE_BYTES", 300_000))

# Retrieval
EMBED_MODEL = os.environ.get("REPOPILOT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
TOP_K = int(os.environ.get("REPOPILOT_TOP_K", 8))
MAX_CONTEXT_CHARS = int(os.environ.get("REPOPILOT_MAX_CONTEXT_CHARS", 24_000))
