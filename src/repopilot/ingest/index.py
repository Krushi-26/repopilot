"""Build and load an on-disk index: chunks.jsonl + optional embeddings.npy + meta.json.

The index is a plain folder so CI can upload it as a build artifact. BM25 is rebuilt
from the chunks on load (it takes well under a second), so only embeddings are stored.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from repopilot import config
from repopilot.ingest.chunk import Chunk, chunk_repo
from repopilot.ingest.clone import fetch_repo, repo_slug


@dataclass
class Index:
    name: str
    root: Path            # where the repo's files live (for read_file / citation checks)
    commit: str
    chunks: list[Chunk]
    embeddings: np.ndarray | None  # (n_chunks, dim), L2-normalised, or None if BM25-only
    embed_model: str | None


def index_dir(name: str) -> Path:
    return config.INDEXES_DIR / name


def build_index(source: str, ref: str | None = None, embed: bool = True, name: str | None = None) -> Index:
    root, commit = fetch_repo(source, ref)
    name = name or repo_slug(source)
    t0 = time.time()
    chunks = chunk_repo(root)
    if not chunks:
        raise ValueError(f"No indexable files found in {root}")

    embeddings, model = None, None
    if embed:
        from repopilot.retrieval.vector import embed_documents, vector_available

        if vector_available():
            embeddings = embed_documents([_embed_text(c) for c in chunks])
            model = config.EMBED_MODEL
        else:
            print("fastembed not installed; building a BM25-only index (pip install 'repopilot[vector]').")

    out = index_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict()) + "\n")
    emb_path = out / "embeddings.npy"
    if embeddings is not None:
        np.save(emb_path, embeddings.astype(np.float32))
    elif emb_path.exists():
        emb_path.unlink()
    meta = {
        "name": name,
        "source": source,
        "root": str(root),
        "commit": commit,
        "n_chunks": len(chunks),
        "n_files": len({c.path for c in chunks}),
        "embed_model": model,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - t0, 2),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return Index(name, root, commit, chunks, embeddings, model)


def load_index(name: str) -> Index:
    d = index_dir(name)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"No index named '{name}'. Run: repopilot index <repo>")
    meta = json.loads((d / "meta.json").read_text())
    with (d / "chunks.jsonl").open() as f:
        chunks = [Chunk(**json.loads(line)) for line in f]
    emb_path = d / "embeddings.npy"
    embeddings = np.load(emb_path) if emb_path.exists() else None
    return Index(meta["name"], Path(meta["root"]), meta["commit"], chunks, embeddings, meta.get("embed_model"))


def list_indexes() -> list[dict]:
    if not config.INDEXES_DIR.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(config.INDEXES_DIR.glob("*/meta.json"))]


def _embed_text(c: Chunk) -> str:
    # Path and symbol carry a lot of meaning for code ("auth/login.py def verify_token").
    header = f"{c.path} {c.symbol}".strip()
    return f"{header}\n{c.text}"[:4000]
