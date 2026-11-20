"""Dense embeddings with a small local ONNX model (fastembed). No API calls, no torch."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from repopilot import config


def vector_available() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def _model(name: str):
    from fastembed import TextEmbedding

    # Keep the model (~70-130 MB) next to indexes, not in a system temp dir, so it is easy
    # to find, delete, and cache in CI.
    return TextEmbedding(model_name=name, cache_dir=str(config.HOME / "models"))


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


def embed_documents(texts: list[str], model: str | None = None) -> np.ndarray:
    m = _model(model or config.EMBED_MODEL)
    return _normalise(np.array(list(m.passage_embed(texts, batch_size=64)), dtype=np.float32))


def embed_query(text: str, model: str | None = None) -> np.ndarray:
    m = _model(model or config.EMBED_MODEL)
    return _normalise(np.array(list(m.query_embed([text])), dtype=np.float32))[0]


def vector_search(query: str, embeddings: np.ndarray, model: str | None, k: int) -> list[tuple[int, float]]:
    """Return [(chunk_index, cosine_score)] best first."""
    q = embed_query(query, model)
    scores = embeddings @ q
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    return sorted(((int(i), float(scores[i])) for i in top), key=lambda x: -x[1])
