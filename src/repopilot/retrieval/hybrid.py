"""Hybrid retrieval: BM25 + dense vectors, fused with Reciprocal Rank Fusion (RRF).

Keyword search wins on exact identifiers ("HTTPAdapter"); vectors win on paraphrases
("where are retries configured"). RRF combines ranked lists without having to calibrate
their very different score scales.
"""

from __future__ import annotations

from dataclasses import dataclass

from repopilot.ingest.chunk import Chunk
from repopilot.ingest.index import Index
from repopilot.retrieval.bm25 import BM25
from repopilot.retrieval.prior import path_prior
from repopilot.retrieval.tokenize import tokenize

RRF_K = 60  # standard constant from the RRF paper; dampens the weight of top ranks
# Per-retriever weights in the fusion. Vectors are the stronger retriever on the golden
# set (see README results log), so they get more say in the final order.
DEFAULT_WEIGHTS = {"bm25": 1.0, "vector": 1.5}


@dataclass
class Hit:
    chunk: Chunk
    score: float
    sources: dict[str, int]  # retriever name -> 1-based rank in that retriever


class Retriever:
    def __init__(self, index: Index, use_prior: bool = True):
        self.index = index
        # Down-weight tests/docs/changelogs (see prior.py); 1.0 everywhere when disabled.
        self.prior = [path_prior(c.path) if use_prior else 1.0 for c in index.chunks]
        # Index the path and symbol alongside the body: file names are strong signals in code.
        docs = [tokenize(f"{c.path} {c.symbol} {c.text}") for c in index.chunks]
        self.bm25 = BM25(docs)

    @property
    def has_vectors(self) -> bool:
        return self.index.embeddings is not None

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = tokenize(query)
        if not q:
            return []
        scores = [s * w for s, w in zip(self.bm25.get_scores(q), self.prior)]
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, float(scores[i])) for i in ranked if scores[i] > 0]

    def vector_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.has_vectors:
            return []
        from repopilot.retrieval.vector import vector_search

        # Over-fetch, then re-rank with the prior so demoted chunks can't fill the pool.
        hits = vector_search(query, self.index.embeddings, self.index.embed_model, k * 3)
        hits = [(i, s * self.prior[i]) for i, s in hits]
        return sorted(hits, key=lambda x: -x[1])[:k]

    def search(
        self, query: str, k: int = 8, mode: str = "hybrid", pool: int = 50, weights: dict[str, float] | None = None
    ) -> list[Hit]:
        """mode: 'bm25', 'vector' or 'hybrid'. `pool` = candidates taken from each retriever."""
        lists: dict[str, list[tuple[int, float]]] = {}
        if mode in ("bm25", "hybrid"):
            lists["bm25"] = self.bm25_search(query, pool)
        if mode in ("vector", "hybrid"):
            lists["vector"] = self.vector_search(query, pool)

        weights = weights or DEFAULT_WEIGHTS
        fused: dict[int, float] = {}
        ranks: dict[int, dict[str, int]] = {}
        for name, results in lists.items():
            for rank, (idx, _) in enumerate(results, start=1):
                fused[idx] = fused.get(idx, 0.0) + weights.get(name, 1.0) / (RRF_K + rank)
                ranks.setdefault(idx, {})[name] = rank

        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.index.chunks[i], fused[i], ranks[i]) for i in best]
