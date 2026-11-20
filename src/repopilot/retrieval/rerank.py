"""Cross-encoder reranking of the top candidates from hybrid search.

First-stage retrievers score the query and each chunk separately; a cross-encoder reads them
together, which is slower but more precise. So it only re-scores the top `depth` candidates.

The final score blends the reranker's opinion with the first-stage rank, rather than replacing
it: web-trained rerankers can misjudge code, and the blend keeps a strong first-stage hit from
being thrown out by one bad score.
"""

from __future__ import annotations

from functools import lru_cache

from repopilot import config


@lru_cache(maxsize=2)
def _model(name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=name, cache_dir=str(config.HOME / "models"))


def rerank_text(path: str, symbol: str, text: str, max_chars: int = 2000) -> str:
    header = f"{path} {symbol}".strip()
    return f"{header}\n{text}"[:max_chars]


def rerank_scores(query: str, docs: list[str], model: str) -> list[float]:
    return [float(s) for s in _model(model).rerank(query, docs, batch_size=32)]


def blend(first_stage_ranks: list[int], rerank: list[float], alpha: float, k: int = 60) -> list[float]:
    """alpha=1 -> reranker only; alpha=0 -> first-stage order only.

    Both sides are turned into ranks and fused RRF-style, so their score scales don't matter.
    """
    order = sorted(range(len(rerank)), key=lambda i: -rerank[i])
    rr_rank = {i: r for r, i in enumerate(order, start=1)}
    return [
        alpha / (k + rr_rank[i]) + (1 - alpha) / (k + first_stage_ranks[i]) for i in range(len(rerank))
    ]
