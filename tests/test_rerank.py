from pathlib import Path

import pytest

from repopilot import config
from repopilot.ingest.index import build_index
from repopilot.retrieval import rerank
from repopilot.retrieval.hybrid import Retriever


def test_blend_extremes_follow_one_side():
    first = [1, 2, 3]
    scores = [0.1, 0.9, 0.5]  # reranker prefers item 1, then 2, then 0
    order = lambda f: sorted(range(3), key=lambda i: -f[i])
    assert order(rerank.blend(first, scores, alpha=1.0)) == [1, 2, 0]
    assert order(rerank.blend(first, scores, alpha=0.0)) == [0, 1, 2]


def test_blend_is_scale_free():
    first = [1, 2, 3]
    a = rerank.blend(first, [1.0, 2.0, 3.0], alpha=0.5)
    b = rerank.blend(first, [-100.0, 0.0, 500.0], alpha=0.5)
    assert a == b  # only the reranker's order matters, not its score scale


@pytest.fixture
def retriever(tmp_path: Path, monkeypatch) -> Retriever:
    monkeypatch.setattr(config, "INDEXES_DIR", tmp_path / "idx")
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in ("alpha", "beta", "gamma"):
        (repo / f"{name}.py").write_text(f"def {name}_handler(request):\n    return 'request {name}'\n")
    return Retriever(build_index(str(repo), embed=False, name="t"))


def test_rerank_mode_reorders_with_reranker(retriever, monkeypatch):
    # Stub cross-encoder: loves "gamma", so with alpha=1 it must come first.
    monkeypatch.setattr(rerank, "rerank_scores", lambda q, docs, model: [10.0 if "gamma" in d else 0.0 for d in docs])
    retriever.rerank_alpha = 1.0
    hits = retriever.search("request handler", k=3, mode="bm25+rerank")
    assert hits[0].chunk.path == "gamma.py"
    assert "rerank" in hits[0].sources and "bm25" in hits[0].sources


def test_unknown_search_option_is_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="bm25+magic")
