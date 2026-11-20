from pathlib import Path

import pytest

from repopilot import config
from repopilot.agent.answer import build_context, extract_citations
from repopilot.ingest.index import build_index, load_index
from repopilot.retrieval.hybrid import Retriever
from repopilot.retrieval.tokenize import tokenize


def test_tokenize_splits_identifiers():
    toks = tokenize("def getHTTPRetry(max_retries): pass")
    assert {"get", "http", "retry", "max", "retries", "gethttpretry", "max_retries"} <= set(toks)
    assert "def" not in toks


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(config, "INDEXES_DIR", tmp_path / "indexes")
    r = tmp_path / "repo"
    (r / "auth").mkdir(parents=True)
    (r / "auth" / "login.py").write_text(
        "def verify_password(user, password):\n    return hash(password) == user.password_hash\n"
    )
    (r / "net").mkdir()
    (r / "net" / "http.py").write_text(
        "class RetryPolicy:\n    def backoff(self, attempt):\n        return 2 ** attempt\n"
    )
    return r


def test_bm25_finds_the_right_file(repo: Path):
    build_index(str(repo), embed=False, name="t")
    r = Retriever(load_index("t"))
    assert r.search("where is the password verified", mode="bm25")[0].chunk.path == "auth/login.py"
    assert r.search("retry backoff policy", mode="bm25")[0].chunk.path == "net/http.py"


def test_citations_are_checked_against_retrieved_chunks(repo: Path):
    build_index(str(repo), embed=False, name="t")
    hits = Retriever(load_index("t")).search("password", mode="bm25")
    ctx = build_context(hits)
    assert "[auth/login.py:1-2]" in ctx and "    1 | def verify_password" in ctx
    cites = extract_citations("See [auth/login.py:1-2] and [auth/login.py:2] and [made/up.py:5-9].", hits)
    assert [c.valid for c in cites] == [True, True, False]
