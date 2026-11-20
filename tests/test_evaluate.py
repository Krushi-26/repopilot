from pathlib import Path

import pytest

from repopilot import config, evaluate
from repopilot.ingest.index import build_index
from repopilot.retrieval.hybrid import Retriever

SRC = '''import os


class Session:
    @property
    def name(self):
        return "s"

    def send(self, request):
        return request


def helper():
    pass
'''


def test_find_symbol_resolves_methods_and_decorators(tmp_path: Path):
    f = tmp_path / "m.py"
    f.write_text(SRC)
    assert evaluate.find_symbol(f, "Session.send") == (9, 10)
    assert evaluate.find_symbol(f, "Session.name") == (5, 7)  # decorator line included
    assert evaluate.find_symbol(f, "helper") == (13, 14)
    with pytest.raises(ValueError):
        evaluate.find_symbol(f, "Session.missing")


def test_target_overlap():
    t = evaluate.Target("a.py", 10, 20)
    assert t.overlaps("a.py", 15, 30) and t.overlaps("a.py", 1, 10)
    assert not t.overlaps("a.py", 21, 40) and not t.overlaps("b.py", 10, 20)
    assert evaluate.Target("a.py", None, None).overlaps("a.py", 500, 600)


def test_end_to_end_scoring(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(config, "INDEXES_DIR", tmp_path / "idx")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text(SRC)
    (repo / "other.py").write_text("def unrelated_thing():\n    return 42\n")
    golden = {
        "questions": [
            {"id": "a", "q": "how does session send a request", "expect": [{"path": "m.py", "symbol": "Session"}]},
            {"id": "b", "q": "zzz nothing matches", "expect": [{"path": "m.py", "symbol": "helper"}]},
        ]
    }
    r = Retriever(build_index(str(repo), embed=False, name="t"))
    res = evaluate.eval_retrieval(golden, r, "bm25")
    assert res[0].span_rank == 1 and res[1].span_rank is None
    s = evaluate.summarise(res)
    assert s["span_hit@1"] == 0.5 and s["mrr@10"] == 0.5
    assert evaluate.badge(s)["message"] == "50%"


def test_golden_labels_are_well_formed():
    g = evaluate.load_golden(Path(__file__).parent.parent / "evals" / "golden_requests.json")
    ids = [q["id"] for q in g["questions"]]
    assert len(ids) == len(set(ids)) >= 30
    for q in g["questions"]:
        assert q["q"].strip() and q["expect"]
        for e in q["expect"]:
            assert e["path"].startswith("src/requests/")
