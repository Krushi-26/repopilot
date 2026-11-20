"""Golden-set evaluation.

Retrieval metrics (no LLM, free, deterministic):
  file hit@k  - an expected file appears in the top k chunks
  span hit@k  - a top-k chunk overlaps an expected symbol's lines (stricter)
  MRR         - mean of 1/rank of the first correct chunk (span-level, within top 10)

Answer metrics (`--answers`, needs an LLM key):
  citation accuracy - share of citations that point inside retrieved excerpts
  grounded rate     - share of answers citing at least one expected span

Expected targets name a symbol ("Session.send"), a line, or just a path. Symbols are
resolved to line ranges from the pinned clone, so a stale label fails loudly instead
of quietly scoring zero.
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from repopilot.ingest.index import Index
from repopilot.retrieval.hybrid import Hit, Retriever

KS = (1, 3, 5, 10)


@dataclass
class Target:
    path: str
    start: int | None  # None = whole file
    end: int | None

    def overlaps(self, path: str, start: int, end: int) -> bool:
        if path != self.path:
            return False
        return self.start is None or (start <= self.end and end >= self.start)


@dataclass
class QResult:
    id: str
    question: str
    file_rank: int | None  # 1-based rank of first chunk in an expected file
    span_rank: int | None  # 1-based rank of first chunk overlapping an expected span
    top: list[str] = field(default_factory=list)
    answer: dict | None = None


def load_golden(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_targets(expect: list[dict], root: Path) -> list[Target]:
    out = []
    for e in expect:
        if "symbol" in e:
            start, end = find_symbol(root / e["path"], e["symbol"])
            out.append(Target(e["path"], start, end))
        elif "line" in e:
            out.append(Target(e["path"], e["line"], e["line"]))
        else:
            if not (root / e["path"]).exists():
                raise ValueError(f"Golden label points at missing file: {e['path']}")
            out.append(Target(e["path"], None, None))
    return out


def find_symbol(file: Path, symbol: str) -> tuple[int, int]:
    """'Class.method' or 'func' -> (start, end) lines, decorators included."""
    tree = ast.parse(file.read_text(encoding="utf-8"))
    nodes = tree.body
    found = None
    for part in symbol.split("."):
        found = next(
            (n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part),
            None,
        )
        if found is None:
            raise ValueError(f"Golden label '{symbol}' not found in {file}")
        nodes = found.body
    start = min([d.lineno for d in found.decorator_list] + [found.lineno])
    return start, found.end_lineno or found.lineno


def _first_rank(hits: list[Hit], match) -> int | None:
    return next((i for i, h in enumerate(hits, 1) if match(h)), None)


def eval_retrieval(golden: dict, retriever: Retriever, mode: str, depth: int = 10) -> list[QResult]:
    root = retriever.index.root
    results = []
    for q in golden["questions"]:
        targets = resolve_targets(q["expect"], root)
        paths = {t.path for t in targets}
        hits = retriever.search(q["q"], k=depth, mode=mode)
        results.append(
            QResult(
                id=q["id"],
                question=q["q"],
                file_rank=_first_rank(hits, lambda h: h.chunk.path in paths),
                span_rank=_first_rank(
                    hits, lambda h: any(t.overlaps(h.chunk.path, h.chunk.start_line, h.chunk.end_line) for t in targets)
                ),
                top=[h.chunk.citation for h in hits[:5]],
            )
        )
    return results


def eval_answers(golden: dict, retriever: Retriever, router, mode: str, results: list[QResult], k: int, agent: bool = False) -> None:
    from repopilot.agent.answer import answer_question
    from repopilot.agent.loop import run_agent

    by_id = {r.id: r for r in results}
    for q in golden["questions"]:
        targets = resolve_targets(q["expect"], retriever.index.root)
        t0 = time.time()
        if agent:
            ans = run_agent(q["q"], retriever, router, mode=mode.partition("+")[0])
        else:
            ans = answer_question(q["q"], retriever, router, k=k, mode=mode)
        grounded = any(t.overlaps(c.path, c.start, c.end) for c in ans.citations for t in targets)
        by_id[q["id"]].answer = {
            "n_citations": len(ans.citations),
            "valid_citations": sum(c.valid for c in ans.citations),
            "grounded": grounded,
            "latency_s": round(time.time() - t0, 2),
            "cached": bool(ans.completion and ans.completion.cached),
            "tokens": ans.tokens,
            "llm_calls": ans.llm_calls,
            "tool_calls": sum("tool" in s and not s.get("seed") for s in ans.steps),
        }


def summarise(results: list[QResult]) -> dict:
    n = len(results)
    s: dict = {"n": n}
    for k in KS:
        s[f"file_hit@{k}"] = sum(r.file_rank is not None and r.file_rank <= k for r in results) / n
        s[f"span_hit@{k}"] = sum(r.span_rank is not None and r.span_rank <= k for r in results) / n
    s["mrr@10"] = sum(1 / r.span_rank for r in results if r.span_rank) / n
    answered = [r.answer for r in results if r.answer]
    if answered:
        cites = sum(a["n_citations"] for a in answered)
        s["citation_accuracy"] = (sum(a["valid_citations"] for a in answered) / cites) if cites else 0.0
        s["grounded_rate"] = sum(a["grounded"] for a in answered) / len(answered)
        s["avg_llm_calls"] = sum(a.get("llm_calls", 1) for a in answered) / len(answered)
        toks = [a["tokens"] for a in answered if a.get("tokens")]
        if toks:
            s["avg_tokens"] = sum(toks) / len(toks)
        s["answered"] = len(answered)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}


def badge(summary: dict, metric: str = "span_hit@5") -> dict:
    """shields.io endpoint JSON: https://shields.io/badges/endpoint-badge"""
    v = summary[metric]
    color = "brightgreen" if v >= 0.8 else "green" if v >= 0.7 else "yellow" if v >= 0.5 else "red"
    return {"schemaVersion": 1, "label": f"retrieval {metric}", "message": f"{v:.0%}", "color": color}


def write_report(out_dir: Path, golden: dict, index: Index, runs: dict[str, list[QResult]]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "repo": golden["repo"],
        "ref": golden["ref"],
        "commit": index.commit,
        "n_chunks": len(index.chunks),
        "embed_model": index.embed_model,
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modes": {m: summarise(rs) for m, rs in runs.items()},
        "per_question": {m: [r.__dict__ for r in rs] for m, rs in runs.items()},
    }
    (out_dir / "results.json").write_text(json.dumps(report, indent=2))
    best = max(report["modes"], key=lambda m: report["modes"][m]["span_hit@5"])
    (out_dir / "badge.json").write_text(json.dumps(badge(report["modes"][best])))
    return report


def format_table(report: dict) -> str:
    cols = ["file_hit@1", "file_hit@5", "span_hit@1", "span_hit@5", "span_hit@10", "mrr@10"]
    extra = ("citation_accuracy", "grounded_rate", "avg_llm_calls", "avg_tokens")
    cols += [c for c in extra if any(c in s for s in report["modes"].values())]
    lines = ["| mode | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m, s in report["modes"].items():
        lines.append(f"| {m} | " + " | ".join(_fmt(c, s[c]) if c in s else "–" for c in cols) + " |")
    return "\n".join(lines)


def _fmt(col: str, v: float) -> str:
    if col == "mrr@10":
        return f"{v:.3f}"
    if col == "avg_llm_calls":
        return f"{v:.1f}"
    if col == "avg_tokens":
        return f"{v:,.0f}"
    return f"{v:.0%}"
