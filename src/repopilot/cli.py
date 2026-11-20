"""Command line: `repopilot index | search | ask | list`."""

from __future__ import annotations

import argparse
import sys

from repopilot import config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="repopilot", description="Ask questions about a codebase, with citations.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="Clone (or read) a repo and build its search index")
    p.add_argument("source", help="GitHub URL or local path")
    p.add_argument("--ref", help="Tag, branch or commit to pin (keeps evals comparable)")
    p.add_argument("--name", help="Index name (default: derived from the source)")
    p.add_argument("--no-embed", action="store_true", help="BM25 only; skip the embedding model")

    for name, help_ in (("search", "Show retrieved chunks (no LLM call)"), ("ask", "Answer a question with citations")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("question")
        p.add_argument("--repo", "-r", help="Index name (default: the only/most recent index)")
        p.add_argument("-k", type=int, default=config.TOP_K)
        p.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid")
        p.add_argument("--rerank", action="store_true", help="Re-score the top 20 with a cross-encoder (~1 s, opt-in)")

    sub.add_parser("list", help="List built indexes")

    p = sub.add_parser("eval", help="Score retrieval (and optionally answers) against a golden set")
    p.add_argument("--golden", default="evals/golden_requests.json")
    p.add_argument("--modes", default="auto", help="Comma list of bm25,vector,hybrid[,hybrid+rerank] (default: all available, no rerank)")
    p.add_argument("--answers", action="store_true", help="Also generate answers and score citations (uses the LLM)")
    p.add_argument("--out", default="evals/results")
    p.add_argument("--no-embed", action="store_true", help="If the index must be built, build it BM25-only")
    p.add_argument("--rebuild", action="store_true", help="Rebuild the index first (after changing chunking)")
    p.add_argument("--min-span-hit5", type=float, default=None, help="Exit non-zero if the best mode scores below this (CI gate)")
    args = ap.parse_args(argv)

    from repopilot.ingest.index import build_index, list_indexes, load_index

    if args.cmd == "eval":
        return _run_eval(args)

    if args.cmd == "index":
        idx = build_index(args.source, ref=args.ref, embed=not args.no_embed, name=args.name)
        files = len({c.path for c in idx.chunks})
        vec = f"vectors: {idx.embed_model}" if idx.embeddings is not None else "BM25 only"
        print(f"Indexed '{idx.name}' @ {idx.commit[:10]}: {len(idx.chunks)} chunks from {files} files ({vec})")
        return 0

    if args.cmd == "list":
        for m in list_indexes():
            print(f"{m['name']:<40} {m['n_chunks']:>6} chunks  {m['commit'][:10]}  {m['embed_model'] or 'bm25-only'}")
        return 0

    name = args.repo or _default_index_name(list_indexes())
    if not name:
        print("No index found. Run: repopilot index <github-url-or-path>", file=sys.stderr)
        return 1

    from repopilot.retrieval.hybrid import Retriever

    retriever = Retriever(load_index(name))
    mode = args.mode if retriever.has_vectors or args.mode == "bm25" else "bm25"
    if args.rerank:
        mode += "+rerank"

    if args.cmd == "search":
        for i, h in enumerate(retriever.search(args.question, k=args.k, mode=mode), 1):
            print(f"{i:>2}. {h.chunk.citation:<55} {h.chunk.symbol:<30} rrf={h.score:.4f} {h.sources}")
        return 0

    from repopilot.agent.answer import answer_question
    from repopilot.llm.router import AllProvidersFailed, Router

    try:
        ans = answer_question(args.question, retriever, Router(), k=args.k, mode=mode)
    except AllProvidersFailed as e:
        print(e, file=sys.stderr)
        return 2
    print(ans.text)
    if ans.citations:
        bad = [str(c) for c in ans.citations if not c.valid]
        print(f"\n— {len(ans.citations)} citations, {len(ans.citations) - len(bad)} verified" + (f"; unverified: {', '.join(bad)}" if bad else ""))
    if ans.completion:
        c = ans.completion
        print(f"— {c.provider}/{c.model}{' (cached)' if c.cached else ''}")
    return 0


def _run_eval(args) -> int:
    from pathlib import Path

    from repopilot import evaluate
    from repopilot.ingest.index import build_index, load_index
    from repopilot.retrieval.hybrid import Retriever

    golden = evaluate.load_golden(Path(args.golden))
    try:
        if args.rebuild:
            raise FileNotFoundError
        index = load_index(golden["index_name"])
        if not index.commit.startswith("unknown"):
            _check_pinned(index, golden)
    except FileNotFoundError:
        print(f"Building index for {golden['repo']} @ {golden['ref']} ...")
        index = build_index(golden["repo"], ref=golden["ref"], embed=not args.no_embed, name=golden["index_name"])

    retriever = Retriever(index)
    if args.modes == "auto":
        modes = ["bm25", "vector", "hybrid"] if retriever.has_vectors else ["bm25"]
    else:
        modes = [m.strip() for m in args.modes.split(",")]
        if not retriever.has_vectors and {m.split("+")[0] for m in modes} - {"bm25"}:
            print("Index has no embeddings; rebuild without --no-embed (needs the 'vector' extra).", file=sys.stderr)
            return 1

    runs = {m: evaluate.eval_retrieval(golden, retriever, m) for m in modes}
    if args.answers:
        from repopilot.llm.router import AllProvidersFailed, Router

        best = modes[-1]
        try:
            evaluate.eval_answers(golden, retriever, Router(), best, runs[best], k=config.TOP_K)
        except AllProvidersFailed as e:
            print(e, file=sys.stderr)
            return 2

    report = evaluate.write_report(Path(args.out), golden, index, runs)
    print(f"{golden['repo']} @ {golden['ref']} ({index.commit[:10]}), {len(golden['questions'])} questions, {len(index.chunks)} chunks\n")
    print(evaluate.format_table(report))
    worst = modes[-1]
    misses = [r for r in runs[worst] if not r.span_rank or r.span_rank > 5]
    if misses:
        print(f"\nMisses in {worst} (no expected span in top 5):")
        for r in misses:
            print(f"  {r.id} {r.question}\n       got: {', '.join(r.top[:3])}")
    print(f"\nWrote {args.out}/results.json and badge.json")

    if args.min_span_hit5 is not None:
        best_score = max(s["span_hit@5"] for s in report["modes"].values())
        if best_score < args.min_span_hit5:
            print(f"FAIL: span_hit@5 {best_score:.0%} < gate {args.min_span_hit5:.0%}", file=sys.stderr)
            return 3
    return 0


def _check_pinned(index, golden: dict) -> None:
    import subprocess

    # Shallow clones fetch the pinned ref as FETCH_HEAD, so the tag may not exist locally.
    proc = subprocess.run(
        ["git", "-C", str(index.root), "rev-parse", "--verify", "-q", f"{golden['ref']}^{{commit}}"],
        capture_output=True, text=True,
    )
    want = proc.stdout.strip() if proc.returncode == 0 else ""
    if want and want != index.commit:
        print(f"Warning: index is at {index.commit[:10]} but golden set is pinned to {golden['ref']} ({want[:10]}).", file=sys.stderr)


def _default_index_name(indexes: list[dict]) -> str | None:
    if not indexes:
        return None
    return max(indexes, key=lambda m: m["built_at"])["name"]


if __name__ == "__main__":
    raise SystemExit(main())
