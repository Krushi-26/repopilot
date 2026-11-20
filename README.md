# RepoPilot

![retrieval](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Krushi-26/repopilot/main/evals/results/badge.json)
![tests](https://github.com/Krushi-26/repopilot/actions/workflows/eval.yml/badge.svg)

An AI teammate that lives in your GitHub repo: ask it about the code and get answers with
file-and-line citations, and (coming next) automatic reviews on every pull request.
Built to run at $0 on free-tier LLM APIs and GitHub Actions.

## Status

| Piece | State |
|---|---|
| Ingest: clone, AST chunking down to methods, skip vendored files | ✅ |
| Hybrid retrieval: BM25 + local embeddings, weighted RRF, path prior | ✅ |
| Q&A with citations, citations verified in code | ✅ |
| LLM layer: retry with backoff, SQLite cache, provider fallback, JSONL traces | ✅ |
| Reranker | ⏳ next |
| Golden eval set (45 questions) + CI gate + badge | ✅ |
| Agent tool loop (search / open file / read diff) | ⏳ week 3 |
| PR review bot (GitHub Action) | ⏳ week 3 |
| MCP server, web demo | stretch |

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[dev,vector]'     # vector extra ≈ 120 MB + 64 MB model; drop it for BM25-only
cp .env.example .env                  # add GROQ_API_KEY (free: console.groq.com/keys)

repopilot index https://github.com/psf/requests --ref v2.32.3
repopilot search "where are HTTP retries configured"     # retrieval only, no API call
repopilot ask "How does Session handle redirects?"
```

## How it works

```
repo ──clone──▶ chunk (functions/classes via ast; line windows otherwise)
                 │
                 ├──▶ BM25 over code-aware tokens (getHTTPRetry → get, http, retry)
                 └──▶ bge-small embeddings (local ONNX, no API)
                          │
question ──▶ hybrid search ── RRF fusion ──▶ top-k excerpts with line numbers
                          │
                          ▼
              router: cache → Groq → Gemini → Cerebras → OpenRouter
                          │
                          ▼
              answer + [path:start-end] citations, each checked against the retrieved code
```

Indexes, clones, the response cache and traces live in `.repopilot/` (gitignored); in CI
the index becomes a build artifact.

## Evaluation

[`evals/golden_requests.json`](evals/golden_requests.json) holds 45 hand-written questions about
[psf/requests](https://github.com/psf/requests) pinned at `v2.32.3`, each labelled with the
function or class that answers it (`sessions.py → Session.get_adapter`). Labels name symbols,
not line numbers: the scorer resolves them with `ast`, so a stale label fails loudly.

| Metric | Meaning |
|---|---|
| file hit@k | an expected file is in the top k chunks |
| span hit@k | a top-k chunk overlaps the expected function's lines (the headline number) |
| MRR@10 | 1 / rank of the first correct chunk, averaged |
| citation accuracy | share of answer citations that point inside retrieved code (`--answers`) |
| grounded rate | share of answers citing the expected code (`--answers`) |

```bash
repopilot eval                 # retrieval only: free, deterministic, runs in CI
repopilot eval --answers       # also generates answers and scores citations (uses the LLM)
```

### Results log

Change one thing at a time; every row is a full run on the same pinned commit and keeps the
changes from the rows above it.

| # | Change | span hit@1 | span hit@5 | span hit@10 | MRR@10 |
|---|---|---|---|---|---|
| 0 | Baseline: BM25, AST chunks for top-level defs, 60-line windows elsewhere | 31% | 58% | 71% | 0.411 |
| 1 | Large classes split into one chunk per method (+ header chunk) | 29% | 60% | 73% | 0.411 |
| 2 | Path prior: demote tests ×0.5, docs ×0.6, changelog ×0.2, license ×0.1 | 47% | 76% | 82% | 0.578 |
| 3 | Vector only: bge-small-en-v1.5 (local ONNX) | 62% | 87% | 96% | 0.729 |
| 4 | Hybrid, equal-weight RRF | 51% | 91% | 96% | 0.684 |
| 5 | **Hybrid, RRF weights bm25 1.0 / vector 1.5 (current default)** | **58%** | **91%** | **96%** | **0.712** |

Notes:
- **Row 0 misses:** tests, docs, `HISTORY.md` and even `LICENSE` outranked source code, which row 2 fixed.
  The prior weights were set up front, not tuned on these questions.
- **Rows 3-5:** hybrid finds more of the right code in the top 5, while vectors alone rank it first
  slightly more often. A weight sweep (1.0 → 3.0) took span hit@1 from 51% to 58% at 1.25 and then
  plateaued; 1.5 was kept rather than chasing the 60% at 3.0, which is one question.
- **Known bias:** questions avoid exact identifiers, which favours vectors. With 45 questions,
  one question is about 2 points, so treat differences under ~5 points as noise.
- **Remaining misses:** SOCKS proxies, response hooks, cookie extraction, IDNA encoding. Next
  step is a reranker over the top 20.

## Development

```bash
pytest -q
```
