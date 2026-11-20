import json
from pathlib import Path

import httpx
import pytest

from repopilot import config
from repopilot.agent.loop import MAX_CALLS_PER_STEP, run_agent
from repopilot.agent.tools import MAX_READ_LINES, RepoTools
from repopilot.ingest.index import build_index
from repopilot.llm.cache import ResponseCache
from repopilot.llm.providers import Provider
from repopilot.llm.router import Completion, Router
from repopilot.retrieval.hybrid import Retriever

LONG = "\n".join(f"    x{i} = {i}" for i in range(200))
SESSION_PY = f'''class Session:
    """A session."""

    def send(self, request):
        r = self.adapter.send(request)
        return dispatch(r)


def helper():
{LONG}
    return 0
'''


@pytest.fixture
def retriever(tmp_path: Path, monkeypatch) -> Retriever:
    monkeypatch.setattr(config, "INDEXES_DIR", tmp_path / "idx")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "session.py").write_text(SESSION_PY)
    (repo / "src" / "hooks.py").write_text("def dispatch(r):\n    # run response hooks\n    return r\n")
    (repo / ".settings.toml").write_text("mode = 'strict'\n")
    (tmp_path / "secret.txt").write_text("TOP SECRET")
    return Retriever(build_index(str(repo), embed=False, name="t"))


# ---- tools -----------------------------------------------------------------------------

def test_read_file_is_scoped_to_the_index(retriever):
    t = RepoTools(retriever)
    assert "not a file in this repository" in t.read_file("../secret.txt")
    assert "not a file in this repository" in t.read_file("/etc/passwd")
    assert "TOP SECRET" not in t.read_file("../secret.txt")
    assert "strict" in t.read_file(".settings.toml")          # leading dot kept, not stripped
    assert "strict" in t.read_file("./.settings.toml")
    assert "Did you mean: src/hooks.py" in t.read_file("hooks.py")
    assert t.seen and t.seen[0].path == ".settings.toml"


def test_read_file_clamps_long_ranges(retriever):
    t = RepoTools(retriever)
    out = t.read_file("src/session.py", 1, 10_000)
    assert f"[src/session.py:1-{MAX_READ_LINES}]" in out and "(file has" in out
    assert "has only" in t.read_file("src/session.py", 9999)


def test_find_definition_merges_split_chunks(retriever):
    t = RepoTools(retriever)
    out = t.find_definition("helper")
    assert out.count("helper") == 1                          # one definition, not one per chunk
    assert "[src/session.py:9-210] helper" in out
    # A method inside a small class (indexed as one chunk) is still found.
    assert "[src/session.py:4-6] Session.send" in t.find_definition("Session.send")
    assert "Session.send" in t.find_definition("send")
    assert "No definition" in t.find_definition("nope")
    assert "Names containing 'help'" in t.find_definition("help") and "helper" in t.find_definition("help")


def test_call_dispatch_handles_bad_input(retriever):
    t = RepoTools(retriever)
    assert "unknown tool" in t.call("rm_rf", {})
    assert "not valid JSON" in t.call("read_file", {"_invalid_json": "{oops"})
    assert "bad arguments" in t.call("read_file", {"wrong": 1})
    assert "src/hooks.py" in t.call("list_files", {"prefix": "src/"})


# ---- loop ------------------------------------------------------------------------------

class ScriptedRouter:
    """Plays back a fixed list of completions and records what it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def complete(self, messages, tools=None, **kw):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        step = self.script.pop(0)
        if isinstance(step, str):
            return Completion(step, "fake", "m", False, {"total_tokens": 100})
        return Completion("", "fake", "m", False, {"total_tokens": 50}, tool_calls=step)


def call(i, tool, **args):
    return {"id": f"c{i}", "name": tool, "arguments": args}


def test_agent_explores_then_answers_with_checked_citations(retriever):
    router = ScriptedRouter([
        [call(1, "find_definition", name="dispatch")],
        [call(2, "read_file", path="src/hooks.py", start_line=1, end_line=3)],
        "Hooks run in dispatch [src/hooks.py:1-3], called from send [src/session.py:4-6]. Also [src/made_up.py:1-2].",
    ])
    ans = run_agent("how are response hooks run?", retriever, router)

    assert [s.get("tool") for s in ans.steps] == ["search_code", "find_definition", "read_file"]
    # hooks.py was read, so citing it is valid. session.py exists but the model was never shown
    # those lines (find_definition lists locations, not code), so that citation is flagged too.
    assert {str(c): c.valid for c in ans.citations} == {
        "src/hooks.py:1-3": True, "src/session.py:4-6": False, "src/made_up.py:1-2": False,
    }
    assert ans.llm_calls == 3 and ans.tokens == 200
    # The model was offered tools on every round, and each tool call got exactly one reply.
    assert all(c["tools"] for c in router.calls)
    last = router.calls[-1]["messages"]
    assert [m["tool_call_id"] for m in last if m["role"] == "tool"] == ["c1", "c2"]
    assert "Initial search results" in last[1]["content"]


def test_agent_is_forced_to_answer_at_step_limit(retriever):
    router = ScriptedRouter([[call(i, "list_files")] for i in range(3)] + ["Done."])
    ans = run_agent("anything", retriever, router, max_steps=3)
    assert ans.text == "Done." and ans.steps[-1] == {"forced_answer": "step limit reached"}
    assert router.calls[-1]["tools"] is None                  # final round offers no tools
    assert "Answer the question now" in router.calls[-1]["messages"][-1]["content"]


def test_agent_caps_parallel_tool_calls_but_answers_every_id(retriever):
    many = [call(i, "list_files") for i in range(MAX_CALLS_PER_STEP + 2)]
    router = ScriptedRouter([many, "ok"])
    ans = run_agent("q", retriever, router)
    tool_msgs = [m for m in router.calls[-1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == len(many)
    assert sum("Skipped" in m["content"] for m in tool_msgs) == 2
    assert sum(s.get("tool") == "list_files" for s in ans.steps) == MAX_CALLS_PER_STEP


# ---- router tool-calling ---------------------------------------------------------------

def test_router_sends_tools_and_parses_and_caches_tool_calls(tmp_path):
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "search_code", "arguments": '{"query": "retry"}'}},
            {"id": "y", "type": "function", "function": {"name": "read_file", "arguments": "{broken"}},
        ]}}]})

    p = Provider("a", "https://a.test/v1", "PATH", "m")
    r = Router(providers=[p], cache=ResponseCache(tmp_path / "c.sqlite"), transport=httpx.MockTransport(handler), sleep=lambda s: None)
    tools = [{"type": "function", "function": {"name": "search_code", "parameters": {}}}]
    c = r.complete([{"role": "user", "content": "q"}], tools=tools)
    assert seen[0]["tools"] == tools and seen[0]["tool_choice"] == "auto"
    assert c.tool_calls[0] == {"id": "x", "name": "search_code", "arguments": {"query": "retry"}}
    assert "_invalid_json" in c.tool_calls[1]["arguments"]
    again = r.complete([{"role": "user", "content": "q"}], tools=tools)
    assert again.cached and again.tool_calls == c.tool_calls and len(seen) == 1
