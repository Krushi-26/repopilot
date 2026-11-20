import httpx
import pytest

from repopilot.llm.cache import ResponseCache
from repopilot.llm.providers import Provider
from repopilot.llm.router import AllProvidersFailed, Router

MSGS = [{"role": "user", "content": "hi"}]


def ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 3}})


def make_router(tmp_path, handler, providers=("a", "b")):
    provs = [Provider(n, f"https://{n}.test/v1", "X", "m") for n in providers]
    for p in provs:
        object.__setattr__(p, "key_env", "PATH")  # any env var that is set
    return Router(
        providers=provs,
        cache=ResponseCache(tmp_path / "c.sqlite"),
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
        max_retries=2,
    )


def test_retries_429_then_succeeds(tmp_path):
    calls = []

    def handler(req):
        calls.append(req.url.host)
        return httpx.Response(429) if len(calls) < 3 else ok("hello")

    r = make_router(tmp_path, handler)
    c = r.complete(MSGS)
    assert c.text == "hello" and c.provider == "a" and calls == ["a.test"] * 3


def test_falls_back_to_next_provider(tmp_path):
    def handler(req):
        return httpx.Response(401) if req.url.host == "a.test" else ok("from b")

    c = make_router(tmp_path, handler).complete(MSGS)
    assert (c.text, c.provider) == ("from b", "b")


def test_cache_avoids_second_call(tmp_path):
    calls = []

    def handler(req):
        calls.append(1)
        return ok("x")

    r = make_router(tmp_path, handler)
    assert not r.complete(MSGS).cached
    assert r.complete(MSGS).cached
    assert len(calls) == 1


def test_all_fail_raises(tmp_path):
    with pytest.raises(AllProvidersFailed):
        make_router(tmp_path, lambda req: httpx.Response(503)).complete(MSGS)
