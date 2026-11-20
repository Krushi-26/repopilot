"""Model router: cache -> provider 1 (with retries) -> provider 2 -> ...

- 429 / 5xx / network errors: retry with exponential backoff + jitter (honours Retry-After).
- Still failing, or 401/403/404 (bad key, unknown model): fall through to the next provider.
- Every call is logged to a JSONL trace with latency, tokens and which provider answered.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field

import httpx

from repopilot import config
from repopilot.llm.cache import ResponseCache
from repopilot.llm.providers import Provider, default_providers

RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


class AllProvidersFailed(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    provider: str
    model: str
    cached: bool
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    # Normalised tool calls: [{"id": str, "name": str, "arguments": dict}]
    tool_calls: list[dict] = field(default_factory=list)


class Router:
    def __init__(
        self,
        providers: list[Provider] | None = None,
        cache: ResponseCache | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,  # injectable for tests
        sleep=time.sleep,
    ):
        self.providers = providers if providers is not None else default_providers()
        self.cache = cache if cache is not None else ResponseCache(config.CACHE_PATH)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.client = httpx.Client(timeout=timeout, transport=transport)
        self.sleep = sleep

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        use_cache: bool = True,
        tools: list[dict] | None = None,
    ) -> Completion:
        """`tools`: OpenAI-format function specs. The model may answer with tool_calls instead of text."""
        if not self.providers:
            raise AllProvidersFailed(
                "No LLM provider configured. Set GROQ_API_KEY (free at console.groq.com) in .env."
            )
        params: dict = {"temperature": temperature, "max_tokens": max_tokens}
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        errors: list[str] = []
        for p in self.providers:
            key = ResponseCache.key(f"{p.name}/{p.model}", messages, params)
            if use_cache and (hit := self.cache.get(key)):
                return Completion(hit["text"], p.name, p.model, True, hit.get("usage", {}), tool_calls=hit.get("tool_calls", []))
            try:
                result = self._call_with_retry(p, messages, params)
            except _ProviderError as e:
                errors.append(f"{p.name}: {e}")
                continue
            self.cache.put(key, p.name, p.model, {"text": result.text, "usage": result.usage, "tool_calls": result.tool_calls})
            return result
        raise AllProvidersFailed("All providers failed:\n  " + "\n  ".join(errors))

    def _call_with_retry(self, p: Provider, messages: list[dict], params: dict) -> Completion:
        last = ""
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self.client.post(
                    f"{p.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {p.api_key}"},
                    json={"model": p.model, "messages": messages, **params},
                )
            except httpx.HTTPError as e:
                last = f"network error: {e}"
                status = None
            else:
                status = resp.status_code
                if status == 200:
                    data = resp.json()
                    msg = data["choices"][0]["message"]
                    c = Completion(
                        msg.get("content") or "", p.name, p.model, False, data.get("usage", {}), time.time() - t0,
                        tool_calls=_parse_tool_calls(msg.get("tool_calls")),
                    )
                    _trace(c)
                    return c
                last = f"HTTP {status}: {resp.text[:200]}"
                if status not in RETRYABLE:
                    break  # auth / bad request / unknown model: retrying won't help
            if attempt < self.max_retries:
                self.sleep(self._delay(attempt, resp if status else None))
        _trace_failure(p, last)
        raise _ProviderError(last)

    def _delay(self, attempt: int, resp: httpx.Response | None) -> float:
        if resp is not None and (ra := resp.headers.get("retry-after")):
            try:
                return min(float(ra), self.max_delay)
            except ValueError:
                pass
        backoff = min(self.base_delay * 2**attempt, self.max_delay)
        return backoff * (0.5 + random.random() / 2)  # jitter: spread out retries


def _parse_tool_calls(raw) -> list[dict]:
    out = []
    for i, tc in enumerate(raw or []):
        fn = tc.get("function") or {}
        args = fn.get("arguments") or "{}"
        try:
            args = json.loads(args) if isinstance(args, str) else dict(args)
        except (json.JSONDecodeError, TypeError, ValueError):
            args = {"_invalid_json": str(args)[:500]}
        out.append({"id": tc.get("id") or f"call_{i}", "name": fn.get("name", ""), "arguments": args})
    return out


class _ProviderError(Exception):
    pass


def _trace(c: Completion) -> None:
    _write_trace({"ok": True, "provider": c.provider, "model": c.model, "latency_s": round(c.latency_s, 3), "usage": c.usage})


def _trace_failure(p: Provider, error: str) -> None:
    _write_trace({"ok": False, "provider": p.name, "model": p.model, "error": error})


def _write_trace(record: dict) -> None:
    try:
        config.TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with config.TRACE_PATH.open("a") as f:
            f.write(json.dumps({"ts": time.time(), **record}) + "\n")
    except OSError:
        pass
