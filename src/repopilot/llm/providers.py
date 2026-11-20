"""Free-tier providers. All four expose an OpenAI-compatible /chat/completions endpoint,
so one small client serves them all. Model names drift often: override with env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str
    model: str

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.key_env) or None


def _p(name: str, base_url: str, key_env: str, default_model: str) -> Provider:
    model = os.environ.get(f"REPOPILOT_{name.upper()}_MODEL", default_model)
    return Provider(name, base_url, key_env, model)


def default_providers() -> list[Provider]:
    """In fallback order. Only providers with a key set are used."""
    all_providers = [
        _p("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
        _p("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-2.5-flash"),
        _p("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "llama-3.3-70b"),
        _p("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free"),
    ]
    order = os.environ.get("REPOPILOT_PROVIDERS")  # e.g. "gemini,groq"
    if order:
        by_name = {p.name: p for p in all_providers}
        all_providers = [by_name[n.strip()] for n in order.split(",") if n.strip() in by_name]
    return [p for p in all_providers if p.api_key]
