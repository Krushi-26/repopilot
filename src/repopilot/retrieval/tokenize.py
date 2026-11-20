"""Code-aware tokenizer for keyword search.

`getHTTPRetry` and `get_http_retry` should both match a query about "http retry", so
identifiers are split on snake_case and camelCase, and the whole identifier is kept too.
"""

from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on", "with", "as", "be",
    "this", "that", "are", "by", "at", "from", "how", "does", "do", "what", "where", "which",
    "when", "why", "i", "we", "you", "self", "return", "if", "else", "not", "none", "true",
    "false", "def", "class", "import",
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for ident in _IDENT.findall(text):
        parts = [p.lower() for piece in ident.split("_") if piece for p in _CAMEL.findall(piece)]
        if len(parts) > 1:  # compound identifier: keep the whole name as well as its parts
            tokens.append(ident.lower().strip("_"))
        tokens.extend(p for p in parts if p not in STOPWORDS and len(p) > 1)
    return tokens
