"""Path prior: how likely a file is to *implement* something, independent of the query.

Questions are usually "how/where does X work", which source code answers. Tests, docs and
changelogs mention the same words constantly, so on raw term overlap they crowd out the
implementation. A multiplicative prior demotes them without hiding them: a test still wins
when it is clearly the best match, and questions about tests or docs still find them.
"""

from __future__ import annotations

import re

RULES: list[tuple[re.Pattern, float]] = [
    (re.compile(r"(^|/)(LICENSE|LICENCE|COPYING|NOTICE|AUTHORS)(\.\w+)?$", re.I), 0.1),
    (re.compile(r"(^|/)(HISTORY|CHANGELOG|CHANGES|NEWS|RELEASE_NOTES)(\.\w+)?$", re.I), 0.2),
    (re.compile(r"(^|/)(tests?|testing|spec|__tests__)/|(^|/)test_[^/]*$|_test\.\w+$|\.spec\.\w+$"), 0.5),
    (re.compile(r"(^|/)(docs?|examples?)/|\.(md|rst|txt)$", re.I), 0.6),
    (re.compile(r"\.(json|ya?ml|toml|ini|cfg)$|(^|/)(setup\.py|conftest\.py|noxfile\.py)$", re.I), 0.7),
]


def path_prior(path: str) -> float:
    for pattern, weight in RULES:
        if pattern.search(path):
            return weight
    return 1.0
