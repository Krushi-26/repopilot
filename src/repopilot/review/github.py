"""Minimal GitHub REST client: fetch a PR's diff and post one review with inline comments.

The bot only ever needs `pull-requests: write`. It posts a COMMENT review (never APPROVE or
REQUEST_CHANGES), so it can't block or approve a merge even if the model is manipulated.
"""

from __future__ import annotations

import httpx

from repopilot.review.reviewer import Review

API = "https://api.github.com"


class GitHubError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status


class GitHub:
    def __init__(self, token: str | None, repo: str, transport: httpx.BaseTransport | None = None):
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "repopilot"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.repo = repo
        self.client = httpx.Client(base_url=API, headers=headers, timeout=30, transport=transport)

    def _check(self, r: httpx.Response) -> httpx.Response:
        if r.status_code >= 400:
            raise GitHubError(r.status_code, r.text[:300])
        return r

    def pr(self, number: int) -> dict:
        return self._check(self.client.get(f"/repos/{self.repo}/pulls/{number}")).json()

    def pr_diff(self, number: int) -> str:
        r = self.client.get(f"/repos/{self.repo}/pulls/{number}", headers={"Accept": "application/vnd.github.diff"})
        return self._check(r).text

    def post_review(self, number: int, commit_sha: str, review: Review) -> dict:
        comments = [
            {"path": c.path, "line": c.line, "side": "RIGHT", "body": _comment_body(c)}
            for c in review.comments
        ]
        payload = {"commit_id": commit_sha, "event": "COMMENT", "body": review.to_markdown(), "comments": comments}
        r = self.client.post(f"/repos/{self.repo}/pulls/{number}/reviews", json=payload)
        if r.status_code == 422 and comments:
            # A line GitHub won't anchor (e.g. diff changed under us): keep the findings in the body.
            payload["comments"] = []
            payload["body"] += "\n\n" + "\n".join(f"- `{c.path}:{c.line}` {_comment_body(c)}" for c in review.comments)
            r = self.client.post(f"/repos/{self.repo}/pulls/{number}/reviews", json=payload)
        return self._check(r).json()


def _comment_body(c) -> str:
    icon = {"bug": "🐛", "security": "🔒", "risk": "⚠️", "suggestion": "💡"}.get(c.severity, "")
    origin = " · automated check" if c.source == "check" else ""
    return f"{icon} **{c.severity}**{origin}\n\n{c.body}"
