"""Minimal GitHub REST client: fetch a PR's diff and publish a review without piling up noise.

Each push to a PR re-runs the bot, so posting is idempotent:
- The summary is ONE sticky PR comment, found by a hidden marker and edited in place.
- Inline findings are posted only if the bot hasn't already left the same text on the same
  file. Matching ignores the line number, so a finding that moved isn't repeated.

The bot only needs `pull-requests: write` and never approves or requests changes, so it can't
block or approve a merge even if the model is manipulated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from repopilot.review.reviewer import Review, ReviewComment

API = "https://api.github.com"
SUMMARY_MARKER = "<!-- repopilot:summary -->"


class GitHubError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status


@dataclass
class PostResult:
    summary_action: str                      # "created" or "updated"
    posted: list[ReviewComment] = field(default_factory=list)
    already_posted: list[ReviewComment] = field(default_factory=list)
    folded: list[ReviewComment] = field(default_factory=list)  # GitHub refused the line; shown in summary


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

    def _paginate(self, url: str) -> list[dict]:
        items, page = [], 1
        while True:
            batch = self._check(self.client.get(url, params={"per_page": 100, "page": page})).json()
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1

    def pr(self, number: int) -> dict:
        return self._check(self.client.get(f"/repos/{self.repo}/pulls/{number}")).json()

    def pr_diff(self, number: int) -> str:
        r = self.client.get(f"/repos/{self.repo}/pulls/{number}", headers={"Accept": "application/vnd.github.diff"})
        return self._check(r).text

    def post_review(self, number: int, commit_sha: str, review: Review) -> PostResult:
        # 1. Inline comments: skip findings the bot already left on this PR.
        existing = {
            (c["path"], c["body"])
            for c in self._paginate(f"/repos/{self.repo}/pulls/{number}/comments")
            if _is_bot(c)
        }
        result = PostResult(summary_action="")
        for c in review.comments:
            body = comment_body(c)
            if (c.path, body) in existing:
                result.already_posted.append(c)
                continue
            r = self.client.post(
                f"/repos/{self.repo}/pulls/{number}/comments",
                json={"body": body, "commit_id": commit_sha, "path": c.path, "line": c.line, "side": "RIGHT"},
            )
            if r.status_code == 422:  # line no longer in the diff (force-push race): keep it in the summary
                result.folded.append(c)
                continue
            self._check(r)
            existing.add((c.path, body))
            result.posted.append(c)

        # 2. Sticky summary: edit the bot's previous summary comment, or create it.
        body = summary_body(review, commit_sha, result)
        previous = next(
            (c for c in self._paginate(f"/repos/{self.repo}/issues/{number}/comments")
             if _is_bot(c) and SUMMARY_MARKER in (c.get("body") or "")),
            None,
        )
        if previous:
            self._check(self.client.patch(f"/repos/{self.repo}/issues/comments/{previous['id']}", json={"body": body}))
            result.summary_action = "updated"
        else:
            self._check(self.client.post(f"/repos/{self.repo}/issues/{number}/comments", json={"body": body}))
            result.summary_action = "created"
        return result


def _is_bot(comment: dict) -> bool:
    # Only ever edit or match our own comments: a human could paste the marker into theirs.
    return (comment.get("user") or {}).get("type") == "Bot"


def comment_body(c: ReviewComment) -> str:
    icon = {"bug": "🐛", "security": "🔒", "risk": "⚠️", "suggestion": "💡"}.get(c.severity, "")
    origin = " · automated check" if c.source == "check" else ""
    return f"{icon} **{c.severity}**{origin}\n\n{c.body}"


def summary_body(review: Review, commit_sha: str, result: PostResult) -> str:
    parts = [SUMMARY_MARKER, review.to_markdown()]
    if result.folded:
        parts.append("**Findings on lines GitHub couldn't anchor:**\n" + "\n".join(
            f"- `{c.path}:{c.line}` {comment_body(c)}" for c in result.folded
        ))
    status = [f"updated for `{commit_sha[:7]}`"]
    if result.posted:
        status.append(f"{len(result.posted)} new inline comment{'s' if len(result.posted) != 1 else ''}")
    if result.already_posted:
        status.append(f"{len(result.already_posted)} already posted")
    parts.append("<sub>" + " · ".join(status) + "</sub>")
    return "\n\n".join(parts)
