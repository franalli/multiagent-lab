---
name: integration-github
description: How to call GitHub via the tool gateway. Auth, rate-limits, and common operations.
---

# Integration skill: GitHub

The agent NEVER calls api.github.com directly. All GitHub operations go
through the tool gateway with action prefix `github.*`.

## Available actions
- `github.create_issue` -- params: `repo`, `title`, `body`, `labels?`
- (future) `github.create_pr`, `github.comment_on_issue`

## Auth
The gateway holds the per-workspace GitHub token (Modal Secret). The
sandbox never sees it. If a call returns 401, escalate to the workspace
admin rather than attempting to refresh the token from inside the sandbox.

## Rate limits
GitHub allows 5000 requests/hour per token. The gateway enforces a
conservative per-workspace cap of 30 actions/minute regardless of
namespace, so issue-spam is bounded.
