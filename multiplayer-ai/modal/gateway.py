# modal/gateway.py
#
# The tool gateway. The ONLY egress path from the sandbox.
#
# WHY THIS MATTERS (the trial talking point):
#   * Sandwich isolation -- the sandbox has no outbound network policy
#     beyond "speak HTTP to the gateway." That gives us one place to:
#       - authenticate per-workspace (look up credentials, never expose them
#         inside the sandbox)
#       - rate-limit per workspace and per action
#       - audit every external call (writes to Convex audit_log)
#       - swap implementations (move slack.send from web API to webhook URL,
#         add new integrations) without touching the agent loop
#
# The dispatcher is intentionally a flat if/elif tree. Action namespaces are
# "<integration>.<verb>", e.g. "slack.send", "github.create_issue".

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import modal
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from common import (  # sibling import; modal/ is intentionally not a package
    app,
    async_convex_post,
    control_plane_image,
    secrets,
)

gateway_app = FastAPI(title="multiplayer-ai tool gateway")


# ---------------------------------------------------------------------------
# Audit + rate limit (POC stubs)
# ---------------------------------------------------------------------------


async def audit(workspace_id: str, action: str, params: dict[str, Any], status: str) -> None:
    """Write one row to Convex audit_log. Keep params compact (truncate)."""
    compact = json.dumps(params)[:512]
    await async_convex_post(
        "/api/audit/log",
        {
            "workspace_id": workspace_id,
            "action": action,
            "params_summary": compact,
            "actor": "gateway",
            "status": status,
        },
    )


# In-process token-bucket (per workspace, per action). Resets on container restart.
_RATE_BUCKETS: dict[tuple[str, str], tuple[int, float]] = {}
RATE_LIMIT_WINDOW_S = 60
RATE_LIMIT_MAX = 30


def rate_limit_ok(workspace_id: str, action: str) -> bool:
    """In-process token bucket: at most RATE_LIMIT_MAX dispatches per
    (workspace, action) per RATE_LIMIT_WINDOW_S. Resets when the
    container restarts -- acceptable for POC; production would back this
    with Convex or a shared Redis."""
    key = (workspace_id, action)
    now = time.time()
    count, started = _RATE_BUCKETS.get(key, (0, now))
    if now - started > RATE_LIMIT_WINDOW_S:
        count, started = 0, now
    count += 1
    _RATE_BUCKETS[key] = (count, started)
    return count <= RATE_LIMIT_MAX


# ---------------------------------------------------------------------------
# Per-workspace credentials (POC stub)
# ---------------------------------------------------------------------------
#
# In production these live in a vault / Modal Secret per workspace.
# For the POC we read from env and label-prefix by action namespace:
#   SLACK_BOT_TOKEN, TEAMS_BOT_APP_PASSWORD, GITHUB_TOKEN, ...


def credentials_for(workspace_id: str, action: str) -> dict[str, str]:
    """Look up per-workspace + per-namespace credentials for the dispatched action.

    Production path: resolve OAuth tokens via Pipedream Connect per
    (workspace, user, integration) -- tokens never enter the sandbox image.
    POC reads namespace-prefixed env vars (SLACK_BOT_TOKEN, TEAMS_BOT_APP_ID,
    GITHUB_TOKEN). The signature is per-workspace-ready; swap the body to
    a vault/Pipedream lookup without touching dispatch().
    """
    namespace = action.split(".", 1)[0]
    if namespace == "slack":
        return {"token": os.environ.get("SLACK_BOT_TOKEN", "")}
    if namespace == "teams":
        return {
            "app_id": os.environ.get("TEAMS_BOT_APP_ID", ""),
            "app_password": os.environ.get("TEAMS_BOT_APP_PASSWORD", ""),
        }
    if namespace == "github":
        return {"token": os.environ.get("GITHUB_TOKEN", "")}
    return {}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


async def dispatch(action: str, params: dict[str, Any], creds: dict[str, str]) -> dict[str, Any]:
    """Route by action namespace. Each handler is small and side-effecting.

    Handlers stub the external call when the relevant creds are absent so
    the gateway is usable in dev without provisioning everything.

    Production path: broker Pipedream Connect's 3000+ integrations here.
    Adding a namespace in POC = one elif + credentials_for case.
    """
    if action == "slack.send":
        if not creds.get("token"):
            return {"stubbed": True, "would_post": params}
        # POC stub -- real impl POSTs https://slack.com/api/chat.postMessage.
        return {"posted": True, "channel": params.get("channel")}

    if action == "teams.send":
        if not (creds.get("app_id") and creds.get("app_password")):
            return {"stubbed": True, "would_post": params}
        # POC stub -- real impl POSTs a Bot Framework Activity to service_url.
        return {"posted": True, "conversation": params.get("conversation")}

    if action == "github.create_issue":
        if not creds.get("token"):
            return {"stubbed": True, "would_create": params}
        return {"created": True, "repo": params.get("repo")}

    raise HTTPException(status_code=400, detail=f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


@gateway_app.post("/dispatch")
async def handle_dispatch(request: Request, background: BackgroundTasks) -> dict[str, Any]:
    body = await request.json()
    workspace_id = body.get("workspace_id")
    action = body.get("action")
    params = body.get("params", {})
    if not (workspace_id and action):
        raise HTTPException(status_code=400, detail="workspace_id and action required")

    # Audit writes go on BackgroundTasks so they run after the response is
    # returned. audit() is async, so FastAPI schedules it as an asyncio task
    # on the same loop -- no threadpool pressure under 2k-tenant burst.
    if not rate_limit_ok(workspace_id, action):
        background.add_task(audit, workspace_id, action, params, "denied")
        raise HTTPException(status_code=429, detail="rate limited")

    creds = credentials_for(workspace_id, action)
    try:
        result = await dispatch(action, params, creds)
        background.add_task(audit, workspace_id, action, params, "ok")
        return {"ok": True, **result}
    except HTTPException:
        background.add_task(audit, workspace_id, action, params, "error")
        raise


@gateway_app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "service": "tool_gateway"}


# ---------------------------------------------------------------------------
# Modal Function declaration
# ---------------------------------------------------------------------------


@app.function(image=control_plane_image, secrets=secrets(), timeout=60)
@modal.asgi_app()
def tool_gateway() -> FastAPI:
    """ASGI entrypoint. Modal serves this at a separate *.modal.run URL.

    The sandbox reads the URL out of TOOL_GATEWAY_URL (an env var the worker
    sets before spawning the sandbox).
    """
    return gateway_app


# ---------------------------------------------------------------------------
# Local entrypoint -- smoke test
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def smoke_gateway() -> None:
    """`modal run modal/gateway.py` exercises a stub dispatch path locally.

    Verifies the routing + rate-limit + audit code without needing real
    Slack/Teams creds.
    """

    async def run() -> None:
        ws = "dev"
        # Walk through the three actions to confirm each branches correctly.
        for action, params in [
            ("slack.send", {"channel": "D01", "text": "hello"}),
            ("teams.send", {"conversation": "a:demo", "text": "hello"}),
            ("github.create_issue", {"repo": "demo/demo", "title": "x"}),
            ("unknown.action", {}),
        ]:
            creds = credentials_for(ws, action)
            try:
                result = await dispatch(action, params, creds)
                print(f"[gateway.smoke] {action} -> {result}")
            except HTTPException as e:
                print(f"[gateway.smoke] {action} -> {e.status_code} {e.detail}")

    asyncio.run(run())
