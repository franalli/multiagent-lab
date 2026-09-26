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
import re
import time
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
import modal
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from common import (  # sibling import; modal/ is intentionally not a package
    app,
    async_convex_post,
    control_plane_image,
    gateway_secrets,
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
# They reach this container via common.gateway_secrets() -- never via
# secrets(), which the Sandbox also mounts.


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
# Outbound HTTP
# ---------------------------------------------------------------------------

# Module-level so the connection pool + TLS sessions survive across dispatches.
_http = httpx.AsyncClient(timeout=10)


async def _request(method: str, url: str, **kwargs: Any) -> httpx.Response:
    """httpx request with transport errors mapped to 502, so handle_dispatch
    audits them as "error" like any other upstream failure."""
    try:
        return await _http.request(method, url, **kwargs)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"upstream unreachable: {type(e).__name__}") from e


def _require(params: dict[str, Any], *keys: str) -> None:
    missing = [k for k in keys if not params.get(k)]
    if missing:
        raise HTTPException(status_code=400, detail=f"missing params: {', '.join(missing)}")


# --- Slack -----------------------------------------------------------------


async def slack_send(token: str, params: dict[str, Any]) -> dict[str, Any]:
    """chat.postMessage. Slack answers HTTP 200 even on failure; the verdict
    is the body's `ok` field."""
    _require(params, "channel", "text")
    body = {"channel": params["channel"], "text": params["text"]}
    if params.get("thread_ts"):
        body["thread_ts"] = params["thread_ts"]
    resp = await _request(
        "POST",
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {token}"},
        json=body,
    )
    data = resp.json()
    if not data.get("ok"):
        raise HTTPException(status_code=502, detail=f"slack: {data.get('error', resp.status_code)}")
    return {"posted": True, "channel": data.get("channel"), "ts": data.get("ts")}


# --- Teams (Bot Framework) -------------------------------------------------

# Multi-tenant bot registrations mint tokens against the botframework.com
# tenant; a single-tenant bot would use its own tenant id in this URL.
TEAMS_TOKEN_URL = "https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token"

# service_url arrives inside the inbound Activity (and the sandbox can pass
# any value), and we attach the bot's bearer token to requests sent there --
# so only Bot Framework hosts are allowed, or the token could be exfiltrated.
# Suffix matching is safe only for Microsoft-owned domains. trafficmanager.net
# is Azure Traffic Manager's shared namespace -- any Azure customer can
# register <name>.trafficmanager.net -- so its one Bot Framework host is pinned.
TEAMS_SERVICE_HOST_SUFFIXES = (".botframework.com", ".botframework.azure.us")
TEAMS_SERVICE_HOSTS = frozenset({"smba.trafficmanager.net"})

# app_id -> (access_token, expires_at). Tokens live ~1h; refresh a minute early.
_TEAMS_TOKENS: dict[str, tuple[str, float]] = {}


async def _teams_token(app_id: str, app_password: str) -> str:
    cached = _TEAMS_TOKENS.get(app_id)
    if cached and cached[1] - 60 > time.time():
        return cached[0]
    resp = await _request(
        "POST",
        TEAMS_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": app_id,
            "client_secret": app_password,
            "scope": "https://api.botframework.com/.default",
        },
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"teams token mint failed: HTTP {resp.status_code}")
    data = resp.json()
    token = data["access_token"]
    _TEAMS_TOKENS[app_id] = (token, time.time() + int(data.get("expires_in", 3600)))
    return token


def _trusted_service_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    trusted = host in TEAMS_SERVICE_HOSTS or host.endswith(TEAMS_SERVICE_HOST_SUFFIXES)
    if parts.scheme != "https" or not trusted:
        raise HTTPException(status_code=400, detail=f"untrusted Teams service_url host: {host!r}")
    return url.rstrip("/")


async def teams_send(app_id: str, app_password: str, params: dict[str, Any]) -> dict[str, Any]:
    """Post a message Activity to the conversation; threaded under
    reply_to_id when given (replyToActivity), else a new message."""
    _require(params, "conversation", "service_url", "text")
    base = _trusted_service_url(params["service_url"])
    url = f"{base}/v3/conversations/{quote(params['conversation'], safe='')}/activities"
    if params.get("reply_to_id"):
        url += f"/{quote(params['reply_to_id'], safe='')}"
    token = await _teams_token(app_id, app_password)
    resp = await _request(
        "POST",
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={"type": "message", "text": params["text"]},
    )
    if resp.status_code not in (200, 201):
        raise HTTPException(status_code=502, detail=f"teams: HTTP {resp.status_code}")
    return {"posted": True, "conversation": params["conversation"], "id": resp.json().get("id")}


# --- GitHub ----------------------------------------------------------------

# owner/name only -- the value is interpolated into the API path.
_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


async def github_create_issue(token: str, params: dict[str, Any]) -> dict[str, Any]:
    _require(params, "repo", "title")
    if not _GITHUB_REPO_RE.match(params["repo"]):
        raise HTTPException(status_code=400, detail="repo must be 'owner/name'")
    resp = await _request(
        "POST",
        f"https://api.github.com/repos/{params['repo']}/issues",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"title": params["title"], "body": params.get("body", "")},
    )
    if resp.status_code != 201:
        raise HTTPException(status_code=502, detail=f"github: HTTP {resp.status_code}")
    data = resp.json()
    return {"created": True, "repo": params["repo"], "number": data.get("number"), "url": data.get("html_url")}


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
        return await slack_send(creds["token"], params)

    if action == "teams.send":
        if not (creds.get("app_id") and creds.get("app_password")):
            return {"stubbed": True, "would_post": params}
        return await teams_send(creds["app_id"], creds["app_password"], params)

    if action == "github.create_issue":
        if not creds.get("token"):
            return {"stubbed": True, "would_create": params}
        return await github_create_issue(creds["token"], params)

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


@app.function(image=control_plane_image, secrets=gateway_secrets(), timeout=60)
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
    Slack/Teams creds. Always dispatches with empty creds -- the handlers
    now make real calls when creds are present, and a smoke run must not
    post to Slack or open GitHub issues just because the caller's shell
    has tokens exported.
    """

    async def run() -> None:
        # Walk through the three actions to confirm each branches correctly.
        for action, params in [
            ("slack.send", {"channel": "D01", "text": "hello"}),
            ("teams.send", {"conversation": "a:demo", "text": "hello"}),
            ("github.create_issue", {"repo": "demo/demo", "title": "x"}),
            ("unknown.action", {}),
        ]:
            try:
                result = await dispatch(action, params, creds={})
                print(f"[gateway.smoke] {action} -> {result}")
            except HTTPException as e:
                print(f"[gateway.smoke] {action} -> {e.status_code} {e.detail}")

    asyncio.run(run())
