# modal/ingress_slack.py
#
# FastAPI Slack ingress, deployed as a Modal ASGI app.
#
# RESPONSIBILITIES (in order, on every event):
#   1. Receive Slack Events API POST.
#   2. Verify X-Slack-Signature HMAC + timestamp replay window.
#   3. Handle url_verification handshake (one-shot, used once during Slack app setup).
#   4. ACK 200 within 3 seconds — Slack's hard deadline.
#   5. Resolve team_id → workspace_id (multi-tenancy routing).
#   6. Function.spawn() the worker asynchronously; return immediately.
#
# WHY THE WORKER IS SPAWNED, NOT CALLED
# -------------------------------------
# Slack drops the connection if we take >3s to reply. Agent work takes
# seconds-to-minutes. Spawn returns a handle without waiting; the worker
# runs in its own Function container.
#
# RUN
# ---
#   modal serve modal/ingress_slack.py            # local dev: serves the ASGI app on a Modal URL
#   modal deploy modal/ingress_slack.py           # production deploy
#   modal run modal/ingress_slack.py              # smoke-test: signs + posts a synthetic event

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any

import modal
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ValidationError

from common import (  # sibling import; modal/ is intentionally not a package
    DEV_SLACK_SIGNING_SECRET,
    app,
    control_plane_image,
    resolve_workspace_from_env,
    secrets,
)


web_app = FastAPI(title="multiplayer-ai Slack ingress")


# Pydantic envelope models -- extra fields tolerated so Slack's additive
# schema changes don't break validation.


class SlackMessageEvent(BaseModel):
    type: str
    user: str = "U00UNKNOWN"
    channel: str = ""
    channel_type: str = "im"
    text: str = ""
    thread_ts: str | None = None
    ts: str = ""


class SlackEnvelope(BaseModel):
    # `challenge` is only set on url_verification; `event` only on
    # event_callback. Both optional so one model covers both branches.
    type: str
    challenge: str | None = None
    team_id: str = "T01DEV0001"
    event: SlackMessageEvent | None = None


# ---------------------------------------------------------------------------
# HMAC verification helpers
# ---------------------------------------------------------------------------

REPLAY_WINDOW_SECONDS = 60 * 5  # Slack docs recommend 5 minutes


def _signing_secret() -> str:
    """Read from env, fall back to the dev default. Falling back is fine for
    the POC; production deploys override via the Modal Secret."""
    return os.environ.get("SLACK_SIGNING_SECRET", DEV_SLACK_SIGNING_SECRET)


def verify_slack_signature(raw_body: bytes, timestamp: str, signature: str) -> bool:
    """Slack's v0 HMAC scheme.

    Returns True on a match (constant-time compare).
    """
    if not timestamp or not signature:
        return False
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    if abs(time.time() - ts) > REPLAY_WINDOW_SECONDS:
        return False

    base = f"v0:{timestamp}:{raw_body.decode()}".encode()
    expected = (
        "v0=" + hmac.new(_signing_secret().encode(), base, hashlib.sha256).hexdigest()
    )
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# team_id → workspace_id resolution
# ---------------------------------------------------------------------------
#
# In production this is a Convex lookup (the `workspaces` table). For the POC we
# accept that mapping in two ways:
#   - if a known team_id is registered in env (POC_TEAM_MAP="T01DEV0001=dev,..."),
#     use that
#   - else fall back to the single demo workspace "dev"
#
# This keeps the code path identical to production (lookup before spawn)
# without requiring a Convex round-trip for every demo event.


def resolve_workspace_id(team_id: str) -> str:
    """Map a Slack team_id to a workspace_id.

    Production reads from a Convex `workspaces` table keyed by
    `slack_team_id`. POC reads from an env-var map (POC_TEAM_MAP="T01=ws,..")
    and falls back to the single "dev" workspace -- same lookup-before-spawn
    code path, no Convex round-trip on the hot ingress path.
    """
    return resolve_workspace_from_env("POC_TEAM_MAP", team_id)


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


@web_app.post("/slack/events")
async def handle_slack_event(request: Request) -> dict[str, Any]:
    # Slack retries on any non-2xx, so unknown event types ACK + drop
    # rather than 4xx-ing -- avoids amplification on benign garbage.
    raw_body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if not verify_slack_signature(raw_body, timestamp, signature):
        # Distinguish stale-vs-bad-signature only loosely — Slack treats both
        # as 401 and so do we.
        raise HTTPException(status_code=401, detail="Bad signature or stale timestamp")

    try:
        envelope = SlackEnvelope.model_validate_json(raw_body)
    except ValidationError as e:
        # Log structured errors server-side; return a generic detail so we
        # don't leak Pydantic error URLs to internet callers.
        print(f"[slack ingress] envelope validation failed: {e.errors()}")
        raise HTTPException(status_code=400, detail="invalid envelope")

    # The one-time challenge Slack sends when configuring the Events API URL.
    if envelope.type == "url_verification":
        return {"challenge": envelope.challenge or ""}

    if envelope.type != "event_callback" or envelope.event is None:
        # Anything we don't understand: acknowledge but don't dispatch. Slack
        # will retry on non-2xx, which would amplify load.
        return {"ok": True, "ignored": envelope.type}

    inner = envelope.event
    if inner.type != "message":
        # We only handle message events in the POC. Same ACK-and-drop logic.
        return {"ok": True, "ignored_event": inner.type}

    workspace_id = resolve_workspace_id(envelope.team_id)

    # Build the structured context the worker consumes. Crucial design point:
    # downstream code never sees the raw Slack envelope. Adding Teams (or any
    # other channel) means parsing a different envelope into the SAME shape.
    context = {
        "workspace_id": workspace_id,
        "channel_origin": "slack",
        "user_id": inner.user,
        "channel": inner.channel,
        "channel_type": inner.channel_type,
        "message": inner.text,
        "thread_ts": inner.thread_ts,
        "ts": inner.ts,
    }

    # .spawn.aio so the control-plane RPC doesn't block the FastAPI loop and
    # miss Slack's 3s ACK under burst load.
    #
    # The sibling import only resolves when both Functions are served from
    # one `modal serve modal/serve_all.py`. Split serves/deploys: swap to
    # `modal.Function.from_name("multiplayer-ai", "agent_worker")`.
    from worker import agent_worker

    await agent_worker.spawn.aio(context)

    return {"ok": True, "workspace_id": workspace_id}


@web_app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "service": "ingress_slack"}


# ---------------------------------------------------------------------------
# Modal Function declaration — the ASGI app
# ---------------------------------------------------------------------------


@app.function(
    image=control_plane_image,
    secrets=secrets(),
    # 30s is plenty — the only thing happening here is signature verification
    # and a Function.spawn() call. The slow work runs in the worker.
    timeout=30,
)
@modal.asgi_app()
def slack_ingress() -> FastAPI:
    """ASGI entrypoint. Modal turns this into a public *.modal.run URL.

    The ingress → worker boundary is documented on `worker.agent_worker`
    (sole owner of the Modal Queue divergence).
    """
    return web_app


# ---------------------------------------------------------------------------
# Local entrypoint — smoke test
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def smoke_slack_ingress() -> None:
    """`modal run modal/ingress_slack.py` exercises the verification path.

    We don't actually POST to the deployed endpoint here (that would require
    knowing the URL); we just confirm the signing math works against the
    harness's signing function. End-to-end testing uses harness/send_event.py
    against the served URL.
    """
    import urllib.error
    import urllib.request

    # Build a tiny synthetic event and sign it the same way the harness does.
    timestamp = str(int(time.time()))
    body_dict = {"type": "url_verification", "challenge": "chal_local"}
    raw = json.dumps(body_dict).encode()
    base = f"v0:{timestamp}:{raw.decode()}".encode()
    sig = "v0=" + hmac.new(_signing_secret().encode(), base, hashlib.sha256).hexdigest()

    ok = verify_slack_signature(raw, timestamp, sig)
    print(f"local HMAC self-check: {'ok' if ok else 'FAILED'}")

    # If the user has SLACK_INGRESS_URL set, POST against it for real.
    url = os.environ.get("SLACK_INGRESS_URL")
    if url:
        req = urllib.request.Request(
            url,
            data=raw,
            headers={
                "Content-Type": "application/json",
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": sig,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"POST {url} -> {resp.status} {resp.read().decode()[:200]}")
        except urllib.error.HTTPError as e:
            print(f"POST {url} -> {e.code} {e.read().decode()[:200]}")
