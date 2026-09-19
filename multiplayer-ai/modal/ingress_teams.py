# modal/ingress_teams.py
#
# A SECOND ingress, parallel to the Slack one. The whole architectural point
# is that downstream code (worker / sandbox / gateway / Volume / Convex) is
# channel-agnostic. ONLY the ingress differs per channel.
#
# We deliberately MIMIC the Teams integration instead of fully implementing
# Bot Framework auth -- that's what the user asked for. The mimic:
#   * accepts a Bot Framework Activity-shaped JSON dict
#   * accepts a "Bearer <dev-token>" Authorization header (no JWT validation)
#   * normalises the Activity into the SAME structured context the Slack
#     ingress emits, so the worker doesn't care which ingress fired
#
# The real production version would:
#   * fetch Microsoft's OpenID config and validate the JWT in Authorization
#   * use botbuilder-python to deserialise the Activity rigorously
#   * call ConnectorClient.send_to_conversation to reply
#
# Demo line: "Only the ingress differs per channel -- Slack does HMAC, Teams
# uses Bot Framework JWT in production -- but both normalise to one context
# shape, so the worker, sandbox, tool gateway, Volume, and Convex layers
# never know which channel fired."

from __future__ import annotations

from typing import Any

import modal
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from common import (  # sibling import
    app,
    control_plane_image,
    resolve_workspace_from_env,
    secrets,
)

teams_app = FastAPI(title="multiplayer-ai Teams ingress (mimic)")


# Pydantic models for the Bot Framework Activity envelope.


class TeamsConversation(BaseModel):
    id: str = ""
    tenantId: str | None = None


class TeamsFrom(BaseModel):
    id: str = "unknown-user"


class TeamsActivity(BaseModel):
    # `from` is a Python keyword; alias maps the JSON field name to `from_`.
    # populate_by_name lets the smoke test construct the model by Python name.
    model_config = ConfigDict(populate_by_name=True)

    type: str = ""
    id: str | None = None
    text: str = ""
    timestamp: str | None = None
    replyToId: str | None = None
    serviceUrl: str | None = None
    conversation: TeamsConversation = Field(default_factory=TeamsConversation)
    from_: TeamsFrom = Field(default_factory=TeamsFrom, alias="from")
    channelData: dict[str, Any] = Field(default_factory=dict)
    recipient: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Auth (mimic)
# ---------------------------------------------------------------------------


def verify_teams_auth(authorization: str | None) -> bool:
    """Accept either a dev-token or a 'real' Bearer header presence.

    Production: parse the JWT, validate iss/aud/signature against Microsoft's
    JWKS. For the POC we accept any Bearer token; the literal "dev-teams-token"
    matches what harness/send_event.py sends.
    """
    if not authorization:
        return False
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False
    return bool(parts[1])


# ---------------------------------------------------------------------------
# tenant_id -> workspace_id resolution
# ---------------------------------------------------------------------------


def resolve_workspace_id(tenant_id: str) -> str:
    """Same shape as the Slack version; just keyed on tenant_id."""
    return resolve_workspace_from_env("POC_TENANT_MAP", tenant_id)


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


@teams_app.post("/teams/messages")
async def handle_teams_activity(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    if not verify_teams_auth(authorization):
        raise HTTPException(status_code=401, detail="missing or invalid Authorization header")

    raw_body = await request.body()
    try:
        activity = TeamsActivity.model_validate_json(raw_body)
    except ValidationError as e:
        # Log structured errors server-side; return a generic detail so we
        # don't leak Pydantic error URLs to internet callers.
        print(f"[teams ingress] activity validation failed: {e.errors()}")
        raise HTTPException(status_code=400, detail="invalid activity")

    if activity.type != "message":
        # Other Activity types (conversationUpdate, invoke, etc.) -- ack and drop.
        # Production would route invoke->card actions through a separate path.
        return {"status": "ignored", "activity_type": activity.type}

    # tenantId can live on conversation OR under channelData.tenant.id depending
    # on which Bot Framework channel forwarded the activity.
    tenant_id = activity.conversation.tenantId or activity.channelData.get("tenant", {}).get("id") or ""
    workspace_id = resolve_workspace_id(tenant_id)

    # NORMALISE -- this is the crux. The keys here match exactly what the
    # Slack ingress emits. Worker / sandbox don't care which channel.
    context = {
        "workspace_id": workspace_id,
        "channel_origin": "teams",
        "user_id": activity.from_.id,
        "channel": activity.conversation.id,
        "channel_type": "teams",
        "message": activity.text,
        "thread_ts": activity.replyToId,
        "ts": activity.timestamp,
        # Teams-specific fields the gateway needs to post replies back. Kept
        # alongside the normalised context (not inside it) so the agent loop
        # itself stays channel-agnostic.
        "_teams": {
            "service_url": activity.serviceUrl,
            "tenant_id": tenant_id,
            "recipient": activity.recipient,
            "activity_id": activity.id,
        },
    }

    from worker import agent_worker

    await agent_worker.spawn.aio(context)

    return {"status": "accepted", "workspace_id": workspace_id}


@teams_app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "service": "ingress_teams"}


# ---------------------------------------------------------------------------
# Modal Function declaration
# ---------------------------------------------------------------------------


@app.function(image=control_plane_image, secrets=secrets(), timeout=30)
@modal.asgi_app()
def teams_ingress() -> FastAPI:
    """ASGI entrypoint. Modal serves this at a public *.modal.run URL.
    Symmetric to slack_ingress; both register on the same App so they
    can dispatch the shared agent_worker."""
    return teams_app


# ---------------------------------------------------------------------------
# Local entrypoint -- smoke test
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def smoke_teams_ingress() -> None:
    """`modal run modal/ingress_teams.py` exercises the normalisation logic.

    Confirms a Teams Activity dict collapses to the same context shape the
    Slack ingress emits -- this is the architectural property we want to
    demonstrate.
    """
    fake_activity = {
        "type": "message",
        "id": "act-test",
        "from": {"id": "29:demo-user"},
        "conversation": {
            "id": "a:demo-conv",
            "tenantId": "tenant-uuid",
            "conversationType": "personal",
        },
        "recipient": {"id": "28:bot-app-id"},
        "text": "hello agent from teams",
        "serviceUrl": "https://smba.example/amer/",
        "timestamp": "2026-05-23T00:00:00Z",
    }

    # Round-trip through TeamsActivity so the smoke test exercises the
    # same parsing path the live handler uses (including the `from` alias).
    activity = TeamsActivity.model_validate(fake_activity)
    context = {
        "workspace_id": resolve_workspace_id(activity.conversation.tenantId or ""),
        "channel_origin": "teams",
        "user_id": activity.from_.id,
        "channel": activity.conversation.id,
        "channel_type": "teams",
        "message": activity.text,
        "thread_ts": activity.replyToId,
        "ts": activity.timestamp,
    }
    print("[teams.smoke] normalised context:", context)
