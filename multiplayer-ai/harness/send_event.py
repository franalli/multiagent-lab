# harness/send_event.py
#
# Slack-shaped (and Teams-shaped) test harness.
#
# WHY THIS EXISTS
# ---------------
# The POC needs to be demoable end-to-end *without* a real Slack workspace or
# a real Bot Framework registration. This script POSTs correctly-formatted
# events at the ingress endpoint — same payload shape, same headers, same
# HMAC signature — so the rest of the pipeline (ingress → worker → sandbox →
# gateway → Convex) exercises exactly the path Slack/Teams would trigger.
#
# It is also the cheapest cold-start verifier on demo morning: one command
# to confirm the entire pipeline is alive.
#
# USAGE
# -----
#   # Slack-shaped event against a locally-served ingress
#   python harness/send_event.py --channel slack \
#     --url http://localhost:8000/slack/events \
#     --text "hello agent"
#
#   # Slack url_verification handshake (one-off, used during Slack app setup)
#   python harness/send_event.py --channel slack --verify --url <ingress-url>
#
#   # Teams-shaped Activity against the Teams ingress
#   python harness/send_event.py --channel teams \
#     --url http://localhost:8000/teams/messages \
#     --text "hello agent from teams"
#
# The signing secret defaults to a dev placeholder so the harness and ingress
# agree out of the box. Override via SLACK_SIGNING_SECRET in the environment.

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

# Dev-default signing secret. The ingress (modal/ingress_slack.py) reads the
# same env var and falls back to this same literal, so the harness and the
# ingress sign/verify with matching keys without any setup step.
DEV_SIGNING_SECRET = "dev-signing-secret-do-not-use-in-prod"  # pragma: allowlist secret


def _slack_signature(body: bytes, timestamp: str, secret: str) -> str:
    """Replicate Slack's v0 HMAC signing.

    sig = "v0=" + HMAC_SHA256(secret, "v0:" + timestamp + ":" + raw_body)
    """
    base = f"v0:{timestamp}:{body.decode()}".encode()
    digest = hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return f"v0={digest}"


def build_slack_message_event(text: str, *, team_id: str, user_id: str, channel: str) -> dict[str, Any]:
    """Construct a Slack Events API "message" envelope.

    Matches the shape Slack actually sends so the ingress can be developed
    against this harness and continue to work when wired to real Slack.
    """
    event_id = f"Ev{uuid.uuid4().hex[:10].upper()}"
    event_ts = f"{time.time():.6f}"
    return {
        "token": "verification-token-unused",
        "team_id": team_id,
        "api_app_id": "A0DEV",
        "event": {
            "type": "message",
            "channel": channel,
            "channel_type": "im",
            "user": user_id,
            "text": text,
            "ts": event_ts,
            "event_ts": event_ts,
        },
        "type": "event_callback",
        "event_id": event_id,
        "event_time": int(time.time()),
        "authed_users": [user_id],
    }


def build_slack_url_verification() -> dict[str, Any]:
    """The one-off challenge Slack sends to confirm the webhook URL."""
    return {
        "token": "verification-token-unused",
        "challenge": f"chal_{uuid.uuid4().hex[:12]}",
        "type": "url_verification",
    }


def build_teams_activity(text: str, *, tenant_id: str, user_id: str, conversation_id: str) -> dict[str, Any]:
    """Construct a Bot Framework Activity envelope (Teams' equivalent of a Slack event).

    We deliberately ship a *plain dict* rather than building it through the
    botbuilder SDK. The ingress treats incoming JSON as the source of truth,
    so the SDK isn't needed for the demo path. See modal/ingress_teams.py for
    the matching parser.
    """
    activity_id = f"act-{uuid.uuid4().hex[:12]}"
    return {
        "type": "message",  # other types: conversationUpdate, invoke, ...
        "id": activity_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "channelId": "msteams",
        "serviceUrl": "https://smba.trafficmanager.net/amer/",
        "from": {"id": user_id, "name": "Demo User"},
        "conversation": {
            "id": conversation_id,
            "tenantId": tenant_id,
            "conversationType": "personal",
        },
        "recipient": {"id": "28:bot-app-id", "name": "Agent"},
        "text": text,
        "locale": "en-US",
    }


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> tuple[int, str]:
    """Tiny stdlib HTTP poster — no httpx dependency to keep the harness portable."""
    raw = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=raw, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def send_slack(args: argparse.Namespace) -> None:
    secret = os.environ.get("SLACK_SIGNING_SECRET", DEV_SIGNING_SECRET)
    if args.verify:
        payload = build_slack_url_verification()
    else:
        payload = build_slack_message_event(
            text=args.text,
            team_id=args.team_id,
            user_id=args.user_id,
            channel=args.slack_channel,
        )

    raw = json.dumps(payload).encode()
    timestamp = str(int(time.time()))
    signature = _slack_signature(raw, timestamp, secret)
    headers = {
        "Content-Type": "application/json",
        "X-Slack-Request-Timestamp": timestamp,
        "X-Slack-Signature": signature,
    }
    status, body = post_json(args.url, payload, headers)
    print(f"[slack ← ingress] {status} {body[:200]}")


def send_teams(args: argparse.Namespace) -> None:
    payload = build_teams_activity(
        text=args.text,
        tenant_id=args.teams_tenant_id,
        user_id=args.user_id,
        conversation_id=args.teams_conversation_id,
    )
    # The Bot Framework auth header is normally a signed JWT. The mimic
    # ingress accepts a literal dev-token (modal/ingress_teams.py) so the
    # full Bot Framework auth dance is not required for the demo path.
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dev-teams-token",
    }
    status, body = post_json(args.url, payload, headers)
    print(f"[teams ← ingress] {status} {body[:200]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Slack/Teams-shaped event harness for multiplayer-ai")
    parser.add_argument("--channel", choices=("slack", "teams"), default="slack")
    parser.add_argument("--url", required=True, help="ingress URL to POST against")
    parser.add_argument("--text", default="hello agent", help="message text body")
    # Slack-specific knobs
    parser.add_argument("--team-id", default="T01DEV0001")
    parser.add_argument("--user-id", default="U01DEMOUSER")
    parser.add_argument("--slack-channel", default="D01DEMODM")
    parser.add_argument("--verify", action="store_true", help="send a Slack url_verification challenge")
    # Teams-specific knobs
    parser.add_argument("--teams-tenant-id", default="00000000-0000-0000-0000-000000000dev")
    parser.add_argument("--teams-conversation-id", default="a:demo-conv")

    args = parser.parse_args()
    if args.channel == "slack":
        send_slack(args)
    else:
        send_teams(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
