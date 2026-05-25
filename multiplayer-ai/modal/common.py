# modal/common.py
#
# Shared bits used by every Modal module in the POC.
#
# Keeping these in one place enforces a single source of truth for:
#   * the Modal App name (one deployed App, Functions grouped by role —
#     the production model the architecture diagram shows)
#   * image definitions (ingress vs sandbox have different deps)
#   * the per-workspace Volume lookup helper
#   * the Convex HTTP client wrapper used by both Modal Functions and the
#     code running INSIDE the sandbox

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

import httpx
import modal

# ---------------------------------------------------------------------------
# App + image definitions
# ---------------------------------------------------------------------------

# One App. All Functions live in it. Multi-tenancy is enforced at sandbox
# *spawn* time (workspace_id -> Volume), not by spinning up per-tenant Apps.
APP_NAME = "multiplayer-ai"
app = modal.App(APP_NAME)

# Image used by Modal Functions (ingress, worker, gateway, scheduler).
#
# The bundled sibling .py files are critical: when `modal serve modal/X.py`
# deploys a single file, Modal does NOT auto-include sibling sources. Our
# "modal/ is intentionally not a Python package" convention means each
# Function does `from common import ...` (and ingress/teams do `from worker
# import ...`). add_local_dir copies the whole modal/ directory to /root in
# the container so those sibling imports resolve at runtime, not just
# during local-machine smoke tests.
#
# The ignore filter excludes Python bytecode caches and the sandbox entry
# script (sandbox_agent.py lives in the separate sandbox_image, mounted at
# /app -- we don't want a stale /root copy shadowing it).
_modal_dir = os.path.dirname(os.path.abspath(__file__))
control_plane_image = (
    modal.Image.debian_slim()
    .uv_pip_install(
        "fastapi[standard]==0.115.5",
        "httpx==0.27.2",
        # google-genai is the new Gemini SDK (replaces google-generativeai).
        # Pinned so scheduler.py Stage-2 LLM extraction has a stable API.
        "google-genai==0.8.0",
    )
    .add_local_dir(
        _modal_dir,
        remote_path="/root",
        ignore=["__pycache__", "*.pyc"],
    )
    # NOTE: sandbox_agent.py is included in the control-plane image too.
    # That's intentional -- the worker re-imports common.py inside its
    # container, which re-evaluates `sandbox_image = ... add_local_file(
    # __file__-relative path)`. Without the file at /root/sandbox_agent.py
    # in the worker, Modal's mount-dedup check fails at Sandbox.create time.
    # The extra ~30KB in the control plane image is the price.
)

# Image baked for the sandbox itself. The sandbox executes generated code,
# so we install the runtime extras here (google-genai for the Gemini agent
# loop, httpx for Convex HTTP writes + gateway egress).
#
# `copy=True` is critical: it bakes sandbox_agent.py into the image at build
# time (locally, on the developer's machine), instead of registering a
# runtime mount. The worker creates the Sandbox from inside another Modal
# Function; a mount would force Modal to re-resolve the local path against
# the worker container's filesystem (where it doesn't exist). With copy=True
# the file lives in the image layer and the worker just references it.
SANDBOX_AGENT_SCRIPT_PATH = "/app/sandbox_agent.py"
sandbox_image = (
    modal.Image.debian_slim()
    .uv_pip_install("google-genai==0.8.0", "httpx==0.27.2")
    .add_local_file(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_agent.py"),
        SANDBOX_AGENT_SCRIPT_PATH,
        copy=True,
    )
)

# ---------------------------------------------------------------------------
# Volume helper — per-workspace storage
# ---------------------------------------------------------------------------

VOLUME_PREFIX = "multiplayer-ai-workspace"
WORKSPACE_MOUNT_PATH = "/workspace"

# Per-process cache: Volume.from_name is a sync control-plane RPC and has no
# .aio() variant. Under autoscale, a single worker container handles many
# invocations -- caching the resolved Volume avoids a per-message lookup.
_VOLUME_CACHE: dict[str, modal.Volume] = {}


def get_workspace_volume(workspace_id: str) -> modal.Volume:
    """Resolve workspace_id → Modal Volume.

    Each tenant gets its own Volume; the agent always mounts at
    WORKSPACE_MOUNT_PATH. This is the multi-tenancy boundary at the infra
    layer. Adding a new tenant is "another Volume," not "another App."

    Per-workspace Volume is the canonical tenant-isolation primitive.
    Per-process cache: Volume.from_name has no .aio() variant, so caching
    avoids a sync control-plane RPC on every invocation under autoscale.
    """
    cached = _VOLUME_CACHE.get(workspace_id)
    if cached is not None:
        return cached
    volume = modal.Volume.from_name(
        f"{VOLUME_PREFIX}-{workspace_id}",
        create_if_missing=True,
    )
    _VOLUME_CACHE[workspace_id] = volume
    return volume


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

# Single Secret holding everything the control plane needs. Configure once
# with `modal secret create multiplayer-ai-secrets ...`. The dev-default
# signing secret here matches harness/send_event.py so the harness and the
# ingress can verify each other out of the box without any setup step.
DEV_SLACK_SIGNING_SECRET = (
    "dev-signing-secret-do-not-use-in-prod"  # pragma: allowlist secret
)

# Secret name — create with:
#   modal secret create multiplayer-ai-secrets \
#       SLACK_SIGNING_SECRET=... \
#       GEMINI_API_KEY=... \
#       GEMINI_MODEL=gemini-3-flash-preview \
#       CONVEX_SITE_URL=https://exuberant-albatross-781.convex.site
SECRETS_NAME = "multiplayer-ai-secrets"  # pragma: allowlist secret


def secrets() -> list[modal.Secret]:
    """Return the secrets list to pass on @app.function(secrets=...).

    We build a `from_dict` Secret at deploy-time, reading each value from
    the local env (or a dev default). This:
      1. Lets a fresh workspace run without `modal secret create ...`.
      2. Lets `modal serve` pick up runtime-only values like the
         dynamically-assigned TOOL_GATEWAY_URL, which can't live in a
         pre-created named Secret.
      3. Lets production override every key by setting it in env before
         `modal deploy`.

    The Sandbox inherits these via the worker's `secrets=` argument; the
    agent_tools module reads TOOL_GATEWAY_URL / CONVEX_SITE_URL from env;
    sandbox_agent reads GEMINI_API_KEY / GEMINI_MODEL.

    `Secret.from_name(SECRETS_NAME)` is the alternative production path
    if you prefer the named-secret workflow -- but it raises at deploy
    time (not import time) if absent, so we don't use it as a default.

    Production path: `Secret.from_name` per workspace + an external vault
    for per-tenant integration creds (Pipedream Connect OAuth tokens never
    enter the sandbox). POC uses `from_dict` with dev fallbacks so a fresh
    clone runs without provisioning -- production-path swap is one line.
    """
    payload: dict[str, str] = {
        "SLACK_SIGNING_SECRET": os.environ.get(
            "SLACK_SIGNING_SECRET", DEV_SLACK_SIGNING_SECRET
        ),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "GEMINI_MODEL": os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        "CONVEX_SITE_URL": os.environ.get(
            "CONVEX_SITE_URL", "https://exuberant-albatross-781.convex.site"
        ),
    }
    gateway_url = os.environ.get("TOOL_GATEWAY_URL", "")
    if gateway_url:
        payload["TOOL_GATEWAY_URL"] = gateway_url
    return [modal.Secret.from_dict(payload)]


# ---------------------------------------------------------------------------
# Convex HTTP client
# ---------------------------------------------------------------------------
#
# The sandbox writes to Convex over HTTP (POST to <CONVEX_SITE_URL>/api/...).
# Why HTTP, not the Python client? Keeps the sandbox image lean and matches
# the egress story: anything leaving the sandbox is a plain HTTP call, so
# the audit story is uniform.


def convex_url(path: str) -> str:
    """Build a fully-qualified URL against the deployed Convex site."""
    base = os.environ.get(
        "CONVEX_SITE_URL", "https://exuberant-albatross-781.convex.site"
    )
    return f"{base.rstrip('/')}{path}"


def convex_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Sync POST helper — used by Modal Functions.

    sandbox_agent.py has a separate, identical stdlib-only convex_post for
    in-sandbox use; the duplication is structural -- the sandbox image
    deliberately cannot import common.py.

    None-valued keys are stripped: Convex `v.optional(...)` validators reject
    explicit JSON `null` and require the field be omitted.

    Modal Functions hit the Convex HTTP layer via the
    `makeMutationRoute` / `makeQueryRoute` wrappers; POC uses stdlib
    urllib to keep the dep surface tiny.
    """
    payload = {k: v for k, v in payload.items() if v is not None}
    raw = json.dumps(payload).encode()
    req = urllib.request.Request(
        convex_url(path),
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode()
    return json.loads(body) if body else {}


def convex_query(route: str, payload: dict[str, Any]) -> Any:
    """Read-side helper. Hits a `makeQueryRoute` HTTP endpoint, which wraps
    the value in `{value: ...}`. Returns None on any failure (network /
    deserialise / non-dict envelope) so callers can compose the fallback.

    Used by scheduler.py + analytics.py for "try Convex, fall back to a
    cold-start default" reads.
    """
    try:
        result = convex_post(route, payload)
    except Exception:  # noqa: BLE001 -- fail-open is the intentional contract
        return None
    return result.get("value") if isinstance(result, dict) else None


# AsyncClient.__init__ doesn't require a running loop; only .post() does.
# Module-level so the connection pool + TLS session survive across calls.
_async_client = httpx.AsyncClient(timeout=15)


async def async_convex_post(path: str, payload: dict[str, Any]) -> None:
    """Fire-and-forget async POST to Convex for hot-path async callers
    (e.g. the gateway audit) where the sync `convex_post` would block the
    FastAPI event loop or saturate the BackgroundTasks threadpool.

    Same None-strip + fail-open contract as `convex_post` / `convex_query`.
    """
    payload = {k: v for k, v in payload.items() if v is not None}
    try:
        await _async_client.post(convex_url(path), json=payload)
    except Exception:  # noqa: BLE001 -- fail-open matches the sync helpers
        pass


# ---------------------------------------------------------------------------
# LLM provider (Gemini) + cost estimate
# ---------------------------------------------------------------------------

# Model id read by sandbox_agent + scheduler. Default matches what
# multiplayer-ai/.env.local pins; override via env to swap models.
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

# Per-call cost approximation -- a function of model, not of caller. Lives
# next to the model id so swapping models updates the cost story too.
# Real token accounting would replace this with usage-aware computation.
COST_PER_LLM_CALL_USD = 0.002


# ---------------------------------------------------------------------------
# Shared tenant routing
# ---------------------------------------------------------------------------


def resolve_workspace_from_env(env_var: str, key: str, default: str = "dev") -> str:
    """Look up a tenant identifier (Slack team_id or Teams tenantId) in an
    env-var map of the form "K1=V1,K2=V2,...". Returns `default` on miss.

    The Slack and Teams ingresses both use this with their respective env
    vars (POC_TEAM_MAP, POC_TENANT_MAP); production resolves through Convex.
    """
    raw = os.environ.get(env_var, "")
    if not raw:
        return default
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        if k.strip() == key:
            return v.strip()
    return default
