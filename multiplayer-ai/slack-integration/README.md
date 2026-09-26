# Slack integration notes

The POC reaches Slack two ways:

1. **For the demo** -- via `harness/send_event.py`, which POSTs
   correctly-signed Slack Events API payloads at `modal/ingress_slack.py`.
   No real Slack workspace is needed.

2. **For a real Slack workspace** -- via a Slack app configured to send
   Events API webhooks at the deployed Modal ingress URL. The steps below.

## Slack app setup (for real-Slack runs only)

1. Go to <https://api.slack.com/apps> -> Create New App -> From scratch.
2. Pick a workspace and a name (e.g. "multiplayer-ai-dev").
3. **OAuth & Permissions** -- add bot token scopes:
   `app_mentions:read`, `channels:history`, `chat:write`, `im:history`,
   `im:read`, `im:write`, `users:read`.
4. **Event Subscriptions** -- enable, set Request URL to the deployed
   ingress URL:

   ```
   https://<your-workspace>--multiplayer-ai-slack-ingress.modal.run/slack/events
   ```

   Subscribe to bot events: `message.im`, `message.channels` (as needed).
5. **Install to Workspace** -- copy the resulting Bot User OAuth Token; it
   is the value for `SLACK_BOT_TOKEN`.
6. **App Credentials** -- copy the Signing Secret; it is the value for
   `SLACK_SIGNING_SECRET`.

## Credentials

`secrets()` in `modal/common.py` builds a `modal.Secret.from_dict(...)` at
deploy time from **your shell's environment**. The named
`multiplayer-ai-secrets` Secret is not read unless you swap in
`Secret.from_name`. So configuring credentials means exporting them
before you deploy:

```bash
cd multiplayer-ai
export SLACK_SIGNING_SECRET=<from-step-6>
export SLACK_BOT_TOKEN=<from-step-5>          # gateway only -- see below
export GEMINI_API_KEY=<your-key>              # already in .env.local
export GEMINI_MODEL=gemini-3-flash-preview    # already in .env.local
export CONVEX_SITE_URL=https://exuberant-albatross-781.convex.site  # optional, this is the default
uv run modal deploy modal/serve_all.py
```

`SLACK_BOT_TOKEN` goes through `common.gateway_secrets()`, which only the
`tool_gateway` Function mounts. It is deliberately **not** in `secrets()`,
because the worker hands that list to the Sandbox, where LLM-generated code
runs. The gateway's `slack.send` calls `chat.postMessage` with the token.
Replies go into the thread when the triggering message was threaded
(`thread_ts`), and otherwise post top-level.

The dev defaults in `modal/common.py` mean the POC still runs with none of
these set. The harness round-trips (the ingress falls back to the same dev
signing secret the harness uses), the LLM call falls back to a stub, and
without `SLACK_BOT_TOKEN` the gateway returns `{"stubbed": true, ...}`
instead of posting.

A Slack-side failure (e.g. `not_in_channel`, `invalid_auth`) comes back
from the gateway as a 502, and the audit log records it as `error`. The
agent still records its reply and `agent_runs` row in Convex. If replies
don't appear in Slack, check `audit_log` first. The usual cause is the bot
not being a member of the channel (`/invite @<bot>`).

## Verifying

```bash
# Local FastAPI dev server (no Modal):
cd multiplayer-ai && uv run --with uvicorn uvicorn ingress_slack:web_app --app-dir modal --reload

# In another shell:
uv run python harness/send_event.py --channel slack \
  --url http://localhost:8000/slack/events \
  --text "hello agent"
```

The ingress should return `{"ok": true, "workspace_id": "dev"}` and the
worker (if Modal-deployed) should pick up the spawned function call.
