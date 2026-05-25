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

## Modal Secret

Create or update the multiplayer-ai Secret in Modal:

```bash
modal secret create multiplayer-ai-secrets \
  SLACK_SIGNING_SECRET=<from-step-6> \
  SLACK_BOT_TOKEN=<from-step-5> \
  ANTHROPIC_API_KEY=<your-key> \
  CONVEX_SITE_URL=https://exuberant-albatross-781.convex.site
```

The dev defaults in `modal/common.py` mean the POC still runs without this
Secret -- the harness will round-trip, the LLM call falls back to a stub,
the gateway returns "stubbed: true" for outbound posts.

## Verifying

```bash
# Local FastAPI dev server (no Modal):
cd multiplayer-ai && uvicorn modal.ingress_slack:web_app --reload

# In another shell:
python harness/send_event.py --channel slack \
  --url http://localhost:8000/slack/events \
  --text "hello agent"
```

The ingress should return `{"ok": true, "workspace_id": "dev"}` and the
worker (if Modal-deployed) should pick up the spawned function call.
