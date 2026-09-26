# Teams integration notes (mimic)

The POC's Teams story is a **mimic**, not a full Bot Framework registration.
The architectural point we demonstrate is: only the *ingress* differs per
channel. Worker, sandbox, gateway, Volume, Convex don't know which channel
fired. So `modal/ingress_teams.py` accepts a Bot Framework Activity-shaped
dict and normalises it to the same context shape `ingress_slack.py` emits.

For the trial demo we exercise the path with `harness/send_event.py
--channel teams`. No Microsoft tenant is required.

## Why the mimic is the right scope for the POC

* Bot Framework JWT validation requires fetching Microsoft's JWKS and
  validating issuer/audience against your tenant. That's plumbing, not
  architecture.
* Outbound replies are real once credentials exist. The gateway's
  `teams.send` mints a Bot Framework token (client-credentials, cached
  until about a minute before expiry) and POSTs a plain-text message
  Activity to the conversation, threaded under the triggering Activity.
  Without `TEAMS_BOT_APP_ID` / `TEAMS_BOT_APP_PASSWORD` it returns
  `{"stubbed": true, ...}`, which is what the harness demo exercises.
  Adaptive Cards aren't built. Replies are text only.

## When you do want to wire real Teams

1. Register an Azure AD app: <https://portal.azure.com> -> Azure AD ->
   App registrations -> New registration.
2. Create a Bot in the **Bot Framework registration**: <https://dev.botframework.com>.
3. Set the messaging endpoint to the deployed Modal ingress:

   ```
   https://<your-workspace>--multiplayer-ai-teams-ingress.modal.run/teams/messages
   ```

4. Sideload the Teams app manifest into your tenant.
5. Export `TEAMS_BOT_APP_ID` and `TEAMS_BOT_APP_PASSWORD` in the shell
   before `uv run modal deploy modal/serve_all.py`.
   `common.gateway_secrets()` forwards them to the `tool_gateway` Function
   only. They never enter the Sandbox. The token is minted against the
   multi-tenant `botframework.com` authority. A single-tenant bot
   registration needs its own tenant id in `TEAMS_TOKEN_URL`
   (`modal/gateway.py`).
6. Replace the dev-token check in `verify_teams_auth` with real JWT
   validation -- the canonical path is `botframework-connector`'s
   `JwtTokenValidation.authenticate_request`. **Do this before step 5 in
   any shared environment.** Until then the ingress trusts whatever
   `serviceUrl` a dev-token caller supplies. The gateway only sends the
   bot token to `https` hosts under `*.botframework.com` or
   `*.botframework.azure.us`, or exactly `smba.trafficmanager.net`. Other
   `*.trafficmanager.net` names are rejected because any Azure customer
   can register one. So it can't be
   pointed at an arbitrary server. But an unauthenticated caller could
   still make the bot post into conversations.

## Verifying the mimic locally

```bash
cd multiplayer-ai && uv run --with uvicorn uvicorn ingress_teams:teams_app --app-dir modal --reload --port 8001

# In another shell:
uv run python harness/send_event.py --channel teams \
  --url http://localhost:8001/teams/messages \
  --text "hello agent from teams"
```

The ingress should return `{"status": "accepted", "workspace_id": "dev"}`
and the worker should pick up the spawn -- same downstream path as Slack.
