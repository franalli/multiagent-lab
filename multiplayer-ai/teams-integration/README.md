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
* Real Adaptive Card responses go via the Bot Framework Connector with a
  freshly-minted access token; the gateway's `teams.send` handler stubs
  this for the demo.

## When you do want to wire real Teams

1. Register an Azure AD app: <https://portal.azure.com> -> Azure AD ->
   App registrations -> New registration.
2. Create a Bot in the **Bot Framework registration**: <https://dev.botframework.com>.
3. Set the messaging endpoint to the deployed Modal ingress:

   ```
   https://<your-workspace>--multiplayer-ai-teams-ingress.modal.run/teams/messages
   ```

4. Sideload the Teams app manifest into your tenant.
5. Provision `TEAMS_BOT_APP_ID` and `TEAMS_BOT_APP_PASSWORD` into the
   Modal Secret.
6. Replace the dev-token check in `verify_teams_auth` with real JWT
   validation -- the canonical path is `botframework-connector`'s
   `JwtTokenValidation.authenticate_request`.

## Verifying the mimic locally

```bash
cd multiplayer-ai && uvicorn modal.ingress_teams:teams_app --reload --port 8001

# In another shell:
python harness/send_event.py --channel teams \
  --url http://localhost:8001/teams/messages \
  --text "hello agent from teams"
```

The ingress should return `{"status": "accepted", "workspace_id": "dev"}`
and the worker should pick up the spawn -- same downstream path as Slack.
