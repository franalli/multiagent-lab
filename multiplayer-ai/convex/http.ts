// convex/http.ts
//
// HTTP endpoints exposed at <CONVEX_SITE_URL>/api/...
//
// The Modal sandbox writes to Convex over HTTP rather than via the Convex
// Python client. Two reasons:
//   1. Keeps the sandbox image lean -- no convex wheel, just httpx.
//   2. Demonstrates the "every egress is an HTTP call" property that
//      matches the sandbox's actual constraints (it can only speak HTTP to
//      the outside world).
//
// makeMutationRoute is a tiny helper that collapses the otherwise repetitive
// "parse JSON body -> runMutation -> return 200" wrapper. Adding a new
// route is a single line.

import { httpRouter, type PublicHttpAction } from "convex/server";
import { httpAction } from "./_generated/server";
import { api } from "./_generated/api";

const http = httpRouter();

// makeMutationRoute -- wraps a mutation reference as an HTTP POST handler.
// If `responseKey` is given, the mutation's return value is wrapped in
// { [responseKey]: <id> }; otherwise an empty {} is returned.
function makeMutationRoute(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  mutationRef: any,
  responseKey?: string,
): PublicHttpAction {
  return httpAction(async (ctx, req) => {
    const body = await req.json();
    const result = await ctx.runMutation(mutationRef, body);
    const payload = responseKey ? { [responseKey]: result } : {};
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });
}

// makeQueryRoute -- mirror of makeMutationRoute for read-side calls.
// The body is the query args; the response is { value: <result> }.
// scheduler.py uses this to read workspace_proactive_state and the
// recent-activity tables without depending on Convex's built-in query
// HTTP surface (which is .convex.cloud, designed for clients not servers).
function makeQueryRoute(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  queryRef: any,
): PublicHttpAction {
  return httpAction(async (ctx, req) => {
    const args = await req.json();
    const value = await ctx.runQuery(queryRef, args);
    return new Response(JSON.stringify({ value }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });
}

http.route({
  path: "/api/messages/append",
  method: "POST",
  handler: makeMutationRoute(api.messages.append_message, "message_id"),
});

http.route({
  path: "/api/threads/get_or_create",
  method: "POST",
  handler: makeMutationRoute(api.messages.get_or_create_thread, "thread_id"),
});

http.route({
  path: "/api/tool_calls/log",
  method: "POST",
  handler: makeMutationRoute(api.messages.log_tool_call, "tool_call_id"),
});

http.route({
  path: "/api/tool_calls/update",
  method: "POST",
  handler: makeMutationRoute(api.messages.update_tool_call_status),
});

http.route({
  path: "/api/debug_traces/append",
  method: "POST",
  handler: makeMutationRoute(api.messages.append_debug_trace),
});

http.route({
  path: "/api/suggestions/save",
  method: "POST",
  handler: makeMutationRoute(api.suggestions.save_suggestion, "suggestion_id"),
});

http.route({
  path: "/api/skills/save_version",
  method: "POST",
  handler: makeMutationRoute(api.skills.save_skill_version, "version_info"),
});

http.route({
  path: "/api/suggestions/accept_rate",
  method: "POST",
  handler: makeQueryRoute(api.suggestions.accept_rate_for_workspace),
});

// Agent run telemetry -- sandbox writes per-turn; analytics reads for NF4.
http.route({
  path: "/api/agent_runs/save",
  method: "POST",
  handler: makeMutationRoute(api.runs.save),
});

http.route({
  path: "/api/agent_runs/recent",
  method: "POST",
  handler: makeQueryRoute(api.runs.recent_for_workspace),
});

http.route({
  path: "/api/agent_runs/recent_for_user",
  method: "POST",
  handler: makeQueryRoute(api.runs.recent_for_user),
});

http.route({
  path: "/api/agent_runs/distinct_users",
  method: "POST",
  handler: makeQueryRoute(api.runs.distinct_users_recent),
});

// Stage-2 input for the proactive scheduler -- real Convex messages, with
// a synthetic fallback in the scheduler if this returns empty.
http.route({
  path: "/api/messages/recent",
  method: "POST",
  handler: makeQueryRoute(api.messages.recent_for_workspace),
});

http.route({
  path: "/api/audit/log",
  method: "POST",
  handler: makeMutationRoute(api.audit.log_call),
});

http.route({
  path: "/api/predictions/save",
  method: "POST",
  handler: makeMutationRoute(api.predictions.save),
});

// Proactive non-invasiveness state. Scheduler reads/writes this every scan.
http.route({
  path: "/api/proactive/ensure_state",
  method: "POST",
  handler: makeMutationRoute(api.proactive.ensure_state, "state_id"),
});

http.route({
  path: "/api/proactive/bump_counter",
  method: "POST",
  handler: makeMutationRoute(api.proactive.bump_counter),
});

http.route({
  path: "/api/proactive/update_trust",
  method: "POST",
  handler: makeMutationRoute(api.proactive.update_trust_from_outcome),
});

// Admin tuning routes -- workspace operators use these to opt channels in,
// set quiet hours, or pause the proactive layer outright.
http.route({
  path: "/api/proactive/set_channel_allow_list",
  method: "POST",
  handler: makeMutationRoute(api.proactive.set_channel_allow_list),
});

http.route({
  path: "/api/proactive/set_quiet_hours",
  method: "POST",
  handler: makeMutationRoute(api.proactive.set_quiet_hours),
});

http.route({
  path: "/api/proactive/set_paused",
  method: "POST",
  handler: makeMutationRoute(api.proactive.set_paused),
});

// Read-side proactive routes -- the scheduler hits these every scan to
// pull workspace state + rejection history + activity recency.
http.route({
  path: "/api/proactive/get_state",
  method: "POST",
  handler: makeQueryRoute(api.proactive.get_state),
});

http.route({
  path: "/api/proactive/recent_rejections",
  method: "POST",
  handler: makeQueryRoute(api.proactive.recent_rejections),
});

http.route({
  path: "/api/proactive/recent_user_activity",
  method: "POST",
  handler: makeQueryRoute(api.proactive.recent_user_activity),
});

http.route({
  path: "/api/proactive/channel_last_activity",
  method: "POST",
  handler: makeQueryRoute(api.proactive.channel_last_activity),
});

// GET /api/health -- liveness probe used by the harness and Modal entrypoints
// before they fire test events.
http.route({
  path: "/api/health",
  method: "GET",
  handler: httpAction(async () =>
    new Response(JSON.stringify({ ok: true, ts: Date.now() }), {
      status: 200,
      headers: { "content-type": "application/json" },
    }),
  ),
});

export default http;
