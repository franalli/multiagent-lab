// convex/runs.ts
//
// agent_runs telemetry. The sandbox writes one row per turn end -- duration,
// tool-call count, exec success, recovery attempts, cost estimate, plus
// skill_reads / skill_writes lists for NF4's diversity / staleness inputs.
//
// modal/analytics.py reads from here to compute the workspace_features row;
// no more synthetic-only data once a few real turns have run.

import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

// save -- one row per agent turn (NF4 telemetry). Written by
// sandbox_agent.run_agent_loop at end-of-turn after the reply has been
// posted through the gateway, so the user-visible latency doesn't pay
// for the Convex write.
//
// Per-turn telemetry to central Convex makes the ephemeral memory layer
// observable. prompt_tokens is the visible manifestation; the POC does
// not compact (per spec line 994) but measures via prompt_token_count
// so the layer is visible.
export const save = mutation({
  args: {
    workspace_id: v.string(),
    user_id: v.string(),
    thread_id: v.id("threads"),
    duration_ms: v.number(),
    tool_calls_count: v.number(),
    exec_success: v.boolean(),
    recovery_attempts: v.number(),
    cost_estimate_usd: v.number(),
    skill_reads: v.optional(v.array(v.string())),
    skill_writes: v.optional(v.array(v.string())),
    prompt_tokens: v.optional(v.number()),
  },
  handler: async (ctx, args) => {
    await ctx.db.insert("agent_runs", {
      ...args,
      started_at: Date.now(),
    });
  },
});

// recent_for_workspace -- consumed by modal/analytics.fetch_agent_runs to
// compute the features row. The `since_ms` bound is pushed into the index
// range so older rows never cross the wire.
export const recent_for_workspace = query({
  args: {
    workspace_id: v.string(),
    since_ms: v.optional(v.number()),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, { workspace_id, since_ms, limit }) => {
    const cap = limit ?? 500;
    const rows = await ctx.db
      .query("agent_runs")
      .withIndex("by_workspace_recent", (q) => {
        const eq = q.eq("workspace_id", workspace_id);
        return since_ms == null ? eq : eq.gte("started_at", since_ms);
      })
      .order("desc")
      .take(cap);
    return rows;
  },
});

// recent_for_user -- per-user analytics, drives user_classifier in
// modal/analytics.py. Uses the composite index so this scales as the
// agent_runs table grows.
export const recent_for_user = query({
  args: {
    workspace_id: v.string(),
    user_id: v.string(),
    since_ms: v.optional(v.number()),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, { workspace_id, user_id, since_ms, limit }) => {
    const cap = limit ?? 200;
    const rows = await ctx.db
      .query("agent_runs")
      .withIndex("by_workspace_user_recent", (q) => {
        const eq = q.eq("workspace_id", workspace_id).eq("user_id", user_id);
        return since_ms == null ? eq : eq.gte("started_at", since_ms);
      })
      .order("desc")
      .take(cap);
    return rows;
  },
});

// distinct_users_recent -- a small server-side aggregation that returns the
// set of user_ids seen in the lookback window. Reads exactly the rows
// inside the window (via the index range), then dedupes.
//
// HARD CAP: take(1000). Workspaces with > 1000 runs inside the window
// will silently truncate -- revisit when peak active_users_count nears
// 1000 (e.g. swap to a per-user reservoir or maintain a denormalised
// `workspace_active_users` table).
export const distinct_users_recent = query({
  args: {
    workspace_id: v.string(),
    since_ms: v.number(),
  },
  handler: async (ctx, { workspace_id, since_ms }) => {
    const rows = await ctx.db
      .query("agent_runs")
      .withIndex("by_workspace_recent", (q) =>
        q.eq("workspace_id", workspace_id).gte("started_at", since_ms),
      )
      .take(1000);
    const seen = new Set<string>();
    for (const r of rows) seen.add(r.user_id);
    return Array.from(seen);
  },
});
