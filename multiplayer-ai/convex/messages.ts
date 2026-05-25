// convex/messages.ts
//
// Thread + message primitives.
//
// Two function types live here side-by-side, by design:
//   - MUTATIONS (write-side) are transactional and CANNOT call external APIs.
//   - QUERIES   (read-side) are reactive — subscribed clients re-render the
//                instant a mutation commits new data.
//
// The architectural point you can land during the trial: the reactive UI for
// thread state "falls out of" this design — you do not write any pub/sub
// glue. The agent mutates from inside the sandbox, the dashboard sees it.

import { v } from "convex/values";
import { mutation, query } from "./_generated/server";
import type { Id } from "./_generated/dataModel";
import {
  channelTypeValidator,
  messageRoleValidator,
  toolCallStatusValidator,
} from "./constants";

// ---------------------------------------------------------------------------
// MUTATIONS
// ---------------------------------------------------------------------------

// get_or_create_thread — the sandbox/agent calls this at the start of every
// turn. We dedupe by (workspace_id, channel) so a single Slack channel maps
// to a single thread row regardless of how many turns it has accumulated.
export const get_or_create_thread = mutation({
  args: {
    workspace_id: v.string(),
    channel: v.string(),
    channel_type: channelTypeValidator,
  },
  handler: async (ctx, args): Promise<Id<"threads">> => {
    const existing = await ctx.db
      .query("threads")
      .withIndex("by_workspace_channel", (q) =>
        q.eq("workspace_id", args.workspace_id).eq("channel", args.channel),
      )
      .first();
    if (existing) return existing._id;

    return await ctx.db.insert("threads", {
      workspace_id: args.workspace_id,
      channel: args.channel,
      channel_type: args.channel_type,
      created_at: Date.now(),
    });
  },
});

// append_message — the only way messages enter Convex. Used both for the
// inbound user turn AND the outbound agent reply, distinguished by `role`.
export const append_message = mutation({
  args: {
    workspace_id: v.string(),
    thread_id: v.id("threads"),
    user_id: v.string(),
    role: messageRoleValidator,
    content: v.string(),
  },
  handler: async (ctx, args): Promise<Id<"messages">> => {
    return await ctx.db.insert("messages", {
      ...args,
      timestamp: Date.now(),
    });
  },
});

// log_tool_call — opens a tool_calls row in "running" state. The sandbox
// owns the lifecycle and calls update_tool_call_status when the script
// finishes (or all retries exhaust).
export const log_tool_call = mutation({
  args: {
    workspace_id: v.string(),
    thread_id: v.id("threads"),
    script: v.string(),
  },
  handler: async (ctx, args): Promise<Id<"tool_calls">> => {
    return await ctx.db.insert("tool_calls", {
      workspace_id: args.workspace_id,
      thread_id: args.thread_id,
      script: args.script,
      status: "running",
      attempts: 0,
      started_at: Date.now(),
    });
  },
});

export const update_tool_call_status = mutation({
  args: {
    tool_call_id: v.id("tool_calls"),
    status: toolCallStatusValidator,
    attempts: v.number(),
    result: v.optional(v.string()),
    error: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const { tool_call_id, ...rest } = args;
    await ctx.db.patch(tool_call_id, rest);
  },
});

// append_debug_trace — one row per execution attempt. The DebugPanel diffs
// consecutive rows by attempt index, so the UI can show "what the recovery
// LLM changed between attempt 1 and attempt 2."
export const append_debug_trace = mutation({
  args: {
    workspace_id: v.string(),
    tool_call_id: v.id("tool_calls"),
    attempt: v.number(),
    script: v.string(),
    stdout: v.string(),
    stderr: v.string(),
    exit_code: v.number(),
    recovery_attempt: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    await ctx.db.insert("debug_traces", { ...args, created_at: Date.now() });
  },
});

// ---------------------------------------------------------------------------
// QUERIES (reactive — drive the dashboard)
// ---------------------------------------------------------------------------

// get_thread_messages — feeds ThreadView. Order ascending so the UI just
// renders in receive order; .collect() is fine at POC scale (we cap thread
// length in practice).
export const get_thread_messages = query({
  args: { thread_id: v.id("threads") },
  handler: async (ctx, { thread_id }) =>
    ctx.db
      .query("messages")
      .withIndex("by_thread", (q) => q.eq("thread_id", thread_id))
      .order("asc")
      .collect(),
});

// get_recent_threads — sidebar listing for the dashboard.
export const get_recent_threads = query({
  args: { workspace_id: v.string() },
  handler: async (ctx, { workspace_id }) =>
    ctx.db
      .query("threads")
      .withIndex("by_workspace", (q) => q.eq("workspace_id", workspace_id))
      .order("desc")
      .take(25),
});

// get_tool_calls_for_thread — feeds the DebugPanel's outer list.
export const get_tool_calls_for_thread = query({
  args: { thread_id: v.id("threads") },
  handler: async (ctx, { thread_id }) =>
    ctx.db
      .query("tool_calls")
      .withIndex("by_thread", (q) => q.eq("thread_id", thread_id))
      .order("asc")
      .collect(),
});

// get_debug_traces — the per-attempt detail. Two consecutive rows are what
// the DebugPanel's diff view is built on.
export const get_debug_traces = query({
  args: { tool_call_id: v.id("tool_calls") },
  handler: async (ctx, { tool_call_id }) =>
    ctx.db
      .query("debug_traces")
      .withIndex("by_tool_call", (q) => q.eq("tool_call_id", tool_call_id))
      .order("asc")
      .collect(),
});

// recent_for_workspace — feeds modal/scheduler.py Stage-2 pattern extraction.
// Returns user messages only (the scheduler never wants to learn patterns
// from the agent's own replies). Caps at 200 rows so a noisy workspace
// doesn't blow the scan budget.
export const recent_for_workspace = query({
  args: {
    workspace_id: v.string(),
    since_ms: v.optional(v.number()),
    limit: v.optional(v.number()),
  },
  handler: async (ctx, { workspace_id, since_ms, limit }) => {
    const cap = limit ?? 200;
    const rows = await ctx.db
      .query("messages")
      .withIndex("by_workspace", (q) => q.eq("workspace_id", workspace_id))
      .order("desc")
      .take(cap);
    return rows.filter((m) => {
      if (m.role !== "user") return false;
      if (since_ms != null && m.timestamp < since_ms) return false;
      return true;
    });
  },
});
