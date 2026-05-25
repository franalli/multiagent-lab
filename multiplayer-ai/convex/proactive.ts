// convex/proactive.ts
//
// The non-invasiveness state layer. NOVEL FEATURE 2 extension.
//
// Encodes the 7 social principles from the prep doc as Convex state so
// every gate the scheduler runs is *data-driven*, not hard-coded:
//
//   start-small-expand-on-trust  -> trust_score
//   channel allow-list           -> channel_allow_list
//   frequency caps               -> suggestions_this_window / max
//   quiet hours                  -> quiet_hours.{start,end}_hour_utc
//   recent rejection cooldown    -> recent rejections + similarity in scheduler
//   SME deference                -> channel last-activity heuristic in scheduler
//   self-canceling on failure    -> paused + consecutive_rejections
//
// "Make 'don't be annoying' impossible to violate by accident, not just
// discouraged."

import { v } from "convex/values";
import { mutation, type MutationCtx, query } from "./_generated/server";

// Default state used when a workspace has never been touched. Conservative
// on purpose: low trust, opt-in channels, weekday-business-hours only.
const DEFAULTS = {
  trust_score: 0.3,
  suggestions_per_window_max: 5,        // per week
  channel_allow_list: [] as string[],
  quiet_hours: {
    start_hour_utc: 7,                  // 7am UTC ~ 9am CET (Warsaw default)
    end_hour_utc: 19,                   // 7pm UTC
    enabled: true,
  },
};

const WEEK_MS = 7 * 24 * 60 * 60 * 1000;
const PAUSE_AFTER_CONSECUTIVE_REJECTIONS = 5;

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

export const get_state = query({
  args: { workspace_id: v.string() },
  handler: async (ctx, { workspace_id }) =>
    ctx.db
      .query("workspace_proactive_state")
      .withIndex("by_workspace", (q) => q.eq("workspace_id", workspace_id))
      .first(),
});

// ---------------------------------------------------------------------------
// Init + admin mutations
// ---------------------------------------------------------------------------

export const ensure_state = mutation({
  args: { workspace_id: v.string() },
  handler: async (ctx, { workspace_id }) => {
    const existing = await ctx.db
      .query("workspace_proactive_state")
      .withIndex("by_workspace", (q) => q.eq("workspace_id", workspace_id))
      .first();
    if (existing) return existing._id;
    const now = Date.now();
    return await ctx.db.insert("workspace_proactive_state", {
      workspace_id,
      trust_score: DEFAULTS.trust_score,
      suggestions_this_window: 0,
      window_resets_at: now + WEEK_MS,
      suggestions_per_window_max: DEFAULTS.suggestions_per_window_max,
      channel_allow_list: DEFAULTS.channel_allow_list,
      quiet_hours: DEFAULTS.quiet_hours,
      paused: false,
      consecutive_rejections: 0,
      last_updated_at: now,
    });
  },
});

// Empty list = opt-in mode (no proactive in any channel).
export const set_channel_allow_list = mutation({
  args: { workspace_id: v.string(), channels: v.array(v.string()) },
  handler: async (ctx, { workspace_id, channels }) => {
    const row = await ctx.db
      .query("workspace_proactive_state")
      .withIndex("by_workspace", (q) => q.eq("workspace_id", workspace_id))
      .first();
    if (!row) throw new Error("workspace_proactive_state not initialised");
    await ctx.db.patch(row._id, {
      channel_allow_list: channels,
      last_updated_at: Date.now(),
    });
  },
});

// enabled=false disables the gate entirely (window still stored).
export const set_quiet_hours = mutation({
  args: {
    workspace_id: v.string(),
    start_hour_utc: v.number(),
    end_hour_utc: v.number(),
    enabled: v.boolean(),
  },
  handler: async (ctx, { workspace_id, ...quiet }) => {
    const row = await ctx.db
      .query("workspace_proactive_state")
      .withIndex("by_workspace", (q) => q.eq("workspace_id", workspace_id))
      .first();
    if (!row) throw new Error("workspace_proactive_state not initialised");
    await ctx.db.patch(row._id, { quiet_hours: quiet, last_updated_at: Date.now() });
  },
});

// set_paused -- the proactive kill switch (NF2 gate 4a). Stage-6
// feedback auto-flips this true after 5 consecutive rejections; admin
// flips it back to false manually after addressing the cause.
export const set_paused = mutation({
  args: { workspace_id: v.string(), paused: v.boolean() },
  handler: async (ctx, { workspace_id, paused }) => {
    const row = await ctx.db
      .query("workspace_proactive_state")
      .withIndex("by_workspace", (q) => q.eq("workspace_id", workspace_id))
      .first();
    if (!row) throw new Error("workspace_proactive_state not initialised");
    await ctx.db.patch(row._id, { paused, last_updated_at: Date.now() });
  },
});

// ---------------------------------------------------------------------------
// Counter ops (called by the scheduler before/after surfacing)
// ---------------------------------------------------------------------------

// Bump the per-window counter. Resets if past window_resets_at.
export const bump_counter = mutation({
  args: { workspace_id: v.string() },
  handler: async (ctx, { workspace_id }) => {
    const row = await ctx.db
      .query("workspace_proactive_state")
      .withIndex("by_workspace", (q) => q.eq("workspace_id", workspace_id))
      .first();
    if (!row) throw new Error("workspace_proactive_state not initialised");
    const now = Date.now();
    const past_window = now >= row.window_resets_at;
    await ctx.db.patch(row._id, {
      suggestions_this_window: past_window ? 1 : row.suggestions_this_window + 1,
      window_resets_at: past_window ? now + WEEK_MS : row.window_resets_at,
      last_updated_at: now,
    });
  },
});

// ---------------------------------------------------------------------------
// Stage 6 -- feedback loop. Triggered by suggestions.set_status (same
// transaction) or by the /api/proactive/update_trust HTTP route (separate
// transaction). The shared helper below is the single source for the EMA
// math + auto-pause threshold, so set_status and update_trust_from_outcome
// can't drift.
// ---------------------------------------------------------------------------

// EMA-style trust update so a single accept/reject doesn't swing the score
// wildly. alpha is small; trust drifts toward the long-run accept rate.
const TRUST_ALPHA = 0.15;

export async function applyTrustUpdate(
  ctx: MutationCtx,
  workspace_id: string,
  accepted: boolean,
): Promise<void> {
  const row = await ctx.db
    .query("workspace_proactive_state")
    .withIndex("by_workspace", (q) => q.eq("workspace_id", workspace_id))
    .first();
  if (!row) return; // tolerate uninitialised; scheduler will init next run
  const newScore =
    (1 - TRUST_ALPHA) * row.trust_score + TRUST_ALPHA * (accepted ? 1 : 0);
  const newConsec = accepted ? 0 : row.consecutive_rejections + 1;
  await ctx.db.patch(row._id, {
    trust_score: Math.max(0, Math.min(1, newScore)),
    consecutive_rejections: newConsec,
    paused: newConsec >= PAUSE_AFTER_CONSECUTIVE_REJECTIONS,
    last_updated_at: Date.now(),
  });
}

// update_trust_from_outcome -- public mutation wrapper around
// applyTrustUpdate. Exposed via /api/proactive/update_trust for
// out-of-band trust nudges (e.g. an admin "this was actually fine"
// override that's not tied to a specific suggestion row).
//
// The trust EMA (alpha=0.15) over accept/reject outcomes encodes the
// "earn trust, then suggest more" philosophy. applyTrustUpdate is the
// single source of math; both this public route and the in-transaction
// call from suggestions.set_status share it so the rule can't drift.
export const update_trust_from_outcome = mutation({
  args: { workspace_id: v.string(), accepted: v.boolean() },
  handler: async (ctx, { workspace_id, accepted }) =>
    applyTrustUpdate(ctx, workspace_id, accepted),
});

// ---------------------------------------------------------------------------
// Queries the scheduler runs each scan (read-only against suggestions +
// messages tables).
// ---------------------------------------------------------------------------

export const recent_rejections = query({
  args: { workspace_id: v.string(), limit: v.optional(v.number()) },
  handler: async (ctx, { workspace_id, limit }) => {
    const rows = await ctx.db
      .query("suggestions")
      .withIndex("by_workspace_status", (q) =>
        q.eq("workspace_id", workspace_id).eq("status", "rejected"),
      )
      .order("desc")
      .take(limit ?? 50);
    return rows.map((r) => ({
      candidate_text: r.candidate_text,
      rejection_reason: r.rejection_reason ?? "other",
      created_at: r.created_at,
    }));
  },
});

// recent_user_activity -- "is the user currently active?" check. Returns
// the timestamp of their most recent message; the scheduler compares
// against a configurable window. Uses the by_workspace_user composite
// index so the lookup is a covering scan, not a workspace-tail filter.
export const recent_user_activity = query({
  args: { workspace_id: v.string(), user_id: v.string() },
  handler: async (ctx, { workspace_id, user_id }) => {
    const latest = await ctx.db
      .query("messages")
      .withIndex("by_workspace_user", (q) =>
        q.eq("workspace_id", workspace_id).eq("user_id", user_id),
      )
      .order("desc")
      .first();
    return latest ? latest.timestamp : null;
  },
});

// channel_last_activity -- "is a human currently active in this channel?"
// powers the SME-deference rule (don't fire while a real expert is in
// flow).
export const channel_last_activity = query({
  args: { workspace_id: v.string(), channel: v.string() },
  handler: async (ctx, { workspace_id, channel }) => {
    const thread = await ctx.db
      .query("threads")
      .withIndex("by_workspace_channel", (q) =>
        q.eq("workspace_id", workspace_id).eq("channel", channel),
      )
      .first();
    if (!thread) return null;
    const latest = await ctx.db
      .query("messages")
      .withIndex("by_thread", (q) => q.eq("thread_id", thread._id))
      .order("desc")
      .first();
    return latest ? latest.timestamp : null;
  },
});
