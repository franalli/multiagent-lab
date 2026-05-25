// convex/suggestions.ts
//
// Backing store for NOVEL FEATURE 2 — the proactive framework.
//
// Lifecycle:
//   scheduler.scan_workspace writes rows in "pending" status.
//   the UI surfaces them (showing specificity_breakdown for transparency).
//   the user accepts → status "accepted"; rejects → status "rejected".
//   rejections feed back into apply_non_invasiveness_filters so similar
//   candidates don't get re-suggested.

import { v } from "convex/values";
import { mutation, query } from "./_generated/server";
import type { Id } from "./_generated/dataModel";
import {
  rejectionReasonValidator,
  suggestionStatusValidator,
} from "./constants";
import { applyTrustUpdate } from "./proactive";

// save_suggestion — written by modal/scheduler.py after candidates pass the
// specificity threshold AND the non-invasiveness filters. Status starts as
// "pending" so the UI can decide when to surface it (e.g. respecting quiet
// hours on the client side).
export const save_suggestion = mutation({
  args: {
    workspace_id: v.string(),
    user_id: v.string(),
    channel: v.optional(v.string()),
    candidate_text: v.string(),
    specificity_score: v.number(),
    specificity_breakdown: v.object({
      named_entity_density: v.number(),
      recurrence_count: v.number(),
      user_attribution: v.boolean(),
      timing_pattern: v.string(),
    }),
    genericness_score: v.optional(v.number()),
  },
  handler: async (ctx, args): Promise<Id<"suggestions">> => {
    return await ctx.db.insert("suggestions", {
      ...args,
      status: "pending",
      created_at: Date.now(),
    });
  },
});

// set_status — accept / reject. Updates the workspace's trust score as a
// side-effect (Stage 6 feedback) so future scans see a calibrated threshold.
// rejection_reason is the causality-tracking field; surfaced when status
// transitions to "rejected" so we can learn what *kind* of mismatch it was.
//
// The trust-update math lives in proactive.applyTrustUpdate so the
// constants (alpha, auto-pause threshold) have one home.
export const set_status = mutation({
  args: {
    suggestion_id: v.id("suggestions"),
    status: suggestionStatusValidator,
    rejection_reason: v.optional(rejectionReasonValidator),
  },
  handler: async (ctx, { suggestion_id, status, rejection_reason }) => {
    const patch: Record<string, unknown> = { status };
    if (rejection_reason) patch.rejection_reason = rejection_reason;
    await ctx.db.patch(suggestion_id, patch);

    if (status === "accepted" || status === "rejected") {
      const suggestion = await ctx.db.get(suggestion_id);
      if (suggestion) {
        await applyTrustUpdate(ctx, suggestion.workspace_id, status === "accepted");
      }
    }
  },
});

// get_pending — drives the SuggestionCard list in ThreadView.
export const get_pending = query({
  args: { workspace_id: v.string() },
  handler: async (ctx, { workspace_id }) =>
    ctx.db
      .query("suggestions")
      .withIndex("by_workspace_status", (q) =>
        q.eq("workspace_id", workspace_id).eq("status", "pending"),
      )
      .order("desc")
      .collect(),
});

// get_all_for_workspace — admin view, sees accepted/rejected history too.
export const get_all_for_workspace = query({
  args: { workspace_id: v.string() },
  handler: async (ctx, { workspace_id }) =>
    ctx.db
      .query("suggestions")
      .withIndex("by_workspace", (q) => q.eq("workspace_id", workspace_id))
      .order("desc")
      .take(100),
});

// accept_rate_for_workspace — feeds NF4 analytics. accept_count over
// accept+reject in the lookback window. Returns 0 when neither has fired
// yet (cold-start). Pending suggestions are excluded because we want a
// *resolved* rate, not "% likely to accept."
export const accept_rate_for_workspace = query({
  args: {
    workspace_id: v.string(),
    since_ms: v.optional(v.number()),  // unix ms; 0 / omitted = all-time
  },
  handler: async (ctx, { workspace_id, since_ms }) => {
    const cutoff = since_ms ?? 0;
    const all = await ctx.db
      .query("suggestions")
      .withIndex("by_workspace", (q) => q.eq("workspace_id", workspace_id))
      .collect();
    let accepted = 0;
    let rejected = 0;
    for (const s of all) {
      if (s.created_at < cutoff) continue;
      if (s.status === "accepted") accepted++;
      else if (s.status === "rejected") rejected++;
    }
    const denom = accepted + rejected;
    return denom === 0 ? 0 : accepted / denom;
  },
});

// get_recent_rejected — used by apply_non_invasiveness_filters (server-side
// from scheduler.py) to suppress near-duplicates of recently rejected
// candidates. POC: text similarity is left to the scheduler; this query just
// hands back the recent rejections.
export const get_recent_rejected = query({
  args: { workspace_id: v.string() },
  handler: async (ctx, { workspace_id }) =>
    ctx.db
      .query("suggestions")
      .withIndex("by_workspace_status", (q) =>
        q.eq("workspace_id", workspace_id).eq("status", "rejected"),
      )
      .order("desc")
      .take(50),
});
