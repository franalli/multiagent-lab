// convex/predictions.ts
//
// Backing store for NOVEL FEATURE 4 (predictive observability).
//
// modal/analytics.py writes one row per workspace per scheduled run.
// ObservabilityDash.tsx subscribes to latest_for_workspace to show the
// current health snapshot + a short trend.

import { v } from "convex/values";
import { mutation, query } from "./_generated/server";

// Drivers and user_classifications are optional so analytics.py can grow
// the schema without breaking historical rows.
export const save = mutation({
  args: {
    workspace_id: v.string(),
    adoption_health_score: v.number(),
    churn_risk_30d: v.number(),
    predicted_stale_skills: v.array(v.string()),
    // Optional driver breakdown -- analytics.py emits this; legacy callers
    // (or a stripped-down stub run) can still write without it.
    drivers: v.optional(
      v.object({
        adoption_contribution: v.number(),
        success_penalty: v.number(),
        cost_penalty: v.number(),
      }),
    ),
    // Per-user segmentation. Optional so cold-start runs still write.
    user_classifications: v.optional(
      v.array(
        v.object({
          user_id: v.string(),
          classification: v.union(
            v.literal("power"),
            v.literal("casual"),
            v.literal("at_risk"),
          ),
          confidence: v.number(),
        }),
      ),
    ),
  },
  handler: async (ctx, args) => {
    await ctx.db.insert("workspace_predictions", {
      ...args,
      computed_at: Date.now(),
    });
  },
});

export const latest_for_workspace = query({
  args: { workspace_id: v.string() },
  handler: async (ctx, { workspace_id }) =>
    ctx.db
      .query("workspace_predictions")
      .withIndex("by_workspace_recent", (q) => q.eq("workspace_id", workspace_id))
      .order("desc")
      .first(),
});
