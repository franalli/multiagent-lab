// convex/audit.ts
//
// Mutations + queries for the audit_log table.
//
// Every call the tool gateway dispatches lands here -- the single chokepoint
// makes the audit story uniform (no "did the agent also call x outside the
// gateway?" gap). Demo line: "by construction the gateway is the only
// egress, so by construction this table is complete."

import { v } from "convex/values";
import { mutation } from "./_generated/server";
import { auditStatusValidator } from "./constants";

// log_call -- one row per external call dispatched by the tool gateway.
// Status comes from `auditStatusValidator` ("ok" | "denied" | "error").
// Called via BackgroundTasks from gateway.py so the inbound request isn't
// blocked on the audit write.
export const log_call = mutation({
  args: {
    workspace_id: v.string(),
    action: v.string(),
    params_summary: v.string(),    // pre-truncated by the caller (POC: 512 chars)
    actor: v.string(),             // sandbox id / "scheduler" / "harness"
    status: auditStatusValidator,
  },
  handler: async (ctx, args) => {
    await ctx.db.insert("audit_log", { ...args, created_at: Date.now() });
  },
});
