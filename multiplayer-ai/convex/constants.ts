// convex/constants.ts
//
// Shared domain literals. Imported by schema.ts (for validators), the
// mutation/query modules, and the frontend -- so each vocabulary lives in
// exactly one place.
//
// This file deliberately exports no Convex `mutation`/`query`/`action`,
// so codegen will not surface it under `api.*` -- it stays a plain TS
// module.

import { v, type VLiteral } from "convex/values";

// literalUnion -- variadic v.union over a string-literal tuple. The cast
// preserves the literal-type narrowing through v.union's typings.
function literalUnion<T extends readonly string[]>(values: T) {
  return v.union(
    ...(values.map((vv) => v.literal(vv)) as VLiteral<T[number], "required">[]),
  );
}

// NOVEL FEATURE 2 -- reasons a user can give when rejecting a proactive
// suggestion. Surfaced as labelled buttons in the UI for causality tracking.
export const REJECTION_REASONS = [
  "too_generic",
  "wrong_timing",
  "not_my_workflow",
  "other",
] as const;
export type RejectionReason = (typeof REJECTION_REASONS)[number];
export const rejectionReasonValidator = literalUnion(REJECTION_REASONS);

// Human-friendly sentence-case labels rendered next to each reject button.
export const REJECTION_REASON_LABELS: Record<RejectionReason, string> = {
  too_generic: "Too generic",
  wrong_timing: "Wrong timing",
  not_my_workflow: "Not my workflow",
  other: "Other",
};

// Slack channel_type values + "teams" for the Bot Framework ingress, which
// folds all Teams conversation types under one label at POC scale.
export const CHANNEL_TYPES = ["im", "channel", "group", "mpim", "teams"] as const;
export type ChannelType = (typeof CHANNEL_TYPES)[number];
export const channelTypeValidator = literalUnion(CHANNEL_TYPES);

// "recovered" means at least one attempt failed but the auto-recovery loop
// produced a working script (NOVEL FEATURE 1).
export const TOOL_CALL_STATUSES = ["running", "success", "error", "recovered"] as const;
export type ToolCallStatus = (typeof TOOL_CALL_STATUSES)[number];
export const toolCallStatusValidator = literalUnion(TOOL_CALL_STATUSES);

// "surfaced" is the optional middle state for clients that surface lazily
// after the scheduler writes pending.
export const SUGGESTION_STATUSES = ["pending", "surfaced", "accepted", "rejected"] as const;
export type SuggestionStatus = (typeof SUGGESTION_STATUSES)[number];
export const suggestionStatusValidator = literalUnion(SUGGESTION_STATUSES);

// "user" = inbound from a real person; "agent" = an outbound bot reply.
export const MESSAGE_ROLES = ["user", "agent"] as const;
export type MessageRole = (typeof MESSAGE_ROLES)[number];
export const messageRoleValidator = literalUnion(MESSAGE_ROLES);

// Per-tenant skill tree categories.
export const SKILL_CATEGORIES = [
  "company",
  "team",
  "user",
  "integration",
  "workflow",
] as const;
export type SkillCategory = (typeof SKILL_CATEGORIES)[number];
export const skillCategoryValidator = literalUnion(SKILL_CATEGORIES);

// Who modified a skill -- the agent itself via file_edit, or a human via
// the admin UX rollback. Used to colour the timeline in SkillHistory.tsx.
export const MODIFIED_BY_KINDS = ["agent", "human"] as const;
export type ModifiedByKind = (typeof MODIFIED_BY_KINDS)[number];
export const modifiedByValidator = literalUnion(MODIFIED_BY_KINDS);

// Audit log status (gateway dispatch outcome).
export const AUDIT_STATUSES = ["ok", "denied", "error"] as const;
export type AuditStatus = (typeof AUDIT_STATUSES)[number];
export const auditStatusValidator = literalUnion(AUDIT_STATUSES);
