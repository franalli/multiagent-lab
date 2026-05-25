// convex/schema.ts
//
// The reactive state backend for the multiplayer-ai POC.
//
// EVERY TABLE IS KEYED BY workspace_id. The central-Convex pattern:
// skill files live in Modal Volume (per-tenant), but agent state
// (threads, messages, tool calls, suggestions, audit) lives centrally so the
// frontend can subscribe to it reactively across all workspaces.
//
// Why a dedicated by_workspace index on most tables:
//   - multi-tenancy is enforced at *query* time (every read filters by
//     workspace_id), so we want a covering index for that filter.
//   - it also leaves room to later add per-tenant rate limits, quotas, and
//     analytics rollups without re-indexing.
//
// During the trial demo, the talking point: "Convex is the agent state plane.
// Skills sit in Modal Volume because the agent reads them as files. Everything
// else sits in Convex because the UI subscribes to it reactively."

import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";
import {
  auditStatusValidator,
  channelTypeValidator,
  messageRoleValidator,
  modifiedByValidator,
  rejectionReasonValidator,
  skillCategoryValidator,
  suggestionStatusValidator,
  toolCallStatusValidator,
} from "./constants";

export default defineSchema({
  // -----------------------------------------------------------------------
  // workspaces — tenant directory. team_id (Slack) and tenant_id (Teams)
  // both resolve to a single workspace_id. This is the "team_id → workspace_id"
  // routing table the ingress consults before spawning the worker.
  // -----------------------------------------------------------------------
  workspaces: defineTable({
    workspace_id: v.string(),         // canonical tenant identifier, e.g. "dev"
    display_name: v.string(),         // human-readable, surfaces in the UI
    slack_team_id: v.optional(v.string()),   // e.g. "T01ABCDEFGH"
    teams_tenant_id: v.optional(v.string()), // e.g. Azure AD tenant UUID
    created_at: v.number(),
  })
    .index("by_workspace_id", ["workspace_id"])
    .index("by_slack_team_id", ["slack_team_id"])
    .index("by_teams_tenant_id", ["teams_tenant_id"]),

  // -----------------------------------------------------------------------
  // threads — one row per Slack/Teams conversation. channel_type lets the UI
  // render DM vs channel posts differently and gives the agent context for
  // tone calibration (DMs allow more verbose reasoning than #general).
  // -----------------------------------------------------------------------
  threads: defineTable({
    workspace_id: v.string(),
    channel: v.string(),              // Slack channel id ("C0..."), Teams conversation.id
    channel_type: channelTypeValidator,
    created_at: v.number(),
  })
    .index("by_workspace", ["workspace_id"])
    .index("by_workspace_channel", ["workspace_id", "channel"]),

  // -----------------------------------------------------------------------
  // messages — both user-sent and agent-emitted. The dashboard subscribes to
  // get_thread_messages and re-renders as the agent writes back.
  // -----------------------------------------------------------------------
  messages: defineTable({
    workspace_id: v.string(),
    thread_id: v.id("threads"),
    user_id: v.string(),              // Slack user_id "U0..." or Teams from.id; "agent" for bot replies
    role: messageRoleValidator,
    content: v.string(),
    timestamp: v.number(),
  })
    .index("by_thread", ["thread_id"])
    .index("by_workspace", ["workspace_id"])
    // by_workspace_user lets proactive.recent_user_activity find the
    // user's latest message via a covering index lookup rather than
    // scanning the workspace tail with a filter.
    .index("by_workspace_user", ["workspace_id", "user_id"]),

  // -----------------------------------------------------------------------
  // tool_calls — one row per code-as-tools execution. Status transitions:
  //   running → success | error | recovered (after the auto-recovery loop fixed it)
  // attempts records how many sandbox runs were needed (>1 means recovery fired).
  // -----------------------------------------------------------------------
  tool_calls: defineTable({
    workspace_id: v.string(),
    thread_id: v.id("threads"),
    message_id: v.optional(v.id("messages")),  // optional because the agent may log before posting
    script: v.string(),
    status: toolCallStatusValidator,
    result: v.optional(v.string()),
    error: v.optional(v.string()),
    attempts: v.number(),
    started_at: v.number(),
  })
    .index("by_workspace", ["workspace_id"])
    .index("by_thread", ["thread_id"]),

  // -----------------------------------------------------------------------
  // debug_traces — NOVEL FEATURE 1. One row per execution *attempt*. The
  // DebugPanel diffs attempt N-1's script against attempt N's script to
  // visualise what the recovery LLM actually changed.
  // -----------------------------------------------------------------------
  debug_traces: defineTable({
    workspace_id: v.string(),
    tool_call_id: v.id("tool_calls"),
    attempt: v.number(),              // 1-indexed
    script: v.string(),               // the script as run on THIS attempt
    stdout: v.string(),
    stderr: v.string(),
    exit_code: v.number(),
    recovery_attempt: v.optional(v.string()), // free-form note about what was fixed
    created_at: v.number(),
  }).index("by_tool_call", ["tool_call_id"]),

  // -----------------------------------------------------------------------
  // suggestions — NOVEL FEATURE 2. Proactive candidates produced by the
  // scheduler. specificity_breakdown is the 4-dimensional score the UI
  // exposes so admins can see *why* a suggestion was surfaced (or filtered).
  // -----------------------------------------------------------------------
  suggestions: defineTable({
    workspace_id: v.string(),
    user_id: v.string(),
    channel: v.optional(v.string()),  // surfaced where; powers SME-deference + channel allow-list
    candidate_text: v.string(),
    specificity_score: v.number(),    // 0.0 - 1.0 aggregate
    specificity_breakdown: v.object({
      named_entity_density: v.number(),
      recurrence_count: v.number(),
      user_attribution: v.boolean(),
      timing_pattern: v.string(),     // e.g. "weekly-monday-morning"
    }),
    // genericness_score is the "negative-space" learned signal -- how
    // closely this candidate resembles past rejections. Higher = filtered.
    genericness_score: v.optional(v.number()),
    status: suggestionStatusValidator,
    // Captured on reject -- powers causality tracking (the "why").
    rejection_reason: v.optional(rejectionReasonValidator),
    created_at: v.number(),
  })
    .index("by_workspace_status", ["workspace_id", "status"])
    .index("by_workspace", ["workspace_id"]),

  // -----------------------------------------------------------------------
  // workspace_proactive_state — encodes the non-invasiveness constraints
  // as Convex state so the scheduler's gates are data-driven, not
  // hard-coded. One row per workspace.
  //
  // The architectural framing: "Make 'don't be annoying' impossible to
  // violate by accident, not just discouraged." Every gate reads here.
  // -----------------------------------------------------------------------
  workspace_proactive_state: defineTable({
    workspace_id: v.string(),
    // Start-small-expand-on-trust. Multiplied into the surfacing threshold:
    // low trust => higher effective threshold (fewer suggestions).
    trust_score: v.number(),                  // 0.0 - 1.0
    // Frequency cap counter. Reset weekly by the scheduler.
    suggestions_this_window: v.number(),
    window_resets_at: v.number(),             // ms timestamp
    suggestions_per_window_max: v.number(),
    // Channel allow-list (Slack channel ids / Teams conversation ids).
    // Empty array => opt-in mode (no proactive in any channel).
    channel_allow_list: v.array(v.string()),
    // Quiet hours: scheduler skips workspaces whose current local hour
    // falls outside [start_hour, end_hour). POC uses UTC integer offsets
    // rather than IANA tz names to keep the demo dependency-free.
    quiet_hours: v.object({
      start_hour_utc: v.number(),             // 0-23
      end_hour_utc: v.number(),               // 0-23
      enabled: v.boolean(),
    }),
    // Hard kill switch flipped by Stage-6 feedback after M consecutive
    // rejections (self-canceling on failure). Admin can flip back manually.
    paused: v.boolean(),
    consecutive_rejections: v.number(),
    last_updated_at: v.number(),
  }).index("by_workspace", ["workspace_id"]),

  // -----------------------------------------------------------------------
  // skill_version_index — NOVEL FEATURE 3. Source of truth is the SKILL.md
  // file in Modal Volume (git-native versioning at the filesystem layer);
  // this table is the *denormalised index* the admin UX reads from to
  // render fast history / diff / rollback views without crawling the
  // filesystem.
  // -----------------------------------------------------------------------
  skill_version_index: defineTable({
    workspace_id: v.string(),
    skill_name: v.string(),           // e.g. "company/SKILL.md", "users/personal_udemo/SKILL.md"
    category: skillCategoryValidator,
    version: v.number(),              // 1-indexed monotonic per (workspace_id, skill_name)
    content: v.string(),              // snapshot of the full SKILL.md after this change
    diff: v.string(),                 // unified diff vs previous version
    modified_by: modifiedByValidator,
    modified_at: v.number(),
    change_summary: v.string(),       // one-line note for the timeline
  })
    .index("by_skill_version", ["workspace_id", "skill_name", "version"])
    .index("by_workspace_category", ["workspace_id", "category"]),

  // -----------------------------------------------------------------------
  // audit_log — every egress call recorded by the tool gateway. The
  // sandbox cannot make outbound calls directly, so by construction the
  // gateway is the single chokepoint we can audit.
  // -----------------------------------------------------------------------
  audit_log: defineTable({
    workspace_id: v.string(),
    action: v.string(),               // e.g. "slack.send", "github.create_issue"
    params_summary: v.string(),       // compact JSON summary; full payload may be large
    actor: v.string(),                // sandbox id or "scheduler"
    status: auditStatusValidator,
    created_at: v.number(),
  })
    .index("by_workspace_recent", ["workspace_id", "created_at"]),

  // -----------------------------------------------------------------------
  // agent_runs / workspace_features / workspace_predictions — NOVEL FEATURE 4.
  // Scaffolded so analytics.py has somewhere to write; the XGBoost model
  // itself stays a stub in this scope.
  // -----------------------------------------------------------------------
  agent_runs: defineTable({
    workspace_id: v.string(),
    user_id: v.string(),
    thread_id: v.id("threads"),
    duration_ms: v.number(),
    tool_calls_count: v.number(),
    exec_success: v.boolean(),
    recovery_attempts: v.number(),
    cost_estimate_usd: v.number(),
    started_at: v.number(),
    // Optional skill telemetry. Powers NF4's skill_diversity_index +
    // stale-skill predictor. Optional so historical rows still validate.
    skill_reads: v.optional(v.array(v.string())),
    skill_writes: v.optional(v.array(v.string())),
    // Sum of `usage_metadata.prompt_token_count` across every LLM call
    // in the turn -- the ephemeral context layer's cost proxy. The doc
    // calls this out as the "context-management" half of the two-layer
    // memory model; we measure but don't compact at POC scale.
    prompt_tokens: v.optional(v.number()),
  })
    .index("by_workspace_recent", ["workspace_id", "started_at"])
    // Lets per-user analytics segment by activity without scanning the
    // workspace tail. Used by the user_classifier in modal/analytics.py.
    .index("by_workspace_user_recent", ["workspace_id", "user_id", "started_at"]),

  workspace_features: defineTable({
    workspace_id: v.string(),
    computed_at: v.number(),
    agent_runs_per_day: v.number(),
    active_users_count: v.number(),
    code_exec_success_rate: v.number(),
    suggestion_accept_rate: v.number(),
    cost_per_run_estimate: v.number(),
    // Optional so previously-written rows still validate.
    skill_diversity_index: v.optional(v.number()),  // Shannon entropy over skill reads
    stale_skills_count: v.optional(v.number()),     // skills with zero reads in window
  }).index("by_workspace_recent", ["workspace_id", "computed_at"]),

  workspace_predictions: defineTable({
    workspace_id: v.string(),
    computed_at: v.number(),
    adoption_health_score: v.number(),
    churn_risk_30d: v.number(),
    predicted_stale_skills: v.array(v.string()),
    // Per-prediction contribution breakdown. The demo line lands here:
    // "explainability matters -- you can see exactly what's driving each
    // score." Optional so prior rows don't fail revalidation.
    drivers: v.optional(
      v.object({
        adoption_contribution: v.number(),  // points added by run volume
        success_penalty: v.number(),        // points subtracted by failures
        cost_penalty: v.number(),           // points subtracted by cost
      }),
    ),
    // Per-user segmentation. Optional so existing rows revalidate.
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
  }).index("by_workspace_recent", ["workspace_id", "computed_at"]),
});
