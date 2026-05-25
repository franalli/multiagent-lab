// convex/skills.ts
//
// NOVEL FEATURE 3 — the skill version *index*.
//
// Architectural framing: SKILL.md files live in the per-workspace Modal
// Volume and are git-versioned at the filesystem layer. That's great for
// engineers but unsuitable as an admin UX. So we mirror every write to
// this denormalised index, which the SkillHistory.tsx surface reads from
// to render timelines / diffs / rollbacks instantly.
//
// Source of truth still lives in the Volume; this table is the *cached view*
// of that source of truth. If the Volume and Convex diverge (rare), the
// Volume wins.

import { v } from "convex/values";
import { mutation, query } from "./_generated/server";
import { modifiedByValidator, skillCategoryValidator } from "./constants";

// save_skill_version — called by `agent_tools.file_edit` after a successful
// SKILL.md write inside the sandbox, and by `rollback_to_version` for human
// restores. Version is computed server-side inside the mutation, which
// runs as a serialized Convex transaction -- so two concurrent edits on
// the same skill produce v(N+1) and v(N+2), never two v(N+1) collisions.
// Returns the new version so the caller can echo it back.
//
// Source-of-truth for skill versions is git on the per-tenant Volume
// (engineer-friendly). The POC keeps Volume canonical but adds this
// denormalised Convex mirror PURELY for the NF3 admin UX (timeline,
// diff, one-click rollback). On divergence, Volume wins.
export const save_skill_version = mutation({
  args: {
    workspace_id: v.string(),
    skill_name: v.string(),
    category: skillCategoryValidator,
    content: v.string(),
    diff: v.string(),
    modified_by: modifiedByValidator,
    change_summary: v.string(),
  },
  handler: async (ctx, args) => {
    const latest = await ctx.db
      .query("skill_version_index")
      .withIndex("by_skill_version", (q) =>
        q.eq("workspace_id", args.workspace_id).eq("skill_name", args.skill_name),
      )
      .order("desc")
      .first();
    const next_version = (latest?.version ?? 0) + 1;
    await ctx.db.insert("skill_version_index", {
      ...args,
      version: next_version,
      modified_at: Date.now(),
    });
    return { version: next_version };
  },
});

// get_current_version — caller uses this to compute the next version number
// before issuing save_skill_version. Returns 0 if the skill has never been
// written (so the next write is version 1).
export const get_current_version = query({
  args: {
    workspace_id: v.string(),
    skill_name: v.string(),
  },
  handler: async (ctx, { workspace_id, skill_name }) => {
    const latest = await ctx.db
      .query("skill_version_index")
      .withIndex("by_skill_version", (q) =>
        q.eq("workspace_id", workspace_id).eq("skill_name", skill_name),
      )
      .order("desc")
      .first();
    return latest?.version ?? 0;
  },
});

// get_skill_history — the version timeline for one skill. Renders the
// left-hand list in SkillHistory.tsx.
export const get_skill_history = query({
  args: {
    workspace_id: v.string(),
    skill_name: v.string(),
  },
  handler: async (ctx, { workspace_id, skill_name }) =>
    ctx.db
      .query("skill_version_index")
      .withIndex("by_skill_version", (q) =>
        q.eq("workspace_id", workspace_id).eq("skill_name", skill_name),
      )
      .order("desc")
      .take(50),
});

// rollback_to_version — NOVEL FEATURE 3. Restore a historical version's
// content by appending a NEW row (never mutate history). The new row's
// `diff` shows what changed from the previous latest content to the
// restored content -- i.e. what the rollback actually reverted.
//
// modified_by="human" by contract: a rollback is an admin action. The
// SkillHistory timeline can colour these rows distinctly from agent edits.
export const rollback_to_version = mutation({
  args: {
    workspace_id: v.string(),
    skill_name: v.string(),
    target_version: v.number(),
  },
  handler: async (ctx, { workspace_id, skill_name, target_version }) => {
    const target = await ctx.db
      .query("skill_version_index")
      .withIndex("by_skill_version", (q) =>
        q
          .eq("workspace_id", workspace_id)
          .eq("skill_name", skill_name)
          .eq("version", target_version),
      )
      .first();
    if (!target) {
      throw new Error(
        `rollback: no v${target_version} for ${skill_name} in ${workspace_id}`,
      );
    }

    const latest = await ctx.db
      .query("skill_version_index")
      .withIndex("by_skill_version", (q) =>
        q.eq("workspace_id", workspace_id).eq("skill_name", skill_name),
      )
      .order("desc")
      .first();
    if (!latest) {
      throw new Error("rollback: timeline empty (impossible if target found)");
    }

    const new_version = latest.version + 1;
    await ctx.db.insert("skill_version_index", {
      workspace_id,
      skill_name,
      category: target.category,
      version: new_version,
      content: target.content,
      diff: diffLines(latest.content, target.content),
      modified_by: "human",
      modified_at: Date.now(),
      change_summary: `rollback to v${target_version}`,
    });

    return { new_version };
  },
});

// Tiny unaligned line-diff. Server-side Convex has no diff library; this
// gives a readable "removed/added" summary without pulling a dependency.
function diffLines(prev: string, next: string): string {
  const prevLines = prev.split("\n");
  const nextLines = next.split("\n");
  const removed = prevLines.filter((l) => !nextLines.includes(l));
  const added = nextLines.filter((l) => !prevLines.includes(l));
  if (removed.length === 0 && added.length === 0) return "(no line changes)";
  return [
    ...removed.map((l) => `- ${l}`),
    ...added.map((l) => `+ ${l}`),
  ].join("\n");
}

// list_skills_in_category — the category filter dropdown.
export const list_skills_in_category = query({
  args: {
    workspace_id: v.string(),
    category: skillCategoryValidator,
  },
  handler: async (ctx, { workspace_id, category }) => {
    // Distinct-by-skill_name. Convex doesn't have native DISTINCT yet, so we
    // collect and dedupe in memory at POC scale — fine for the demo.
    const rows = await ctx.db
      .query("skill_version_index")
      .withIndex("by_workspace_category", (q) =>
        q.eq("workspace_id", workspace_id).eq("category", category),
      )
      .collect();
    const seen = new Set<string>();
    const result: typeof rows = [];
    for (const r of rows) {
      if (seen.has(r.skill_name)) continue;
      seen.add(r.skill_name);
      result.push(r);
    }
    return result;
  },
});
