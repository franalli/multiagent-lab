# multiplayer-ai -- agent navigation (repo root)

> **Role: MIRROR.** This file exists so tooling that prefers `AGENTS.md`
> (Codex, etc.) finds something useful. The full ramp guide for *every*
> coding agent is in **[`CLAUDE.md`](./CLAUDE.md)** — one source, no
> drift. See the [Documentation file map](./CLAUDE.md#documentation-file-map)
> in `CLAUDE.md` for how all four CLAUDE/AGENTS files relate.

## Before reading `CLAUDE.md`

- Architecture narrative + full run guide are in
  [`README.md`](./README.md), not in `CLAUDE.md`.
- Project source code lives in `multiplayer-ai/`. Other top-level
  folders (`agent-basics/`, `modal-examples-main/`) are vendored
  read-only examples and out of scope.
- The 4 novel features (debug surface, proactive framework, skill
  version UX, predictive observability) are summarised in `CLAUDE.md`
  with the key files for each.
- Convex agent skills install with `npx convex ai-files install` (run
  from inside `multiplayer-ai/`). When touching Convex code, read
  `multiplayer-ai/convex/_generated/ai/guidelines.md` first — it
  overrides training-data assumptions.

Everything else lives in [`CLAUDE.md`](./CLAUDE.md). Do not add
multiplayer-ai conventions or recipes to this file.
