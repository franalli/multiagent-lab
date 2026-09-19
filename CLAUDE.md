# multiplayer-ai -- Claude / agent ramp notes (repo root)

> **Role: SOURCE OF TRUTH.** This is the canonical agent ramp for every
> coding agent (Claude Code, Codex, etc.) working on `multiplayer-ai/`.
> All other CLAUDE.md / AGENTS.md files in this repo are thin pointers
> to here — see [Documentation file map](#documentation-file-map) below.
> Architecture narrative + run guide live in `README.md` next to this
> file.

> **Path convention**: relative paths in this doc (`modal/`, `convex/schema.ts`,
> `frontend/`, etc.) refer to entries under `multiplayer-ai/`. Shell snippets
> assume `cd multiplayer-ai` first. The other top-level folders
> (`agent-basics/`, `modal-examples-main/`) are vendored read-only examples
> and out of scope for this guide.

## Documentation file map

| File | Role |
|---|---|
| **`CLAUDE.md` (this file)** | **Source of truth.** Full ramp for all coding agents. |
| `AGENTS.md` (repo root) | One-screen mirror of this file's headline rules for tools that prefer `AGENTS.md`. No unique content — links back here. |
| `multiplayer-ai/CLAUDE.md` | Pointer stub. Exists only so the Convex CLI (`npx convex ai-files install/update`) has a place to write its auto-managed block. Do not add ramp content here. |
| `multiplayer-ai/AGENTS.md` | Same role as `multiplayer-ai/CLAUDE.md`, for the `AGENTS.md` half of the Convex CLI's auto-managed pair. |

When in doubt: edit only this file. The other three either auto-regenerate
(Convex CLI) or mirror this file's headlines.

## What this is

A Slack/Teams multiplayer-AI demo: Slack/Teams → Modal ingress →
per-workspace Sandbox → Convex reactive state, plus four novel features
(debug surface, proactive framework, skill version UX, predictive
observability). POC scope; the deliverable is the demo.

## Folder map

| Path                          | Owner                                            |
|-------------------------------|--------------------------------------------------|
| `modal/`                      | Compute control plane. **Not** a Python package. |
| `convex/`                     | Reactive state backend. `npx convex dev`.        |
| `volume/workspace-dev/`       | Local repr of the per-tenant Modal Volume.       |
| `frontend/`                   | Next.js components (drop-in scaffolds).          |
| `harness/`                    | Slack-/Teams-shaped event poster.                |
| `slack-integration/`, `teams-integration/` | Real-channel setup notes.           |

## The four novel features at a glance

| # | Surface                                                              | Key files                                                            |
|---|----------------------------------------------------------------------|----------------------------------------------------------------------|
| 1 | **Code-execution debug surface** — per-attempt trace + LLM recovery  | `modal/sandbox_agent.execute_with_recovery`, `convex.debug_traces`, `DebugPanel.tsx` |
| 2 | **Proactive framework** — 6-stage pipeline + 7 non-invasiveness gates + trust EMA + genericness classifier + causality tracking | `modal/scheduler.py`, `convex/proactive.ts`, `convex/suggestions.ts`, `ThreadView.tsx` |
| 3 | **Skill version UX** — denormalised index + timeline + diff + rollback | `agent_tools.file_edit`, `convex/skills.ts` (`save_skill_version`, `rollback_to_version`), `SkillHistory.tsx` |
| 4 | **Predictive observability** — telemetry → features → predictions + drivers + user classifications | `convex/runs.ts`, `modal/analytics.py`, `convex/predictions.ts`, `ObservabilityDash.tsx` |

## Critical conventions

1. **`modal/` is NOT a Python package**. There's no `__init__.py`. Files
   import siblings as `from common import ...`. This is deliberate -- a
   `modal` package would shadow the Modal SDK. Run files with
   `modal run modal/<file>.py`; **never** `python -m modal.<x>`.

2. **Sandbox → Convex is HTTP** (`<CONVEX_SITE_URL>/api/...` routes in
   `convex/http.ts`), not the Convex Python client. Sandbox image stays
   lean; egress is uniform.

3. **All outbound calls from the sandbox go through the tool gateway**
   (`send_via_gateway(action, params)` in `agent_tools.py`). The
   sandbox image has no direct Slack/Teams SDK -- there is no alternative
   path. The system prompt says so; the tool injection makes it true.

4. **Multi-tenancy is at the Volume layer**. `get_workspace_volume(workspace_id)`
   resolves to one Volume per workspace. Adding a tenant is "another
   Volume," not "another App."

5. **`agent_tools.py` is single-source**. `AGENT_TOOLS_SOURCE` (a string
   constant in `modal/sandbox_agent.py`) is materialised at `/tmp/agent_tools.py`
   once at sandbox boot. Both the parent agent loop and the subprocess
   executor import from the same materialised module -- no drift.

6. **Shared vocabularies live in `convex/constants.ts`**. Channel types,
   rejection reasons, suggestion statuses, classification labels, etc.
   Each vocabulary exports a literal-array, a TypeScript type, and a
   Convex validator -- one source for schema / mutations / frontend.

7. **HTTP routes use `makeMutationRoute(api.X.Y)` and `makeQueryRoute(api.X.Y)`**
   in `convex/http.ts`. Adding a new endpoint is a single line. The
   `convex_query` helper in `modal/common.py` is the matching read-side
   wrapper (fail-open: returns `None` on any error).

8. **Two memory layers, kept distinct.** Durable = per-workspace Modal
   Volume; `file_read`/`file_edit` on SKILL.md is the Anthropic
   memory-tool pattern with the Volume as backing store. Ephemeral =
   the `messages` list inside `run_agent_loop`, bounded by `MAX_TURNS`
   and measured via `prompt_tokens` on every agent_runs row. The POC
   doesn't compact (per spec line 994); we measure so the layer is
   visible. See the articulation block above `run_agent_loop` in
   `modal/sandbox_agent.py`.

## Convex module map

| File                | Owns                                                                       |
|---------------------|----------------------------------------------------------------------------|
| `schema.ts`         | All 12 tables, every index. Reads validators from `constants.ts`.          |
| `constants.ts`      | Shared literal unions + validator factories + label dicts.                 |
| `messages.ts`       | Thread/message/tool_call/debug_trace mutations + reactive queries.         |
| `suggestions.ts`    | NF2 suggestions table + `set_status` → trust-update side-effect.           |
| `proactive.ts`      | `workspace_proactive_state` CRUD + `applyTrustUpdate` helper + scheduler read queries (`recent_rejections`, `recent_user_activity`, `channel_last_activity`). |
| `skills.ts`         | NF3 `skill_version_index` + `save_skill_version` + `rollback_to_version`.  |
| `audit.ts`          | Gateway audit log.                                                         |
| `runs.ts`           | NF4 `agent_runs` save + recent/per-user/distinct-users queries.            |
| `predictions.ts`    | NF4 `workspace_predictions` save + latest_for_workspace query.             |
| `llm.ts`            | `call_gemini` action (only Convex function type that can fetch).            |
| `http.ts`           | All `/api/...` HTTP routes. Uses `makeMutationRoute` + `makeQueryRoute`.   |

## Modal module map

| File                | Role                                                                            |
|---------------------|---------------------------------------------------------------------------------|
| `common.py`         | App + image defs + Volume helper + `convex_post` + `convex_query` + secrets + `COST_PER_LLM_CALL_USD` + `resolve_workspace_from_env`. |
| `ingress_slack.py`  | FastAPI Slack ingress -- HMAC, ACK <3s, `Function.spawn(agent_worker)`.         |
| `ingress_teams.py`  | Bot Framework Activity ingress (mimic) -- normalises to same context.           |
| `worker.py`         | Spawns the Sandbox with per-workspace Volume mounted.                           |
| `sandbox_agent.py`  | The agent loop -- runs **inside** the Sandbox. Owns `AGENT_TOOLS_SOURCE`; writes per-turn `agent_runs` telemetry. |
| `gateway.py`        | Tool gateway -- single egress. Audit writes via `BackgroundTasks`.              |
| `scheduler.py`      | NF2 6-stage proactive pipeline + 7 non-invasiveness gates.                      |
| `analytics.py`      | NF4 features + predictions + per-user classifier (band-midpoint anchors).       |

## Where to make changes (recipes)

- **New agent-callable tool**: extend `AGENT_TOOLS_SOURCE` in
  `modal/sandbox_agent.py` (function + add to system prompt) and add a
  branch in `dispatch()` in `modal/gateway.py` if it needs to egress.
  Per-workspace creds belong in `credentials_for(workspace_id, action)`.

- **New Convex table**: add to `convex/schema.ts`. If the vocabulary is
  a literal union, extract to `convex/constants.ts` (validator + type +
  label dict). Run `npx convex codegen` after every schema change.
  New mutations/queries go in their own module.

- **New HTTP endpoint**: add one line to `convex/http.ts` --
  `makeMutationRoute(api.X.Y, "responseKey")` for writes,
  `makeQueryRoute(api.X.Y)` for reads. Don't hand-roll the wrapper.
  Then call from Python via `convex_post` or `convex_query` in
  `common.py`.

- **New ingress channel**: a new file `modal/ingress_<channel>.py` that
  normalises whatever envelope the channel speaks into the same
  structured-context dict Slack/Teams emit. Downstream (worker /
  sandbox / gateway / Convex) is channel-agnostic by construction.

- **New non-invasiveness gate (NF2)**: add `gate_<name>(state, candidate)`
  in `modal/scheduler.py` returning `None | "reason"`. Append to the
  `gates` list in `scan_workspace`. If it needs Convex state, add the
  column to `workspace_proactive_state` in `convex/schema.ts` plus a
  query in `convex/proactive.ts` + an HTTP route in `convex/http.ts`.

- **New telemetry field (NF4)**: add the column to `agent_runs` in
  `convex/schema.ts` (optional, so historical rows revalidate). Update
  the `runs.save` mutation args. Have `sandbox_agent.run_agent_loop`
  populate it in the per-turn save block. Consume it in
  `modal/analytics.compute_features` or `analytics.classify_user`.

- **New prediction driver (NF4)**: extend `predict_health()` in
  `modal/analytics.py` to compute the contribution. Add the field to
  the `drivers` object in `convex/schema.ts` + `convex/predictions.ts`
  (both optional). Render in `frontend/components/ObservabilityDash.tsx`.

- **New shared literal union**: add to `convex/constants.ts` --
  `FOO_VALUES as const`, `type Foo`, `fooValidator` (via the
  `literalUnion` helper). If the frontend needs labels, add a `FOO_LABELS`
  record next to the validator.

## Python environment (uv)

> **Path exception**: paths in this section are **repo-root**, not
> under `multiplayer-ai/`. One environment serves the whole lab.

`uv` owns the Python side. One `pyproject.toml`, one `uv.lock`, one
`.venv` -- all at the repo root, shared by `multiplayer-ai/`, `LLM/`
and `agent-basics/`. There is no `requirements.txt` anywhere and no
per-folder virtualenv; both were removed 2026-09-17. Python is pinned
to **3.12** (`requires-python = ">=3.12,<3.13"`) because
`mistral-common` caps `numpy<2.4` below 3.13 -- that and the rest of
the dependency log live in `LLM/decisions.md`. There's no
`.python-version`; uv resolves the interpreter from `requires-python`,
downloading a managed CPython if the machine has no 3.12.

**Dependency groups.** `[project.dependencies]` is the agent /
control-plane stack; everything else is a group, so the training stack
installs without the tooling and vice versa.

| Command | Gets you |
|---|---|
| `uv sync` | project deps **+ the `dev` group** (ruff, mypy, pytest, jupyter, pre-commit) -- `dev` is on unless you pass `--no-dev` |
| `uv sync --group llm` | the above + the training stack (torch, transformers, tiktoken, pandas, ...) for `LLM/` |
| `uv sync --all-groups` | `dev` + `llm` + `audio`; this is what the working `.venv` currently holds |
| `uv sync --no-dev` | project deps only -- CI / minimal runtime |

`uv sync` is *exact*: it uninstalls whatever isn't in the selected set.
A bare `uv sync` in an all-groups `.venv` strips the `llm` and `audio`
stacks; `--no-dev` additionally strips the tooling. If you're working
in `LLM/`, keep `--group llm` (or `--all-groups`) on every sync.

**Running things.** `uv run <cmd>` works from anywhere in the tree (uv
walks up to the root `pyproject.toml`) -- `uv run modal serve
multiplayer-ai/modal/serve_all.py`, `uv run ruff check .`, `uv run
pytest`. Unlike `uv sync`, `uv run` syncs *inexactly*: it installs
whatever the default groups are missing and removes nothing (`--exact`
opts into removal), so a bare `uv run` is safe in an all-groups
`.venv`. The subtractive trap is `uv sync` alone. `--no-sync` skips the
sync entirely. `source .venv/bin/activate` still works -- it just stops
guaranteeing the env matches the lock.

**Changing dependencies.** `uv add <pkg>`, `uv add --group llm <pkg>`,
`uv add --group dev <pkg>`; `uv lock --upgrade` to re-resolve
everything; `uv sync` to apply. Never `pip install` into `.venv` -- it
isn't recorded in `uv.lock`, so the next sync deletes it without
comment. `pyproject.toml` and `uv.lock` are both tracked and belong in
the same commit; `.venv/` is gitignored. `ruff` is pinned in two places
on purpose: `ruff==0.16.8` in the `dev` group and `rev: v0.16.8` in
`.pre-commit-config.yaml`. Bump them together, or the hook and your
shell lint with different rule sets.

**What stays out of this env.** Container dependencies. The Modal
images declare their own inline --
`.uv_pip_install("google-genai==0.8.0", "httpx==0.27.2")` on both
`control_plane_image` and `sandbox_image` in `modal/common.py`.
`google-genai` is deliberately absent from the root `pyproject.toml`:
it runs in the container, never on your laptop, so `uv add
google-genai` is the wrong fix for an import error in sandbox code --
edit the image. Same for GPU wheels (CUDA torch, bitsandbytes, vllm).

## How to run end-to-end

See **[`README.md#quickstart-run-the-slack-agent`](./README.md#quickstart-run-the-slack-agent)**
for the unified `modal serve modal/serve_all.py` flow + harness fire.
The README is the single source of truth for run instructions; this
file deliberately doesn't re-state them so they can't drift. Every
Python entry point runs through the root uv environment -- prefix it
with `uv run` (see [Python environment (uv)](#python-environment-uv)).

LLM provider: **Gemini 3 Flash Preview** via `google-genai`. Reads
`GEMINI_API_KEY` + `GEMINI_MODEL` from env (both live in
`multiplayer-ai/.env.local`). Without the key, falls back to a
deterministic stub.

NF2 + NF4 smoke tests live in the README too; one-liners are
`uv run modal run modal/scheduler.py::smoke_scheduler` (one proactive
scan, prints which gates fired) and
`uv run modal run modal/analytics.py::smoke_analytics` (one prediction
run).

## Convex agent skills

Convex agent skills for common tasks can be installed by running
`npx convex ai-files install` (from inside `multiplayer-ai/`). When
working on Convex code, **always read `convex/_generated/ai/guidelines.md`
first** for important guidelines on how to correctly use Convex APIs
and patterns. That file overrides what you may have learned from
training data.

The install populates three gitignored paths at the `multiplayer-ai/`
npm root: `skills/` (skill bundles), `ai/` (guidelines + state),
`skills-lock.json` (lockfile). They live next to `package.json`, not
inside `convex/`, because the CLI anchors to the npm root.

## Three things named "skills" -- disambiguation

| Thing | Where | What it is |
|---|---|---|
| `multiplayer-ai/skills/` | Filesystem, npm-root | CLI-installed agent skills the IDE reads while you author code. Tooling material, not deployed. |
| `convex/skills.ts` + `skill_version_index` table | Convex runtime | NF3's *user-skill versioning* backing store -- save / list / rollback. Production code. |
| `/api/skills/save_version` HTTP route | URL surface | The egress NF3 uses to push a new skill version from the sandbox into Convex. Not a filesystem path despite starting with `skills/`. |

When something says "skills," ask which of the three. They share a brand,
nothing else.
