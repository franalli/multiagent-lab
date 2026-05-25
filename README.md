# multiagent-lab

A Slack / Microsoft Teams multiplayer-AI demo project: an agent that
lives in your team's chat surface, runs in a per-workspace Modal Sandbox,
and exposes a reactive Convex-backed dashboard for debugging, proactive
suggestions, skill versioning, and observability.

Workspace with three top-level folders:

| Folder | What it is |
|---|---|
| `multiplayer-ai/` | **The POC.** Slack/Teams → Modal ingress → per-workspace Sandbox → Convex reactive state, with four novel features (debug surface, proactive framework, skill version UX, predictive observability). Everything below this line is about this folder. |
| `agent-basics/` | Tutorial track that ramps the prep: `01_async_basics`, `02_reliability`, `03_anthropic_agents`, `04_langchain`, `05_langgraph`, `06_elevenlabs`, `07_mcp`. Self-contained exercises -- run individually. |
| `modal-examples-main/` | Vendored copy of the [`modal-labs/modal-examples`](https://github.com/modal-labs/modal-examples) repository, **MIT-licensed, © Modal Labs**. Used as a syntax/idiom reference while ramping on Modal Sandboxes, Volumes, scheduled functions, and web endpoints. The original `LICENSE` is preserved inside the folder. |

Top-level `requirements.txt` pins the Python deps used across all three.
`.venv/` is the shared virtualenv -- activate with `source .venv/bin/activate`.

---

## Agent governance (CLAUDE.md / AGENTS.md / `.claude/agents/`)

Agent guidance lives at the **repo root**, scoped to the
`multiplayer-ai/` subproject by content convention:

| File | Purpose |
|---|---|
| `CLAUDE.md` | The agent ramp guide for Claude Code. A header note declares that bare relative paths (`modal/`, `convex/schema.ts`) refer to entries under `multiplayer-ai/`; shell snippets assume `cd multiplayer-ai` first. |
| `AGENTS.md` | Thin pointer file so tooling that scans for `AGENTS.md` (Codex, Cursor, etc.) lands on `CLAUDE.md`. Not a byte-equivalent mirror -- intentionally short. |
| `.claude/agents/convention-reviewer.md` | Project-level subagent encoding the 7 critical conventions from `CLAUDE.md`. Auto-discovered by Claude Code from anywhere in the repo; its own instructions restrict review scope to `multiplayer-ai/`. |

The lab is effectively a single project (`multiplayer-ai/`) with two
vendored read-only example folders (`agent-basics/`,
`modal-examples-main/`) that don't carry their own conventions.
Hoisting puts the ramp in front of the model regardless of cwd; the
in-file path convention keeps the dense doc accurate without verbose
repo-relative paths.

**Conventions going forward:**

| Rule | Why |
|---|---|
| If an example folder later grows into a real subproject with non-obvious conventions, give it its own `<folder>/CLAUDE.md` rather than expanding the root one. | Claude Code loads CLAUDE.md hierarchically (root + every ancestor of cwd), so a subproject-scoped file layers on top of the root one when you `cd` in. Keep each file scoped to what's true *at its level*. |
| Keep `CLAUDE.md` focused on **how to navigate** and **what NOT to do**. Architecture narrative, ops, and demo flow live here in the README. | Agent files should be short enough to actually load into context. |
| Add new subagents under `.claude/agents/<name>.md` with YAML frontmatter (`name`, `description`, `tools`). The subagent's own description controls when Claude Code auto-invokes it. | Convention reviewer is the template -- copy its frontmatter shape. |

If `AGENTS.md` and `CLAUDE.md` ever drift apart in content claims,
treat that as a bug -- `CLAUDE.md` is the source, `AGENTS.md` only
points at it.

---

# multiplayer-ai — the POC

A Slack/Teams bot whose backend is a FastAPI ingress on Modal, an agent
that runs in a Modal Sandbox with a per-workspace Volume mounted, Convex
for reactive state, and a Next.js + Convex dashboard.

The architecture is a symmetric gateway (in/out) wrapped around a
per-tenant Volume with central Convex for state, then four novel features
layered on top.

## Table of contents

1. [Architecture in one screen](#architecture-in-one-screen)
2. [The four novel features](#the-four-novel-features)
3. [The granular proactive framework](#the-granular-proactive-framework)
4. [File tree](#file-tree)
5. [Entry-point scripts](#entry-point-scripts)
6. [Prereqs](#prereqs)
7. [Quickstart: run the Slack agent](#quickstart-run-the-slack-agent)
8. [Running end to end (the full procedure)](#running-end-to-end-the-full-procedure)
9. [Ingress points](#ingress-points)
10. [The harness](#the-harness)
11. [Smoke-testing each Modal module](#smoke-testing-each-modal-module)
12. [Demo flow](#demo-flow)
13. [Design points to land](#design-points-to-land)
14. [What this POC does not ship](#what-this-poc-does-not-ship)

---

## Architecture in one screen

```
  [harness/send_event.py]   OR    [Slack workspace]    OR   [Teams tenant]
              │                            │                         │
              ▼                            ▼                         ▼
  ┌───────────────────────────────────┐   ┌─────────────────────────────────────┐
  │  modal/ingress_slack.py           │   │  modal/ingress_teams.py (mimic)     │
  │  HMAC verify · ACK <3s            │   │  Bearer-token check (dev-mimic)     │
  │  team_id → workspace_id           │   │  tenant_id → workspace_id           │
  └───────────────────────────────────┘   └─────────────────────────────────────┘
                                  │ both normalise to the SAME context
                                  ▼
                          modal/worker.py
                          spawn Sandbox + mount per-workspace Volume
                                  │
                                  ▼
                          modal/sandbox_agent.py (INSIDE sandbox)
                          progressive disclosure on /workspace/skills/
                          code-as-tools with execute_with_recovery (NF1)
                          tools via /tmp/agent_tools.py (single source)
                                  │
                  ┌───────────────┼────────────────┐
                  ▼               ▼                ▼
            volume/         Convex (state)   modal/gateway.py
            workspace-dev/  threads,         single egress path
            skills/...      messages,        slack.send · teams.send
            channels/       tool_calls,      audit + rate limit
            logs/           debug_traces,    BackgroundTasks for audit writes
                            suggestions,
                            skill_versions,
                            audit_log,
                            workspace_proactive_state,
                            workspace_predictions
                                  │ reactive subscriptions
                                  ▼
                          frontend/components/
                          ThreadView · DebugPanel ·
                          SkillHistory · ObservabilityDash

         modal/scheduler.py (every 6h cron) ──fan-out──► scan_workspace
         6-stage pipeline: activity-delta → extract → score → gate → surface → feedback

         modal/analytics.py (every 6h cron) ──fan-out──► predict_workspace_health
         telemetry → features → XGBoost-style → workspace_predictions
```

The architectural property to land in conversation: **symmetric gateway,
sandwich-isolated sandbox.** Ingress in + tool gateway out = the sandbox
does only LLM work; everything that needs auth / audit / rate-limit lives
at the edges.

---

## The four novel features

| # | Surface | What it solves |
|---|---|---|
| 1 | `multiplayer-ai/modal/sandbox_agent.execute_with_recovery` + `convex.debug_traces` + `DebugPanel.tsx` | Code-as-tools loses the structured tool_use/tool_result trace. We rebuild that surface with per-attempt instrumentation + LLM-driven retry. |
| 2 | `multiplayer-ai/modal/scheduler.py` + `convex.proactive` + `convex.suggestions` + `ThreadView SuggestionCard` | Six-stage proactive pipeline with explicit specificity scoring, encoded non-invasiveness constraints, trust-adjusted threshold, negative-space genericness classifier, causality-tracking rejections. |
| 3 | `sandbox_agent.file_edit` → `convex.skill_version_index` + `SkillHistory.tsx` | Admin UX over git-native skill versioning. Timeline + diff + rollback for workspace admins. |
| 4 | `multiplayer-ai/modal/analytics.py` + `convex.workspace_predictions` + `ObservabilityDash.tsx` | XGBoost-style predictive observability (workspace health, churn risk, skill staleness). Bonus -- cuttable. |

---

## The granular proactive framework

NF2 is the strongest demo piece. The pipeline runs every 6 hours via
`multiplayer-ai/modal/scheduler.py::proactive_scan_all`, which fans out
to `scan_workspace(workspace_id)` per active tenant.

### Stage 1 -- activity-delta gate

`has_meaningful_activity_since_last_scan(workspace_id)`. Cheap, no LLM.
The 80%-exit gate.

### Stage 2 -- candidate extraction

`extract_candidate_patterns(messages)` -- liberal mining for repeated
patterns. POC uses a per-user-repetition heuristic; production substitutes
LLM-as-judge with the same output shape.

### Stage 3 -- specificity scoring (the visible novel piece)

Each candidate scored on 4 dimensions in [0, 1]:

- **named_entity_density** -- specific names, quarter labels, weekdays → high score
- **recurrence_count** -- has happened multiple times → high score
- **user_attribution** -- a specific person is responsible → bool
- **timing_pattern** -- intervals near-constant → high score

The breakdown is stored on every `suggestions` row so the UI can render
*why* a suggestion was surfaced. Demo line: "explainability matters --
admins can see exactly which dimension is driving the decision."

### Stage 4 -- non-invasiveness gates (the encoded constraints)

State lives in `convex.workspace_proactive_state` (one row per workspace).
Every gate is a pure function in `multiplayer-ai/modal/scheduler.py`. First
gate to fire wins; the rejection reason is surfaced in the scan result for
transparency.

| Sub-stage | Gate                       | Source field(s)                                          |
|-----------|----------------------------|----------------------------------------------------------|
| 4a        | workspace paused           | `paused` (auto-flips after 5 consecutive rejections)     |
| 4b        | weekly frequency cap       | `suggestions_this_window`, `_per_window_max`, `window_resets_at` |
| 4c        | channel allow-list         | `channel_allow_list` (empty = opt-in mode)               |
| 4d        | quiet hours (workspace tz) | `quiet_hours.{start_hour_utc, end_hour_utc, enabled}`    |
| 4e        | recent-rejection cooldown  | bag-of-words cosine vs `suggestions[status="rejected"]`  |
| 4f        | user currently active      | `messages.by_workspace_user` covering index              |
| 4g        | SME deference              | last message timestamp on the candidate's channel        |

The trust score (`trust_score` field) multiplies into the surfacing
threshold:

```
effective_threshold = BASE + 0.3 * (0.5 - trust_score)
```

So a workspace at trust 0.0 needs candidates 30% higher-scoring than the
base threshold; a workspace at trust 1.0 sees its threshold drop 30%.
"Start small, expand on trust" encoded as code, not culture.

### Stage 5 -- surface in Convex

`convex.suggestions.save_suggestion` plus `proactive.bump_counter` (so
subsequent candidates in the same scan see the updated frequency state).
The UI subscribes to `suggestions.get_pending` and re-renders reactively.

### Stage 6 -- outcome feedback (the learning loop)

Accept/reject flows through `convex.suggestions.set_status`, which calls
the shared `applyTrustUpdate` helper in `convex/proactive.ts` -- single
source for the EMA math (alpha = 0.15) and the auto-pause threshold (5
consecutive rejections).

Rejection captures a **rejection_reason** -- one of `too_generic`,
`wrong_timing`, `not_my_workflow`, `other`. That's the causality-tracking
data needed for learning from rejection, not just acceptance rates.

### The negative-space genericness classifier

`gate_recent_rejection` computes a bag-of-words cosine between each
candidate and the workspace's last 50 rejections (pre-tokenised once per
scan, not per candidate). If the max cosine exceeds 0.7 the candidate is
filtered. The score is recorded on every suggestion as
`genericness_score` -- visible in the UI alongside the specificity
breakdown.

This is the "negative-space classifier" approach at POC scale.
Production would swap to a learned embedding model; the gate interface
stays identical.

### Surface tuning knobs

- `SUGGESTION_SPECIFICITY_THRESHOLD` (env var) -- base threshold; default 0.45.
- `api.proactive.set_channel_allow_list(workspace_id, channels)` -- opt channels in.
- `api.proactive.set_quiet_hours(workspace_id, start, end, enabled)` -- the time gate.
- `api.proactive.set_paused(workspace_id, paused)` -- manual override.

### What's left for production

- Stage 1's watermark logic is a stub (`has_meaningful_activity_since_last_scan`
  returns True). Wiring a real `messages.since(last_scan_ts)` query is ~10 min.
- Stage 2 uses synthetic messages from `get_recent_messages()` so the
  scoring path is exercisable without a real Slack history. Swap to a
  Convex `messages.by_workspace_recent` query when ready.
- The genericness classifier uses bag-of-words cosine; embeddings come later.

---

## File tree

```
multiagent-lab/
├── README.md                this file (consolidated repo overview + POC docs)
├── CLAUDE.md                agent ramp notes (paths relative to multiplayer-ai/)
├── AGENTS.md                pointer file for AGENTS.md-aware tooling → CLAUDE.md
├── .claude/
│   └── agents/
│       └── convention-reviewer.md   project subagent enforcing CLAUDE.md conventions
├── requirements.txt         shared Python deps
├── agent-basics/            tutorial track (01_async_basics ... 07_mcp), read-only
├── modal-examples-main/     Modal Labs reference (MIT, © Modal Labs), read-only
└── multiplayer-ai/          the POC
    ├── package.json         Convex + frontend deps (npm root for this subproject)
    ├── modal/               compute plane (NOT a Python package)
    │   ├── common.py        shared App + image defs + Volume helper + Convex HTTP wrapper + resolve_workspace_from_env
    │   ├── ingress_slack.py FastAPI Slack ingress (HMAC, ACK, spawn worker)
    │   ├── ingress_teams.py Bot Framework Activity ingress (mimic, normalises to same context)
    │   ├── worker.py        spawns the agent Sandbox with the per-workspace Volume mounted
    │   ├── sandbox_agent.py the agent loop -- runs INSIDE the Sandbox
    │   ├── gateway.py       tool gateway -- the ONLY egress path
    │   ├── scheduler.py     6-stage proactive pipeline + non-invasiveness gates (NF2)
    │   └── analytics.py     feature compute + XGBoost-style predictions (NF4)
    ├── convex/              reactive state backend
    │   ├── schema.ts        tables: workspaces / threads / messages / tool_calls /
    │   │                    debug_traces / suggestions / skill_version_index /
    │   │                    audit_log / workspace_proactive_state /
    │   │                    workspace_predictions / agent_runs / workspace_features
    │   ├── constants.ts     shared literal unions + validator factory (single source for vocabularies)
    │   ├── messages.ts      mutations + reactive queries for thread surface
    │   ├── suggestions.ts   NF2 backing store + Stage-6 trust update on set_status
    │   ├── proactive.ts     non-invasiveness state mutations + queries (NF2 extension)
    │   ├── skills.ts        NF3 backing store (skill_version_index)
    │   ├── audit.ts         gateway audit_log mutations
    │   ├── predictions.ts   NF4 backing store
    │   ├── llm.ts           Action: call_anthropic
    │   └── http.ts          HTTP endpoints (makeMutationRoute / makeQueryRoute helpers)
    ├── volume/
    │   └── workspace-dev/   the single POC tenant (workspace_id = "dev")
    │       ├── skills/      company / team / users / integration / workflow
    │       ├── channels/    Slack log dumps
    │       └── logs/        agent working memory
    ├── frontend/
    │   ├── convexClient.ts  ConvexReactClient wiring
    │   └── components/
    │       ├── ThreadView.tsx       live thread + pending suggestions + accept/reject
    │       ├── DebugPanel.tsx       per-attempt stdout/stderr + diff (NF1)
    │       ├── SkillHistory.tsx     version timeline + rollback (NF3)
    │       └── ObservabilityDash.tsx adoption / churn / staleness (NF4)
    ├── harness/
    │   └── send_event.py    Slack- and Teams-shaped event harness
    ├── slack-integration/README.md  how to wire a real Slack app
    └── teams-integration/README.md  why this is a mimic + how to wire real Teams
```

---

## Entry-point scripts

Skim this table before reading run instructions -- it's the answer to
"which file do I actually invoke?":

| Surface                       | Entry-point script                            | How to invoke                                                                                                                                                          |
|-------------------------------|-----------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **All Modal Functions, unified** | `multiplayer-ai/modal/serve_all.py`        | **The canonical run.** `modal serve modal/serve_all.py` brings up the ingress + worker + gateway + scheduler + analytics in **one** process, sharing one App instance. `modal deploy modal/serve_all.py` for production. |
| **Event harness**             | `multiplayer-ai/harness/send_event.py`        | `python harness/send_event.py --channel slack --url <ingress-url>/slack/events --text "..."`                                                                            |
| **Convex backend**            | `multiplayer-ai/convex/` (no single script)   | `npx convex dev` from `multiplayer-ai/` -- watches `convex/*.ts` and hot-redeploys. Only needed when editing schema/routes; the agent reads deployed Convex over HTTPS. |
| **Proactive scan (one-shot)** | `multiplayer-ai/modal/scheduler.py`           | Cron in production; `modal run modal/scheduler.py::smoke_scheduler` to fire a one-shot scan.                                                                            |
| **Predictive observability (one-shot)** | `multiplayer-ai/modal/analytics.py` | Cron in production; `modal run modal/analytics.py::smoke_analytics` to fire a one-shot prediction.                                                                      |
| **Slack ingress (ASGI-only)** | `multiplayer-ai/modal/ingress_slack.py`       | Advanced: `uvicorn modal.ingress_slack:web_app --reload` runs the FastAPI app on localhost without Modal -- HMAC + routing only, no worker spawn.                       |

The worker (`modal/worker.py`), gateway (`modal/gateway.py`),
ingresses, and `sandbox_agent.py` are all valid Modal Functions but
**not standalone entry points** -- they share one App and need to be
served together (via `serve_all.py`) so cross-Function dispatch
(`agent_worker.spawn(...)`) resolves. Serving them in separate
processes fails because each `modal serve` registers an *ephemeral* App
that the others can't look up by name.

Each Function file still has its own `@app.local_entrypoint() def
smoke_<role>()` for `modal run`-style isolated smoke tests -- they
exercise the module's logic without bringing up the full pipeline.

---

## Prereqs

* **Modal CLI**, authenticated: `modal token set ...`
* **Convex CLI** + project linked (this repo's `multiplayer-ai/.env.local`
  points to `https://exuberant-albatross-781.convex.cloud`). The agent
  talks to *deployed* Convex over HTTPS -- you don't need a local Convex
  server. `npx convex dev` is only needed if you're editing `convex/*.ts`
  and want hot-redeploys. If the schema has never been pushed from this
  machine, run `npx convex dev --once` once to deploy it.
* **Python venv** at repo root with `modal`, `anthropic`, `httpx`, `fastapi`
  installed -- `source .venv/bin/activate`
* **`GEMINI_API_KEY` + `GEMINI_MODEL`** in env (or in
  `multiplayer-ai/.env.local`, where they already live) for non-stubbed
  LLM calls; the POC falls back to a deterministic stub when missing.
  Default model: `gemini-3-flash-preview`. Convex's `npx convex dev`
  reads .env.local automatically; Modal does not, so export them in
  the shell before `modal serve` (see Quickstart).
* **Convex agent skills (fresh clone only)**: run `cd multiplayer-ai &&
  npx convex ai-files install` once to populate `multiplayer-ai/skills/`,
  `multiplayer-ai/ai/`, and `multiplayer-ai/skills-lock.json`. These are
  gitignored regenerable artifacts — installing them gives Claude/Cursor
  the Convex-specific guidance referenced from `CLAUDE.md`.

---

## Quickstart: run the Slack agent

One `modal serve modal/serve_all.py` brings up the whole Modal control
plane (ingress + worker + gateway + scheduler + analytics) in a single
process. All paths below are relative to `multiplayer-ai/`.

> **About Convex.** The agent talks to a deployed Convex backend over
> HTTPS (the URL in `multiplayer-ai/.env.local`, currently
> `exuberant-albatross-781.convex.cloud`). It does NOT need a local
> Convex server. `npx convex dev` is only needed when you're **editing
> `convex/*.ts`** -- it watches the directory and redeploys on change.
> If the schema and routes are already deployed (they are after any
> prior `npx convex dev --once`), skip the Convex terminal entirely.

> **About the LLM.** The sandbox agent loop calls **Gemini 3 Flash
> Preview** via the `google-genai` SDK when `GEMINI_API_KEY` is in env
> (it lives in `multiplayer-ai/.env.local`). Without it, the loop falls
> back to a deterministic echo stub (`[stub-llm] received: …`) so the
> pipeline is exercisable offline. The stub path verifies *plumbing
> only* (HMAC, spawn, Volume mount, Convex writes, gateway dispatch)
> -- it never generates code-as-tools. The real-LLM path produces
> ```python``` blocks that `execute_with_recovery` runs inside the
> sandbox; verified end-to-end on the last run with `tool_calls=4,
> exec_success=True` in `agent_runs`.

> **`TOOL_GATEWAY_URL` chicken-and-egg.** The sandbox needs to know
> the gateway's URL before it can post replies through it, but Modal
> assigns that URL only after `modal serve` starts. Two-pass on first
> bootstrap: serve once, copy the gateway URL from the output, restart
> with it exported. After that the URL is sticky (Modal "steals" the
> label across restarts) so you can hardcode it.

### Demo path -- harness fires a synthetic Slack event (no real Slack app)

**First-time bootstrap (one extra step):**

```bash
cd multiplayer-ai && modal serve modal/serve_all.py
# Look in the output for:
#   🔨 Created web function tool_gateway =>
#     https://<workspace>--multiplayer-ai-tool-gateway-dev.modal.run
# Ctrl-C, copy that URL.
```

**Steady-state run:**

```bash
# Terminal 1 -- Modal control plane (ingress + worker + gateway + crons)
cd multiplayer-ai
# GEMINI_API_KEY + GEMINI_MODEL are read from .env.local automatically by
# `npx convex dev`, but Modal doesn't read .env.local -- export them here.
# Pick whichever pattern fits your shell:
export $(grep -E '^(GEMINI_API_KEY|GEMINI_MODEL)=' .env.local | sed 's/"//g')
export TOOL_GATEWAY_URL=https://<workspace>--multiplayer-ai-tool-gateway-dev.modal.run/dispatch
modal serve modal/serve_all.py

# (Optional) Terminal 2 -- Convex watcher, only if editing convex/*.ts
cd multiplayer-ai && npx convex dev

# Terminal 3 -- fire the signed Slack event at the live ingress URL
cd multiplayer-ai && python harness/send_event.py --channel slack \
  --url https://<workspace>--multiplayer-ai-slack-ingress-dev.modal.run/slack/events \
  --text "summarise the last week"
```

What happens after the harness fires:

1. `ingress_slack.py` verifies the HMAC signature (the harness signs
   with the same `dev-signing-secret-do-not-use-in-prod` the ingress
   verifies with -- no setup required), resolves `team_id → workspace_id`,
   and `Function.spawn(agent_worker)`.
2. `worker.py` mounts the per-workspace Modal Volume and spawns the
   Sandbox. The Sandbox image bakes `sandbox_agent.py` + `google-genai`.
3. `sandbox_agent.py` calls Gemini, emits ```python``` code-as-tools,
   `execute_with_recovery` runs them, writes thread / messages /
   tool_calls / debug_traces / agent_runs rows to Convex, and posts the
   final reply through the tool gateway.
4. The Convex dashboard shows new rows in real time. If `frontend/`
   is wired into a Next app, the reactive components re-render.

Verify Convex landed the rows:

```bash
python3 -c "
import json, urllib.request
def post(p, b):
    r = urllib.request.Request('https://exuberant-albatross-781.convex.site' + p,
        data=json.dumps(b).encode(), headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=15).read())
runs = post('/api/agent_runs/recent', {'workspace_id':'dev','limit':3})
for r in runs['value'][:3]:
    print(f\"  user={r['user_id']:14s} dur={r['duration_ms']}ms tool_calls={r['tool_calls_count']} success={r['exec_success']}\")
"
```

`tool_calls > 0, success=True` means the LLM path ran successfully. `tool_calls=0` with the agent still replying means the stub path (no GEMINI_API_KEY in env).

### Fully-local alternative (no Modal at all)

For the fastest iteration loop on `ingress_slack.py` itself, skip Modal
and run the FastAPI app directly with uvicorn:

```bash
# (Optional) Terminal 0 -- Convex watcher if you're editing convex/*.ts
cd multiplayer-ai && npx convex dev

# Terminal 1 -- raw FastAPI on localhost:8000
cd multiplayer-ai && uvicorn modal.ingress_slack:web_app --reload

# Then -- harness posts at localhost
cd multiplayer-ai && python harness/send_event.py --channel slack \
  --url http://localhost:8000/slack/events \
  --text "hello agent"
```

Caveat: under uvicorn the ingress can't actually spawn the worker (no
Modal context), so this verifies HMAC + routing only -- not the full
sandbox path. Use it for ingress iteration; switch to `modal serve
modal/serve_all.py` for the full end-to-end.

### Real Slack workspace (vs. harness)

The demo path uses the harness because it requires zero Slack-side
setup. To point a *real* Slack workspace at the agent:

1. Deploy stable URLs: `modal deploy modal/serve_all.py`. (One command
   deploys ingress + worker + gateway + crons together.)
2. Follow `multiplayer-ai/slack-integration/README.md` -- it walks
   through Slack-app creation, scopes (`app_mentions:read`,
   `chat:write`, ...), Event Subscriptions URL, and the Modal Secret
   that holds `SLACK_SIGNING_SECRET` + `SLACK_BOT_TOKEN`.

The harness keeps working against the deployed URL too, which is the
right way to smoke-test a new deployment before pointing real users at
it.

---

## Running end to end (the full procedure)

The quickstart above is enough to demo. This section is the *full*
procedure including smoke-testing every Modal module and optionally
running the dashboard.

### 1. (Optional) Start the Convex watcher

```bash
cd multiplayer-ai
npx convex dev
```

Optional -- see [Prereqs](#prereqs) for why the watcher isn't required
to run the agent. Leave it running if you're editing `convex/*.ts`.

### 2. Smoke-test each Modal module

See [Smoke-testing each Modal module](#smoke-testing-each-modal-module)
for the per-file `modal run` table -- one-shot verifies that catch
broken images / Volume mounts / signature math before you go to the
full pipeline.

> **Note on imports inside `modal/`.** The folder is named `modal/` to
> match the architecture diagram, but it is intentionally NOT a Python
> package -- there is no `__init__.py`. Each file inside imports siblings
> with `from common import ...`. This works under `modal run
> modal/<file>.py` because Modal CLI puts the file's directory on
> `sys.path`. **Do not** invoke these files via `python -m modal.<x>` --
> that treats the folder as a package and shadows the Modal SDK.

### 3. Serve + fire (the Slack pipeline)

The 3-terminal serve + harness fire is documented in
[Quickstart](#quickstart-run-the-slack-agent). Use that section verbatim
once the smoke tests pass; the full procedure adds nothing beyond it
except the dashboard step below.

### 4. (optional) Dashboard

```bash
cd multiplayer-ai/frontend && npx create-next-app@latest . --typescript --use-npm --yes
# drop convexClient.ts + components/*.tsx into the new app
npm run dev
```

---

## Ingress points

| Ingress              | URL pattern (deployed)                                                          | Auth                                    | What it does                                                |
|----------------------|---------------------------------------------------------------------------------|-----------------------------------------|-------------------------------------------------------------|
| Slack `ingress_slack`| `https://<workspace>--multiplayer-ai-slack-ingress.modal.run/slack/events`     | `X-Slack-Signature` HMAC v0 + 5-min replay window | Verify HMAC, handle `url_verification`, ACK <3s, `team_id → workspace_id`, `Function.spawn(agent_worker)`. |
| Teams `ingress_teams`| `https://<workspace>--multiplayer-ai-teams-ingress.modal.run/teams/messages`   | Bearer token (mimic; real = Bot Framework JWT) | Accept Activity JSON dict, validate Bearer, normalise into the SAME context Slack emits, spawn worker. |
| Tool gateway         | `https://<workspace>--multiplayer-ai-tool-gateway.modal.run/dispatch`          | (internal; sandbox-only audience)       | Single egress for sandbox. Dispatches by action namespace (`slack.send`, `teams.send`, `github.create_issue`). Rate limit + audit (background task). |
| `/health` on each    | `<ingress-url>/health`                                                          | none                                    | Liveness probe.                                             |

Local dev: each Modal file can be served locally with `modal serve` and
hit via the URL Modal prints. The harness defaults to `localhost`-style
URLs but works against any URL.

---

## The harness

`multiplayer-ai/harness/send_event.py` synthesises correctly-formatted
events and POSTs them at the ingress. No real Slack workspace or Teams
tenant required.

```bash
# Slack message event (default)
python multiplayer-ai/harness/send_event.py --channel slack \
  --url <ingress-url>/slack/events \
  --text "hello agent"

# Slack url_verification challenge (one-shot, used during Slack app setup)
python multiplayer-ai/harness/send_event.py --channel slack --verify \
  --url <ingress-url>/slack/events

# Teams Activity
python multiplayer-ai/harness/send_event.py --channel teams \
  --url <teams-ingress-url>/teams/messages \
  --text "hello agent from teams"

# Override the tenant identifiers (the resolve_workspace_id maps live in env)
python multiplayer-ai/harness/send_event.py --channel slack --url <url> --text "x" \
  --team-id T0YOUR --user-id U0YOU --slack-channel D0YOURDM
```

The harness signs Slack events with the same `SLACK_SIGNING_SECRET` the
ingress verifies with. Both read it from env and fall back to the same
dev default (`dev-signing-secret-do-not-use-in-prod`) so the harness and
ingress agree out of the box -- no setup step.

---

## Smoke-testing each Modal module

Every file in `multiplayer-ai/modal/` has its own
`@app.local_entrypoint()`, runnable standalone with `modal run`.

| Command (from `multiplayer-ai/`)        | What it verifies                                                                  |
|-----------------------------------------|-----------------------------------------------------------------------------------|
| `modal run modal/ingress_slack.py`      | HMAC math is internally consistent. POSTs against `SLACK_INGRESS_URL` if set.      |
| `modal run modal/ingress_teams.py`      | Teams Activity normalises to the same context shape Slack emits.                  |
| `modal run modal/worker.py`             | Sandbox spawns, image builds, Volume mounts, sandbox_agent runs end-to-end.       |
| `modal run modal/gateway.py`            | Dispatcher routes by action namespace; stub branches return without external API. |
| `modal run modal/scheduler.py`          | Full 6-stage pipeline. Prints which gates dropped which candidates.               |
| `modal run modal/analytics.py`          | Feature compute + prediction; writes a row to `workspace_predictions`.            |

Direct Python invocation (for the loop itself):

```bash
cd multiplayer-ai/modal && python sandbox_agent.py '{"workspace_id":"dev","channel":"D01","channel_type":"im","user_id":"U01","message":"hello","channel_origin":"slack"}'
```

Useful when iterating on the agent loop without paying the Modal
Sandbox spawn cost.

---

## End-to-end gotchas (issues found running this for real)

A live `modal serve modal/serve_all.py` + harness fire on a clean
clone surfaces architectural pitfalls that no unit test or local-only
smoke catches. They're fixed in this codebase; documenting them so the
next operator doesn't rediscover them:

1. **Sibling imports inside Modal containers.** Modal serves a single
   `.py` file by default; sibling `from common import …` fails in the
   deployed container because the source files aren't bundled. Fix:
   `control_plane_image.add_local_dir(modal_dir, "/root", …)` in
   `common.py` copies the whole `modal/` directory into the image.

2. **Duplicate `@app.local_entrypoint() def main()` across files.**
   When one file imports another at request time, both files' `main`
   entrypoints register on the same App → `InvalidError: Duplicate
   local entrypoint name`. Fix: every Function file's smoke entrypoint
   was renamed to a file-specific name (`smoke_slack_ingress`,
   `smoke_worker`, …).

3. **Cross-process `Function.spawn` doesn't work in `modal serve`.**
   `Function.from_name("multiplayer-ai", "agent_worker")` looks for a
   *deployed* App, not an ephemeral served one — and three separate
   `modal serve` processes register three *ephemeral* Apps that can't
   look each other up. Fix: `modal/serve_all.py` aggregates all
   Function modules so one `modal serve` registers them in a single
   in-process App, and sibling `from worker import agent_worker`
   resolves directly.

4. **`Image.add_local_file` re-resolves the source path at Sandbox
   creation time.** Even with `copy=True`, the worker container's
   common.py re-evaluates `sandbox_image = … add_local_file(…)`, which
   resolves `__file__`-relative against the worker's filesystem. If
   `sandbox_agent.py` isn't present at `/root/sandbox_agent.py` in the
   worker, Modal's mount dedup raises `FileNotFoundError`. Fix:
   include `sandbox_agent.py` in *both* images (the control-plane
   image bundles `modal/` wholesale; the sandbox image copies it
   explicitly with `copy=True`).

5. **`TOOL_GATEWAY_URL` chicken-and-egg.** Modal assigns the gateway
   URL only after `serve` starts, but `secrets()` builds the
   sandbox-facing Secret at import time. First-time bootstrap is
   two-pass; the URL is sticky after that ("label stolen" across
   restarts), so the second pass is the only one you ever need.

6. **Convex AI Files clobber CLAUDE.md / AGENTS.md on first install.**
   `npx convex ai-files install` overwrites both files from scratch
   if it doesn't find its `<!-- convex-ai-start --><!-- convex-ai-end -->`
   markers. Restored content + embedded the markers; do not edit
   between them.

7. **Slack agent runs without LLM by default.** With `GEMINI_API_KEY`
   unset, `call_gemini` returns a stub echo; the agent reply is
   `[stub-llm] received: …`, no `tool_calls` row gets written, and
   `exec_success=False`. Plumbing works (HMAC, spawn, Volume, Convex
   writes, gateway dispatch) but no real LLM reasoning. Export the
   key to exercise the full code-as-tools path.

## Demo flow

1. **Architecture (90s)** -- show this diagram. Land "symmetric gateway,
   sandwich isolation."
2. **Basic flow (90s)** -- harness fires an event, sandbox runs, response
   posts back, dashboard updates live.
3. **NF1 -- debug surface (2m)** -- trigger a flaky script, watch attempt
   1 fail, attempt 2 succeed, show the diff in the DebugPanel.
4. **NF2 -- proactive (2m)** -- run a scan, point at the 4-score
   breakdown, then at the trust score and frequency cap in the panel,
   then at a filtered candidate's `genericness_score`. Demo Stage 6 by
   rejecting one with a reason and watching the trust score tick down.
5. **NF3 -- skill versioning (2m)** -- show the SkillHistory timeline,
   diff between two versions, rollback button.
6. **NF4 (optional, 90s)** -- ObservabilityDash with the latest
   prediction row.

---

## How code-as-tools actually runs inside the sandbox

The system prompt tells the LLM that `file_read`, `file_edit`, and
`send_via_gateway` are available. Those helpers live in
`/tmp/agent_tools.py`, materialised at sandbox boot by
`write_agent_tools_module()`. Every generated script is prepended with
`from agent_tools import *` before `subprocess.run`, and the subprocess
is launched with `PYTHONPATH=/tmp` so the import resolves. The
architectural consequence: the agent literally cannot call external APIs
without going through `send_via_gateway` -- there is no alternative path.

---

## Design points to land

1. **Symmetric gateway = sandwich isolation.** Sandbox cannot make
   outbound calls except through the gateway. Audit / rate-limit / per-
   workspace creds all live at the edges.
2. **Multi-tenancy at the Volume layer.** One Volume per workspace.
   Spawn-time parameterisation.
3. **Progressive disclosure on skills.** Descriptions in the prompt;
   full `SKILL.md` only on demand via `file_read`.
4. **Reactive UI falls out of Convex.** No polling, no manual
   websockets.
5. **Channel-agnostic downstream.** Slack and Teams ingresses normalise
   to the same context. Worker / sandbox / gateway never know which
   channel fired.
6. **The proactive layer is data-driven, not hard-coded.** Every
   non-invasiveness gate reads from `workspace_proactive_state`. Adding
   a gate is "one function + one column."
7. **Trust earned, not assumed.** EMA-updated `trust_score`
   transactionally adjusts the surfacing threshold. Five consecutive
   rejections auto-pauses the workspace.
8. **Causality tracked.** Every rejection carries a `rejection_reason`
   so we can learn from negative outcomes, not just acceptance rates.

---

## What this POC does not ship

* Real Bot Framework JWT validation (the Teams ingress accepts a dev
  Bearer token; see `multiplayer-ai/teams-integration/README.md`).
* A trained XGBoost model -- analytics is the *pattern*, not a finished
  model. Predictions today come from a transparent linear blend so the
  demo audience can see what drives each score.
* Pipedream Connect (the architecture supports it; we don't need 3000+
  integrations to demo the loop).
* A Next.js shell -- frontend components are scaffolds you drop into a
  fresh Next app.
* Embedding-based rejection similarity (bag-of-words cosine is the POC
  stand-in; the gate interface stays the same).

---

## Attributions

* `modal-examples-main/` is a vendored copy of
  <https://github.com/modal-labs/modal-examples>, **MIT-licensed,
  © 2022 Modal Labs**. The original `LICENSE` file is preserved inside
  the folder.

---

## Disclaimer

Everything in this repository was built from publicly available
materials and reflects the repo owner's own interpretation and
implementation choices. **No confidential, proprietary, or non-public
information of any kind was used.** Any resemblance to an internal
system, architecture, or product is coincidental or derived strictly
from public documentation, talks, blog posts, and product testimony.
