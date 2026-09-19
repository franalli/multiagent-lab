# modal/sandbox_agent.py
#
# THIS SCRIPT RUNS INSIDE THE MODAL SANDBOX.
#
# It is NOT a Modal Function definition. The worker (modal/worker.py) bakes
# this file into the sandbox image at /app/sandbox_agent.py and starts it as
# the entrypoint process:
#
#   sandbox.<exec>("python", "/app/sandbox_agent.py", json.dumps(context))
#
# WHAT IT DOES (the synchronous agent loop, POC version)
#   1. Load skill DESCRIPTIONS from /workspace/skills/**/SKILL.md frontmatter.
#      Only descriptions go in the system prompt -- progressive disclosure.
#   2. Call Gemini with system prompt + user message.
#   3. If the reply contains a fenced ```python code block, treat it as a
#      tool invocation (code-as-tools).
#   4. Execute the script as a subprocess IN THIS sandbox with per-attempt
#      tracing + LLM-driven recovery (NOVEL FEATURE 1).
#   5. Feed the tool result back into the conversation and loop.
#   6. When the reply has no code, treat it as the final answer: post it
#      via the tool gateway and write it to Convex.
#
# WHY NO NESTED Sandboxes:
#   The doc was explicit -- "no nested Sandbox.create calls; same container."
#   We run the generated code as subprocess.run() inside the already-spawned
#   sandbox. That keeps the per-turn latency bounded.

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# google-genai is installed in the sandbox image; httpx too.
try:
    from google import genai as _genai
    from google.genai import types as _genai_types
except ImportError:  # in dev / unit testing the import might be missing
    _genai = None  # type: ignore[assignment]
    _genai_types = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE = "/workspace"
SKILLS_ROOT = f"{WORKSPACE}/skills"

# Mirrors the COST_PER_LLM_CALL_USD import-or-fallback pattern below: the
# defining home is common.py, but standalone `python sandbox_agent.py` runs
# may not be able to import common (its `import modal` can fail outside the
# Modal container). Fall back to a direct env read in that case.
try:
    from common import DEFAULT_GEMINI_MODEL as DEFAULT_MODEL
except ImportError:
    DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

MAX_TURNS = 4  # cap the agent loop to avoid runaway costs
MAX_RECOVERY_ATTEMPTS = 3  # retries inside execute_with_recovery (NF1)

# Shared telemetry path: agent_tools.py appends one JSON line per skill
# read/write; the parent loop reads it back at end-of-turn for agent_runs.
SKILL_TELEMETRY_PATH = "/tmp/skill_log.jsonl"

# Per-LLM-call cost approximation. Lives in common.py so analytics agrees
# on the same number; importable here because sandbox_agent runs INSIDE the
# sandbox image which has common.py mounted via the same Volume.
# (This file is also runnable standalone with `python sandbox_agent.py`
# during dev, where common.py's `import modal` may fail -- ImportError
# subsumes ModuleNotFoundError, so that's the precise gate.)
# WARNING: the fallback literal must stay in sync with common.COST_PER_LLM_CALL_USD.
try:
    from common import COST_PER_LLM_CALL_USD
except ImportError:
    COST_PER_LLM_CALL_USD = 0.002  # keep in sync with common.COST_PER_LLM_CALL_USD

# Tool gateway URL is injected via env by the worker. For local-only test
# runs (no gateway deployed yet), we stub the post and log the action.
TOOL_GATEWAY_URL = os.environ.get("TOOL_GATEWAY_URL", "")
CONVEX_SITE_URL = os.environ.get("CONVEX_SITE_URL", "https://exuberant-albatross-781.convex.site")


# ---------------------------------------------------------------------------
# Skill loading (progressive disclosure)
# ---------------------------------------------------------------------------


def parse_frontmatter(text: str) -> dict[str, str]:
    """Tolerant YAML-frontmatter parser. Just key:value pairs, no nesting."""
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    out: dict[str, str] = {}
    for line in parts[1].strip().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_skill_index() -> dict[str, str]:
    """Return {skill_path: one-line description}.

    The full body of each SKILL.md is NOT loaded here. The system prompt
    gets only the descriptions; the agent uses file_read() on demand to
    pull a full skill when it picks one. That's the progressive-disclosure
    pattern.
    """
    index: dict[str, str] = {}
    root = Path(SKILLS_ROOT)
    if not root.exists():
        return index
    for path in root.rglob("SKILL.md"):
        try:
            text = path.read_text()
        except OSError:
            continue
        meta = parse_frontmatter(text)
        description = meta.get("description") or text.splitlines()[0][:140]
        index[str(path)] = description
    return index


# Convex HTTP wrapper -- best-effort writes, swallowed errors are fine for
# telemetry. agent_tools.send_via_gateway has the matching contract.
#
# None-valued keys are stripped before serialising: Convex `v.optional(...)`
# validators reject explicit JSON `null` and require the field be omitted.
def convex_post(route: str, payload: dict[str, Any]) -> dict[str, Any]:
    payload = {k: v for k, v in payload.items() if v is not None}
    raw = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{CONVEX_SITE_URL.rstrip('/')}{route}",
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode()
        return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"convex_post {route} HTTP {e.code}: {e.read().decode()[:200]}\n")
        return {}
    except urllib.error.URLError as e:
        sys.stderr.write(f"convex_post {route} URLError: {e}\n")
        return {}


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def build_system_prompt(skill_index: dict[str, str], context: dict[str, Any]) -> str:
    """Compose the system prompt: identity + skill descriptions + context.

    The system prompt tells the LLM that tools live in the `agent_tools`
    module. write_agent_tools_module() drops that module on disk before any
    script runs, and execute_with_recovery prepends the import to every
    generated script -- so `send_via_gateway(...)` resolves correctly when
    the subprocess executes.
    """
    skill_lines = "\n".join(f"- {p}: {d}" for p, d in sorted(skill_index.items()))
    return f"You are a multiplayer-AI assistant running inside a Modal sandbox.\nYou can call tools by emitting Python in a single ```python ...``` block.\nEvery script you emit is prepended with `from agent_tools import *`\nbefore execution, so the following helpers are available:\n  file_read(path)                  -> read a file from the workspace volume\n  file_edit(path, content)         -> write a file to the workspace volume\n  send_via_gateway(action, params) -> proxied egress through the tool gateway\nDO NOT call external HTTP APIs directly -- always go through send_via_gateway.\n\nIf you need a skill, file_read the full SKILL.md by path. Only DESCRIPTIONS\nare shown below; full files load on demand.\n\nAvailable skills:\n{skill_lines or '(none yet)'}\n\nWorkspace: {context.get('workspace_id')}\nUser: {context.get('user_id')}\nChannel: {context.get('channel')} ({context.get('channel_type')})\n\nWhen you have a final answer with no more tool calls, reply in plain text.\n"


# ---------------------------------------------------------------------------
# Tool injection -- the subprocess can't see this file's local functions, so
# we materialise an `agent_tools` module at /tmp and prepend `from agent_tools
# import *` to every script before running it. Matches the system prompt
# above.
# ---------------------------------------------------------------------------

AGENT_TOOLS_DIR = "/tmp"
AGENT_TOOLS_PATH = f"{AGENT_TOOLS_DIR}/agent_tools.py"

AGENT_TOOLS_SOURCE = '''
"""agent_tools -- helpers injected into every generated script.

This file is materialised at sandbox boot by sandbox_agent.py. Generated
scripts get `from agent_tools import *` prepended automatically, so they
can call file_read / file_edit / send_via_gateway without redeclaring them.
"""

import difflib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

WORKSPACE = "/workspace"
SKILLS_ROOT = "/workspace/skills"
TOOL_GATEWAY_URL = os.environ.get("TOOL_GATEWAY_URL", "")
WORKSPACE_ID = os.environ.get("WORKSPACE_ID", "dev")
CONVEX_SITE_URL = os.environ.get("CONVEX_SITE_URL", "")
# Skill telemetry log. The parent agent loop creates and drains this file
# at end-of-turn so per-run skill_reads / skill_writes land in agent_runs.
SKILL_TELEMETRY_PATH = os.environ.get("SKILL_TELEMETRY_PATH", "/tmp/skill_log.jsonl")

# Maps the top-level directory under SKILLS_ROOT to the schema's
# skillCategoryValidator. The volume uses pluralised "users/" while the
# schema validates against the singular "user" -- this is the one and only
# place that mapping lives. Folders not in the map are non-skill writes and
# the version indexer skips them.
_SKILL_DIR_TO_CATEGORY = {
    "company": "company",
    "team": "team",
    "users": "user",
    "integration": "integration",
    "workflow": "workflow",
}


def _log_skill_event(kind, path):
    """Append a `{kind, path}` record to the per-turn telemetry log so the
    parent loop can include skill_reads / skill_writes in the agent_runs
    row. Best-effort -- a logging failure must not break the script."""
    if not SKILL_TELEMETRY_PATH:
        return
    try:
        with open(SKILL_TELEMETRY_PATH, "a") as f:
            f.write(json.dumps({"kind": kind, "path": path}) + "\\n")
    except OSError:
        pass


def file_read(path):
    """Read `path` from the per-workspace Volume. If `path` is a SKILL.md
    under SKILLS_ROOT, emit a skill_reads telemetry row so the NF4
    skill_diversity_index can see what the agent consulted."""
    text = Path(path).read_text()
    if path.startswith(SKILLS_ROOT + "/") and path.endswith("/SKILL.md"):
        _log_skill_event("read", path)
    return text


def file_edit(path, content):
    """Write `content` to `path`. If `path` is a SKILL.md under SKILLS_ROOT,
    additionally append a row to the Convex skill_version_index so the
    NF3 admin surface can render the new version + diff.

    The Volume write happens *first* so that a Convex outage doesn't lose
    the actual skill edit -- worst case the index is stale until the next
    edit. (Source of truth is the Volume; Convex is a denormalised view.)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    path_str = str(p)

    category = _skill_category(path_str)
    prev_content = p.read_text() if (category and p.exists()) else None

    p.write_text(content)

    if category is not None:
        _log_skill_event("write", path_str)
        _record_skill_version(path_str, category, prev_content or "", content)


def _skill_category(path_str):
    """Return the skill category if `path_str` is a SKILL.md under
    SKILLS_ROOT, else None."""
    if not (path_str.startswith(SKILLS_ROOT + "/") and path_str.endswith("/SKILL.md")):
        return None
    parts = path_str[len(SKILLS_ROOT) + 1 :].split("/")
    return _SKILL_DIR_TO_CATEGORY.get(parts[0])


def _record_skill_version(path_str, category, prev_content, new_content):
    """Compute a unified diff + POST a new row to skill_version_index.
    Best-effort: never raises out of file_edit -- the volume write already
    succeeded, the index is allowed to drift on transient HTTP failure."""
    if not CONVEX_SITE_URL:
        sys.stderr.write("[skill_version] CONVEX_SITE_URL unset; skipping index\\n")
        return
    skill_name = path_str[len(SKILLS_ROOT) + 1 :]
    diff = "".join(
        difflib.unified_diff(
            prev_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{skill_name}",
            tofile=f"b/{skill_name}",
            n=2,
        )
    ) or "(no line changes)"
    payload = {
        "workspace_id": WORKSPACE_ID,
        "skill_name": skill_name,
        "category": category,
        "content": new_content,
        "diff": diff,
        "modified_by": "agent",
        "change_summary": "agent edit via file_edit",
    }
    try:
        raw = json.dumps(payload).encode()
        req = urllib.request.Request(
            CONVEX_SITE_URL.rstrip("/") + "/api/skills/save_version",
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        sys.stderr.write(f"[skill_version] index write failed: {e}\\n")


def send_via_gateway(action, params):
    """Single chokepoint. If TOOL_GATEWAY_URL is unset, log and stub."""
    payload = {"action": action, "params": params, "workspace_id": WORKSPACE_ID}
    if not TOOL_GATEWAY_URL:
        sys.stderr.write(f"[gateway:stub] {action} {json.dumps(params)[:200]}\\n")
        return {"stubbed": True}
    raw = json.dumps(payload).encode()
    req = urllib.request.Request(
        TOOL_GATEWAY_URL,
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode()
    return json.loads(body) if body else {}
'''


def _agent_tools():
    """Materialise + import agent_tools. Single source for both the parent
    agent loop and the subprocess executor -- the helpers don't drift."""
    write_agent_tools_module(os.environ.get("WORKSPACE_ID", "dev"))
    if AGENT_TOOLS_DIR not in sys.path:
        sys.path.insert(0, AGENT_TOOLS_DIR)
    import agent_tools  # deferred: module path only valid after writing it

    return agent_tools


def write_agent_tools_module(workspace_id: str) -> None:
    """Drop /tmp/agent_tools.py once at startup. PYTHONPATH=/tmp picks it up
    when subprocess.run launches the script."""
    Path(AGENT_TOOLS_PATH).write_text(AGENT_TOOLS_SOURCE.lstrip())
    # The agent_tools module reads WORKSPACE_ID and CONVEX_SITE_URL from the
    # subprocess env. Worker injects TOOL_GATEWAY_URL the same way.
    os.environ.setdefault("WORKSPACE_ID", workspace_id)
    os.environ.setdefault("CONVEX_SITE_URL", CONVEX_SITE_URL)


def call_gemini(system: str, messages: list[dict[str, Any]]) -> tuple[str, int]:
    """Best-effort LLM call against Gemini. Returns (text, prompt_tokens).

    `prompt_tokens` comes from `usage_metadata.prompt_token_count` on
    the response -- the ephemeral context layer's per-turn cost proxy.
    The agent loop accumulates it across all LLM calls in the turn
    (initial + recovery retries) and persists the total on the
    agent_runs row.

    Falls back to a deterministic stub when no API key is configured;
    in that path prompt_tokens=0 (no API call was made).

    Gemini's API takes a single `contents` list (alternating user/model
    roles) plus a `system_instruction` config. We map the loop's
    {role: "user"|"assistant", content: str} messages into Gemini's
    Content+Part shape, treating "assistant" as Gemini's "model" role.

    POC uses Gemini 3 Flash Preview. Provider swap is one function (this
    one) plus the convex/llm.ts action -- the agent-loop shape is
    provider-agnostic.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or _genai is None or _genai_types is None:
        # Deterministic stub: echo the last user message back.
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"[stub-llm] received: {last_user[:160]}", 0

    contents = [
        _genai_types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[_genai_types.Part.from_text(text=m["content"])],
        )
        for m in messages
    ]
    client = _genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=contents,
        config=_genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1024,
        ),
    )
    # usage_metadata may be absent on some SDK error paths -- treat as 0.
    usage = getattr(resp, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    return resp.text or "", prompt_tokens


# ---------------------------------------------------------------------------
# Code extraction (code-as-tools)
# ---------------------------------------------------------------------------

CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code_block(reply: str) -> str | None:
    """Return the first ```python``` block, or None."""
    m = CODE_BLOCK_RE.search(reply)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Execute-with-recovery (NOVEL FEATURE 1)
# ---------------------------------------------------------------------------


SCRIPT_PREAMBLE = "from agent_tools import *  # injected by sandbox_agent.py\n"


def execute_with_recovery(
    script: str,
    *,
    tool_call_id: str,
    workspace_id: str,
    max_attempts: int = MAX_RECOVERY_ATTEMPTS,
) -> dict[str, Any]:
    """Run generated code with per-attempt tracing and LLM-driven recovery.

    Every attempt writes a debug_traces row in Convex so the DebugPanel can
    diff consecutive attempts and show what the recovery LLM changed.

    The generated script is prepended with `from agent_tools import *` so
    that file_read / file_edit / send_via_gateway resolve correctly. That
    module is materialised at sandbox boot by write_agent_tools_module().

    Code-as-tools with LLM repair is the underlying pattern; POC's NOVEL
    contribution (NF1) is the per-attempt debug_traces surface -- making
    the recovery loop visible/reviewable instead of opaque. The
    subprocess-in-same-sandbox approach (not a nested Sandbox.create)
    keeps repair cycles cheap.
    """
    current_script = script
    recovery_prompt_tokens = 0
    for attempt in range(1, max_attempts + 1):
        # Persist the script to /tmp so subprocess.run can execute it. We
        # prepend the import unconditionally; if the LLM already wrote one,
        # `import *` is idempotent.
        script_path = Path("/tmp/script.py")
        script_path.write_text(SCRIPT_PREAMBLE + current_script)

        # PYTHONPATH=/tmp ensures the subprocess can find agent_tools.
        sub_env = {
            **os.environ,
            "PYTHONPATH": f"{AGENT_TOOLS_DIR}:{os.environ.get('PYTHONPATH', '')}",
        }
        completed = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=sub_env,
        )

        convex_post(
            "/api/debug_traces/append",
            {
                "workspace_id": workspace_id,
                "tool_call_id": tool_call_id,
                "attempt": attempt,
                "script": current_script,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "exit_code": completed.returncode,
            },
        )

        if completed.returncode == 0:
            return {
                "status": "success" if attempt == 1 else "recovered",
                "result": completed.stdout,
                "attempts": attempt,
                "prompt_tokens": recovery_prompt_tokens,
            }

        # Failed -- ask Gemini to fix it. The recovery prompt deliberately
        # contains the full stderr so the model can see the actual error.
        recovery_user = f"The script you wrote failed.\nSTDERR:\n{completed.stderr.strip()}\nSTDOUT:\n{completed.stdout.strip()}\nReturn ONLY a corrected ```python``` block."
        fix_reply, fix_tokens = call_gemini(
            system="You are debugging a failed Python script. Return ONLY a corrected ```python``` block.",
            messages=[{"role": "user", "content": recovery_user}],
        )
        recovery_prompt_tokens += fix_tokens
        fixed = extract_code_block(fix_reply) or current_script
        current_script = fixed

    return {
        "status": "error",
        "error": "max recovery attempts exhausted",
        "attempts": max_attempts,
        "last_script": current_script,
        "prompt_tokens": recovery_prompt_tokens,
    }


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------


def _reset_skill_telemetry() -> None:
    """Truncate the per-turn skill telemetry log. Called once at run start so
    a previous run's events don't bleed into this one's agent_runs row."""
    try:
        Path(SKILL_TELEMETRY_PATH).write_text("")
    except OSError:
        pass


def _drain_skill_telemetry() -> tuple[list[str], list[str]]:
    """Read + reset the telemetry log. Returns (reads, writes) for agent_runs."""
    reads: list[str] = []
    writes: list[str] = []
    try:
        raw = Path(SKILL_TELEMETRY_PATH).read_text()
    except OSError:
        return reads, writes
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("kind") == "read":
            reads.append(evt.get("path", ""))
        elif evt.get("kind") == "write":
            writes.append(evt.get("path", ""))
    return reads, writes


# ---------------------------------------------------------------------------
# Two memory layers (the spec's articulation point)
# ---------------------------------------------------------------------------
# The agent loop below interacts with TWO architecturally distinct memory
# stores; conflating them is the most common newcomer mistake:
#
#   DURABLE -- the per-workspace Modal Volume. `file_read` / `file_edit` on
#   SKILL.md files (see agent_tools.py inside AGENT_TOOLS_SOURCE) is
#   literally the Anthropic memory-tool pattern with the Volume as the
#   backing store. Survives across turns, workspaces, and sandbox
#   terminations; git-versioned at the filesystem layer; mirrored to
#   skill_version_index in Convex for the NF3 admin UX.
#
#   EPHEMERAL -- the `messages` list this loop appends to each turn. Lives
#   only for the duration of run_agent_loop. In production this is what
#   Anthropic's context-management beta header / Opus 4.5's auto-compaction
#   would manage. POC keeps it bounded by MAX_TURNS (=4) instead -- per
#   spec line 994, no compactor needed at POC scale. We *measure* the
#   layer's cost via `prompt_tokens_total` for visibility, but don't
#   compact: when the ratio of (peak prompt tokens / model context window)
#   approaches 0.5+, that's the signal to wire a real compactor.
# ---------------------------------------------------------------------------


def run_agent_loop(context: dict[str, Any]) -> str:
    """The agent loop. Reads SKILL.md memory, calls Gemini, executes emitted
    python via execute_with_recovery, posts the final answer through the
    gateway, writes telemetry to Convex. See the two-memory-layers block
    above for the durable/ephemeral story.
    """
    workspace_id = context["workspace_id"]
    started_at = time.time()
    # Drop /tmp/agent_tools.py once, before any tool-using script runs.
    write_agent_tools_module(workspace_id)
    os.environ["SKILL_TELEMETRY_PATH"] = SKILL_TELEMETRY_PATH
    _reset_skill_telemetry()

    skills = load_skill_index()
    system = build_system_prompt(skills, context)

    # Open / find the thread, append the inbound user message.
    thread_resp = convex_post(
        "/api/threads/get_or_create",
        {
            "workspace_id": workspace_id,
            "channel": context.get("channel", "unknown"),
            "channel_type": context.get("channel_type", "im"),
        },
    )
    thread_id = thread_resp.get("thread_id")
    if thread_id:
        convex_post(
            "/api/messages/append",
            {
                "workspace_id": workspace_id,
                "thread_id": thread_id,
                "user_id": context.get("user_id", "U00UNKNOWN"),
                "role": "user",
                "content": context.get("message", ""),
            },
        )

    messages: list[dict[str, Any]] = [{"role": "user", "content": context.get("message", "")}]
    final_text = ""

    tool_calls_count = 0
    total_recovery_attempts = 0
    llm_calls = 0
    prompt_tokens_total = 0
    any_tool_call_failed = False

    for _turn in range(MAX_TURNS):
        llm_calls += 1
        reply, reply_tokens = call_gemini(system, messages)
        prompt_tokens_total += reply_tokens
        code = extract_code_block(reply)
        if not code:
            final_text = reply
            break

        tool_calls_count += 1
        # Open a tool_calls row, then run with recovery.
        tc_resp = convex_post(
            "/api/tool_calls/log",
            {"workspace_id": workspace_id, "thread_id": thread_id, "script": code},
        )
        tool_call_id = tc_resp.get("tool_call_id", "")
        result = execute_with_recovery(code, tool_call_id=tool_call_id, workspace_id=workspace_id)
        attempts = result.get("attempts", 1)
        if attempts > 1:
            # Each retry triggered a fresh LLM repair call inside execute_with_recovery.
            llm_calls += attempts - 1
            total_recovery_attempts += attempts - 1
        prompt_tokens_total += result.get("prompt_tokens", 0)
        if result["status"] == "error":
            any_tool_call_failed = True
        convex_post(
            "/api/tool_calls/update",
            {
                "tool_call_id": tool_call_id,
                "status": result["status"],
                "attempts": attempts,
                "result": result.get("result"),
                "error": result.get("error"),
            },
        )

        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {
                "role": "user",
                "content": f"TOOL RESULT:\n{result.get('result') or result.get('error')}",
            }
        )

    # Post the final answer back through the gateway AND record it in Convex.
    if final_text:
        tools = _agent_tools()
        gateway_action = "slack.send" if context.get("channel_origin") == "slack" else "teams.send"
        tools.send_via_gateway(
            gateway_action,
            {"channel": context.get("channel", ""), "text": final_text},
        )
        if thread_id:
            convex_post(
                "/api/messages/append",
                {
                    "workspace_id": workspace_id,
                    "thread_id": thread_id,
                    "user_id": "agent",
                    "role": "agent",
                    "content": final_text,
                },
            )

    # Write the agent_runs telemetry row -- feeds NF4 analytics.
    # We do this last so failures upstream still produce a row with the
    # exec_success=False signal that the analytics layer learns from.
    #
    # exec_success is True only when at least one tool call ran AND none
    # of them failed. Turns that never invoked code-as-tools are excluded
    # from "executed successfully" -- they neither succeed nor fail at
    # execution by definition. Analytics treats `True` as positive signal,
    # `False` as a real failure; turns with no tool calls just don't lift
    # the success_rate either way (they reduce the denominator if you
    # filter them out -- which analytics.py can do later).
    if thread_id:
        skill_reads, skill_writes = _drain_skill_telemetry()
        duration_ms = int((time.time() - started_at) * 1000)
        exec_success = tool_calls_count > 0 and not any_tool_call_failed
        convex_post(
            "/api/agent_runs/save",
            {
                "workspace_id": workspace_id,
                "user_id": context.get("user_id", "U00UNKNOWN"),
                "thread_id": thread_id,
                "duration_ms": duration_ms,
                "tool_calls_count": tool_calls_count,
                "exec_success": exec_success,
                "recovery_attempts": total_recovery_attempts,
                "cost_estimate_usd": round(llm_calls * COST_PER_LLM_CALL_USD, 4),
                "prompt_tokens": prompt_tokens_total,
                "skill_reads": skill_reads,
                "skill_writes": skill_writes,
            },
        )

    return final_text


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: sandbox_agent.py <context-json>\n")
        sys.exit(2)
    context = json.loads(sys.argv[1])
    started = time.time()
    output = run_agent_loop(context)
    elapsed_ms = int((time.time() - started) * 1000)
    sys.stdout.write(output + "\n")
    sys.stderr.write(f"[sandbox_agent] done in {elapsed_ms}ms\n")


if __name__ == "__main__":
    main()
