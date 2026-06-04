"""External tool sources: MCP servers and SKILL.md skills.

Sibling of 07_production_agent.py. 07's tools were defined in-process with the
`@tool` decorator. Real agents also pull capability from *outside* the code:

  - MCP servers expose tools over a protocol (one process, or a remote URL),
    so you can compose third-party capability without importing an SDK per
    integration. Here we drive the repo's own KV server over stdio.

  - Agent Skills are filesystem folders, each with a `SKILL.md`. They package
    task-specific *instructions* (and optional scripts/references) that load on
    demand, keeping the base prompt small while staying discoverable.

Both share one principle this file is built around — **progressive disclosure**,
in three stages so the context window only ever holds what the task needs:

  Skills:   metadata (name + description)  ->  full SKILL.md on activation
            ->  bundled scripts/references only on execution
  MCP:      tool name + description + schema is fetched once per session and
            held open (re-spawning the server per turn would be wasteful).

The split mirrors Anthropic's tool-search/skills design: keep the fixed surface
tiny, fetch detail when (and only when) it's needed.

Requires: ANTHROPIC_API_KEY, `pip install anthropic mcp`. The MCP half drives
../07_mcp/04_stateful_kv_server.py; the skills half reads ./skills/.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from anthropic import AsyncAnthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
SKILLS_DIR = Path(__file__).parent / "skills"
# Reuse an MCP server that already exists in this repo (stateful key/value store).
MCP_SERVER_PATH = str(
    Path(__file__).parent.parent / "07_mcp" / "04_stateful_kv_server.py"
)


# ===========================================================================
# SOURCE 1: SKILLS (filesystem)
# ===========================================================================


@dataclass
class Skill:
    """One skill folder: its advertised metadata plus where it lives on disk.

    `name`/`description` are the only things that sit in context by default
    (cheap). The body and any bundled files are loaded later, on demand.
    """

    name: str
    description: str
    path: Path  # the skill's directory (contains SKILL.md + any resources)


def parse_skill_md(text: str) -> tuple[dict[str, str], str]:
    """Split a SKILL.md into its YAML-ish frontmatter and its body.

    We hand-parse the simple `key: value` frontmatter between `---` fences
    rather than depend on PyYAML — SKILL.md frontmatter is intentionally flat
    (name + description), so a real YAML parser would be overkill here. Returns
    (metadata_dict, body_text); metadata is empty if there's no frontmatter.
    """
    if not text.startswith("---"):
        return {}, text
    # Split into ["", frontmatter, body] on the first two `---` fences.
    _, _, rest = text.partition("---")
    front, sep, body = rest.partition("---")
    if not sep:  # malformed (no closing fence) — treat the whole thing as body
        return {}, text

    meta: dict[str, str] = {}
    for line in front.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("'\"")
    return meta, body.strip()


def scan_skills(root: Path) -> list[Skill]:
    """Discover skills by reading ONLY each SKILL.md's frontmatter (stage 1).

    This is the cheap pass: for N skills we read N small files and keep just the
    name + description. Bodies and bundled resources stay on disk until the model
    asks for them. A malformed skill (missing name/description) is skipped with a
    warning rather than breaking discovery.
    """
    skills: list[Skill] = []
    for skill_md in sorted(root.glob("*/SKILL.md")):
        meta, _body = parse_skill_md(skill_md.read_text())
        name, description = meta.get("name"), meta.get("description")
        if not name or not description:
            log.warning(
                "skipping %s: missing name/description in frontmatter", skill_md
            )
            continue
        skills.append(Skill(name=name, description=description, path=skill_md.parent))
    return skills


def load_skill_body(skill: Skill) -> str:
    """Stage 2 (activation): return the full SKILL.md body, sans frontmatter.

    Called when the model decides a skill is relevant. This is the instruction
    payload — the steps, house rules, and pointers to bundled resources.
    """
    _meta, body = parse_skill_md((skill.path / "SKILL.md").read_text())
    return body


def read_skill_resource(skill: Skill, relpath: str) -> str:
    """Stage 3 (execution): read one bundled file from the skill's folder.

    Bundled scripts/references load only when the model is about to use them.
    The model supplies `relpath`, so we MUST prevent path traversal: resolve the
    target and confirm it stays inside the skill directory before reading
    (otherwise `../../etc/passwd` would escape the sandbox of the skill folder).
    """
    base = skill.path.resolve()
    target = (skill.path / relpath).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"{relpath!r} escapes the skill directory")
    if not target.is_file():
        raise FileNotFoundError(f"no such resource: {relpath}")
    return target.read_text()


# Skill access is mediated by three meta-tools (stable across the run, like
# 07's meta-tool surface). The skill *metadata* is also injected into the system
# prompt so the model knows skills exist without a discovery round-trip.
SKILL_TOOLS = [
    {
        "name": "load_skill",
        "description": (
            "Load the full instructions for a skill by name (from the list in "
            "your system prompt). Do this when a skill is relevant to the task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "read_skill_resource",
        "description": (
            "Read a bundled file (script or reference) from a skill's folder. "
            "Only do this when you are about to use that file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "skill": {"type": "string"},
                "path": {
                    "type": "string",
                    "description": "file path relative to the skill folder",
                },
            },
            "required": ["skill", "path"],
        },
    },
]
SKILL_TOOL_NAMES = {t["name"] for t in SKILL_TOOLS}


def dispatch_skill_tool(name: str, args: dict, skills_by_name: dict[str, Skill]) -> str:
    """Execute a skill meta-tool call and return the text result.

    Centralises the activation/execution stages and the not-found handling so
    the agent loop below stays a thin router.
    """
    if name == "load_skill":
        skill = skills_by_name.get(args.get("name", ""))
        if skill is None:
            return f"Unknown skill: {args.get('name')!r}"
        return load_skill_body(skill)

    if name == "read_skill_resource":
        skill = skills_by_name.get(args.get("skill", ""))
        if skill is None:
            return f"Unknown skill: {args.get('skill')!r}"
        try:
            return read_skill_resource(skill, args.get("path", ""))
        except (ValueError, FileNotFoundError) as e:
            return f"Error: {e}"

    return f"Unknown skill tool: {name}"


# ===========================================================================
# SOURCE 2: MCP (Model Context Protocol)
# ===========================================================================
# Same two conversions as 07_mcp/07_anthropic_with_mcp.py — kept here so this
# file is self-contained.


def mcp_tools_to_anthropic(mcp_tools) -> list[dict]:
    """Convert an MCP tool catalogue into Anthropic tool definitions.

    The only structural difference is the schema field name: MCP calls it
    `inputSchema`, the Messages API calls it `input_schema`. Everything else
    (name, description, the JSON schema itself) carries over unchanged.
    """
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in mcp_tools
    ]


async def execute_mcp_tool(
    session: ClientSession, name: str, args: dict, tool_use_id: str
) -> dict:
    """Forward a tool_use to the MCP session and wrap the reply as a tool_result.

    Failures on either side (the subprocess died, the server raised) all land
    here and become an is_error result, so the model can recover the same way it
    would from any tool error.
    """
    try:
        result = await session.call_tool(name, arguments=args)
        # CallToolResult.content is a list of blocks; concatenate the text parts.
        text = "\n".join(getattr(b, "text", str(b)) for b in result.content)
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": text,
            "is_error": bool(result.isError),
        }
    except Exception as e:  # noqa: BLE001 — surface any MCP failure to the model
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": f"MCP error: {e}",
            "is_error": True,
        }


# ===========================================================================
# THE AGENT: both sources behind one loop
# ===========================================================================


def build_system_prompt(skills: list[Skill]) -> str:
    """Inject skill *metadata* (stage 1) into the system prompt.

    This is the "metadata first" half of skills: the model sees that each skill
    exists and what it's for, but not its full instructions — those arrive only
    when it calls load_skill. Kept stable/frozen so it remains a cacheable prefix.
    """
    catalog = "\n".join(f"  - {s.name}: {s.description}" for s in skills) or "  (none)"
    return (
        "You are an assistant with two sources of capability:\n"
        "  • TOOLS from an MCP server (use them directly).\n"
        "  • SKILLS — task playbooks loaded on demand. Available skills:\n"
        f"{catalog}\n\n"
        "When a skill is relevant, call load_skill(name) to get its full "
        "instructions, then follow them. Use read_skill_resource only for a "
        "bundled file you actually need. Answer the user directly when done."
    )


async def run_agent(prompt: str, *, max_iterations: int = 10) -> str:
    """Drive one task with tools sourced from MCP + skills from the filesystem.

    The MCP session is opened once and held for the whole conversation (cheap,
    and re-spawning per turn would be slow). Each turn we route every tool_use
    to its source by name: skill meta-tools to the local dispatcher, everything
    else to the MCP session.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")
    client = AsyncAnthropic()

    # Stage 1 for skills: scan the directory for metadata only.
    skills = scan_skills(SKILLS_DIR)
    skills_by_name = {s.name: s for s in skills}
    log.info("discovered %d skills: %s", len(skills), [s.name for s in skills])

    params = StdioServerParameters(command=sys.executable, args=[MCP_SERVER_PATH])
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        # Fetch the MCP catalogue once; combine with the skill meta-tools.
        mcp_tools = (await session.list_tools()).tools
        anthropic_tools = mcp_tools_to_anthropic(mcp_tools) + SKILL_TOOLS
        log.info(
            "MCP exposes %d tools: %s", len(mcp_tools), [t.name for t in mcp_tools]
        )

        system = build_system_prompt(skills)
        messages: list[dict] = [{"role": "user", "content": prompt}]

        for _ in range(max_iterations):
            response = await client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=system,
                tools=anthropic_tools,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return "\n".join(b.text for b in response.content if b.type == "text")

            if response.stop_reason == "tool_use":
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    if block.name in SKILL_TOOL_NAMES:
                        # Local source: skills on the filesystem.
                        text = dispatch_skill_tool(
                            block.name, block.input, skills_by_name
                        )
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": text,
                            }
                        )
                    else:
                        # Remote source: MCP server.
                        results.append(
                            await execute_mcp_tool(
                                session, block.name, block.input, block.id
                            )
                        )
                    log.info("  %s(%s)", block.name, json.dumps(block.input)[:80])
                messages.append({"role": "user", "content": results})
                continue

            raise RuntimeError(f"unexpected stop_reason: {response.stop_reason}")

        raise RuntimeError(f"agent did not finish in {max_iterations} iterations")


async def main() -> None:
    # A task that touches both sources: the skill supplies the *format*, the MCP
    # KV server supplies the *storage*.
    answer = await run_agent(
        "Draft a changelog entry for fixing a race condition in the cache "
        "(PR #1421) using the changelog-entry skill, then save the finished "
        "entry to the KV store under the key 'changelog'."
    )
    print(f"\nFINAL:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
