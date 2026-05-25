"""End-to-end: Claude consumes an MCP server as its tool source.

Same agent loop as 03_anthropic_agents/01_agent_loop.py, but the tools
come from an MCP server instead of being declared inline. The two
conversions that make it work:

  1. MCP tool catalogue -> Anthropic tools list (rename inputSchema).
  2. Anthropic tool_use blocks -> MCP call_tool() invocations.

The MCP session is held open for the WHOLE conversation — re-spawning
the subprocess per Claude turn would be wasteful and slow.

Requires: ANTHROPIC_API_KEY.
"""

import asyncio
import os
import sys
from pathlib import Path

from anthropic import AsyncAnthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "claude-haiku-4-5-20251001"
# We drive the KV server so Claude can chain SET + LIST + GET — visible
# multi-step reasoning rather than a single tool call.
SERVER_PATH = str(Path(__file__).parent / "04_stateful_kv_server.py")


def check_env():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")


def mcp_tools_to_anthropic(mcp_tools):
    """Rename inputSchema -> input_schema; otherwise identical."""
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema,
        }
        for t in mcp_tools
    ]


async def execute_mcp_tool(session, name, args, tool_use_id):
    """Forward Claude's tool_use to MCP and wrap the result as a tool_result.

    Failures on EITHER side (MCP subprocess died, server raised) all land
    here and become tool_result with is_error=True. Claude then decides
    whether to retry or apologise — same pattern as a normal tool error.
    """
    try:
        result = await session.call_tool(name, arguments=args)
        # CallToolResult.content is a list of blocks; concatenate text parts.
        text = "\n".join(getattr(b, "text", str(b)) for b in result.content)
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": text,
            "is_error": bool(result.isError),
        }
    except Exception as e:
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": f"MCP call error: {e}",
            "is_error": True,
        }


async def run_agent(prompt, *, max_iterations=8):
    check_env()
    client = AsyncAnthropic()

    params = StdioServerParameters(command=sys.executable, args=[SERVER_PATH])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Catalogue is stable for the session — fetch + convert once.
            mcp_tools = (await session.list_tools()).tools
            anthropic_tools = mcp_tools_to_anthropic(mcp_tools)
            print(
                f"MCP exposes {len(anthropic_tools)} tools: "
                f"{[t['name'] for t in anthropic_tools]}\n"
            )

            messages = [{"role": "user", "content": prompt}]
            for iteration in range(max_iterations):
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    tools=anthropic_tools,
                    messages=messages,
                )
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn":
                    return "\n".join(
                        b.text for b in response.content if b.type == "text"
                    )

                if response.stop_reason == "tool_use":
                    blocks = [b for b in response.content if b.type == "tool_use"]
                    # Parallel execution — same pattern as 03_anthropic_agents/02.
                    results = await asyncio.gather(
                        *(
                            execute_mcp_tool(session, b.name, b.input, b.id)
                            for b in blocks
                        )
                    )
                    for b, r in zip(blocks, results, strict=True):
                        print(
                            f"  [iter {iteration}] {b.name}({b.input}) "
                            f"=> {r['content'][:60]}"
                        )
                    messages.append({"role": "user", "content": results})
                    continue

                raise RuntimeError(f"unexpected stop_reason: {response.stop_reason}")

            raise RuntimeError(f"agent did not finish in {max_iterations} iterations")


async def main():
    answer = await run_agent(
        "Use the KV store: set 'project' to 'ElevenLabs', set 'role' to "
        "'Research Engineer', list keys with prefix 'p', then fetch 'project'. "
        "Summarise what you did and the value you got back."
    )
    print(f"\nFINAL:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
