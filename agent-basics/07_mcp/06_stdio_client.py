"""MCP client: spawns a server subprocess and drives it.

You write a custom client when:
  - Building your own agent loop (not delegating to Claude Desktop).
  - Testing a server you authored (this file's main use).
  - Composing multiple MCP servers in one agent.

This script drives 01_hello_server_stdio.py — list tools, call each, exit.
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = str(Path(__file__).parent / "01_hello_server_stdio.py")


async def main():
    # StdioServerParameters declares HOW to launch the subprocess.
    # We use sys.executable so the child runs in the SAME Python interpreter.
    params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        env=None,
    )

    # Both context managers reap the subprocess on exit — no orphan procs.
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        # 1. Handshake. The server replies with its capabilities.
        info = await session.initialize()
        print(f"connected to: {info.serverInfo.name} v{info.serverInfo.version}")

        # 2. Discover tools. Every MCP server supports list_tools().
        tools = (await session.list_tools()).tools
        print(f"\ntools exposed ({len(tools)}):")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        # 3. Call tools. `arguments` must match each tool's input_schema
        # (which FastMCP generated from the server's type hints).
        for name, args in [
            ("add", {"a": 3, "b": 4}),
            ("greet", {"name": "Demo"}),
            ("greet", {"name": "World", "greeting": "Hi"}),
        ]:
            print(f"\ncalling {name}({args})...")
            result = await session.call_tool(name, arguments=args)
            # result.content is a list of content blocks; for simple
            # returns it's a single TextContent.
            for block in result.content:
                if hasattr(block, "text"):
                    print(f"  -> {block.text}")

    print("\nsession closed; subprocess reaped.")


if __name__ == "__main__":
    asyncio.run(main())
