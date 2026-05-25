"""MCP hello-world: minimal stdio server exposing two tools.

MCP (Model Context Protocol) is Anthropic's standard for letting an LLM
client discover and call external capabilities. A server exposes three
primitives: tools (functions), resources (read-only data), prompts
(parameterised templates). This file just does tools.

stdio transport: the client (Claude Desktop, etc.) spawns this script
as a subprocess and exchanges JSON-RPC over stdin/stdout. Don't print()
from tool bodies — it corrupts the stream. Use ctx.info() (see file 04).

Test via 06_stdio_client.py.
"""

from mcp.server.fastmcp import FastMCP

# `name` is what clients see in capability lists.
mcp = FastMCP("hello-world")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers.

    The @mcp.tool() decorator reflects on the type annotations to generate
    the tool's JSON schema, and uses the docstring as the description.
    Stick to JSON-friendly types (int, float, str, bool, list, dict, BaseModel).
    """
    return a + b


@mcp.tool()
def greet(name: str, greeting: str = "Hello") -> str:
    """Build a personalised greeting. The default makes `greeting` optional."""
    return f"{greeting}, {name}!"


if __name__ == "__main__":
    # `run()` is blocking — owns the event loop until the parent closes us.
    mcp.run(transport="stdio")
