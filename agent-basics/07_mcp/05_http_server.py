"""Streamable HTTP transport — the modern web alternative to stdio.

Three transports in MCP:
  - stdio (default): subprocess + JSON-RPC over stdin/stdout. Trivial deploy,
                     no ports, but 1:1 client:server, no remote access.
  - streamable-http: long-running HTTP service, multi-client, remote-OK.
                     Modern; you own auth and TLS.
  - sse: legacy HTTP transport. Avoid for new servers.

The decorator API is identical across transports — only `mcp.run(transport=...)`
changes. Default address: http://127.0.0.1:8000/mcp/

Test with curl:
    curl -L -X POST http://127.0.0.1:8000/mcp \\
         -H 'Content-Type: application/json' \\
         -H 'Accept: application/json, text/event-stream' \\
         -d '{"jsonrpc":"2.0","id":1,"method":"initialize",
              "params":{"protocolVersion":"2025-06-18",
                        "capabilities":{},
                        "clientInfo":{"name":"curl","version":"1"}}}'
"""

import math

from mcp.server.fastmcp import FastMCP

# Defaults bind to loopback only — explicit safety choice. For production
# expose `mcp.streamable_http_app()` under your own uvicorn/gunicorn + auth.
mcp = FastMCP("http-demo", host="127.0.0.1", port=8000)


@mcp.tool()
def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


@mcp.tool()
def normalize_email(email: str) -> str:
    """Lowercase + strip — a 'data hygiene' tool a backend team might expose
    so every consumer applies the same rules."""
    return email.strip().lower()


@mcp.resource("config://units")
def supported_units() -> str:
    """Static JSON resource. Over HTTP it could plausibly sit behind a CDN."""
    return '{"length": ["m", "ft", "in"], "mass": ["kg", "lb"]}'


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
