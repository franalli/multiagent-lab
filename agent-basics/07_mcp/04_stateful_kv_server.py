"""MCP server with mutable state: an in-memory KV store with TTL.

Five tools (set, get, delete, list_keys, clear) + a snapshot resource.
The state lives at module scope — fine for one subprocess (stdio) or one
uvicorn worker. Cross-process state needs Redis/Postgres.

Also demonstrates the `Context` parameter: when a tool function takes
`ctx: Context`, the SDK injects per-request lifecycle hooks (logging,
progress). The Context parameter is invisible to the LLM's schema.
"""

import time
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("kv-store")


@dataclass
class _Entry:
    value: str
    expires_at: float | None  # None = never expires

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.monotonic() >= self.expires_at


_STORE: dict[str, _Entry] = {}


@mcp.tool()
async def set(key: str, value: str, ttl_seconds: float | None = None, ctx: Context | None = None) -> str:
    """Write a value under `key`. Optional TTL in seconds.

    `ctx: Context` is auto-injected by the SDK and NOT exposed in the
    tool schema — the model never sees it. Use ctx.info() / ctx.warning()
    instead of print() (print corrupts the stdio JSON-RPC channel).
    """
    expires_at = (time.monotonic() + ttl_seconds) if ttl_seconds is not None else None
    _STORE[key] = _Entry(value=value, expires_at=expires_at)
    if ctx is not None:
        await ctx.info(f"set {key!r} (ttl={ttl_seconds})")
    return f"OK ({key!r})"


@mcp.tool()
def get(key: str) -> str:
    """Read `key`. Reaps expired entries lazily (no background sweeper)."""
    entry = _STORE.get(key)
    if entry is None:
        return "<not found>"
    if entry.is_expired():
        del _STORE[key]
        return "<expired>"
    return entry.value


@mcp.tool()
def delete(key: str) -> str:
    """Remove `key`. Idempotent — succeeds even if absent."""
    existed = key in _STORE
    _STORE.pop(key, None)
    return "deleted" if existed else "<not found, no-op>"


@mcp.tool()
def list_keys(prefix: str = "") -> list[str]:
    """Return live keys (skipping expired), sorted for determinism."""
    return sorted(key for key, entry in _STORE.items() if not entry.is_expired() and key.startswith(prefix))


@mcp.tool()
async def clear(ctx: Context | None = None) -> str:
    """Remove every key. Irreversible.

    Name destructive tools descriptively — the model is more likely to
    confirm with the user before calling 'clear' than 'reset'.
    """
    n = len(_STORE)
    _STORE.clear()
    if ctx is not None:
        await ctx.warning(f"cleared {n} keys")
    return f"cleared {n} keys"


@mcp.resource("kv://snapshot")
def snapshot() -> str:
    """Live snapshot — tools mutate, resources observe (re-computed each read)."""
    import json

    live = {k: e.value for k, e in _STORE.items() if not e.is_expired()}
    return json.dumps(live, indent=2, sort_keys=True)


if __name__ == "__main__":
    mcp.run(transport="stdio")
