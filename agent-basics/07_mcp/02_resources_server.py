"""MCP resources — read-only data exposed via URIs.

Tools = active actions (mutate things). Resources = passive data (read things).
A client can call a tool whenever it wants; resources are typically fetched
by the client and folded into context BEFORE the model runs.

Three shapes shown below:
  - Static URI:      `config://app/settings`
  - Templated URI:   `user://{user_id}/profile`
  - Computed:        `status://server/{component}` (re-computed on every read)
"""

import json
from datetime import UTC, datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("resources-demo")

# In-memory "database". Resources READ from it; the tool below MUTATES it.
USERS = {
    "u1": {"name": "Alice", "role": "editor", "team": "voice-dubbing"},
    "u2": {"name": "Bob", "role": "reviewer", "team": "voice-dubbing"},
    "u3": {"name": "Carol", "role": "actor", "team": "english-voices"},
}


@mcp.resource("config://app/settings")
def app_settings() -> str:
    """Static URI: no path parameters, singleton resource. JSON-encoded text is the standard."""
    return json.dumps(
        {
            "version": "1.0.0",
            "max_dubbing_languages": 10,
            "default_voice_model": "eleven_flash_v2_5",
        },
        indent=2,
    )


@mcp.resource("user://{user_id}/profile")
def user_profile(user_id: str) -> str:
    """Templated URI: {user_id} binds to the parameter of the same name.

    Raise on unknown ids — the SDK translates to a JSON-RPC error. Don't
    swallow exceptions in resource handlers; let the framework surface them.
    """
    user = USERS.get(user_id)
    if user is None:
        raise ValueError(f"unknown user_id {user_id!r}")
    return json.dumps(user, indent=2)


@mcp.resource("status://server/{component}")
def component_status(component: str) -> str:
    """Computed on every read — resources don't have to be stored data."""
    return json.dumps(
        {
            "component": component,
            "status": "ok",
            "checked_at": datetime.now(UTC).isoformat(),
        },
        indent=2,
    )


@mcp.tool()
def update_user_role(user_id: str, new_role: str) -> str:
    """Tools mutate; resources observe. Subsequent reads of user_profile
    reflect this change because resources are computed at read time."""
    user = USERS.get(user_id)
    if user is None:
        raise ValueError(f"unknown user_id {user_id!r}")
    user["role"] = new_role
    return f"updated {user_id} role -> {new_role}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
