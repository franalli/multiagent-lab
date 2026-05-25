# modal/serve_all.py
#
# Single-process aggregator for `modal serve`. Imports every Function
# module so they all register on the same in-process App
# (`multiplayer-ai`). This is the dev/demo runtime: one terminal, all
# Functions live, sibling imports (e.g. ingress_slack -> worker) resolve
# without `Function.from_name(...)` lookups.
#
# In production, the equivalent move is `modal deploy modal/serve_all.py`
# -- the import surface is identical; only the lifecycle differs.
#
# Files NOT imported here:
#   * sandbox_agent.py -- runs INSIDE the Sandbox, not as a Modal Function.
#                         Bundled into sandbox_image via add_local_file.
#   * common.py        -- imported transitively by every Function module.

from common import app  # noqa: F401  -- re-export so `modal serve` discovers the App

# Each import below registers one or more @app.function declarations.
# We never call these names; the @app.function side-effect is the point.
from ingress_slack import slack_ingress  # noqa: F401
from ingress_teams import teams_ingress  # noqa: F401
from worker import agent_worker  # noqa: F401
from gateway import tool_gateway  # noqa: F401
from scheduler import (  # noqa: F401
    scan_workspace,
    proactive_scan_all,
)
from analytics import (  # noqa: F401
    compute_workspace_features,
    predict_workspace_health,
    predict_all_workspaces,
)
