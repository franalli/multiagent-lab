# modal/worker.py
#
# The spawner. Sits between the ingress (Slack or Teams) and the Sandbox.
#
# Why a separate Function (not "spawn directly from the ingress")?
#   - The ingress's only job is "ACK in 3s." Sandbox.create has variable
#     latency; isolating it here means the ingress stays a thin signature-
#     verifier that's easy to reason about.
#   - The worker is also the right place for any control-plane operations
#     that need a real Function (vs an ASGI request): rate limit checks,
#     idempotency dedupe, fan-out, etc. The POC keeps those as stubs.
#
# RUN
# ---
#   modal run modal/worker.py        # smoke-test: spawn a sandbox with a hard-coded context

from __future__ import annotations

import json
from typing import Any

import modal

from common import (  # sibling import; modal/ is intentionally not a package
    SANDBOX_AGENT_SCRIPT_PATH,
    WORKSPACE_MOUNT_PATH,
    app,
    control_plane_image,
    get_workspace_volume,
    sandbox_image,
    secrets,
)


@app.function(
    image=control_plane_image,
    secrets=secrets(),
    timeout=600,
)
def agent_worker(context: dict[str, Any]) -> dict[str, Any]:
    """Spawn the agent Sandbox with the per-workspace Volume mounted.

    `context` is the channel-agnostic dict produced by the ingress. Whatever
    came in -- Slack event, Teams Activity -- collapses to this single shape.
    The sandbox itself never knows which channel fired.

    Production path: insert a Modal Queue partitioned by team_id between
    ingress and worker (per-tenant fairness + backpressure). POC collapses
    ingress → Function.spawn.aio(agent_worker) directly. The Sandbox
    primitive + per-workspace Volume mount stays identical; only the
    in-front-of-worker queueing surface is simpler.

    Sync-blocking on purpose: at POC scope, one worker container handles one
    sandbox at a time and Modal autoscales worker containers up to 2k under
    burst. The blocking wait gives us a clean per-invocation success/failure
    signal that any caller (ingress, tests, smoke tooling) can rely on.
    Production-scale move: `async def` + `@modal.concurrent(max_inputs=N)`
    so one container pipelines N sandboxes (cheaper at 2k tenants) -- at the
    cost of losing the per-invocation error envelope this function returns.

    Note on Sandbox API: modal.Sandbox.create(...) returns a stateful
    container; we invoke processes inside it via the documented argv-style
    process-launch method (no shell, takes a list of strings -- it is the
    safe-by-default pattern, equivalent to subprocess.run with shell=False).
    """
    workspace_id = context["workspace_id"]
    volume = get_workspace_volume(workspace_id)

    # Spawn-and-wait: the worker blocks on the sandbox finishing so we get
    # stdout/stderr back in the Modal logs and can return success/failure to
    # any caller that did .get() on the FunctionCall. For ingress callers,
    # the worker is spawned async -- they don't wait on this return.
    sandbox = modal.Sandbox.create(
        app=app,
        image=sandbox_image,
        volumes={WORKSPACE_MOUNT_PATH: volume},
        secrets=secrets(),
        timeout=300,
    )
    try:
        # Pass the structured context as an argv blob so the entrypoint can
        # parse it directly. Production would also stream this via an env
        # var if the payload grew larger than argv-safe.
        # Modal Sandbox.exec takes argv (list of strings), not a shell command --
        # so this is execFile-equivalent, no shell injection surface.
        runner = sandbox.exec("python", SANDBOX_AGENT_SCRIPT_PATH, json.dumps(context))
        runner.wait()
        stdout = runner.stdout.read()
        stderr = runner.stderr.read()
    finally:
        sandbox.terminate()

    return {
        "workspace_id": workspace_id,
        "stdout": stdout,
        "stderr": stderr,
    }


# ---------------------------------------------------------------------------
# Local entrypoint -- smoke test
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def smoke_worker() -> None:
    """`modal run modal/worker.py` spawns a sandbox against the demo workspace.

    Useful for verifying the image + Volume + script-mount work end-to-end
    without involving the ingress or harness.
    """
    demo_context = {
        "workspace_id": "dev",
        "channel_origin": "harness",
        "user_id": "U01DEMOUSER",
        "channel": "D01DEMODM",
        "channel_type": "im",
        "message": "hello agent (worker smoke test)",
        "thread_ts": None,
        "ts": "0.0",
    }
    result = agent_worker.remote(demo_context)
    print("[worker.smoke] stdout =", result["stdout"][:400])
    if result["stderr"]:
        print("[worker.smoke] stderr =", result["stderr"][:400])
