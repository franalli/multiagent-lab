"""Sandboxed code execution: running model-written code with resource limits.

Sibling of 07_production_agent.py. When you give an agent a "run this code"
tool, the code is *untrusted* — the model can be wrong, or steered by injected
content (see 07's quarantine layer). This file is the tool you'd register for
that, built around an OS-process sandbox.

THREAT MODEL — read this before trusting it anywhere.

  This sandbox runs code in a separate process and applies POSIX resource
  limits (`setrlimit`) plus a wall-clock timeout, a stripped environment, and
  output caps. Concretely it DEFENDS AGAINST:
    - runaway CPU (RLIMIT_CPU)                 - fork bombs (RLIMIT_NPROC)
    - memory blowups (RLIMIT_AS, where honoured) - giant files (RLIMIT_FSIZE)
    - infinite loops (wall-clock timeout -> kill the whole process group)
    - log flooding (stdout/stderr truncation)
    - leaking the parent's secrets via os.environ (minimal env + python -I)

  It DOES NOT provide real isolation. Same user, same filesystem, and — most
  importantly — `setrlimit` cannot block network access. A determined payload
  can still read your files or call out to the network. For genuinely untrusted
  code use a real sandbox: a container, gVisor, a Firecracker microVM, nsjail +
  seccomp-bpf, or Anthropic's *server-side* code execution tool
  (`{"type": "code_execution_20260120", "name": "code_execution"}`), which runs
  in an Anthropic-hosted container with no client-side execution at all.

  Treat this as the right *shape* (process boundary, limits, timeout, kill) with
  honest gaps — not as a security boundary.

Platform note: `resource` / `setrlimit` are POSIX-only (Linux, macOS). On macOS
RLIMIT_AS is frequently not enforced, so the memory cap is best-effort there;
the CPU, file-size, process-count, and timeout limits still apply. On Windows
you'd swap the preexec approach for a Job Object.

Requires: ANTHROPIC_API_KEY (for the demo loop). The sandbox itself is stdlib.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from dataclasses import dataclass

# `resource` is POSIX-only. Import defensively so the file at least loads (and
# documents itself) on Windows; the sandbox raises a clear error if used there.
try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]


# Defaults for the sandbox. Tuned small on purpose — bump per call as needed.
DEFAULT_TIMEOUT_S = 5.0  # wall-clock; the hard backstop for infinite loops
DEFAULT_CPU_SECONDS = 2  # RLIMIT_CPU: SIGXCPU then SIGKILL past this
DEFAULT_MEMORY_MB = 256  # RLIMIT_AS: address space (best-effort on macOS)
DEFAULT_MAX_OUTPUT = 10_000  # chars of stdout/stderr kept before truncation
DEFAULT_MAX_PROCS = 64  # RLIMIT_NPROC: fork-bomb guard
DEFAULT_MAX_FILE_MB = 10  # RLIMIT_FSIZE: cap files the code can write


@dataclass
class SandboxResult:
    """Everything the caller (and the model) needs to know about a run.

    `timed_out` is surfaced separately from `returncode` because a timeout is a
    different failure than a non-zero exit — the model should react differently
    (simplify vs. fix a bug).
    """

    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool
    duration_seconds: float


def _build_limits(cpu_seconds: int, memory_mb: int, max_procs: int, max_file_mb: int):
    """Return a `preexec_fn` that locks down the child *before* it execs.

    `preexec_fn` runs in the forked child, in between fork() and exec(), so the
    limits are in force for the entire lifetime of the executed code. We also
    `setsid()` to put the child in its own process group — that's what lets the
    timeout path kill the whole tree (including anything it spawned) in one call.

    Caveats: `preexec_fn` is POSIX-only and runs in a fragile post-fork context
    (don't allocate/log/lock in here); keep it to syscalls. It's also not
    thread-safe, which is fine for our single-purpose use.
    """

    def _apply() -> None:
        # New session + process group => os.killpg(pid) reaps the whole tree.
        os.setsid()

        def lim(which, soft_hard) -> None:
            # Set (soft, hard) but never raise above the inherited hard limit,
            # or setrlimit raises EPERM for unprivileged processes.
            try:
                _cur_soft, cur_hard = resource.getrlimit(which)
                hard = cur_hard if cur_hard != resource.RLIM_INFINITY else soft_hard
                resource.setrlimit(which, (min(soft_hard, hard), hard))
            except (ValueError, OSError):
                pass  # some limits aren't enforceable on every platform

        lim(resource.RLIMIT_CPU, cpu_seconds)  # CPU seconds
        lim(resource.RLIMIT_AS, memory_mb * 1024 * 1024)  # address space (memory)
        lim(resource.RLIMIT_FSIZE, max_file_mb * 1024 * 1024)  # max file write
        lim(resource.RLIMIT_NPROC, max_procs)  # process count (fork bomb)
        lim(resource.RLIMIT_CORE, 0)  # no core dumps

    return _apply


async def run_in_sandbox(
    code: str,
    *,
    args: list[str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_MB,
    max_output: int = DEFAULT_MAX_OUTPUT,
    max_procs: int = DEFAULT_MAX_PROCS,
    max_file_mb: int = DEFAULT_MAX_FILE_MB,
) -> SandboxResult:
    """Execute `code` in a resource-limited subprocess and capture its output.

    The layers, in order:
      1. `python -I -c code` — *isolated mode*: ignores PYTHONPATH/PYTHON* env
         vars, doesn't put cwd on sys.path, and skips the user site dir. Cuts off
         the easiest ways to smuggle code/config in via the environment.
      2. A stripped `env` (just PATH) so the child can't read the parent's
         secrets out of os.environ.
      3. `preexec_fn` setrlimits + setsid (see `_build_limits`).
      4. A wall-clock timeout; on expiry we SIGKILL the whole process group, so
         even a child that ignores SIGTERM or spawned helpers gets reaped.
      5. Output truncation so a noisy program can't blow up the caller's context.

    `args`, if given, are appended after `code` on the command line, so they
    arrive as `sys.argv[1:]` inside the program (`sys.argv[0]` is "-c"). That's
    how a skill's bundled script (run via 07's run_skill_script) receives, e.g.,
    the path of a file to operate on.

    Returns a `SandboxResult`; never raises on the *code's* behalf (a crash or
    timeout is data, not an exception). Raises only if the platform can't sandbox.
    """
    if resource is None:
        raise RuntimeError("run_in_sandbox requires a POSIX platform (resource module).")

    start = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",  # isolated mode — see docstring
        "-c",
        code,
        *(args or []),  # appended after -c => sys.argv[1:] inside the child
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # Minimal environment: no inherited secrets. PATH only so the
        # interpreter can find shared libs / helper binaries it needs.
        env={"PATH": "/usr/bin:/bin"},
        # Lock down the child before it runs the untrusted code.
        preexec_fn=_build_limits(cpu_seconds, memory_mb, max_procs, max_file_mb),
        # Don't leak the parent's open file descriptors into the sandbox.
        close_fds=True,
    )

    timed_out = False
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        timed_out = True
        # Kill the entire process group so spawned children die too. SIGKILL
        # because a wedged process may be ignoring softer signals. Suppress the
        # benign teardown race where the process/group is already gone.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), 9)
        # Reap to avoid a zombie; collect whatever output it managed to flush.
        stdout_b, stderr_b = await proc.communicate()

    return SandboxResult(
        stdout=_truncate(stdout_b.decode(errors="replace"), max_output),
        stderr=_truncate(stderr_b.decode(errors="replace"), max_output),
        returncode=proc.returncode,
        timed_out=timed_out,
        duration_seconds=time.perf_counter() - start,
    )


def _truncate(text: str, limit: int) -> str:
    """Cap captured output, telling the reader exactly how much was dropped."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"


# ---------------------------------------------------------------------------
# Wiring the sandbox into an agent loop
# ---------------------------------------------------------------------------
# Same loop shape as 07/01 — kept lean here so the focus stays on the sandbox.
# The one tool we expose runs code through `run_in_sandbox`. In 07's runtime
# this tool would carry ToolAnnotations(read_only=False, destructive=True) so it
# requires human approval; here we keep the loop minimal.

import logging  # imported here to keep the sandbox section stdlib-only above

from anthropic import AsyncAnthropic

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"

CODE_TOOL = {
    "name": "run_python",
    "description": ("Execute a short Python 3 program in a resource-limited sandbox and return its stdout/stderr. No network or installed third-party packages; stdlib only. Print results — return values are not captured."),
    "input_schema": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source to run"}},
        "required": ["code"],
    },
}


async def _execute_code_tool(code: str) -> str:
    """Adapt a `SandboxResult` into the string the model reads as a tool result.

    We hand back stdout, plus stderr and a timeout flag only when relevant, so a
    success stays terse and a failure carries the diagnostic the model needs.
    """
    result = await run_in_sandbox(code)
    parts = [f"exit={result.returncode} timed_out={result.timed_out} ({result.duration_seconds:.2f}s)"]
    if result.stdout:
        parts.append(f"stdout:\n{result.stdout}")
    if result.stderr:
        parts.append(f"stderr:\n{result.stderr}")
    return "\n".join(parts)


async def run_agent(prompt: str, *, max_iterations: int = 6) -> str:
    """Minimal agent loop that can write and run code to answer a question."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")
    client = AsyncAnthropic()
    messages: list[dict] = [{"role": "user", "content": prompt}]

    for _ in range(max_iterations):
        response = await client.messages.create(model=MODEL, max_tokens=1024, tools=[CODE_TOOL], messages=messages)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return "\n".join(b.text for b in response.content if b.type == "text")

        if response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                log.info(
                    "running sandboxed code (%d chars)",
                    len(block.input.get("code", "")),
                )
                out = await _execute_code_tool(block.input["code"])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
            messages.append({"role": "user", "content": results})
            continue

        raise RuntimeError(f"unexpected stop_reason: {response.stop_reason}")

    raise RuntimeError(f"agent did not finish in {max_iterations} iterations")


async def _demo_sandbox_directly() -> None:
    """Exercise the sandbox without the model, including the failure modes.

    Useful as documentation: it shows the timeout and resource limits actually
    firing, which is the whole point of the file.
    """
    print("--- normal program ---")
    print(await run_in_sandbox("print(sum(range(100)))"))

    print("\n--- infinite loop (killed by the wall-clock timeout) ---")
    print(await run_in_sandbox("while True: pass", timeout_s=1.0))

    print("\n--- memory grab (RLIMIT_AS where enforced; MemoryError otherwise) ---")
    print(await run_in_sandbox("x = bytearray(10**9); print(len(x))", memory_mb=64))


async def main() -> None:
    # First show the sandbox primitives directly, then drive it via the model.
    await _demo_sandbox_directly()
    print("\n--- agent using the sandbox ---")
    answer = await run_agent("What is the 25th Fibonacci number? Write and run Python to compute it, then state the answer.")
    print(f"\nFINAL:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
