"""Multi-step orchestration: planner -> generator -> evaluator, subagents, resume.

Sibling of 07_production_agent.py. 07 is one `agent.run()`. Some tasks exceed a
single context window or a single attempt; this file is the layer *above* the
loop that makes them tractable. Three patterns, composed:

  1. PLANNER -> GENERATOR -> EVALUATOR — Anthropic's recommended shape for
     long-running tasks. A planner decomposes the goal into steps; a generator
     executes each step; an evaluator grades the result and sends it back for
     revision until it passes. This is the lineage of 05 (planner/executor) and
     06 (generator/critic), fused into one controller. The evaluator returns a
     STRUCTURED verdict ({"approved", "feedback"}) — the machine-readable upgrade
     06 flagged over its brittle "APPROVED" string.

  2. SUBAGENTS — the generator runs each step in a subagent with *isolated
     context*: it sees only the step and a compact digest of prior results, not
     the whole transcript. That keeps the controller's context small and lets
     sub-work run on a cheaper model. (Spawning a cheap-model subagent is also
     the right answer to "I want a cheaper model for this sub-task" — you can't
     swap the model under a live loop without busting its cache; see 07.)

  3. PERSISTENT STATE + RESUMPTION — state is checkpointed to disk after every
     completed step. A crash, a timeout, or a deliberate stop can be resumed:
     finished steps reload from the checkpoint, and execution picks up at the
     first incomplete one. This is what lets a run survive a process restart or
     span multiple sessions.

Cost note: planning quality compounds across every downstream step, so in
production the planner runs on a stronger model while generators/evaluators run
cheap. We keep the whole series on Haiku so it's runnable with one key, and mark
where the stronger tier (STRONG_MODEL) belongs.

Requires: ANTHROPIC_API_KEY, `pip install anthropic`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

FAST_MODEL = "claude-haiku-4-5-20251001"  # generators, evaluators, subagents
STRONG_MODEL = "claude-opus-4-8"  # where a production planner would run
# Planning quality compounds, so prod would set PLANNER_MODEL = STRONG_MODEL.
# Kept on FAST_MODEL here so the demo runs with a Haiku-only key (like 05/06).
PLANNER_MODEL = FAST_MODEL

MAX_REVISIONS = 2  # evaluator->generator retries per step before accepting


def get_client() -> AsyncAnthropic:
    """Construct the async client, failing fast if the key is missing."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")
    return AsyncAnthropic()


async def _complete(client: AsyncAnthropic, model: str, system: str, user: str, *, max_tokens: int = 800) -> str:
    """One-shot, non-streaming completion returning the text. The shared
    primitive for every role below (planner/generator/evaluator/subagent).

    Non-streaming is fine here: each role caps `max_tokens` well under the ~16K
    threshold where the SDK would require streaming to dodge HTTP timeouts.
    """
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "\n".join(b.text for b in response.content if b.type == "text")


def _extract_json(text: str, opener: str, closer: str) -> Any:
    """Pull the first JSON value out of a possibly-chatty model response.

    Models often wrap JSON in prose or fences even when told not to. We slice
    from the first opener to the last closer and parse that. `opener`/`closer`
    are "[","]" for arrays or "{","}" for objects.
    """
    start, end = text.find(opener), text.rfind(closer)
    if start == -1 or end == -1:
        raise ValueError(f"no JSON {opener}{closer} found in:\n{text}")
    return json.loads(text[start : end + 1])


# ===========================================================================
# STATE — the unit of persistence
# ===========================================================================


@dataclass
class Step:
    """One planned step. `result` is None until the step completes.

    Plain, JSON-friendly fields so the whole state serialises with `asdict`.
    """

    id: int
    description: str
    result: str | None = None


@dataclass
class OrchestrationState:
    """The full, serialisable state of a run — the thing we checkpoint.

    `current_index` is the resume cursor: the number of steps already completed,
    i.e. the index of the first step still to do. Reloading this is what makes a
    run crash-safe.
    """

    task: str
    steps: list[Step] = field(default_factory=list)
    current_index: int = 0
    final: str = ""

    def to_json(self) -> str:
        """Serialise to JSON (dataclasses -> dicts) for the checkpoint file."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> OrchestrationState:
        """Rehydrate from a checkpoint, rebuilding the nested Step dataclasses."""
        raw = json.loads(text)
        raw["steps"] = [Step(**s) for s in raw["steps"]]
        return cls(**raw)


# ===========================================================================
# THE THREE ROLES
# ===========================================================================


async def plan(client: AsyncAnthropic, task: str) -> list[Step]:
    """PLANNER: decompose the task into 3-5 ordered steps (structured JSON).

    The plan is an inspectable, replayable artifact — far easier to debug than a
    monolithic chain of thought, and it's what we checkpoint. In production this
    is the call you'd point at a stronger model (see PLANNER_MODEL note up top).
    """
    raw = await _complete(
        client,
        PLANNER_MODEL,
        system=('You are a planner. Decompose the user\'s task into 3-5 sequential steps. Output ONLY a JSON array; each item is {"id": int, "description": str}. No prose outside the JSON.'),
        user=task,
    )
    return [Step(id=item["id"], description=item["description"]) for item in _extract_json(raw, "[", "]")]


class Subagent:
    """A focused worker with ISOLATED context, run on the cheap model.

    The point of a subagent is context hygiene: the parent controller does not
    hand it the full transcript, only the one subtask plus a compact digest of
    what's already been decided. So the parent's window stays small no matter how
    long the run gets, and each subagent's window holds only what its step needs.
    A real subagent could carry its own tools (07's runtime) — here it's text-only.
    """

    def __init__(self, client: AsyncAnthropic, model: str = FAST_MODEL):
        self.client = client
        self.model = model

    async def run(self, subtask: str, context_digest: str) -> str:
        """Execute one subtask against a fresh, minimal context; return a result.

        Note the prompt is built from scratch every call — no accumulated history
        leaks in. That isolation is the feature, not a limitation.
        """
        prompt = f"Subtask: {subtask}\n\nRelevant context so far (digest, not full history):\n{context_digest or '(none)'}\n\nProduce a concise, complete result for THIS subtask only."
        return await self._invoke(prompt)

    async def _invoke(self, prompt: str) -> str:
        """Thin wrapper over the model call — isolated to keep `run` readable."""
        return await _complete(
            self.client,
            self.model,
            system="You are a focused worker. Do exactly the subtask; no preamble.",
            user=prompt,
            max_tokens=600,
        )


async def evaluate_step(client: AsyncAnthropic, task: str, step: Step, output: str) -> tuple[bool, str]:
    """EVALUATOR: grade a step's output, returning (approved, feedback).

    Returns a STRUCTURED verdict rather than sniffing for the word "APPROVED" in
    prose (the brittleness 06 calls out). We ask for JSON and parse it; if the
    model returns something unparseable we fail safe to "not approved" with the
    raw text as feedback, so a bad verdict can't masquerade as a pass.

    Using the cheap model on purpose: criticism is usually easier than generation,
    so the evaluator is an asymmetric-cost win.
    """
    raw = await _complete(
        client,
        FAST_MODEL,
        system=('You are a strict reviewer. Judge whether the output satisfies the step in service of the overall task. Output ONLY JSON: {"approved": bool, "feedback": "specific, actionable issues or why it passes"}.'),
        user=f"Overall task: {task}\n\nStep: {step.description}\n\nOutput to review:\n{output}",
        max_tokens=400,
    )
    try:
        verdict = _extract_json(raw, "{", "}")
        return bool(verdict.get("approved", False)), str(verdict.get("feedback", ""))
    except (ValueError, json.JSONDecodeError):
        # Unparseable verdict => treat as not-approved; never silently pass.
        return False, raw


async def generate_step(client: AsyncAnthropic, step: Step, prior_results: list[str]) -> str:
    """GENERATOR: produce a step's output via a subagent, then revise on feedback.

    This is where the three roles meet: we delegate the actual work to a subagent
    (isolated context, cheap model), then run the evaluator. If the evaluator
    rejects, we feed its feedback back into another subagent attempt — bounded by
    MAX_REVISIONS so a hard-to-satisfy step can't loop forever.
    """
    subagent = Subagent(client)
    # Compact digest of prior steps — this is all the subagent sees of history.
    digest = "\n".join(f"- step {i}: {r}" for i, r in enumerate(prior_results)) or "(none)"

    output = await subagent.run(step.description, digest)
    for attempt in range(MAX_REVISIONS):
        approved, feedback = await evaluate_step(client, step.description, step, output)
        if approved:
            log.info("  step %d approved (after %d revision(s))", step.id, attempt)
            return output
        log.info("  step %d revising: %s", step.id, feedback[:80])
        # Re-run with the critique folded into the subtask — still isolated.
        output = await subagent.run(
            f"{step.description}\n\nYour previous attempt was rejected. Feedback: {feedback}\nPrevious attempt:\n{output}",
            digest,
        )
    # Out of revisions: return the best effort, clearly marked for the synthesiser.
    log.warning("  step %d hit MAX_REVISIONS; using best effort", step.id)
    return output


async def synthesize(client: AsyncAnthropic, task: str, steps: list[Step]) -> str:
    """Fold the completed step results into one final answer to the task."""
    transcript = "\n\n".join(f"[step {s.id}] {s.description}\n{s.result}" for s in steps)
    return await _complete(
        client,
        FAST_MODEL,
        system="Synthesise the step results into a single coherent answer to the user's task.",
        user=f"Task: {task}\n\nStep results:\n{transcript}",
        max_tokens=800,
    )


# ===========================================================================
# THE ORCHESTRATOR — ties roles + persistence together
# ===========================================================================


class Orchestrator:
    """Runs plan -> (generate -> evaluate)* -> synthesise, checkpointing as it goes."""

    def __init__(self, client: AsyncAnthropic, checkpoint_path: Path | None = None):
        self.client = client
        # Default checkpoint to a temp file so demos don't litter the repo.
        self.checkpoint_path = checkpoint_path or Path(tempfile.gettempdir()) / "orchestration_checkpoint.json"

    def _save(self, state: OrchestrationState) -> None:
        """Persist state after each completed step (the crash-safety point).

        Written atomically-ish: a real system would write-temp-then-rename to
        avoid a torn file if it dies mid-write. We keep it one call for clarity.
        """
        self.checkpoint_path.write_text(state.to_json())

    def _load(self) -> OrchestrationState | None:
        """Load a checkpoint if one exists, else None (start fresh)."""
        if self.checkpoint_path.exists():
            return OrchestrationState.from_json(self.checkpoint_path.read_text())
        return None

    async def run(self, task: str, *, resume: bool = False) -> str:
        """Execute (or resume) the task and return the synthesised answer.

        With resume=True we reload the checkpoint and continue from its
        `current_index`: completed steps are NOT recomputed (their results come
        off disk), execution restarts at the first incomplete step. With
        resume=False we plan from scratch and overwrite any old checkpoint.
        """
        state = self._load() if resume else None
        if state is None:
            # Fresh run: plan, then persist the plan before doing any work, so
            # even a crash during step 0 leaves a resumable checkpoint.
            steps = await plan(self.client, task)
            state = OrchestrationState(task=task, steps=steps)
            self._save(state)
            log.info("planned %d steps", len(steps))
        else:
            log.info(
                "resuming at step %d (%d already complete)",
                state.current_index,
                state.current_index,
            )

        # Execute from the resume cursor forward. Completed steps are skipped.
        prior_results = [s.result or "" for s in state.steps[: state.current_index]]
        for step in state.steps[state.current_index :]:
            log.info("step %d: %s", step.id, step.description)
            step.result = await generate_step(self.client, step, prior_results)
            prior_results.append(step.result)
            state.current_index += 1
            self._save(state)  # checkpoint AFTER the step succeeds

        state.final = await synthesize(self.client, task, state.steps)
        self._save(state)
        return state.final


def _simulate_crash(checkpoint_path: Path, rewind_to: int) -> None:
    """Demo helper: rewind a checkpoint so resumption has something to do.

    Mutates the on-disk checkpoint to look like the process died after
    `rewind_to` steps — drops later results and moves the cursor back. This lets
    `main()` show resume reloading the kept steps and recomputing only the rest,
    without us having to actually crash the interpreter.
    """
    state = OrchestrationState.from_json(checkpoint_path.read_text())
    for step in state.steps[rewind_to:]:
        step.result = None
    state.current_index = rewind_to
    state.final = ""
    checkpoint_path.write_text(state.to_json())


async def main() -> None:
    client = get_client()
    # Use a demo-local checkpoint we can inspect and rewind.
    ckpt = Path(tempfile.gettempdir()) / "orchestration_demo.json"
    ckpt.unlink(missing_ok=True)
    orch = Orchestrator(client, checkpoint_path=ckpt)

    task = "Design a simple onboarding email sequence (3 emails) for a developer-tools startup: outline each email's goal, subject line, and key message."

    print("=== INITIAL RUN ===")
    answer = await orch.run(task)
    print(f"\n{answer}\n")

    # Demonstrate resumption: pretend we crashed after step 1, then resume.
    print("=== SIMULATED CRASH after step 1, then RESUME ===")
    _simulate_crash(ckpt, rewind_to=1)
    resumed = await Orchestrator(client, checkpoint_path=ckpt).run(task, resume=True)
    print(f"\n{resumed}")

    ckpt.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
