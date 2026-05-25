"""Planner / executor split: separate the planning model from the doing model.

A "planner" agent decomposes the task into discrete steps (often as
structured JSON). A separate "executor" agent runs each step,
optionally with tools.

Why split:
  - The planner can be a stronger model (better at decomposition) while
    executors run on a cheaper model (per-step cost matters at scale).
  - Each step is a smaller context: the executor only sees the current
    step and a digest of prior results, not the whole conversation.
  - Plan output is inspectable / cacheable / replayable — much easier
    to debug than a monolithic agent's chain of thought.

Production extension: have the planner re-plan after each step
(replanning). That's where LangGraph-style state machines shine.
"""

import asyncio
import json
import os
import re

from anthropic import AsyncAnthropic

PLANNER_MODEL = "claude-haiku-4-5-20251001"  # could use a stronger model here
EXECUTOR_MODEL = "claude-haiku-4-5-20251001"


def get_client() -> AsyncAnthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")
    return AsyncAnthropic()


def extract_json_array(text: str) -> list[dict]:
    """Tolerate prose around the JSON — find the first [...] block."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON array found in:\n{text}")
    return json.loads(match.group(0))


async def plan(client: AsyncAnthropic, task: str) -> list[dict]:
    response = await client.messages.create(
        model=PLANNER_MODEL,
        max_tokens=800,
        system=(
            "You are a planner. Decompose the user's task into 3-5 sequential steps. "
            'Output ONLY a JSON array. Each item: {"id": int, "description": "..."}. '
            "No prose outside the JSON."
        ),
        messages=[{"role": "user", "content": task}],
    )
    return extract_json_array(response.content[0].text)


async def execute_step(
    client: AsyncAnthropic,
    step: dict,
    prior_results: list[dict],
) -> str:
    context = "\n".join(f"- step {r['id']}: {r['summary']}" for r in prior_results)
    prompt = (
        f"You are executing one step of a larger plan.\n"
        f"Current step ({step['id']}): {step['description']}\n\n"
        f"Prior results so far:\n{context if context else '(none)'}\n\n"
        f"Produce a 1-2 sentence result for THIS step only."
    )
    response = await client.messages.create(
        model=EXECUTOR_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def plan_and_execute(task: str) -> str:
    client = get_client()

    print(f"PLANNER thinking about: {task}\n")
    steps = await plan(client, task)
    for s in steps:
        print(f"  step {s['id']}: {s['description']}")

    prior_results: list[dict] = []
    for step in steps:
        print(f"\nEXECUTOR running step {step['id']}...")
        summary = await execute_step(client, step, prior_results)
        print(f"  -> {summary}")
        prior_results.append({"id": step["id"], "summary": summary})

    return "\n".join(f"[{r['id']}] {r['summary']}" for r in prior_results)


async def main() -> None:
    final = await plan_and_execute(
        "Plan and 'execute' a 30-minute morning routine for someone aiming "
        "to feel more energetic by 10am. Steps should be concrete actions."
    )
    print(f"\n=== ALL STEP RESULTS ===\n{final}")


if __name__ == "__main__":
    asyncio.run(main())
