"""Generator-critic verifier loop: one agent writes, another grades.

Pattern:
  1. Generator produces output.
  2. Critic reviews it; emits "APPROVED" or concrete feedback.
  3. If approved -> return. Otherwise pass feedback back to generator
     and try again. Bounded by max_iterations.

Two senior signals when using this pattern:
  - Use a CHEAPER model for the critic (asymmetric cost; criticism is
    often easier than generation).
  - The 'APPROVED' string is brittle in real systems — replace with
    structured tool_use output ({"approved": bool, "feedback": "..."})
    so you get machine-readable verdicts.

This is the "close the loop" pattern: don't trust raw generator output,
make it survive scrutiny first.
"""

import asyncio
import os

from anthropic import AsyncAnthropic

GENERATOR_MODEL = "claude-haiku-4-5-20251001"
CRITIC_MODEL = "claude-haiku-4-5-20251001"  # in prod: cheaper than generator


def get_client() -> AsyncAnthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")
    return AsyncAnthropic()


async def generate(
    client: AsyncAnthropic,
    task: str,
    prev: str | None,
    feedback: str | None,
) -> str:
    if prev is None:
        user = task
    else:
        user = (
            f"Original task: {task}\n\nYour previous attempt:\n{prev}\n\n"
            f"Critic feedback:\n{feedback}\n\nProduce an improved version."
        )
    response = await client.messages.create(
        model=GENERATOR_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


async def critique(client: AsyncAnthropic, task: str, output: str) -> str:
    response = await client.messages.create(
        model=CRITIC_MODEL,
        max_tokens=300,
        system=(
            "You are a strict reviewer. Identify concrete, actionable issues with "
            "the work. Be specific. If the work is genuinely good enough, reply "
            "with exactly the single word: APPROVED"
        ),
        messages=[
            {"role": "user", "content": f"Task: {task}\n\nWork to review:\n{output}"}
        ],
    )
    return response.content[0].text


async def generator_critic_loop(task: str, max_iterations: int = 3) -> str:
    client = get_client()
    output: str | None = None
    feedback: str | None = None

    for i in range(max_iterations):
        print(f"\n--- iteration {i + 1} ---")
        output = await generate(client, task, output, feedback)
        print(f"GENERATOR:\n{output}")

        feedback = await critique(client, task, output)
        print(f"\nCRITIC: {feedback}")

        if "APPROVED" in feedback.upper():
            print(f"\napproved after {i + 1} iteration(s)")
            return output

    print(f"\nmax_iterations ({max_iterations}) reached; returning best effort")
    return output  # type: ignore[return-value]


async def main() -> None:
    final = await generator_critic_loop(
        "Write a tweet (max 280 chars) explaining what Python's GIL is "
        "to an experienced JavaScript developer. Be technically precise."
    )
    print(f"\n=== FINAL ({len(final)} chars) ===\n{final}")


if __name__ == "__main__":
    asyncio.run(main())
