"""Fan-out / fan-in: many sub-agents explore independently, one synthesises.

Pattern:
  1. Spawn N sub-agents in parallel, each with a different angle on the task.
  2. asyncio.gather their outputs.
  3. Pass all N outputs to a synthesiser agent that reconciles them.

Why N branches: agents are stochastic. One branch may miss something
obvious; the synthesiser sees disagreements and resolves them. Cost
scales linearly with N — pick N based on stakes vs. budget.

This is what production "deep research" features look like under the
hood, simplified to ~50 lines.
"""

import asyncio
import os

from anthropic import AsyncAnthropic

MODEL = "claude-haiku-4-5-20251001"


def get_client() -> AsyncAnthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")
    return AsyncAnthropic()


async def explore_branch(client: AsyncAnthropic, query: str, angle: str) -> str:
    """One sub-agent's perspective."""
    response = await client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=(
            f"You are exploring a question from this angle: {angle}. "
            "Be concise (3-5 bullet points)."
        ),
        messages=[{"role": "user", "content": query}],
    )
    return response.content[0].text


async def synthesise(
    client: AsyncAnthropic,
    query: str,
    branch_outputs: list[tuple[str, str]],
) -> str:
    """Reconcile the N branches into a coherent answer."""
    body = "\n\n".join(f"### {angle}\n{output}" for angle, output in branch_outputs)
    prompt = (
        f"Original question: {query}\n\n"
        f"You received {len(branch_outputs)} independent perspectives below. "
        f"Synthesise them: where do they agree? Where do they diverge? "
        f"Produce a final answer no longer than 6 bullet points.\n\n"
        f"{body}"
    )
    response = await client.messages.create(
        model=MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def fan_out_research(query: str, angles: list[str]) -> str:
    client = get_client()

    # FAN OUT: parallel branch exploration.
    branch_texts = await asyncio.gather(
        *(explore_branch(client, query, a) for a in angles)
    )
    for angle, text in zip(angles, branch_texts, strict=True):
        print(f"\n--- branch: {angle} ---\n{text}")

    # FAN IN: synthesise.
    return await synthesise(client, query, list(zip(angles, branch_texts, strict=True)))


async def main() -> None:
    query = "Should a startup adopt Kubernetes from day one?"
    angles = [
        "pro-Kubernetes engineering perspective",
        "skeptical operational-cost perspective",
        "early-stage startup founder perspective",
    ]
    final = await fan_out_research(query, angles)
    print(f"\n=== SYNTHESIS ===\n{final}")


if __name__ == "__main__":
    asyncio.run(main())
