"""Checkpointing: persistent state via InMemorySaver, keyed by thread_id.

The big LangGraph feature you can't easily build yourself: each step
of graph execution writes a checkpoint. Pass a `thread_id` config and
the graph resumes from where it left off — even across process restarts
if you swap InMemorySaver for SqliteSaver / PostgresSaver.

This is what enables:
  - **Conversational memory** — invoke the same thread_id again, history is preserved.
  - **Human-in-the-loop** — pause the graph at a point, await human review,
    resume from the checkpoint. (Set up via `interrupt_before=[...]` on compile.)
  - **Crash recovery** — long-running agents resume after worker restarts.
  - **Time travel** — inspect or replay any prior checkpoint by thread_id + checkpoint_id.

This demo uses a ReAct agent + InMemorySaver to show conversational memory:
two separate .ainvoke() calls on the same thread_id, with the second
turn able to reference the first.
"""

import asyncio
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

MODEL = "claude-haiku-4-5-20251001"


def check_env() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")


@tool
def remember_fact(fact: str) -> str:
    """Record a fact for later. Use when the user shares info they may reference later."""
    return f"Recorded: {fact}"


async def main() -> None:
    check_env()

    checkpointer = InMemorySaver()  # in-memory; swap for SqliteSaver in real apps
    agent = create_react_agent(
        model=ChatAnthropic(model=MODEL, max_tokens=512),
        tools=[remember_fact],
        prompt=("You are a helpful assistant. When the user shares a fact about themselves, call remember_fact. When they ask a question, answer using your memory."),
        checkpointer=checkpointer,
    )

    # Same thread_id across both turns -> the second invocation sees the first's history.
    config = {"configurable": {"thread_id": "user-42"}}

    print("--- turn 1: introduce a fact ---")
    r1 = await agent.ainvoke(
        {"messages": [("user", "My favourite language is Python and I'm based in Amsterdam.")]},
        config=config,
    )
    print(f"AI: {r1['messages'][-1].content}\n")

    print("--- turn 2: ask about it ---")
    r2 = await agent.ainvoke(
        {"messages": [("user", "Where am I based and what language do I prefer?")]},
        config=config,
    )
    print(f"AI: {r2['messages'][-1].content}")

    # Inspect what the checkpointer is holding.
    print("\n--- checkpointer state ---")
    state = await agent.aget_state(config)
    print(f"messages in thread: {len(state.values['messages'])}")
    print(f"next nodes: {state.next}")

    # Different thread_id -> no memory of the above.
    print("\n--- different thread, no memory ---")
    other_config = {"configurable": {"thread_id": "user-99"}}
    r3 = await agent.ainvoke(
        {"messages": [("user", "Where am I based?")]},
        config=other_config,
    )
    print(f"AI: {r3['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
