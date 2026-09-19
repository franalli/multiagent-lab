"""create_react_agent: the prebuilt agent loop, in one line.

LangGraph's `create_react_agent` is the prebuilt ReAct loop — model
reasons, requests tools, framework executes, results feed back, repeat
until end_turn. Under the hood it builds a graph with:

  START -> agent (LLM call)
  agent -> tools  (if response.tool_calls)
  agent -> END    (if no tool calls)
  tools -> agent  (loop back after execution)

Compare to:
  03_anthropic_agents/01_agent_loop.py — same logic, ~50 lines hand-written
  04_langchain/03_tool_calling.py      — same logic, LangChain primitives

What you get for free with the prebuilt:
  - Parallel tool execution within a turn
  - Automatic message-history management
  - Standard hooks for checkpointing (script 04 here uses this)
  - Streaming intermediate states (.astream())

What you give up:
  - Custom error handling at each step (you'd subclass or build your own graph)
  - Total control over the prompt structure (system prompt is the main lever)
"""

import asyncio
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

MODEL = "claude-haiku-4-5-20251001"


def check_env() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")


@tool
def get_weather(location: str) -> dict:
    """Get the current weather for a city."""
    fake = {"Amsterdam": 14, "Tokyo": 22, "Sydney": 26}
    return {
        "location": location,
        "temp_c": fake.get(location.split(",")[0].strip(), 18),
    }


@tool
def get_time(timezone: str) -> dict:
    """Get the current local time for an IANA timezone."""
    fakes = {"Europe/Amsterdam": "14:32", "Asia/Tokyo": "22:32"}
    return {"timezone": timezone, "time": fakes.get(timezone, "12:00")}


@tool
def get_population(city: str) -> dict:
    """Get the population of a city."""
    fakes = {"Amsterdam": 905_000, "Tokyo": 13_960_000, "Sydney": 5_312_000}
    return {"city": city, "population": fakes.get(city, "unknown")}


async def main() -> None:
    check_env()

    model = ChatAnthropic(model=MODEL, max_tokens=1024)
    agent = create_react_agent(
        model=model,
        tools=[get_weather, get_time, get_population],
        prompt="You answer briefly. Use tools to look up facts you don't know.",
    )

    question = "Give me a quick fact sheet on Tokyo: weather, local time, and population."
    result = await agent.ainvoke({"messages": [("user", question)]})

    # The result["messages"] list is the full conversation including
    # tool calls and tool results. The final assistant message has the answer.
    final = result["messages"][-1]
    print(f"FINAL:\n{final.content}\n")

    # Show the tool calls that happened.
    print("--- tool calls during the run ---")
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                print(f"  {call['name']}({call['args']})")


if __name__ == "__main__":
    asyncio.run(main())
