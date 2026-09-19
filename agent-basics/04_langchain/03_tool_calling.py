"""Tool calling with LangChain's @tool decorator + .bind_tools().

The same pattern as 03_anthropic_agents/01_agent_loop.py, but expressed
through LangChain primitives. The agent loop logic is identical — what
changes is the *shape* of the input/output:

Raw Anthropic SDK                  LangChain
-----------------                  ---------
list[dict] messages                list[BaseMessage] (HumanMessage, AIMessage, ToolMessage)
response.content[0].type           message.tool_calls
response.stop_reason == "tool_use" not message.tool_calls (empty list when done)
tool_use_id                        tool_call_id (in ToolMessage)

The advantage isn't fewer lines — it's that swapping `ChatAnthropic`
for `ChatOpenAI` makes the same loop work against a different provider.
"""

import asyncio
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

MODEL = "claude-haiku-4-5-20251001"


def check_env() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")


# @tool: type annotations + docstring become the schema the model sees.
@tool
def get_weather(location: str, units: str = "celsius") -> dict:
    """Get the current weather for a city."""
    fake = {"Amsterdam": 14, "Tokyo": 22, "Sydney": 26}
    temp = fake.get(location.split(",")[0].strip(), 18)
    if units == "fahrenheit":
        temp = temp * 9 / 5 + 32
    return {"location": location, "temp": temp, "units": units}


@tool
def get_time(timezone: str) -> dict:
    """Get the current local time for an IANA timezone (e.g. 'Asia/Tokyo')."""
    fakes = {"Europe/Amsterdam": "14:32", "Asia/Tokyo": "22:32"}
    return {"timezone": timezone, "time": fakes.get(timezone, "12:00")}


TOOLS = [get_weather, get_time]
TOOL_REGISTRY = {t.name: t for t in TOOLS}


async def run_agent(question: str, max_iterations: int = 6) -> str:
    check_env()
    model = ChatAnthropic(model=MODEL, max_tokens=1024).bind_tools(TOOLS)

    messages: list[BaseMessage] = [HumanMessage(content=question)]

    for iteration in range(max_iterations):
        response: AIMessage = await model.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # No tools requested -> we're done. Extract text.
            return response.content if isinstance(response.content, str) else str(response.content)

        # Execute every requested tool call.
        for call in response.tool_calls:
            print(f"  [iter {iteration}] -> {call['name']}({call['args']})")
            try:
                result = await TOOL_REGISTRY[call["name"]].ainvoke(call["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            except Exception as e:  # noqa: BLE001 -- tool errors go back to the model, never kill the loop
                messages.append(
                    ToolMessage(content=f"Error: {e}", tool_call_id=call["id"], status="error"),
                )

    raise RuntimeError(f"agent did not terminate within {max_iterations} iterations")


async def main() -> None:
    answer = await run_agent("What's the weather in Tokyo, and what local time is it there right now?")
    print(f"\nFINAL:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
