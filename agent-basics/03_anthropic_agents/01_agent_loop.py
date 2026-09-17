"""Canonical async agent loop with Anthropic tool calling.

The agent loop is the heart of every tool-using LLM system:
  1. Send messages + tool definitions to the model.
  2. If stop_reason == "end_turn": return the text.
  3. If stop_reason == "tool_use": execute each tool, append results,
     loop back to step 1.

Critical invariants:
  - Append the assistant response IN FULL (tool_use blocks included).
    The model needs to see its own past tool calls.
  - Return a tool_result for EVERY tool_use, even on failure.
    Missing results -> API error. Failures use is_error=True.
  - Cap iterations. A misbehaving model can loop forever.

Requires: ANTHROPIC_API_KEY env var.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import Any

from anthropic import AsyncAnthropic

MODEL = "claude-haiku-4-5-20251001"  # cheap + fast for playground


def get_client() -> AsyncAnthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")
    return AsyncAnthropic()


# --- tool implementations ---


async def get_weather(location: str, units: str = "celsius") -> dict[str, Any]:
    fake = {"Amsterdam": 14, "Tokyo": 22, "Sydney": 26}
    temp_c = fake.get(location.split(",")[0].strip(), 18)
    temp = temp_c if units == "celsius" else temp_c * 9 / 5 + 32
    return {"location": location, "temp": temp, "units": units, "conditions": "cloudy"}


async def get_time(timezone: str) -> dict[str, str]:
    fakes = {
        "Europe/Amsterdam": "14:32",
        "Asia/Tokyo": "22:32",
        "Australia/Sydney": "00:32",
    }
    return {"timezone": timezone, "time": fakes.get(timezone, "12:00")}


TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. 'Amsterdam, NL'",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_time",
        "description": "Get current local time for an IANA timezone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA tz, e.g. 'Asia/Tokyo'",
                },
            },
            "required": ["timezone"],
        },
    },
]

TOOL_IMPLS: dict[str, Callable[..., Awaitable[Any]]] = {
    "get_weather": get_weather,
    "get_time": get_time,
}


async def run_agent(prompt: str, max_iterations: int = 8) -> str:
    client = get_client()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

    for iteration in range(max_iterations):
        response = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        # Append the model's full content, including tool_use blocks.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return "\n".join(b.text for b in response.content if b.type == "text")

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                print(f"  [iter {iteration}] -> {block.name}({block.input})")
                try:
                    result = await TOOL_IMPLS[block.name](**block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        }
                    )
                except Exception as e:  # noqa: BLE001 -- tool errors go back to the model, never kill the loop
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: {e}",
                            "is_error": True,
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
            continue

        raise RuntimeError(f"unexpected stop_reason: {response.stop_reason}")

    raise RuntimeError(f"agent did not terminate within {max_iterations} iterations")


async def main() -> None:
    answer = await run_agent(
        "What's the weather in Tokyo right now, and what local time is it there?"
    )
    print(f"\nFINAL:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
