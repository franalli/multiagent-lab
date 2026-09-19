"""Canonical async agent loop with the OpenAI SDK (Chat Completions) tool calling.

Same three-step loop as the Anthropic version
(03_anthropic_agents/01_agent_loop.py) — Chat Completions maps almost 1:1
onto the Anthropic messages loop, which is why it's the cleanest mirror
(vs the newer Responses API, whose item/stateful model is a worse fit):
  1. Send messages + tool definitions to the model.
  2. If the assistant message has no tool_calls: return the text.
  3. If it has tool_calls: execute each, append one tool message per
     call, loop back to step 1.

OpenAI-specific gotchas:
  - Append the assistant message object IN FULL — it carries the
    `tool_calls` the model must see next turn (same invariant as
    appending Anthropic's content blocks). Appending the SDK's message
    object directly is the documented pattern.
  - Return a tool message for EVERY tool_call, keyed by `tool_call_id`.
    A missing answer -> 400 on the next request (Anthropic errors the
    same way on a missing tool_result).
  - `tool_call.function.arguments` is a JSON STRING — json.loads it.
    (Anthropic's `.input` and Gemini's `.args` are already dicts; OpenAI
    hands you text.)
  - Cap iterations. A misbehaving model can loop forever.

Requires: OPENAI_API_KEY env var.  Install: pip install openai
"""

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI

MODEL = "gpt-4.1-mini"  # cheap + fast for playground; swap to "gpt-4o-mini" if needed


def get_client() -> AsyncOpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run this demo.")
    return AsyncOpenAI()


# --- tool implementations (identical to the Anthropic demo, for comparison) ---


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


# OpenAI wraps each tool in a {"type": "function", "function": {...}} envelope;
# the inner "parameters" is plain JSON schema (same dialect as Anthropic's
# input_schema), so the schemas below are the Anthropic ones, re-wrapped.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country, e.g. 'Amsterdam, NL'",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current local time for an IANA timezone.",
            "parameters": {
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
    },
]

TOOL_IMPLS: dict[str, Callable[..., Awaitable[Any]]] = {
    "get_weather": get_weather,
    "get_time": get_time,
}


async def run_agent(prompt: str, max_iterations: int = 8) -> str:
    client = get_client()
    # Mixed list: dict messages + the SDK's assistant message object (appended below).
    messages: list[Any] = [{"role": "user", "content": prompt}]

    for iteration in range(max_iterations):
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        message = response.choices[0].message
        # Append the assistant message IN FULL — it carries the tool_calls the
        # model must see next turn. The SDK serialises the object on resend.
        messages.append(message)

        if not message.tool_calls:
            return message.content or ""

        # Answer EVERY tool_call, keyed by tool_call_id, or the next call 400s.
        for tc in message.tool_calls:
            print(f"  [iter {iteration}] -> {tc.function.name}({tc.function.arguments})")
            try:
                # arguments is a JSON STRING; a hallucinated call can be malformed,
                # so parse INSIDE the try so a bad parse becomes a recoverable error.
                args = json.loads(tc.function.arguments)
                content = str(await TOOL_IMPLS[tc.function.name](**args))
            except Exception as e:  # noqa: BLE001 -- tool errors go back to the model, never kill the loop
                content = f"Error: {e}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

    raise RuntimeError(f"agent did not terminate within {max_iterations} iterations")


async def main() -> None:
    answer = await run_agent("What's the weather in Tokyo right now, and what local time is it there?")
    print(f"\nFINAL:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
