"""Canonical async agent loop with the Gemini SDK (google-genai) tool calling.

The agent loop is identical in *shape* across providers — only the wire
format changes. Same three steps as the Anthropic version
(03_anthropic_agents/01_agent_loop.py):
  1. Send contents + function declarations to the model.
  2. If the model returns no function calls: return the text.
  3. If it returns function calls: execute each, append the results,
     loop back to step 1.

Gemini-specific gotchas (why this isn't a copy-paste of the Anthropic loop):
  - Append the model's full turn (`response.candidates[0].content`,
    role="model") INCLUDING its function_call parts — same invariant as
    Anthropic: the model must see its own past calls.
  - Function *results* go back in a Content with role="user", NOT
    "tool"/"function". This trips everyone up.
  - `function_call.args` is ALREADY a dict (so is Anthropic's `.input`;
    OpenAI hands you a JSON *string* you must json.loads).
  - There's no is_error flag — encode failures inside the response dict
    ({"error": ...}) so the model can see and react to them.
  - Cap iterations. A misbehaving model can loop forever.

Passing `function_declarations` (rather than Python callables) keeps tool
execution MANUAL: the SDK returns function_call parts instead of running
them for you, which is what lets us own the loop.

Requires: GEMINI_API_KEY env var.  Install: pip install google-genai
(verified against google-genai 0.8.0; the function-calling surface used
here is stable through 1.x).
"""

import asyncio
import os
from collections.abc import Awaitable, Callable
from typing import Any

from google import genai
from google.genai import types

MODEL = "gemini-3-flash-preview"  # repo default; swap to gemini-2.5-flash


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY to run this demo.")
    return genai.Client(api_key=api_key)


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


# Gemini wants typed Schema objects rather than raw JSON-schema dicts: the
# `type` is an enum (types.Type.STRING), not the lowercase string "string".
FUNCTIONS = [
    types.FunctionDeclaration(
        name="get_weather",
        description="Get current weather for a city.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "location": types.Schema(
                    type=types.Type.STRING,
                    description="City and country, e.g. 'Amsterdam, NL'",
                ),
                "units": types.Schema(
                    type=types.Type.STRING,
                    enum=["celsius", "fahrenheit"],
                    description="Temperature units",
                ),
            },
            required=["location"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_time",
        description="Get current local time for an IANA timezone.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "timezone": types.Schema(
                    type=types.Type.STRING,
                    description="IANA tz, e.g. 'Asia/Tokyo'",
                ),
            },
            required=["timezone"],
        ),
    ),
]

CONFIG = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=FUNCTIONS)]
)

TOOL_IMPLS: dict[str, Callable[..., Awaitable[Any]]] = {
    "get_weather": get_weather,
    "get_time": get_time,
}


async def run_agent(prompt: str, max_iterations: int = 8) -> str:
    client = get_client()
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    for iteration in range(max_iterations):
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=contents,
            config=CONFIG,
        )
        # Append the model's full turn (role="model"), including function_call parts.
        contents.append(response.candidates[0].content)

        calls = response.function_calls
        if not calls:
            return response.text or ""

        # Answer EVERY function_call. Gemini quirk: results ride in a role="user"
        # Content, gathered into a single message (mirrors collecting tool_results).
        tool_parts = []
        for fc in calls:
            print(f"  [iter {iteration}] -> {fc.name}({fc.args})")
            try:
                result = await TOOL_IMPLS[fc.name](**(fc.args or {}))
                payload: dict[str, Any] = {"result": result}
            except Exception as e:  # noqa: BLE001 -- tool errors go back to the model, never kill the loop
                payload = {"error": str(e)}  # no is_error flag; encode it in the body
            tool_parts.append(
                types.Part.from_function_response(name=fc.name, response=payload)
            )
        contents.append(types.Content(role="user", parts=tool_parts))

    raise RuntimeError(f"agent did not terminate within {max_iterations} iterations")


async def main() -> None:
    answer = await run_agent(
        "What's the weather in Tokyo right now, and what local time is it there?"
    )
    print(f"\nFINAL:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
