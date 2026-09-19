"""Pydantic-defined tool schemas — validated, self-documenting.

Two wins over raw JSON schema dicts:
  1. Pydantic *validates* the input on the way in — catches model
     hallucinations (wrong type, missing field) before your handler runs.
  2. The schema is auto-generated from the type annotations, so adding
     a new field touches one place.

The pattern: declare a `BaseModel` for the tool's input, generate the
schema with `model_json_schema()`, and use the model to parse the
`tool_use.input` dict before dispatching to the handler.
"""

import asyncio
import os
from typing import Any, Literal

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field, ValidationError

MODEL = "claude-haiku-4-5-20251001"


def get_client() -> AsyncAnthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")
    return AsyncAnthropic()


# --- Tool input schemas via Pydantic ---


class WeatherInput(BaseModel):
    location: str = Field(..., description="City and country, e.g. 'Amsterdam, NL'")
    units: Literal["celsius", "fahrenheit"] = Field("celsius", description="Temperature units")


class SearchInput(BaseModel):
    query: str = Field(..., min_length=1, description="Search query")
    max_results: int = Field(5, ge=1, le=20, description="Max results to return")


# --- Tool registry: schema model + async handler ---


class Tool:
    def __init__(self, name, description, input_model, handler):
        self.name = name
        self.description = description
        self.input_model = input_model
        self.handler = handler

    def to_anthropic(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }

    async def execute(self, raw_input):
        # Validate FIRST — this is where Pydantic earns its keep.
        validated = self.input_model.model_validate(raw_input)
        return await self.handler(validated)


async def weather_handler(inp: WeatherInput) -> dict[str, Any]:
    return {"location": inp.location, "temp": 18, "units": inp.units}


async def search_handler(inp: SearchInput) -> dict[str, Any]:
    return {
        "query": inp.query,
        "results": [f"result-{i}" for i in range(inp.max_results)],
    }


TOOLS = [
    Tool("get_weather", "Get current weather.", WeatherInput, weather_handler),
    Tool("search", "Search the web.", SearchInput, search_handler),
]
REGISTRY = {t.name: t for t in TOOLS}


async def run_agent(prompt: str) -> str:
    client = get_client()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    anthropic_tools = [t.to_anthropic() for t in TOOLS]

    for _ in range(6):
        response = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=anthropic_tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return "\n".join(b.text for b in response.content if b.type == "text")

        if response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool = REGISTRY[block.name]
                try:
                    out = await tool.execute(block.input)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(out),
                        }
                    )
                except ValidationError as e:
                    # Tell the model exactly what was wrong — it can fix and retry.
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Validation error: {e.errors()}",
                            "is_error": True,
                        }
                    )
            messages.append({"role": "user", "content": results})
            continue
    raise RuntimeError("agent did not terminate")


async def main() -> None:
    print("--- generated schema for WeatherInput ---")
    import json

    print(json.dumps(WeatherInput.model_json_schema(), indent=2))
    print()
    answer = await run_agent("Search for 'asyncio TaskGroup', return 3 results.")
    print(f"\nFINAL:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
