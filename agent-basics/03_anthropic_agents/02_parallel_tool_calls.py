"""Parallel tool execution within a single agent turn.

Claude can emit multiple tool_use blocks in one response (e.g., "fetch
weather AND time"). The naive loop executes them sequentially. The
right move: asyncio.gather them so total latency = max(tools), not sum.

This is the single biggest perf win in agent loops with independent
tool calls. Compare to 13_anthropic_agent_loop.py which already does
this implicitly — here we make the parallelism visible with timing.
"""

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from anthropic import AsyncAnthropic

MODEL = "claude-haiku-4-5-20251001"
TOOL_LATENCY = 1.0  # seconds — simulate slow upstream so parallelism shows


def get_client() -> AsyncAnthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")
    return AsyncAnthropic()


async def slow_weather(location: str) -> dict[str, Any]:
    await asyncio.sleep(TOOL_LATENCY)
    return {"location": location, "temp": 18, "conditions": "cloudy"}


async def slow_stocks(ticker: str) -> dict[str, Any]:
    await asyncio.sleep(TOOL_LATENCY)
    return {"ticker": ticker, "price": 123.45}


async def slow_news(topic: str) -> dict[str, Any]:
    await asyncio.sleep(TOOL_LATENCY)
    return {"topic": topic, "headline": f"Recent {topic} development"}


TOOLS = [
    {
        "name": "get_weather",
        "description": "Weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "Latest stock price.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_news",
        "description": "Recent news on a topic.",
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
]

TOOL_IMPLS: dict[str, Callable[..., Awaitable[Any]]] = {
    "get_weather": slow_weather,
    "get_stock_price": slow_stocks,
    "get_news": slow_news,
}


async def execute_one(block) -> dict[str, Any]:
    try:
        result = await TOOL_IMPLS[block.name](**block.input)
        return {"type": "tool_result", "tool_use_id": block.id, "content": str(result)}
    except Exception as e:  # noqa: BLE001 -- tool errors go back to the model, never kill the loop
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": f"Error: {e}",
            "is_error": True,
        }


async def run_agent(prompt: str, *, parallel: bool) -> tuple[str, float]:
    client = get_client()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tool_wall_time = 0.0

    for _ in range(6):
        response = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            text = "\n".join(b.text for b in response.content if b.type == "text")
            return text, tool_wall_time

        if response.stop_reason == "tool_use":
            blocks = [b for b in response.content if b.type == "tool_use"]
            print(f"  model wants {len(blocks)} tool(s): {[b.name for b in blocks]}")

            t0 = time.perf_counter()
            if parallel:
                results = await asyncio.gather(*(execute_one(b) for b in blocks))
            else:
                results = [await execute_one(b) for b in blocks]
            tool_wall_time += time.perf_counter() - t0

            messages.append({"role": "user", "content": results})
            continue
    raise RuntimeError("agent did not terminate")


async def main() -> None:
    prompt = "Give me a quick briefing: weather in Amsterdam, current AAPL stock price, and recent news on Mars exploration. Use the tools."

    print("--- sequential tool execution ---")
    _, seq = await run_agent(prompt, parallel=False)
    print(f"tool wall-clock: {seq:.2f}s\n")

    print("--- parallel (asyncio.gather) ---")
    _, par = await run_agent(prompt, parallel=True)
    print(f"tool wall-clock: {par:.2f}s")
    print(f"\nspeedup: {seq / par:.1f}x (3 tools * {TOOL_LATENCY}s each)")


if __name__ == "__main__":
    asyncio.run(main())
