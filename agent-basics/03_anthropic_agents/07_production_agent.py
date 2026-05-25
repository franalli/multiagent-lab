"""Production-grade agent: everything from earlier scripts, composed.

Pulls in:
  - Parallel tool execution (asyncio.gather)              [14]
  - Per-tool timeouts (asyncio.timeout)                   [05]
  - Concurrency cap (asyncio.Semaphore)                   [07]
  - Retries on the model call with backoff + jitter       [09]
  - Structured error returns (is_error=True)              [13]
  - Iteration cap to bound runaway loops
  - Per-run observability (token count, tool calls, time)

This is the shape a senior interviewer expects you to whiteboard for
"design a production agent runtime." Each piece exists in isolation
in scripts 05-14; this one composes them.
"""

import asyncio
import logging
import os
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from anthropic import APIError, AsyncAnthropic

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"


def get_client() -> AsyncAnthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")
    return AsyncAnthropic()


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Awaitable[Any]]
    timeout_seconds: float = 10.0


@dataclass
class AgentRun:
    task: str
    final_output: str = ""
    iterations: int = 0
    total_tokens: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0


class Agent:
    def __init__(
        self, tools: list[Tool], *, max_iterations: int = 8, max_parallel_tools: int = 5
    ):
        self.client = get_client()
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations
        self.semaphore = asyncio.Semaphore(max_parallel_tools)

    async def _call_model(self, messages: list, anthropic_tools: list) -> Any:
        """Retry the model call on transient API errors."""
        for attempt in range(3):
            try:
                return await self.client.messages.create(
                    model=MODEL,
                    max_tokens=2048,
                    tools=anthropic_tools,
                    messages=messages,
                )
            except APIError as e:
                if attempt == 2:
                    raise
                delay = random.uniform(0, min(1.0 * (2**attempt), 8.0))
                log.warning(f"model call failed ({e}); retrying in {delay:.2f}s")
                await asyncio.sleep(delay)

    async def _execute_tool(self, block) -> dict:
        tool = self.tools.get(block.name)
        if tool is None:
            return {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": f"Unknown tool: {block.name}",
                "is_error": True,
            }

        async with self.semaphore:  # cap concurrent tool work
            try:
                async with asyncio.timeout(tool.timeout_seconds):
                    result = await tool.handler(**block.input)
                log.info(f"tool {block.name} ok")
                return {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                }
            except TimeoutError:
                log.warning(f"tool {block.name} timed out")
                return {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Timeout after {tool.timeout_seconds}s",
                    "is_error": True,
                }
            except Exception as e:
                log.error(f"tool {block.name} failed: {e}")
                return {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Error: {e}",
                    "is_error": True,
                }

    async def run(self, task: str) -> AgentRun:
        run = AgentRun(task=task)
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        anthropic_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tools.values()
        ]
        start = time.perf_counter()

        for iteration in range(self.max_iterations):
            run.iterations = iteration + 1
            response = await self._call_model(messages, anthropic_tools)
            messages.append({"role": "assistant", "content": response.content})
            run.total_tokens += (
                response.usage.input_tokens + response.usage.output_tokens
            )

            if response.stop_reason == "end_turn":
                run.final_output = "\n".join(
                    b.text for b in response.content if b.type == "text"
                )
                break

            if response.stop_reason == "tool_use":
                blocks = [b for b in response.content if b.type == "tool_use"]
                run.tool_calls.extend(
                    {"name": b.name, "input": b.input} for b in blocks
                )
                # Parallel execution; semaphore caps real concurrency.
                results = await asyncio.gather(*(self._execute_tool(b) for b in blocks))
                messages.append({"role": "user", "content": results})
                continue

            raise RuntimeError(f"unexpected stop_reason: {response.stop_reason}")
        else:
            run.final_output = "(max iterations exceeded)"

        run.duration_seconds = time.perf_counter() - start
        return run


# --- demo tools ---


async def search_web(query: str) -> dict:
    await asyncio.sleep(0.3)
    return {"query": query, "results": [f"Result about {query} #{i}" for i in range(3)]}


async def convert_temperature(value: float, from_unit: str, to_unit: str) -> dict:
    units = {"celsius", "fahrenheit", "kelvin"}
    if from_unit not in units or to_unit not in units:
        raise ValueError(f"unit must be one of {units}")
    # normalise to celsius
    c = (
        value
        if from_unit == "celsius"
        else ((value - 32) * 5 / 9 if from_unit == "fahrenheit" else value - 273.15)
    )
    out = (
        c
        if to_unit == "celsius"
        else (c * 9 / 5 + 32 if to_unit == "fahrenheit" else c + 273.15)
    )
    return {"value": value, "from": from_unit, "to": to_unit, "result": round(out, 2)}


TOOLS = [
    Tool(
        name="search_web",
        description="Search the web for recent information.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=search_web,
    ),
    Tool(
        name="convert_temperature",
        description="Convert a temperature between celsius, fahrenheit, and kelvin.",
        input_schema={
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit", "kelvin"],
                },
                "to_unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit", "kelvin"],
                },
            },
            "required": ["value", "from_unit", "to_unit"],
        },
        handler=convert_temperature,
    ),
]


async def main() -> None:
    agent = Agent(TOOLS)
    run = await agent.run(
        "Search the web for current asyncio best practices, and also convert "
        "100 degrees fahrenheit to celsius and kelvin."
    )
    print(f"\n=== FINAL ===\n{run.final_output}")
    print(
        f"\niterations={run.iterations}  tokens={run.total_tokens}  "
        f"tool_calls={len(run.tool_calls)}  duration={run.duration_seconds:.2f}s"
    )


if __name__ == "__main__":
    asyncio.run(main())
