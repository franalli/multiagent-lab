"""Token streaming with .astream() — yields each chunk as it arrives.

For chat UIs you want to display text as the model emits it, not wait
for the full response. LangChain wraps the provider-specific streaming
API (Anthropic's SSE in this case) behind a uniform async iterator.

Two streaming methods worth knowing:
  - `.astream(input)`           — yields chunks of output (default: AIMessageChunk)
  - `.astream_events(input)`    — yields fine-grained events from EVERY runnable
                                  in the chain (on_chat_model_start, on_tool_end, etc.)
                                  — useful for instrumentation and UI.

This script demonstrates .astream(); .astream_events() is mentioned for
when you need step-level observability instead of just token output.
"""

import asyncio
import os
import sys
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

MODEL = "claude-haiku-4-5-20251001"


def check_env() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")


async def main() -> None:
    check_env()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a poetic explainer. Write a short stanza."),
            ("user", "{topic}"),
        ]
    )
    model = ChatAnthropic(model=MODEL, max_tokens=300)
    chain = prompt | model

    print("Streaming response (each char appears as it arrives):\n")
    t0 = time.perf_counter()
    first_token_at: float | None = None
    chunk_count = 0
    total_text = ""

    async for chunk in chain.astream({"topic": "the asyncio event loop"}):
        if first_token_at is None:
            first_token_at = time.perf_counter() - t0
        # AIMessageChunk.content can be a string OR a list of content blocks
        # (when extended thinking / tool calls are enabled). For plain text it's a str.
        text = chunk.content if isinstance(chunk.content, str) else ""
        total_text += text
        sys.stdout.write(text)
        sys.stdout.flush()
        chunk_count += 1

    total_elapsed = time.perf_counter() - t0
    print("\n\n--- timing ---")
    print(f"first token after: {first_token_at:.2f}s")
    print(f"full response:     {total_elapsed:.2f}s")
    print(f"chunks:            {chunk_count}")
    print(f"chars:             {len(total_text)}")


if __name__ == "__main__":
    asyncio.run(main())
