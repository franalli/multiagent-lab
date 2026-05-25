"""LangChain LCEL — composing prompt | model | parser with the | operator.

LCEL (LangChain Expression Language) is the modern primitive: every
component is a `Runnable`, and you compose them with `|` (pipe).
Each Runnable has:
  - `.invoke(input)`     — sync
  - `.ainvoke(input)`    — async (what async code uses)
  - `.batch(inputs)`     — concurrent batch, sync
  - `.abatch(inputs)`    — concurrent batch, async
  - `.stream(input)`     — sync token streaming
  - `.astream(input)`    — async token streaming

Compare to script 03_anthropic_agents/01_agent_loop.py: the agent loop
is ~30 lines of raw Anthropic SDK. LangChain doesn't shorten it
dramatically — it standardises the *shape* across providers (swap
ChatAnthropic for ChatOpenAI, same chain works).
"""

import asyncio
import os
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

MODEL = "claude-haiku-4-5-20251001"


def check_env() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run this demo.")


async def main() -> None:
    check_env()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise technical writer. Answer in 2 sentences."),
            ("user", "{question}"),
        ]
    )
    model = ChatAnthropic(model=MODEL, max_tokens=200)
    parser = StrOutputParser()

    # LCEL composition: prompt | model | parser
    # Each `|` wires the previous step's output into the next step's input.
    chain = prompt | model | parser

    print("--- single ainvoke ---")
    answer = await chain.ainvoke({"question": "What is asyncio's event loop?"})
    print(answer)

    print("\n--- abatch: 3 questions concurrently ---")
    questions = [
        {"question": "What is a coroutine?"},
        {"question": "What is the GIL?"},
        {"question": "What is structured concurrency?"},
    ]
    t0 = time.perf_counter()
    answers = await chain.abatch(questions)
    elapsed = time.perf_counter() - t0
    for q, a in zip(questions, answers, strict=True):
        print(f"\nQ: {q['question']}\nA: {a}")
    print(f"\nabatch wall-clock: {elapsed:.2f}s (concurrent under the hood)")


if __name__ == "__main__":
    asyncio.run(main())
