"""LangGraph fundamentals: StateGraph, nodes, edges.

A LangGraph graph has:
  - A **state** (a TypedDict): the data that flows between nodes.
  - **Nodes**: functions that take state and return *partial* state updates.
    The graph merges updates into the running state.
  - **Edges**: declare transitions between nodes. START is the entrypoint,
    END terminates execution.

This script builds a 3-node linear graph that progressively enriches
the state. Compare to a hand-written async pipeline:
  - LangGraph adds: serialisable state, automatic merging, async/sync
    parity, and (in script 04) checkpointing for resumability.
  - Hand-written wins on simplicity for trivial pipelines — but every
    abstraction you'd add (state passing, error handling, resume-from-step)
    is what LangGraph gives you for free.

No API key needed — this demo uses pure-Python nodes, no LLM call.
"""

import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class PipelineState(TypedDict):
    raw_text: str
    word_count: int
    sentiment: str
    summary: str


async def count_words(state: PipelineState) -> dict:
    """Node 1: derive word count from raw_text. Returns a partial update."""
    return {"word_count": len(state["raw_text"].split())}


async def classify_sentiment(state: PipelineState) -> dict:
    """Node 2: trivial keyword-based sentiment (would be an LLM in real life)."""
    text = state["raw_text"].lower()
    positive = sum(text.count(w) for w in ("great", "love", "excellent", "amazing"))
    negative = sum(text.count(w) for w in ("bad", "hate", "terrible", "awful"))
    if positive > negative:
        return {"sentiment": "positive"}
    if negative > positive:
        return {"sentiment": "negative"}
    return {"sentiment": "neutral"}


async def summarise(state: PipelineState) -> dict:
    """Node 3: build a summary using state populated by earlier nodes."""
    return {
        "summary": (f"Text ({state['word_count']} words) classified as {state['sentiment']}."),
    }


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("count_words", count_words)
    graph.add_node("classify_sentiment", classify_sentiment)
    graph.add_node("summarise", summarise)

    # Linear flow: START -> count_words -> classify_sentiment -> summarise -> END
    graph.add_edge(START, "count_words")
    graph.add_edge("count_words", "classify_sentiment")
    graph.add_edge("classify_sentiment", "summarise")
    graph.add_edge("summarise", END)

    return graph.compile()


async def main() -> None:
    app = build_graph()

    initial: PipelineState = {
        "raw_text": ("LangGraph is a great framework. I love how the state machine model makes complex flows tractable."),
        "word_count": 0,
        "sentiment": "",
        "summary": "",
    }
    final = await app.ainvoke(initial)
    print("--- final state ---")
    for k, v in final.items():
        print(f"  {k}: {v}")

    # Stream intermediate states as nodes complete
    print("\n--- streaming each node's output ---")
    async for chunk in app.astream(initial):
        for node_name, partial in chunk.items():
            print(f"  after {node_name}: {partial}")


if __name__ == "__main__":
    asyncio.run(main())
