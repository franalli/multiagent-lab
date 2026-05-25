"""Conditional edges: branching based on state — the router pattern.

A `conditional_edge` lets a node's output decide which node runs next.
This is the primitive behind agent loops: after the model runs, check
whether it requested tools; if yes -> tool node, if no -> END.

The pattern:
  1. Define a router function: state -> str (the next node's name).
  2. Register it with `add_conditional_edges(source, router, mapping)`.
  3. LangGraph dispatches.

This is what raw asyncio cannot express ergonomically: state-driven
control flow with multiple possible next steps, visible to outside
tooling (so you can draw the graph, replay it, etc.).

No API key needed — uses a pure-Python "model" that just inspects input.
"""

import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class TriageState(TypedDict):
    query: str
    category: str
    response: str


async def classify(state: TriageState) -> dict:
    """A real implementation would call an LLM here."""
    q = state["query"].lower()
    if any(w in q for w in ("error", "crash", "bug", "broken")):
        return {"category": "bug"}
    if any(w in q for w in ("how", "what", "why")):
        return {"category": "question"}
    return {"category": "feedback"}


async def handle_bug(state: TriageState) -> dict:
    return {
        "response": f"Logged bug report: '{state['query']}'. Ticket #12345 created."
    }


async def handle_question(state: TriageState) -> dict:
    return {"response": f"Answer to '{state['query']}': here are some docs..."}


async def handle_feedback(state: TriageState) -> dict:
    return {"response": f"Thanks for the feedback: '{state['query']}'"}


def route_by_category(state: TriageState) -> str:
    """Router: returns the name of the next node to execute."""
    return state["category"]


def build_graph():
    graph = StateGraph(TriageState)
    graph.add_node("classify", classify)
    graph.add_node("handle_bug", handle_bug)
    graph.add_node("handle_question", handle_question)
    graph.add_node("handle_feedback", handle_feedback)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_category,
        {
            # Map router output -> destination node.
            "bug": "handle_bug",
            "question": "handle_question",
            "feedback": "handle_feedback",
        },
    )
    # All three handlers terminate.
    for handler in ("handle_bug", "handle_question", "handle_feedback"):
        graph.add_edge(handler, END)
    return graph.compile()


async def main() -> None:
    app = build_graph()
    queries = [
        "Why does my app crash on startup?",
        "How do I configure routing in LangGraph?",
        "The new dashboard layout feels great!",
    ]
    for q in queries:
        result = await app.ainvoke({"query": q, "category": "", "response": ""})
        print(f"Q: {q}")
        print(f"   category: {result['category']}")
        print(f"   response: {result['response']}\n")


if __name__ == "__main__":
    asyncio.run(main())
