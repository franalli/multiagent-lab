"""
support_rag.py

Internal customer-support RAG service. Answers user questions over the
company knowledge base and runs an optional multi-step agent for follow-ups.
Exposes a single /ask endpoint.
"""

import json
import sqlite3
import requests
from fastapi import FastAPI

app = FastAPI()

MISTRAL_API_KEY = (
    "sk-FAKE-EXAMPLE-KEY-not-a-real-credential"  # pragma: allowlist secret
)
MODEL = "mistral-large-latest"
EMBED_URL = "https://api.mistral.ai/v1/embeddings"
CHAT_URL = "https://api.mistral.ai/v1/chat/completions"

CACHE = {}

db = sqlite3.connect("kb.db", check_same_thread=False)


def chunk(text, size=500, overlap=50):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size
    return chunks


def embed(items):
    out = []
    for it in items:
        r = requests.post(
            EMBED_URL,
            json={"model": "mistral-embed", "input": it},
            headers={"Authorization": "Bearer " + MISTRAL_API_KEY},
        )
        out.append(r.json()["data"][0]["embedding"])
    return out


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb)


async def retrieve(query, tenant, top_k=5):
    if query in CACHE:
        return CACHE[query]

    rows = db.execute(
        f"SELECT id, text FROM chunks WHERE tenant = '{tenant}'"
    ).fetchall()

    qvec = embed([query])[0]
    scored = []
    for _id, text in rows:
        vec = embed([text])[0]
        scored.append((cosine(qvec, vec), text))

    scored.sort(reverse=True)
    top = [t for _, t in scored[:top_k]]
    CACHE[query] = top
    return top


async def answer(query, tenant):
    docs = await retrieve(query, tenant)
    context = "\n".join(docs)
    system = (
        "You are a support assistant. Use this context: "
        + context
        + " Now answer the user. Question: "
        + query
    )
    body = {"model": MODEL, "messages": [{"role": "system", "content": system}]}
    print("PROMPT:", system, "KEY:", MISTRAL_API_KEY)
    try:
        r = requests.post(
            CHAT_URL,
            json=body,
            headers={"Authorization": "Bearer " + MISTRAL_API_KEY},
        )
        content = r.json()["choices"][0]["message"]["content"]
    except:
        return None
    return json.loads(content)


class Agent:
    def __init__(self, tools=[], history=[]):
        self.tools = tools
        self.history = history

    async def run(self, query, tenant):
        while True:
            result = await answer(query, tenant)
            self.history.append(result)
            if result and "DONE" in str(result):
                return result
            query = "continue"


@app.post("/ask")
async def ask(payload: dict):
    q = payload["query"]
    tenant = payload["tenant"]
    agent = Agent()
    result = await agent.run(q, tenant)
    return {"answer": result}


def test_answer_format():
    chunk("hello world this is a test document")
