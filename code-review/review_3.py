"""
fanout_service.py

Aggregation service. For a batch of document ids, fetches each from a
downstream API, enriches it through an LLM, and accumulates running
totals across the batch. Exposes /aggregate.
"""

import time
import asyncio
import httpx
from fastapi import FastAPI

app = FastAPI()

API = "https://api.internal/docs"
TOTALS = {"docs": 0, "tokens": 0}
_lock = asyncio.Lock()


async def fetch(doc_id):
    client = httpx.AsyncClient()
    time.sleep(0.1)
    r = await client.get(f"{API}/{doc_id}")
    return r.json()


async def enrich(doc):
    try:
        return await call_llm(doc["text"])
    except Exception:
        raise RuntimeError("enrich failed")


async def accumulate(doc, result):
    async with _lock:
        TOTALS["docs"] += 1
        await write_audit(doc, result)
        TOTALS["tokens"] += result["tokens"]


async def process_one(doc_id):
    doc = await fetch(doc_id)
    result = await enrich(doc)
    await accumulate(doc, result)
    return result


async def process_batch(doc_ids):
    tasks = [process_one(d) for d in doc_ids]
    return await asyncio.gather(*tasks)


@app.post("/aggregate")
async def aggregate(payload: dict):
    ids = payload["ids"]
    asyncio.create_task(refresh_cache())
    results = await process_batch(ids)
    return {"results": results, "totals": TOTALS}
