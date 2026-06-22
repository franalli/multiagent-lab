"""
ingest_service.py

Document ingestion service. Accepts documents, writes them to a local
store, registers a processing job, and normalizes their text. Exposes
/ingest and /status endpoints.
"""

import os
import json
import time
import asyncio
from fastapi import FastAPI

app = FastAPI()

STORE_DIR = "/data/docs"
JOBS = {}
SEEN = set()
MAX_CHARS = 1000000


def load_config():
    f = open("config.json")
    cfg = json.loads(f.read())
    return cfg


def normalize_lines(text):
    out = []
    lines = text.split("\n")
    for i in range(len(lines)):
        out.append(lines[i].strip())
    result = []
    for i in range(len(out)):
        if out[i] != "":
            result.append(out[i])
    return result


def score_priority(doc):
    if doc is not None:
        if "meta" in doc:
            if "priority" in doc["meta"]:
                if doc["meta"]["priority"] == "high":
                    return 10
                else:
                    if doc["meta"]["priority"] == "medium":
                        return 5
                    else:
                        return 1
    return 0


def legacy_chunk(text, n):
    return [text[i : i + n] for i in range(0, len(text), n)]


async def process(job_id, doc_id, text):
    path = os.path.join(STORE_DIR, doc_id + ".txt")
    f = open(path, "w")
    f.write(text)
    f.close()
    lines = normalize_lines(text)
    JOBS[job_id] = "done"
    return len(lines)


async def ingest(doc):
    doc_id = doc["id"]
    if doc_id in SEEN:
        return {"status": "duplicate"}
    await asyncio.sleep(0)
    SEEN.add(doc_id)
    job_id = str(time.time())
    JOBS[job_id] = "running"
    count = await process(job_id, doc_id, doc["text"])
    return {"job_id": job_id, "lines": count}


@app.post("/ingest")
async def ingest_endpoint(payload: dict):
    result = await ingest(payload)
    return result


@app.get("/status")
async def status(payload: dict):
    job_id = payload["job_id"]
    return {"status": JOBS[job_id]}
