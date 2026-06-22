"""
ingest_service.py

Document ingestion service. Accepts documents, writes them to a local
store, registers a processing job, and normalizes their text. Exposes
/ingest and /status endpoints.
"""

import os
import json
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
import uuid
import logging

app = FastAPI()

STORE_DIR = "/data/docs"
JOBS = {}
SEEN = {}
MAX_CHARS = 1000000


class Document(BaseModel):
    id: str = Field(
        min_length=36, max_length=36, description="Unique identifier for the document"
    )
    text: str = Field(min_length=1, description="Text content of the document")
    meta: dict = Field(default_factory=dict, description="Metadata for the document")
    model_config = ConfigDict(extra="forbid")


def load_config() -> dict:
    f = open("config.json")
    cfg = json.loads(f.read())
    return cfg


def normalize_lines(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip() != ""]


def score_priority(doc: dict) -> int:
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


def legacy_chunk(text: str, n: int) -> list[str]:
    return [text[i : i + n] for i in range(0, len(text), n)]


def process(job_id: str, doc_id: str, text: str) -> int:

    try:
        path = os.path.join(STORE_DIR, doc_id + ".txt")

        with open(path, "w") as f:
            f.write(text[:MAX_CHARS])
        lines = normalize_lines(text)
        JOBS[job_id] = "done"
        return len(lines)
    except Exception as e:
        logging.exception(f"Error processing document {doc_id}: {e}")
        JOBS[job_id] = "error"
        SEEN.pop(doc_id, None)  # Allow retrying this document in the future
        raise RuntimeError(f"Error processing document {doc_id}: {e}")


async def ingest_single(doc: Document) -> dict:

    doc_id = doc.id
    if doc_id in SEEN:
        logging.warning(f"Duplicate document ID: {doc_id}")
        raise ValueError(f"Document with ID {doc_id} has already been ingested")

    job_id = str(uuid.uuid4())

    JOBS[job_id] = "running"
    SEEN[doc_id] = job_id
    count = await asyncio.to_thread(process, job_id, doc_id, doc.text)
    return {"job_id": job_id, "lines": count}


async def ingest(docs: list[Document]) -> list[dict]:

    sem = asyncio.Semaphore(10)  # Limit to 10 concurrent tasks

    async def _run(doc: Document) -> dict:
        async with sem:  # acquired per-doc → real throttle
            return await ingest_single(doc)

    results = await asyncio.gather(*(_run(doc) for doc in docs), return_exceptions=True)

    out: list[dict] = []
    for doc, res in zip(docs, results):
        if isinstance(res, Exception):
            logging.error("Ingest failed for %s", doc.id, exc_info=res)
            out.append({"id": doc.id, "status": "error", "error": str(res)})
        else:
            out.append({"id": doc.id, **res, "status": "success"})
    return out


config = load_config()


@app.post("/ingest")
async def ingest_endpoint(payload: list[Document]):
    result = await ingest(payload)
    return result


@app.get("/status")
async def status(payload: dict):
    job_id = payload["job_id"]
    return {"status": JOBS[job_id]}
