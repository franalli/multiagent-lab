"""
fanout_service.py

Aggregation service. For a batch of document ids, fetches each from a
downstream API, enriches it through an LLM, and accumulates running
totals across the batch. Exposes /aggregate.
"""

import os
import asyncio
import aiohttp
import httpx
from fastapi import FastAPI
import logging
from pydantic import BaseModel, Field, ConfigDict
from tiktoken import count_tokens

app = FastAPI()

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
API = "https://api.internal/docs"
MODEL = "mistral-large-latest"
EMBED_URL = "https://api.mistral.ai/v1/embeddings"
CHAT_URL = "https://api.mistral.ai/v1/chat/completions"

TOTALS = {"docs": 0, "tokens": 0}

SYSTEM_MESSAGE = "You are a helpful assistant that enriches document text with additional context and information about current events, relevant background, and related topics. Provide clear and concise explanations, examples, and references where appropriate. Avoid repeating the original text verbatim; instead, focus on adding value and insight. Ensure that the enriched content is accurate, informative, and engaging for the reader."
_lock = asyncio.Lock()


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {MISTRAL_API_KEY or ''}"}


class Document(BaseModel):
    id: str = Field(..., min_length=36, max_length=36)
    text: str = Field(..., max_length=1000000)
    model_config = ConfigDict(extra="forbid")


async def fetch(doc_id):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{API}/{doc_id}")
            r.raise_for_status()  # Raise an exception for non-2xx responses
            result = await r.json()
            return Document(text=result["text"], id=doc_id)
    except httpx.RequestError as e:
        logging.error(f"An error occurred while requesting {e.request.url!r}.")
        raise RuntimeError(
            f"An error occurred while requesting {e.request.url!r}."
        ) from e


async def call_llm(doc_text: str):

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": f"Please enrich the following document text: {doc_text}",
        },
    ]

    body = {"model": MODEL, "messages": messages, "tools": [], "max_tokens": 512}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                CHAT_URL,
                json=body,
                headers=_auth_headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as r:
                r.raise_for_status()  # surface HTTP errors, like embed_batch
                content = (await r.json())["choices"][0]["message"][
                    "content"
                ]  # read while connection is open
                return content
    except aiohttp.ClientError as e:
        logging.error(f"HTTP error during Mistral API call: {e}")
        raise RuntimeError(f"HTTP error during Mistral API call: {e}") from e
    except KeyError as e:
        logging.error(f"Unexpected response format from Mistral API: {e}")
        raise RuntimeError(f"Unexpected response format from Mistral API: {e}") from e
    except Exception as e:
        logging.exception(f"Error during Mistral API call: {e}")
        raise RuntimeError("Error during Mistral API call") from e


async def enrich(doc):
    try:
        result = await call_llm(doc.text)
        return Document(text=result, id=doc.id)
    except Exception as e:
        logging.exception(f"Error during enrich: {e}")
        raise RuntimeError("enrich failed") from e


def write_audit(doc: Document) -> dict:
    result = {"tokens": count_tokens(doc.text)}
    return result


async def accumulate(doc, result):
    async with _lock:
        TOTALS["docs"] += 1
        result = await asyncio.to_thread(write_audit, doc)
        TOTALS["tokens"] += result["tokens"]


async def process_one(doc_id: str) -> Document:
    try:
        doc = await fetch(doc_id)
        result = await enrich(doc)
        await accumulate(doc, result)
        return result
    except Exception as e:
        logging.error(f"Error processing document {doc_id}: {e}")
        raise RuntimeError(f"Error processing document {doc_id}") from e


async def process_batch(doc_ids: list[str]) -> list[Document]:
    tasks = [process_one(d) for d in doc_ids]
    result = await asyncio.gather(*tasks, return_exceptions=True)

    return result


@app.post("/aggregate")
async def aggregate(payload: list[Document]):

    ids = [d.id for d in payload]
    asyncio.create_task(refresh_cache())  # why cache here?

    results = await process_batch(ids)
    return {"results": results, "totals": TOTALS}
