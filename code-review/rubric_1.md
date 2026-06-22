# Answer Key — `review_me.py`

**Open only after your 30-minute review.** Issues are grouped by severity tier; each lists the location, the problem, and the fix. The ones marked **[MUST-CATCH]** are what separate a strong review from a weak one.

There are ~20 distinct issues. You will not get all of them in 30 minutes, and you are not meant to. The grade is about hitting the high-severity ones, naming the structural and scalability problems, prioritizing, and refactoring something.

---

## Tier 1 — Security (blocking)

1. **[MUST-CATCH] Hardcoded API key.** `MISTRAL_API_KEY = "sk-..."` in source. Move to an env var or secret manager, rotate the leaked key, and never commit or log it. <!-- pragma: allowlist secret -->
2. **[MUST-CATCH] SQL injection.** `retrieve` builds `f"... WHERE tenant = '{tenant}'"` from a user-controlled value. Parameterize the query, validate/whitelist `tenant`, and use least-privilege DB credentials.
3. **[MUST-CATCH] Prompt injection.** `answer` concatenates the user `query` *and* retrieved `docs` straight into the **system** role. A malicious user message or a poisoned KB document can override instructions or hijack tools. Keep the system prompt fixed and trusted; pass the question and the retrieved context as **user**-role data; treat retrieved content as untrusted; add input/output guardrails.
4. **[MUST-CATCH] Secret and full prompt printed to logs.** `print("PROMPT:", system, "KEY:", MISTRAL_API_KEY)` leaks the key and any PII in the context. Use a real logger, never log secrets or PII, redact.
5. **Cross-tenant data leak via the cache** (also a scalability bug, see #12). The cache key is the bare `query`, so tenant A's answer is served to tenant B for the same question. Key on `(tenant, query, top_k)` at minimum.

## Tier 1 — Correctness (blocking / high)

6. **Bare `except` swallows everything.** `answer`'s `except: pass`-style returns `None` on any failure, hiding the traceback and the real error. Catch specific exceptions, log with context, and either handle (fallback) or re-raise.
7. **`json.loads` on raw model output.** `return json.loads(content)` crashes on any non-JSON response. Use structured/JSON mode, validate against a schema (Pydantic), repair or retry on failure, and handle the give-up path.
8. **`embed` never checks the response.** `r.json()["data"][0]["embedding"]` raises `KeyError`/`IndexError` the moment the API errors or rate-limits. Check status, handle errors, retry.
9. **No input validation on `/ask`.** `payload: dict` then `payload["query"]` / `payload["tenant"]` — missing fields are an unhandled `KeyError`, and nothing is validated. Use a Pydantic request model.
10. **`cosine` divides by zero** on a zero vector (`na * nb == 0`). Guard it.
11. **`chunk` ignores its `overlap` parameter.** `i += size` (not `size - overlap`) means the overlap argument does nothing, so chunks never overlap and context is lost at boundaries. Use `i += size - overlap` with a guard that `overlap < size`.

## Tier 2 — Scalability / concurrency (the ones that fail under load)

12. **[MUST-CATCH] `retrieve` re-embeds every chunk on every query.** `for _id, text in rows: vec = embed([text])[0]` makes one LLM embedding call **per chunk, per request**. This is O(corpus) model calls per query and will not survive any load. Embeddings must be computed **once at ingestion** and stored; at query time you embed only the query and do a nearest-neighbor lookup. **This is the headline issue.**
13. **No vector index; Python-side scoring over all rows.** Even with stored vectors, scanning every row and sorting in Python does not scale. Use an ANN index (the vector store returns top-k).
14. **`embed` is N+1 and unbatched.** One HTTP call per item in a loop. Use the batch embeddings endpoint (a single call for many inputs).
15. **[MUST-CATCH] Synchronous `requests` inside `async` functions.** `retrieve` and `answer` are `async` but call blocking `requests.post`, which stalls the event loop and kills concurrency. Use an async client (httpx/aiohttp) and `asyncio.gather` where work is parallel.
16. **No timeouts, no retries/backoff** on any API call. A hung provider hangs the request forever; a 429 just fails. Add timeouts, exponential backoff with jitter, a max attempt count, and honor `Retry-After`.
17. **Query is re-embedded every call** (`embed([query])`) even on cache misses with identical queries — cache the query embedding too.
18. **Unbounded in-memory `CACHE`.** No size cap, TTL, or eviction — it grows forever (memory leak) and is not shared across workers. Use a bounded LRU with TTL (or a shared cache like Redis).
19. **[MUST-CATCH] Unbounded agent loop.** `Agent.run` is `while True` with no max-iteration cap, no cost/token budget, and a weak termination condition (`"DONE" in str(result)`). Add a step limit, an explicit stop condition, and a budget with a circuit breaker.

## Tier 3 — Structure / quality

20. **[MUST-CATCH] `answer` is a god function.** It does retrieval, prompt building, logging, the HTTP call, and parsing in one place. Split into `retrieve` / `build_messages` / `call_llm` / `parse_answer`, each independently testable. (This is the "structure" half of their grading.)
21. **Mutable default arguments.** `Agent.__init__(self, tools=[], history=[])` shares the same list objects across every instance. Default to `None` and initialize inside.
22. **Magic numbers and config scattered.** `500`, `50`, `top_k=5`, the model string, and the URLs are hardcoded inline. Centralize in a settings object or constants.
23. **`print` instead of logging** throughout; no structured logs, no correlation IDs.
24. **`test_answer_format` asserts nothing.** It calls `chunk(...)` and checks no behavior. Add real assertions, edge cases, and error paths.
25. **Missing type hints and docstrings; thin names** (`embed`, `cosine`, `it`) — minor, mention briefly, do not lead with these.

---

## Scoring bands

- **Strong (clear hire signal).** Caught all four Tier-1 security issues (secret, SQL injection, prompt injection, secret-in-logs), the headline scalability catastrophe (#12 re-embedding every request), the sync-in-async + no-timeout/retry problems, the cross-tenant cache bug, and the unbounded agent loop. Named the god-function structure problem. Refactored at least one function live. Prioritized blocking vs nit and gave a verdict.
- **Pass bar.** Caught most Tier-1 issues, named the major scalability problems (re-embedding and/or N+1, sync-in-async), spotted the god function, refactored one thing, and prioritized.
- **Below bar.** Mostly line-level nits (naming, magic numbers, missing types). Missed SQL injection and/or prompt injection. Missed the re-embedding scalability catastrophe. Proposed no structural refactor.

## What a strong review sounds like (model opening)

> "Before any line-level comments, four blockers. The API key is hardcoded and printed to the logs. The tenant goes straight into a SQL string, so this is injectable. The user query and retrieved docs are concatenated into the system prompt, so it's prompt-injectable, and a poisoned KB doc could hijack it. And the cache is keyed only on the query, so it leaks answers across tenants.
>
> The biggest scalability problem is that `retrieve` re-embeds every chunk on every request — that's O(corpus) model calls per query and won't survive any load; embeddings belong at ingestion behind a vector index. On top of that the service is `async` but uses blocking `requests` with no timeouts or retries, so concurrency and resilience are both broken.
>
> Structurally, `answer` is doing five jobs; I'd split it. Let me walk the rest by tier, then show how I'd rewrite `retrieve` and `answer`."

That ordering — security and the headline scalability issue first, structure next, nits last — is most of the grade.

---

## Refactor models (the "improve" half)

**`retrieve` — embed once, parameterize, index, cache correctly, async:**

```python
async def retrieve(query: str, tenant: str, top_k: int = settings.top_k) -> list[str]:
    key = (tenant, query, top_k)
    if (cached := cache.get(key)) is not None:
        return cached
    qvec = await embed_query(query)                      # embed ONLY the query
    hits = await store.search(tenant=tenant,             # ANN over a prebuilt, tenant-scoped index
                              vector=qvec, top_k=top_k)   # parameterized + scoped in the store
    top = [h.text for h in hits]
    cache.set(key, top)                                  # bounded LRU + TTL, keyed on (tenant, query, top_k)
    return top
```

**`answer` — role separation, structured output, error handling, no god function:**

```python
async def answer(query: str, tenant: str) -> Answer:
    docs = await retrieve(query, tenant)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},               # fixed, trusted
        {"role": "user", "content": build_user_message(query, docs)},  # question + context as data
    ]
    raw = await call_llm(messages, timeout=10, max_retries=3)        # async client, timeout, backoff
    return parse_answer(raw)                                         # schema-validate + repair, explicit failure
```

Where `call_llm` owns the HTTP/timeout/retry concern, `build_user_message` owns prompt assembly, and `parse_answer` owns validation — each testable on its own.
