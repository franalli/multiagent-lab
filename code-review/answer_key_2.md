# Answer Key — `review_me_2.py` (variant 2)

**Open only after your 30-minute review.** Same protocol as `how_to_run.md`: 30 minutes, no AI, narrate aloud, review and improve.

This variant deliberately targets the categories the first mock under-covered: **async races, context managers, Python idioms (comprehensions / enumerate), nesting depth, dead code, and idempotency on a write path.** It is a document-ingestion service, so it has a write path (unlike the read-mostly first mock), which is where idempotency and races have a real home.

---

## Tier 1 — Concurrency and correctness (blocking)

1. **[MUST-CATCH] Async race condition (check-then-act).** `ingest` does `if doc_id in SEEN:` then `await asyncio.sleep(0)` then `SEEN.add(doc_id)`. Two concurrent ingests of the same `doc_id` both pass the check before either adds, so the document is processed twice. The `await` between the check and the mutation is exactly where the interleave happens. Fix: serialize the check-and-add under an `asyncio.Lock`, or make the registry an atomic operation.
2. **[MUST-CATCH] Write path is not idempotent.** `/ingest` has no idempotency key, so a client retry (timeout, network blip) re-ingests the same document and creates a second job and a second write. Fix: make `doc_id` (or a client-supplied idempotency key) the dedup key, and return the existing job on a repeat instead of starting a new one. Combine with #1 so the dedup is race-safe.
3. **Weak, collision-prone job id.** `job_id = str(time.time())` collides under concurrency (two requests in the same tick get the same id, overwriting each other in `JOBS`). Fix: `uuid4().hex` or a monotonic counter behind the lock.
4. **Shared mutable state only works in one process.** `JOBS` and `SEEN` are module-level dicts/sets, so the dedup and status break the moment you run more than one worker. Fix: a shared store (Redis/DB) with an atomic upsert; this is the correct home for both the idempotency key and the status.
5. **No input validation; `KeyError`s.** `ingest` reads `doc["id"]` / `doc["text"]`, and `/status` reads `payload["job_id"]` then `JOBS[job_id]`, all unguarded. A missing field or unknown job is a 500. Fix: Pydantic request models and a not-found path.

## Tier 2 — Resources and idioms

6. **[MUST-CATCH] Files opened without a context manager.** `load_config` does `f = open("config.json")` and never closes it; `process` does `f = open(path, "w"); f.write(...); f.close()`, which leaks the handle if `write` raises. Fix: `with open(...) as f:` in both, so the file closes on success and on error.
7. **[MUST-CATCH] `range(len(...))` index juggling.** `normalize_lines` loops `for i in range(len(lines))` twice to strip and filter. This is the textbook comprehension/`enumerate` case. Fix: `return [line.strip() for line in text.split("\n") if line.strip()]`.
8. **Deep nesting that should be guard clauses.** `score_priority` is four `if`s deep. Fix: flatten with early returns and a lookup table (below).
9. **Dead code.** `legacy_chunk` is defined and never called; `MAX_CHARS` is declared and never used. Remove them (or wire them in if intended).
10. **Missing type hints and thin structure.** No type hints on any function; bare dicts passed across boundaries. Add hints and a `@dataclass`/Pydantic model for the document and the job. Minor; mention, do not lead with it.

---

## Scoring bands

- **Strong.** Caught the async race (#1) and tied it to the non-idempotent write path (#2), named the single-process shared-state problem (#4), spotted both missing-`with` cases (#6), refactored `normalize_lines` to a comprehension (#7) and `score_priority` to guard clauses (#8), and flagged the dead code (#9). Led with the race and idempotency, nits last.
- **Pass.** Caught the missing context managers and the `range(len())` idiom, spotted the nesting and dead code, and noticed the write path can double-ingest, even if you did not fully articulate the await-interleave.
- **Below bar.** Treated it as a style review (naming, hints) and missed that two concurrent ingests double-process, and that a retry re-ingests.

## What a strong review sounds like (model opening)

> "The headline is a concurrency and idempotency bug. `ingest` checks `SEEN`, then awaits, then adds, so two concurrent ingests of the same doc both get through and process it twice. And `/ingest` has no idempotency key, so a client retry does the same thing. Both need to be fixed together: dedup on `doc_id` under a lock, and ideally in a shared store since `JOBS` and `SEEN` only work in one worker. After that: both `open` calls leak file handles without a `with`, `normalize_lines` should be a comprehension, `score_priority` should be guard clauses, and `legacy_chunk` and `MAX_CHARS` are dead. Let me show the ingest and normalize refactors."

---

## Refactor models (the "improve" half)

**`normalize_lines` — one comprehension:**

```python
def normalize_lines(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]
```

**`score_priority` — guard clause + lookup table (no nesting):**

```python
PRIORITY = {"high": 10, "medium": 5}

def score_priority(doc: dict | None) -> int:
    meta = (doc or {}).get("meta") or {}
    if "priority" not in meta:
        return 0
    return PRIORITY.get(meta["priority"], 1)
```

**File write — context manager:**

```python
with open(path, "w") as f:
    f.write(text)
```

**`ingest` — race-safe, idempotent, unique id:**

```python
from uuid import uuid4

_lock = asyncio.Lock()
JOB_BY_DOC: dict[str, str] = {}

async def ingest(doc: dict) -> dict:
    doc_id = doc["id"]
    async with _lock:                          # serialize the check-and-register
        if doc_id in JOB_BY_DOC:               # idempotent: same doc -> same job
            return {"status": "duplicate", "job_id": JOB_BY_DOC[doc_id]}
        job_id = uuid4().hex                    # unique, collision-free
        JOBS[job_id] = "running"
        JOB_BY_DOC[doc_id] = job_id
    count = await process(job_id, doc_id, doc["text"])
    return {"job_id": job_id, "lines": count}
```

The in-process lock plus dict makes it correct in **one** worker. For real deployments, move the dedup to a shared store with an atomic upsert keyed on `doc_id` (or a client idempotency key), since multiple workers do not share Python memory. Saying that last point out loud is part of the strong signal.
