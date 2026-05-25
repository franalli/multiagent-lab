"""Exponential backoff with jitter — hand-rolled.

Why jitter? Without it, if 100 concurrent agents all hit the same rate
limit at t=0, they all back off identically (1s, 2s, 4s, 8s...) and
stampede the API together when they retry. Jitter randomises the
distribution so retries spread out.

`tenacity` does this for you, but hand-rolling is the version
interviewers ask for — and it's only ~15 lines.
"""

import asyncio
import random


class TransientError(Exception):
    """Errors worth retrying (network blip, 5xx, timeout)."""


async def call_with_retry(operation, *, max_attempts=5, base_delay=1.0, max_delay=30.0):
    """Retry an async callable with capped exponential backoff + jitter."""
    for attempt in range(max_attempts):
        try:
            return await operation()
        except TransientError as e:
            if attempt == max_attempts - 1:
                raise
            base = min(base_delay * (2**attempt), max_delay)
            # Full jitter: pick a delay in [0, base]. Spreads retries widely.
            delay = random.uniform(0, base)
            print(f"  attempt {attempt + 1} failed ({e}); sleeping {delay:.2f}s")
            await asyncio.sleep(delay)


async def main() -> None:
    attempts = {"count": 0}

    async def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 4:
            raise TransientError(f"503 service unavailable (try {attempts['count']})")
        return f"success on attempt {attempts['count']}"

    random.seed(7)
    result = await call_with_retry(flaky, max_attempts=6, base_delay=0.1, max_delay=2.0)
    print(f"final: {result}")


if __name__ == "__main__":
    asyncio.run(main())
