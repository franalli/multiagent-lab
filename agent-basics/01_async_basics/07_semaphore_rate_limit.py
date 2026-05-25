"""Bounded concurrency with asyncio.Semaphore.

Use case: you need to fire many requests but the upstream API has a
"max N in-flight" limit. A Semaphore is the cleanest way to express
"acquire one of N slots, do work, release." Tasks queue automatically
on the semaphore — no manual scheduling.

Contrast with asyncio.Queue: a queue moves *data* between producers
and consumers. A semaphore gates *concurrency* — same code path,
N slots, callers serialize on acquisition.
"""

import asyncio
import time


class RateLimitedClient:
    """Mock API client allowing only `max_concurrent` requests in-flight."""

    def __init__(self, max_concurrent: int) -> None:
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.in_flight = 0
        self.peak = 0

    async def get(self, url: str) -> str:
        async with self.semaphore:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            try:
                await asyncio.sleep(0.2)  # simulate network
                return f"ok {url}"
            finally:
                self.in_flight -= 1


async def main() -> None:
    client = RateLimitedClient(max_concurrent=5)
    urls = [f"https://example.com/{i}" for i in range(30)]

    t0 = time.perf_counter()
    results = await asyncio.gather(*(client.get(u) for u in urls))
    elapsed = time.perf_counter() - t0

    print(f"fetched {len(results)} urls in {elapsed:.2f}s")
    print(f"peak in-flight: {client.peak} (should be == max_concurrent=5)")
    # 30 urls / 5 slots * 0.2s ≈ 1.2s — the semaphore serialises beyond 5.


if __name__ == "__main__":
    asyncio.run(main())
