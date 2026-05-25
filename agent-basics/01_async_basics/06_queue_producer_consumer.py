"""Bounded asyncio.Queue: backpressure between producers and consumers.

A bounded queue (maxsize=N) gives free backpressure — producers block on
put() when the queue is full, naturally slowing them to consumer speed.
This is the canonical pattern for rate-limited pipelines.
"""

import asyncio
import random

QUEUE_SIZE = 5
N_ITEMS = 20
N_CONSUMERS = 3
SENTINEL = None  # one per consumer to signal shutdown


async def producer(q: asyncio.Queue[int | None]) -> None:
    for i in range(N_ITEMS):
        await asyncio.sleep(random.uniform(0, 0.05))
        await q.put(i)
        print(f"produced {i} (qsize={q.qsize()})")
    # Tell every consumer to stop.
    for _ in range(N_CONSUMERS):
        await q.put(SENTINEL)


async def consumer(name: str, q: asyncio.Queue[int | None]) -> int:
    processed = 0
    while True:
        item = await q.get()
        if item is SENTINEL:
            q.task_done()
            print(f"{name} stopping, processed {processed}")
            return processed
        await asyncio.sleep(random.uniform(0.05, 0.15))
        print(f"  {name} consumed {item}")
        processed += 1
        q.task_done()


async def main() -> None:
    q: asyncio.Queue[int | None] = asyncio.Queue(maxsize=QUEUE_SIZE)
    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer(q))
        for i in range(N_CONSUMERS):
            tg.create_task(consumer(f"c{i}", q))


if __name__ == "__main__":
    asyncio.run(main())
