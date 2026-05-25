"""Stream results as they finish with asyncio.as_completed.

gather() waits for the slowest task. as_completed() yields each task
as soon as it's done — ideal for streaming partial output to a user
(or to another sub-agent) while remaining tasks still run.

Common in agent systems: sub-agent dispatch where you'd like to surface
early answers before the slow ones finish.
"""

import asyncio
import random
import time


async def sub_agent(name: str) -> str:
    delay = random.uniform(0.2, 1.5)
    await asyncio.sleep(delay)
    return f"{name} done after {delay:.2f}s"


async def with_gather() -> None:
    print("--- gather: print only after ALL finish ---")
    t0 = time.perf_counter()
    results = await asyncio.gather(*(sub_agent(f"agent-{i}") for i in range(5)))
    for r in results:
        print(f"  {r}")
    print(f"total: {time.perf_counter() - t0:.2f}s\n")


async def with_as_completed() -> None:
    print("--- as_completed: print each as it finishes ---")
    t0 = time.perf_counter()
    coros = [sub_agent(f"agent-{i}") for i in range(5)]
    for fut in asyncio.as_completed(coros):
        result = await fut
        print(f"  [{time.perf_counter() - t0:.2f}s] {result}")
    print(f"total: {time.perf_counter() - t0:.2f}s")


async def main() -> None:
    random.seed(42)
    await with_gather()
    random.seed(42)
    await with_as_completed()


if __name__ == "__main__":
    asyncio.run(main())
