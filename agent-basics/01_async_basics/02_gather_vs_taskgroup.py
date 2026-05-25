"""gather() vs TaskGroup (3.11+): error semantics differ.

- gather(..., return_exceptions=False): one failure cancels siblings, others
  may still run to completion in the background. You get the first exception.
- TaskGroup: structured concurrency. One failure cancels all siblings AND the
  group await re-raises an ExceptionGroup. Cleaner cleanup, no orphan tasks.
"""

import asyncio


async def worker(name: str, delay: float, *, fail: bool = False) -> str:
    try:
        await asyncio.sleep(delay)
        if fail:
            raise RuntimeError(f"{name} blew up")
        print(f"  {name} finished after {delay}s")
        return name
    except asyncio.CancelledError:
        print(f"  {name} was cancelled")
        raise


async def with_gather() -> None:
    print("--- gather ---")
    try:
        await asyncio.gather(
            worker("g1", 0.5),
            worker("g2", 1.0, fail=True),
            worker("g3", 1.5),
        )
    except RuntimeError as e:
        print(f"caught: {e}\n")


async def with_taskgroup() -> None:
    print("--- TaskGroup ---")
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(worker("t1", 0.5))
            tg.create_task(worker("t2", 1.0, fail=True))
            tg.create_task(worker("t3", 1.5))
    except* RuntimeError as eg:
        # `except*` unpacks ExceptionGroup (PEP 654).
        for e in eg.exceptions:
            print(f"caught: {e}")


async def main() -> None:
    await with_gather()
    await with_taskgroup()


if __name__ == "__main__":
    asyncio.run(main())
