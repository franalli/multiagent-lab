"""When asyncio CAN'T help: CPU-bound work.

The GIL serializes pure-Python bytecode. `asyncio.gather()` on CPU-bound
coroutines won't speed anything up — they all run on one thread/core.

Solutions:
  - ProcessPoolExecutor: spawn worker processes (true parallelism, OS-scheduled).
  - ThreadPoolExecutor:  only helps if the CPU work releases the GIL
                         (numpy, hashlib, native extensions, file/socket I/O).

`loop.run_in_executor()` bridges sync work into an async program without
blocking the event loop — critical for staying responsive.
"""

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor


def fib(n: int) -> int:
    """Deliberately naive — CPU-bound, no I/O, no GIL release."""
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


async def fib_async_wrapper(n: int) -> int:
    """Wrapping a sync function in `async def` does NOT make it concurrent."""
    return fib(n)


async def run_in_event_loop_only() -> None:
    """All work runs on the single event loop thread — no parallelism."""
    t0 = time.perf_counter()
    results = await asyncio.gather(*(fib_async_wrapper(35) for _ in range(4)))
    print(f"asyncio.gather on CPU work: {time.perf_counter() - t0:.2f}s, {results}")


async def run_in_process_pool() -> None:
    """Offload to worker processes — true parallelism across cores."""
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as pool:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(loop.run_in_executor(pool, fib, 35) for _ in range(4)),
        )
        print(f"ProcessPoolExecutor:        {time.perf_counter() - t0:.2f}s, {results}")


async def main() -> None:
    await run_in_event_loop_only()
    await run_in_process_pool()


if __name__ == "__main__":
    asyncio.run(main())
