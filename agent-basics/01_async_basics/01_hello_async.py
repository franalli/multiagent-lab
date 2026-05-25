"""Coroutines, the event loop, and why `await` doesn't block."""

import asyncio
import time


async def slow_task(name: str, seconds: float) -> str:
    print(f"[{time.strftime('%X')}] {name} starting (will sleep {seconds}s)")
    await asyncio.sleep(seconds)
    print(f"[{time.strftime('%X')}] {name} done")
    return f"{name}-result"


async def main() -> None:
    # Sequential: each await blocks this coroutine until done.
    # Total wall time ≈ 1 + 2 = 3s.
    t0 = time.perf_counter()
    a = await slow_task("seq-A", 1.0)
    b = await slow_task("seq-B", 2.0)
    print(f"sequential took {time.perf_counter() - t0:.2f}s -> {a}, {b}\n")

    # Concurrent: schedule both, then await both. Wall time ≈ max(1, 2) = 2s.
    t0 = time.perf_counter()
    a, b = await asyncio.gather(
        slow_task("par-A", 1.0),
        slow_task("par-B", 2.0),
    )
    print(f"gather took {time.perf_counter() - t0:.2f}s -> {a}, {b}")


if __name__ == "__main__":
    asyncio.run(main())
