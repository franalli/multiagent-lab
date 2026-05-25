"""Cancellation, timeouts, and shielding.

Cancellation in asyncio is cooperative: a CancelledError is raised at the
next `await` point. Code can intercept it (catch + cleanup + re-raise) but
should NOT suppress it. Use `asyncio.shield()` to protect critical sections.
"""

import asyncio


async def cleanup() -> None:
    print("  running cleanup...")
    # Important: shield this so an outer cancel doesn't truncate cleanup.
    await asyncio.shield(asyncio.sleep(0.3))
    print("  cleanup done")


async def cancellable_work() -> str:
    try:
        await asyncio.sleep(5)  # cancellation will land here
        return "completed"
    except asyncio.CancelledError:
        print("  caught cancellation, cleaning up")
        await cleanup()
        raise  # always re-raise CancelledError


async def with_timeout() -> None:
    print("--- asyncio.timeout (3.11+) ---")
    try:
        async with asyncio.timeout(1.0):
            result = await cancellable_work()
            print(f"got {result}")
    except TimeoutError:
        print("timed out (TimeoutError, not CancelledError)\n")


async def manual_cancel() -> None:
    print("--- manual task.cancel() ---")
    task = asyncio.create_task(cancellable_work())
    await asyncio.sleep(0.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("task confirmed cancelled\n")


async def main() -> None:
    await with_timeout()
    await manual_cancel()


if __name__ == "__main__":
    asyncio.run(main())
