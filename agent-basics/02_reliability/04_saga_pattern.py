"""Saga pattern: multi-step workflow with compensating rollback.

When a multi-step workflow has side effects and step N fails, steps
0..N-1 already happened. A saga pairs each forward action with a
compensating action; on failure, completed steps roll back in reverse.

Note: compensations are best-effort and can themselves fail. Log loudly,
alert humans, and accept that perfect transactional semantics across
external systems is impossible without distributed-txn machinery (2PC,
which most APIs don't speak).

This pattern lives at the layer above retries: retries can't help if
the upstream succeeded but your *next* step fails — you need rollback.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class SagaStep:
    name: str
    action: Callable[[], Awaitable[Any]]
    compensate: Callable[[Any], Awaitable[None]]


async def run_saga(steps: list[SagaStep]) -> list[Any]:
    """Run forward; on first failure, compensate completed steps in reverse."""
    completed: list[tuple[SagaStep, Any]] = []
    try:
        for step in steps:
            print(f"  -> {step.name}")
            result = await step.action()
            completed.append((step, result))
        return [r for _, r in completed]
    except Exception as e:
        print(f"  FAILED at {steps[len(completed)].name}: {e}")
        print("  rolling back...")
        for step, result in reversed(completed):
            try:
                await step.compensate(result)
                print(f"  <- compensated {step.name}")
            except Exception as comp_err:
                # Compensation can fail. Log and continue — don't lose the original error.
                print(f"  !! compensation for {step.name} failed: {comp_err}")
        raise


# --- fake services with controllable failures ---


async def book_flight() -> dict:
    await asyncio.sleep(0.1)
    return {"booking_id": "FL-001"}


async def cancel_flight(result: dict) -> None:
    await asyncio.sleep(0.05)


async def book_hotel() -> dict:
    await asyncio.sleep(0.1)
    return {"booking_id": "HT-001"}


async def cancel_hotel(result: dict) -> None:
    await asyncio.sleep(0.05)


async def book_car_fails() -> dict:
    await asyncio.sleep(0.1)
    raise RuntimeError("no cars available")


async def main() -> None:
    steps = [
        SagaStep("book_flight", book_flight, cancel_flight),
        SagaStep("book_hotel", book_hotel, cancel_hotel),
        SagaStep("book_car", book_car_fails, lambda _: asyncio.sleep(0)),
    ]
    try:
        await run_saga(steps)
    except RuntimeError as e:
        print(f"\nsaga aborted: {e}")


if __name__ == "__main__":
    asyncio.run(main())
