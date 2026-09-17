"""Circuit breaker: stop hammering a tool that's clearly broken.

Three states:
  CLOSED    — normal, calls pass through
  OPEN      — too many failures; reject calls immediately for `recovery_timeout`
  HALF_OPEN — recovery window elapsed; let one call through to probe
              (success -> CLOSED; failure -> OPEN again)

Pair with retries: retries handle *transient* failures within one call;
the breaker handles *persistent* failures across many calls. Without
the breaker, retries amplify load on a struggling upstream.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpen(Exception):
    """Raised when the breaker is OPEN and rejecting calls."""


class CircuitBreaker:
    def __init__(
        self, *, failure_threshold: int = 3, recovery_timeout: float = 2.0
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at: datetime | None = None

    async def call(self, operation):
        if self.state == CircuitState.OPEN:
            assert self.opened_at is not None
            if datetime.now(UTC) - self.opened_at > timedelta(
                seconds=self.recovery_timeout
            ):
                print("  breaker: OPEN -> HALF_OPEN (probing)")
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpen("circuit breaker open")

        try:
            result = await operation()
        except Exception:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = datetime.now(UTC)
                print(f"  breaker: tripped OPEN after {self.failure_count} failures")
            raise

        # Success path
        if self.state == CircuitState.HALF_OPEN:
            print("  breaker: HALF_OPEN -> CLOSED (recovered)")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        return result


async def main() -> None:
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    failing = True

    async def upstream() -> str:
        if failing:
            raise RuntimeError("upstream down")
        return "ok"

    # Trip the breaker
    for i in range(5):
        try:
            await breaker.call(upstream)
        except RuntimeError as e:
            print(f"call {i}: error -> {e}")
        except CircuitOpen as e:
            print(f"call {i}: rejected -> {e}")

    # Wait through recovery_timeout; upstream now healthy
    await asyncio.sleep(1.1)
    failing = False
    result = await breaker.call(upstream)
    print(f"after recovery: {result}")


if __name__ == "__main__":
    asyncio.run(main())
