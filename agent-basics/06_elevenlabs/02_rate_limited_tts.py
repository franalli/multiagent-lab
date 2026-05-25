"""Token-bucket rate limiter in front of an async TTS call.

Extension of 01_streaming_chunker.py: "TTS is rate-limited to 10/sec. Add backpressure."

Mental model:
  - Bucket holds up to `capacity` tokens; tokens regen at `rate`/sec.
  - Each TTS call consumes one token.
  - When dry, callers queue. The drain loop refills lazily and dispatches.

Why this shape (Event + timed wait) beats `while True: sleep`:
  - Bursts: Event wakes the drainer immediately when work arrives.
  - Steady-state: timed wait sleeps EXACTLY long enough for one token to refill.
"""

import asyncio
import time
from collections import deque


class RateLimitedTTS:
    """Token bucket in front of `tts_call`.

    Bucket math:
      tokens starts at `capacity` (allow initial burst).
      tokens regenerate continuously at `rate` per second.
      tokens are clamped to `capacity` (no banking — that's why it's
        called a LEAKY bucket; excess just disappears).
      Each call consumes 1 token.

    Concurrency shape:
      submit() is SYNC because callers (the chunker) are sync.
      The drainer is a background asyncio.Task; submit() pings it via
      `wake` (asyncio.Event), and the drainer either dispatches now or
      waits exactly long enough for the next token to refill.
    """

    def __init__(self, tts_call, *, rate=10.0, capacity=10):
        self.tts_call = tts_call
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)  # start full -> first `capacity` go free
        self.last_refill = time.monotonic()
        self.queue: deque[str] = deque()
        self.wake = asyncio.Event()  # submit() pings, drainer waits
        self.drainer = None
        self.stopped = False

    def start(self):
        """Spawn the drain loop as a background task. Call once at startup."""
        self.drainer = asyncio.create_task(self._drain_loop())

    async def stop(self):
        """Signal shutdown, then await the drainer so queued chunks finish."""
        self.stopped = True
        self.wake.set()
        if self.drainer:
            await self.drainer

    def submit(self, chunk):
        """Sync entry point — the chunker calls this from add_token().

        We don't make this async because the chunker IS sync (token arrivals
        are synchronous events). Pinging the Event un-blocks the drainer
        immediately if it was waiting; if it's already running, it's a no-op.
        """
        self.queue.append(chunk)
        self.wake.set()

    def _refill(self):
        """Top up tokens proportional to elapsed wall-time. Idempotent."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        # Clamp to capacity — can't bank arbitrary burst (the "leaky" part).
        self.tokens = min(float(self.capacity), self.tokens + elapsed * self.rate)
        self.last_refill = now

    async def _drain_loop(self):
        """Owner of the queue. The ONLY place tokens are consumed.

        Two waits, both crucial:
          - When queue is EMPTY: block on the Event (no busy-poll).
          - When queue is NON-EMPTY but bucket dry: wait exactly long
            enough for one token. This keeps the rate accurate without
            oversleeping.
        """
        # Exit when shutdown requested AND nothing pending (don't drop work).
        while not (self.stopped and not self.queue):
            self._refill()

            # Dispatch as many chunks as we have tokens for.
            while self.queue and self.tokens >= 1:
                chunk = self.queue.popleft()
                self.tokens -= 1
                # If tts_call raises, the drainer raises — intentional. A
                # production version would wrap with retries + a DLQ.
                await self.tts_call(chunk)

            # No work pending -> block until submit() pings us.
            if not self.queue:
                self.wake.clear()
                await self.wake.wait()
                continue

            # Work pending but no tokens -> wait for refill OR new submission.
            wait = (1.0 - self.tokens) / self.rate
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=max(0.0, wait))
            except TimeoutError:
                pass  # token refilled; loop and dispatch
            self.wake.clear()


# --- demo: 30 chunks submitted at burst; limiter caps to 10/sec ------------


async def main():
    arrivals = []
    t0 = time.monotonic()

    async def fake_tts(_chunk):
        arrivals.append(time.monotonic() - t0)

    limiter = RateLimitedTTS(fake_tts, rate=10.0, capacity=5)
    limiter.start()
    for i in range(30):
        limiter.submit(f"chunk-{i}")

    # Poll until drained, then stop. (Could wrap queue in an Event for prod.)
    while limiter.queue:  # noqa: ASYNC110
        await asyncio.sleep(0.05)
    await limiter.stop()

    print(f"30 chunks in {arrivals[-1]:.2f}s (5 burst, then ~10/sec -> ~2.5s)")
    print(f"first 6 arrivals: {[round(a, 2) for a in arrivals[:6]]}")
    print(f"last 4 arrivals:  {[round(a, 2) for a in arrivals[-4:]]}")


if __name__ == "__main__":
    asyncio.run(main())
