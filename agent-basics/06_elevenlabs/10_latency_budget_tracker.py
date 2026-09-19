"""Latency budget tracker for a voice-agent pipeline.

ElevenLabs' published targets:
  - end-to-end p50 <800ms, p95 <1500ms (deployed agent)
  - Flash v2.5 TTS first-byte ~75ms

Senior-interview pattern: per-stage budgets (STT/LLM/TTS), per-call spans,
p50/p95 across many calls, alert on threshold breach. Averages lie;
percentiles are what users feel.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self):
        return self.end_ms - self.start_ms


@dataclass
class Call:
    spans: list[Span] = field(default_factory=list)
    started_at_ms: float = 0.0

    @property
    def total_ms(self):
        return max((s.end_ms for s in self.spans), default=self.started_at_ms) - self.started_at_ms


def now_ms():
    # perf_counter is monotonic — never NTP-jumps backwards.
    return time.perf_counter() * 1000.0


def percentile(sorted_data, pct):
    """Linear-interpolation percentile (matches numpy default)."""
    if not sorted_data:
        return 0.0
    if len(sorted_data) == 1:
        return sorted_data[0]
    k = (len(sorted_data) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


class BudgetTracker:
    def __init__(self, budgets_ms, total_budget_ms):
        self.budgets_ms = budgets_ms  # {stage_name: budget_ms}
        self.total_budget_ms = total_budget_ms
        self.calls = []
        self.alerts = []

    @asynccontextmanager
    async def track_call(self):
        """Context manager for ONE request. Yields a `span` helper."""
        call = Call(started_at_ms=now_ms())

        @asynccontextmanager
        async def span(name):
            start = now_ms()
            try:
                yield
            finally:
                call.spans.append(Span(name, start, now_ms()))

        try:
            yield span
        finally:
            self._finalise(call)

    def _finalise(self, call):
        self.calls.append(call)
        for s in call.spans:
            budget = self.budgets_ms.get(s.name)
            if budget is not None and s.duration_ms > budget:
                self.alerts.append(f"stage '{s.name}' overran: {s.duration_ms:.0f}ms > budget {budget:.0f}ms")
        if call.total_ms > self.total_budget_ms:
            self.alerts.append(f"end-to-end overran: {call.total_ms:.0f}ms > {self.total_budget_ms:.0f}ms")

    def stage_p50_p95(self, name):
        durations = sorted(s.duration_ms for c in self.calls for s in c.spans if s.name == name)
        return percentile(durations, 50), percentile(durations, 95), len(durations)


# --- demo + tests -----------------------------------------------------------


async def handle_call(tracker, *, slow_llm=False):
    async with tracker.track_call() as span:
        async with span("stt"):
            await asyncio.sleep(0.12)
        async with span("llm"):
            await asyncio.sleep(0.6 if slow_llm else 0.35)
        async with span("tts_first_byte"):
            await asyncio.sleep(0.075)


def test_percentile():
    data = list(range(10, 101, 10))
    assert abs(percentile(data, 50) - 55.0) < 1e-9
    assert abs(percentile(data, 95) - 95.5) < 1e-9
    print("percentile math OK")


async def main():
    test_percentile()
    tracker = BudgetTracker(
        budgets_ms={"stt": 200, "llm": 400, "tts_first_byte": 100},
        total_budget_ms=800,
    )
    # 8 healthy calls + 2 with a slow LLM.
    await asyncio.gather(*(handle_call(tracker) for _ in range(8)))
    await asyncio.gather(*(handle_call(tracker, slow_llm=True) for _ in range(2)))

    print(f"\ncalls: {len(tracker.calls)}")
    for stage in ("stt", "llm", "tts_first_byte"):
        p50, p95, _n = tracker.stage_p50_p95(stage)
        budget = tracker.budgets_ms[stage]
        ok = "OK" if p95 <= budget else "OVER"
        print(f"  {stage:>16}: p50={p50:5.0f}  p95={p95:5.0f}  budget={budget:4.0f}  [{ok}]")

    totals = sorted(c.total_ms for c in tracker.calls)
    print(f"  {'end-to-end':>16}: p50={percentile(totals, 50):5.0f}  p95={percentile(totals, 95):5.0f}  budget={tracker.total_budget_ms:4.0f}")

    print(f"\nalerts ({len(tracker.alerts)}):")
    for a in tracker.alerts[:5]:
        print(f"  - {a}")


if __name__ == "__main__":
    asyncio.run(main())
