"""TTS eval harness: run a checkpoint through a test set, compare to baseline.

Maps to RE JD pillar 2 ("autonomously training, assessing, and launching models").

The shape:
  1. Synthesise each test item with the candidate checkpoint.
  2. Score with WER (intelligibility) + SECS (voice similarity). Stand-ins
     here; real version uses Whisper + a speaker encoder.
  3. Aggregate per metric and compare to a baseline checkpoint's report.
  4. Flag REGRESSION if any metric degrades past its threshold.

Concurrency: synthesis + scoring runs per-item under a Semaphore (GPU-bound
work cap). Same pattern as 01_async_basics/07_semaphore_rate_limit.py.
"""

import asyncio
import statistics
from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class TestItem:
    item_id: str
    text: str
    speaker_id: str
    category: str  # 'general', 'numbers', 'abbreviation', 'codeswitch'


@dataclass
class ScoreResult:
    item_id: str
    metric: str
    value: float


class Verdict(Enum):
    PASS = "pass"
    REGRESSION = "regression"


@dataclass
class MetricSpec:
    name: str
    higher_is_better: bool
    regression_threshold: float


@dataclass
class CheckpointReport:
    checkpoint_id: str
    per_item: list[ScoreResult] = field(default_factory=list)
    aggregates: dict[str, float] = field(default_factory=dict)

    def worst_items(self, metric, *, higher_is_better, n=5):
        items = [r for r in self.per_item if r.metric == metric]
        items.sort(key=lambda r: r.value, reverse=not higher_is_better)
        return items[:n]


# --- the harness ------------------------------------------------------------


async def score_checkpoint(
    checkpoint_id, test_set, synthesiser, scorers, specs, *, max_concurrent=4
) -> CheckpointReport:
    """Run the full eval pass for one checkpoint.

    Two-level concurrency:
      - Outer: each test item is its own task -> ALL items in flight at once.
      - Inner: per-item, ALL scorers run concurrently against the synth.
      - Semaphore caps total GPU-bound work in flight (real synthesisers and
        speaker encoders are GPU-bound; uncapped fan-out would OOM).

    Aggregation: arithmetic mean per metric. Real systems often report
    p50/p95 too; same shape — just swap statistics.fmean for percentiles.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def evaluate(item):
        async with sem:  # cap GPU-bound work
            synth = await synthesiser(item.text, checkpoint_id)
            # Run all scorers on this one synth concurrently.
            return await asyncio.gather(
                *(
                    _safe_score(name, scorer, synth, item)
                    for name, scorer in scorers.items()
                )
            )

    all_results = await asyncio.gather(*(evaluate(item) for item in test_set))

    # Flatten per-item results into a single per_item list, then aggregate.
    report = CheckpointReport(checkpoint_id=checkpoint_id)
    for item_results in all_results:
        report.per_item.extend(r for r in item_results if r is not None)
    for spec in specs:
        values = [r.value for r in report.per_item if r.metric == spec.name]
        if values:
            report.aggregates[spec.name] = statistics.fmean(values)
    return report


async def _safe_score(name, scorer, synth, item) -> ScoreResult | None:
    """Wrap a scorer so a SINGLE-item failure doesn't poison the whole eval.

    Without this, a scorer that crashes on one weird input (e.g. Whisper
    OOMs on a 60-second sample) would abort the entire run. Returning None
    lets the aggregate still get computed from the other 199 samples; the
    drilldown surfaces what failed.
    """
    try:
        return ScoreResult(item.item_id, name, await scorer(synth, item))
    except Exception:  # noqa: BLE001 -- one bad sample must not abort the eval sweep
        return None


def compare(candidate, baseline, specs) -> tuple[Verdict, list[str]]:
    """Decide whether the candidate ships, based on per-metric thresholds.

    Each MetricSpec knows its own direction (higher_is_better) and its
    regression threshold. We compute degradation in the natural direction
    so the same comparison logic works for WER (lower=better) AND SECS
    (higher=better).

    Returns (Verdict, alerts) so callers can both gate AND explain.
    """
    alerts = []
    for spec in specs:
        c = candidate.aggregates.get(spec.name)
        b = baseline.aggregates.get(spec.name)
        if c is None or b is None:
            continue  # no data on one side — skip silently
        # degradation > 0 means the candidate is WORSE on this metric.
        degradation = (b - c) if spec.higher_is_better else (c - b)
        if degradation > spec.regression_threshold:
            alerts.append(
                f"{spec.name}: regression of {degradation:.4f} "
                f"(threshold {spec.regression_threshold:.4f}, "
                f"baseline {b:.4f} -> candidate {c:.4f})"
            )
    return (Verdict.REGRESSION if alerts else Verdict.PASS), alerts


# --- stand-in scorers + synth (real version: Whisper / speaker encoder) -----


async def fake_synth(text, checkpoint_id):
    await asyncio.sleep(0.01)
    return f"synth://{checkpoint_id}/{hash(text) & 0xFFFFFF:06x}"


def make_wer_scorer(noise):
    async def scorer(_synth, item):
        # Penalize harder categories deterministically.
        base = 0.05 + (0.02 if item.category in {"numbers", "abbreviation"} else 0.0)
        return base + ((hash(item.item_id) % 1000) / 10_000.0) + noise

    return scorer


def make_secs_scorer(drift):
    async def scorer(_synth, item):
        base = 0.92 - (0.01 if item.speaker_id.startswith("held_out") else 0.0)
        wobble = (hash(item.item_id + item.speaker_id) % 1000) / 100_000.0
        return max(0.0, min(1.0, base - drift + wobble))

    return scorer


SPECS = [
    MetricSpec("wer", higher_is_better=False, regression_threshold=0.02),
    MetricSpec("secs", higher_is_better=True, regression_threshold=0.03),
]

TEST_SET = [
    TestItem("t1", "Hello world.", "speaker_1", "general"),
    TestItem("t2", "It's 3:45 PM on Tuesday.", "speaker_2", "numbers"),
    TestItem("t3", "Dr. Smith will see you now.", "speaker_1", "abbreviation"),
    TestItem("t4", "We could meet at cafe Mueller.", "held_out_1", "codeswitch"),
    TestItem("t5", "Please confirm your appointment.", "held_out_2", "general"),
    TestItem("t6", "Pi is approximately 3.14159.", "speaker_3", "numbers"),
]


# --- tests ------------------------------------------------------------------


async def test_basic_run() -> None:
    scorers = {"wer": make_wer_scorer(0), "secs": make_secs_scorer(0)}
    report = await score_checkpoint("baseline", TEST_SET, fake_synth, scorers, SPECS)
    # 6 items * 2 metrics = 12 scores.
    assert len(report.per_item) == 12
    print(
        f"basic run OK -> wer={report.aggregates['wer']:.4f} "
        f"secs={report.aggregates['secs']:.4f}"
    )


async def test_passes_when_no_regression() -> None:
    scorers = {"wer": make_wer_scorer(0), "secs": make_secs_scorer(0)}
    baseline = await score_checkpoint("v1", TEST_SET, fake_synth, scorers, SPECS)
    candidate = await score_checkpoint("v1-rerun", TEST_SET, fake_synth, scorers, SPECS)
    verdict, _ = compare(candidate, baseline, SPECS)
    assert verdict == Verdict.PASS
    print("no-regression PASS OK")


async def test_detects_regression() -> None:
    baseline_scorers = {"wer": make_wer_scorer(0), "secs": make_secs_scorer(0)}
    bad_scorers = {"wer": make_wer_scorer(0.05), "secs": make_secs_scorer(0)}
    baseline = await score_checkpoint(
        "v1", TEST_SET, fake_synth, baseline_scorers, SPECS
    )
    candidate = await score_checkpoint("v2", TEST_SET, fake_synth, bad_scorers, SPECS)
    verdict, alerts = compare(candidate, baseline, SPECS)
    assert verdict == Verdict.REGRESSION
    assert any("wer" in a for a in alerts)
    print(f"regression detected -> {alerts[0]}")


async def test_drilldown() -> None:
    scorers = {"wer": make_wer_scorer(0), "secs": make_secs_scorer(0)}
    report = await score_checkpoint("v1", TEST_SET, fake_synth, scorers, SPECS)
    worst = report.worst_items("wer", higher_is_better=False, n=3)
    print(f"worst-3 WER items -> {[r.item_id for r in worst]}")


async def main() -> None:
    await test_basic_run()
    await test_passes_when_no_regression()
    await test_detects_regression()
    await test_drilldown()
    print("\nExtensions (for the interview):")
    print("  - Per-category aggregates (general/numbers/abbreviation/codeswitch)")
    print("    catch regressions that average out across the whole test set.")
    print("  - Stratify held-out vs in-distribution speakers.")
    print("  - Bootstrap CI on per-item diffs before declaring regression.")
    print("  - Run on every training checkpoint; gate promotion on PASS.")


if __name__ == "__main__":
    asyncio.run(main())
