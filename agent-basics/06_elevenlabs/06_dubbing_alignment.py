"""Practical-round variant 4: dubbing alignment.

Problem: original video has segments (start, end, text). The dubbing
pipeline produces translated segments with synthesised durations. Align
the translated segments to original timing under constraints:

  - Can speed up / slow down each translated segment by up to MAX_RATE_RATIO.
  - Translated segments cannot overlap.
  - Silence may be inserted between segments.
  - Goal: produce output where each translated segment's mid-point is
    as close as possible to the original's mid-point.

Greedy algorithm:
  1. For each translated segment, target its mid-point to match the
     original's mid-point.
  2. If that would overlap the previous translated segment, push it
     right (introduce silence after previous, accept mid-point drift).
  3. If the translated duration is too long, compress within rate limit.
  4. If still doesn't fit, push the previous segment's end-time earlier
     by speeding it up (within rate limit) — this is the bidirectional
     adjustment a senior candidate flags.

Returns (placed_segments, total_drift_ms, feasible: bool).

A real production system would use DP for the global optimum; greedy
is the right pick for a 30-min interview answer with a "discuss
DP extension" remark.
"""

from dataclasses import dataclass

MAX_RATE_RATIO = 1.2


@dataclass
class SourceSegment:
    start_ms: int
    end_ms: int
    text: str


@dataclass
class TranslatedSegment:
    text: str
    synthesised_duration_ms: int  # the raw TTS duration at neutral speed


@dataclass
class PlacedSegment:
    start_ms: int
    end_ms: int
    text: str
    speed_factor: float  # 1.0 = neutral; >1 = sped up; <1 = slowed down


def _clamp_speed(desired_duration_ms: int, raw_duration_ms: int) -> tuple[int, float]:
    """Pick a speed factor and achievable duration for one translated segment.

    Inputs:
      - desired_duration_ms: how long we'd LIKE this segment to be (the source window).
      - raw_duration_ms: how long the synth's voice naturally takes for this text.

    Output:
      - (achievable_duration_ms, speed_factor)
      - speed_factor > 1  -> sped up (talking faster). Acceptable up to MAX_RATE_RATIO.
      - speed_factor < 1  -> slowed down. Acceptable down to 1/MAX_RATE_RATIO.

    The clamp is the *physical constraint*: humans tolerate ~20% time-stretch
    before it sounds unnatural ("alvin and the chipmunks" territory).
    """
    # Edge case: zero or negative target window (shouldn't happen in valid input,
    # but degrade gracefully — pick the max-speed achievable duration).
    if desired_duration_ms <= 0:
        speed = MAX_RATE_RATIO
        return max(1, int(raw_duration_ms / speed)), speed

    # Raw arithmetic: if speed_factor = X, the segment plays in raw/X ms.
    # To fit `desired_duration_ms`, we'd need speed = raw/desired.
    speed = raw_duration_ms / desired_duration_ms
    # Clamp to the human-tolerable range. After clamping, achievable
    # may not equal desired — that's the "we couldn't fit" signal.
    speed = max(1.0 / MAX_RATE_RATIO, min(MAX_RATE_RATIO, speed))
    achievable = int(raw_duration_ms / speed)
    return achievable, speed


def align_dub(
    source: list[SourceSegment],
    translated: list[TranslatedSegment],
) -> tuple[list[PlacedSegment], int, bool]:
    """Greedy single-pass alignment of translated segments to source timing.

    Returns:
      placed: list of PlacedSegment in time order
      total_drift: cumulative ms that we had to push segments LATER than ideal
                   (proxy for "how much did we deviate from source timing?")
      feasible: False if any segment had to be compressed past the rate cap
                in a way that suggests the language pair just won't fit
    """
    if len(source) != len(translated):
        raise ValueError("source and translated must have the same length")

    placed: list[PlacedSegment] = []
    total_drift = 0
    feasible = True
    # `prev_end` tracks the latest "occupied" timestamp — every new segment
    # must start at-or-after this to maintain the no-overlap invariant.
    prev_end = 0

    for src, tr in zip(source, translated, strict=True):
        original_window = src.end_ms - src.start_ms
        # Ideal placement: center the translated segment on the SAME midpoint
        # as the source segment. Mid-point alignment preserves perceptual
        # synchrony better than start-aligned (lips and audio drift less).
        ideal_mid = (src.start_ms + src.end_ms) // 2

        # ---- Step 1: figure out duration + speed ----
        # We aim for the original window, with the speed clamp picking up
        # whatever stretch/compression is needed AND allowed.
        duration, speed = _clamp_speed(original_window, tr.synthesised_duration_ms)

        # ---- Step 2: place around the ideal mid-point ----
        target_start = ideal_mid - duration // 2
        target_end = target_start + duration

        # ---- Step 3: enforce no-overlap with the previous placed segment ----
        # If our desired start would collide, push us to right after `prev_end`.
        # The shift accumulates into `total_drift` — that's our quality signal.
        if target_start < prev_end:
            shift = prev_end - target_start
            target_start = prev_end
            target_end = target_start + duration
            total_drift += shift

        # ---- Step 4: feasibility check ----
        # Heuristic: if we're at MAX_RATE_RATIO (i.e. we wanted to compress more
        # but the clamp stopped us) AND the duration is still more than 2x the
        # original window, we genuinely cannot fit. A real bidirectional pass
        # would try to speed up EARLIER segments to make room here; the greedy
        # single-pass version just flags `feasible = False` and lets the caller
        # decide (re-translate, allow overlap, skip the segment).
        if speed >= MAX_RATE_RATIO and duration > original_window * 2:
            feasible = False

        placed.append(PlacedSegment(target_start, target_end, tr.text, speed))
        prev_end = target_end

    return placed, total_drift, feasible


# --- tests ---


def test_no_compression_needed() -> None:
    src = [
        SourceSegment(0, 2000, "Hello world"),
        SourceSegment(2000, 4000, "How are you"),
    ]
    tr = [
        TranslatedSegment("Hola mundo", 2000),
        TranslatedSegment("Cómo estás", 2000),
    ]
    placed, drift, ok = align_dub(src, tr)
    assert ok
    assert drift == 0
    assert all(p.speed_factor == 1.0 for p in placed)
    print(f"no-compression OK -> placed {[(p.start_ms, p.end_ms) for p in placed]}")


def test_compression_within_limit() -> None:
    src = [SourceSegment(0, 1000, "Hi")]
    # Translated is 1100ms raw -> needs 1.1x speed up (within 1.2x limit).
    tr = [TranslatedSegment("Hola mundo amigo", 1100)]
    placed, _, ok = align_dub(src, tr)
    assert ok
    assert 1.05 < placed[0].speed_factor <= MAX_RATE_RATIO
    print(f"compression OK -> speed={placed[0].speed_factor:.3f}")


def test_overlap_pushes_with_silence() -> None:
    src = [
        SourceSegment(0, 1000, "A"),
        SourceSegment(1000, 2000, "B"),
    ]
    # First translation is 1500ms raw -> compresses to ~1.2x = ~1250ms, exceeds window.
    # After clamp speed to MAX, achievable = 1500/1.2 = 1250ms. Placed end > 1000.
    # Second segment must wait, introducing drift.
    tr = [
        TranslatedSegment("First long translation", 1500),
        TranslatedSegment("Second", 800),
    ]
    placed, drift, _ = align_dub(src, tr)
    # Second segment cannot start before first ends.
    assert placed[1].start_ms >= placed[0].end_ms
    # First segment was compressed to max rate.
    assert placed[0].speed_factor == MAX_RATE_RATIO
    print(
        f"overlap-push OK -> drift={drift}ms, "
        f"segments={[(p.start_ms, p.end_ms) for p in placed]}"
    )


def main() -> None:
    test_no_compression_needed()
    test_compression_within_limit()
    test_overlap_pushes_with_silence()


if __name__ == "__main__":
    main()
