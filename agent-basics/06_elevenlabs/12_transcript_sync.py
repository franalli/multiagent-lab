"""Reported ElevenLabs problem: sync audio playback with transcript highlighting.

  "Design and implement logic to synchronize audio playback timing
   with transcript highlighting." -- Exponent

The non-trivial part:
  - Playback fires position updates at ~30 Hz; binary search per tick
    is O(log N), naive scan is O(N).
  - User can scrub (jump anywhere). Cache-the-last-index trick keeps
    sequential playback at O(1), falls back to binary search on scrubs.
  - Silence between segments -> active segment is None, not the last one.
  - Pick half-open intervals [start, end) so adjacent segments don't both
    claim the boundary timestamp.
"""

import bisect
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start_ms: int
    end_ms: int
    text: str

    def contains(self, t_ms):
        # Half-open: t == start_ms is IN, t == end_ms is OUT.
        return self.start_ms <= t_ms < self.end_ms


class TranscriptSync:
    """Maps playback timestamp -> active segment index (or None for silence).

    Performance properties (this is what the interviewer is probing):
      - Cold lookup (user scrubbed):  O(log N) via binary search.
      - Hot path (sequential playback at 30 Hz): O(1) via the last-index cache.
        ~99% of lookups during normal playback hit the cache because the
        playhead has only moved ~33ms since the last call.
      - Memory: O(N) for the parallel starts array; O(1) for cache state.
    """

    def __init__(self, segments):
        # Sort defensively. Real systems usually guarantee sorted input,
        # but the cost of sorting once at construction is trivial vs.
        # the cost of debugging an out-of-order edge case.
        self.segments = sorted(segments, key=lambda s: s.start_ms)
        # Pre-extract starts into a flat list so bisect operates on a
        # homogeneous int array (faster + simpler than passing a key= func).
        self.starts = [s.start_ms for s in self.segments]
        self.last_idx = -1  # -1 sentinel = no prior lookup

    def find_active(self, t_ms):
        """Return index of the segment containing t_ms, or None for silence.

        Three fast paths cover ~99% of sequential-playback ticks:
          1. Still in the cached segment.
          2. Advanced into the immediately-next segment.
          3. In the silent gap right after the cached segment.
        Anything else falls through to binary search.
        """
        if not self.segments:
            return None

        # Fast paths — only meaningful if we have a cached index.
        if self.last_idx >= 0:
            seg = self.segments[self.last_idx]
            if seg.contains(t_ms):
                return self.last_idx  # path 1
            nxt = self.last_idx + 1
            if nxt < len(self.segments) and self.segments[nxt].contains(t_ms):
                self.last_idx = nxt
                return nxt  # path 2
            # In the silent gap right after the cached segment?
            if t_ms >= seg.end_ms and (nxt >= len(self.segments) or t_ms < self.segments[nxt].start_ms):
                return None  # path 3

        # Slow path: user scrubbed. Binary search by start_ms.
        # bisect_right(arr, x) returns the FIRST i with arr[i] > x; so the
        # last segment that COULD contain t_ms is at index pos-1.
        pos = bisect.bisect_right(self.starts, t_ms)
        if pos == 0:
            return None  # before any segment started
        candidate = pos - 1
        self.last_idx = candidate  # anchor cache for the next tick
        # candidate.start_ms <= t_ms, but it may have already ended (gap).
        return candidate if self.segments[candidate].contains(t_ms) else None


# --- tests ------------------------------------------------------------------


def make_sync():
    return TranscriptSync(
        [
            Segment(0, 500, "Hello"),
            Segment(500, 1000, "world"),
            # 200ms gap
            Segment(1200, 1800, "how"),
            Segment(1800, 2400, "are"),
            Segment(2400, 3000, "you"),
        ]
    )


def test_inside_segment():
    s = make_sync()
    assert s.find_active(0) == 0
    assert s.find_active(250) == 0
    assert s.find_active(499) == 0
    print("inside segment OK")


def test_half_open_boundary():
    s = make_sync()
    # At t == 500: seg 0 ends (exclusive), seg 1 starts.
    assert s.find_active(500) == 1
    print("half-open boundary OK")


def test_silence_gap():
    s = make_sync()
    assert s.find_active(1000) is None
    assert s.find_active(1100) is None
    assert s.find_active(1200) == 2
    print("silence gap returns None OK")


def test_before_and_after():
    s = make_sync()
    s.last_idx = -1  # cold lookup
    assert s.find_active(-1) is None
    s.last_idx = -1
    assert s.find_active(3000) is None
    print("before/after returns None OK")


def test_sequential_30hz():
    s = make_sync()
    transitions, last = 0, None
    for t in range(0, 3000, 33):
        idx = s.find_active(t)
        if idx != last:
            transitions += 1
            last = idx
    # seg0 -> seg1 -> None -> seg2 -> seg3 -> seg4 = 6 transitions.
    assert transitions == 6
    print(f"sequential 30Hz playback OK -> {transitions} transitions")


def test_scrub():
    s = make_sync()
    assert s.find_active(2500) == 4  # last segment
    assert s.find_active(100) == 0  # scrubbed back to first
    assert s.find_active(600) == 1  # forward again
    print("scrub OK")


def main():
    test_inside_segment()
    test_half_open_boundary()
    test_silence_gap()
    test_before_and_after()
    test_sequential_30hz()
    test_scrub()


if __name__ == "__main__":
    main()
