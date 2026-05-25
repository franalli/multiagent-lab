"""Audio segment timeline manager (variant 1 from the practical-round prompts).

Operations:
  - insert(start, end, ...) with snap-to-grid
  - find_overlaps()  -> list of overlapping (i, j) pairs
  - find_gaps(start, end)
  - merge_adjacent_same_voice()

Invariant: self.segments is always sorted by start_ms. Every op preserves it.
"""

import bisect
from dataclasses import dataclass


@dataclass(order=True)
class Segment:
    """order=True so segments compare by start_ms (then end_ms) automatically."""

    start_ms: int
    end_ms: int
    audio_url: str
    voice_id: str

    def __post_init__(self):
        if self.end_ms <= self.start_ms:
            raise ValueError(f"end {self.end_ms} must be > start {self.start_ms}")

    def overlaps(self, other):
        # Canonical "two intervals overlap" test.
        return self.start_ms < other.end_ms and other.start_ms < self.end_ms


def snap(t_ms, grid=100):
    return round(t_ms / grid) * grid


class Timeline:
    """Sorted list of segments + standard interval operations.

    The sorted invariant is what makes find_overlaps O(n + overlaps) instead
    of O(n^2), and find_gaps a single linear sweep. Every mutator preserves it.
    """

    def __init__(self, grid_ms=100):
        self.grid_ms = grid_ms
        self.segments = []  # always sorted by start_ms

    def insert(self, start_ms, end_ms, audio_url, voice_id):
        """Snap to the grid, then insert into the right sorted position."""
        seg = Segment(
            snap(start_ms, self.grid_ms),
            snap(end_ms, self.grid_ms),
            audio_url,
            voice_id,
        )
        # bisect.insort = O(log n) to find the slot + O(n) shift. Acceptable
        # for editor scale (1k-10k segments). For 1M+, switch to a balanced
        # tree (e.g. sortedcontainers.SortedList).
        bisect.insort(self.segments, seg)
        return seg

    def delete(self, idx):
        return self.segments.pop(idx)

    def find_overlaps(self):
        """Return all (i, j) index pairs with overlapping windows.

        Early-exit trick: because the list is sorted by start_ms, as soon as
        we see segments[j].start_ms >= segments[i].end_ms, NO further j>j can
        overlap i. That's what makes this O(n + overlaps), not O(n^2).
        """
        out = []
        n = len(self.segments)
        for i in range(n):
            for j in range(i + 1, n):
                if self.segments[j].start_ms >= self.segments[i].end_ms:
                    break  # early exit thanks to sort order
                if self.segments[i].overlaps(self.segments[j]):
                    out.append((i, j))
        return out

    def find_gaps(self, range_start, range_end):
        """Return silent intervals inside [range_start, range_end).

        Single-pass sweep with a `cursor` = "earliest time not yet covered."
        Each segment either advances the cursor (if it covers cursor's
        timestamp) or reveals a gap from cursor to the segment's start.
        """
        gaps = []
        cursor = range_start
        for seg in self.segments:
            # Skip segments entirely outside the query window.
            if seg.end_ms <= range_start or seg.start_ms >= range_end:
                continue
            if seg.start_ms > cursor:
                gaps.append((cursor, min(seg.start_ms, range_end)))
            cursor = max(cursor, seg.end_ms)
            if cursor >= range_end:
                break
        # Tail gap after the last covering segment.
        if cursor < range_end:
            gaps.append((cursor, range_end))
        return gaps

    def merge_adjacent_same_voice(self):
        """Fold touching/overlapping segments that share a voice_id.

        Sort invariant makes this O(n): only check each segment against
        the IMMEDIATELY-previous accepted one. Returns the number of merges
        for diagnostics (0 = nothing changed).
        """
        if not self.segments:
            return 0
        merged = [self.segments[0]]
        merges = 0
        for seg in self.segments[1:]:
            top = merged[-1]
            # Same voice AND (touching OR overlapping) -> fold.
            # `<=` covers both touching (seg.start == top.end) and overlap.
            if seg.voice_id == top.voice_id and seg.start_ms <= top.end_ms:
                top.end_ms = max(top.end_ms, seg.end_ms)
                merges += 1
            else:
                merged.append(seg)
        self.segments = merged
        return merges


# --- tests ------------------------------------------------------------------


def test_snap_and_insert():
    t = Timeline()
    s = t.insert(127, 583, "u1", "v1")
    assert (s.start_ms, s.end_ms) == (100, 600)
    print("snap + insert OK")


def test_overlaps():
    t = Timeline()
    t.insert(0, 1000, "u1", "v1")
    t.insert(500, 1500, "u2", "v2")  # overlaps first
    t.insert(2000, 3000, "u3", "v1")
    t.insert(2500, 2700, "u4", "v2")  # overlaps third
    assert t.find_overlaps() == [(0, 1), (2, 3)]
    print("find_overlaps OK")


def test_gaps():
    t = Timeline()
    t.insert(200, 500, "u1", "v1")
    t.insert(700, 900, "u2", "v1")
    assert t.find_gaps(0, 1000) == [(0, 200), (500, 700), (900, 1000)]
    print("find_gaps OK")


def test_merge_same_voice():
    t = Timeline()
    t.insert(0, 500, "u1", "v1")
    t.insert(500, 1000, "u2", "v1")  # touching, same voice -> merge
    t.insert(1100, 1500, "u3", "v2")  # different voice -> no merge
    t.insert(1400, 1800, "u4", "v2")  # overlap, same voice -> merge
    merges = t.merge_adjacent_same_voice()
    assert merges == 2
    assert [(s.start_ms, s.end_ms, s.voice_id) for s in t.segments] == [
        (0, 1000, "v1"),
        (1100, 1800, "v2"),
    ]
    print(f"merge_adjacent_same_voice OK -> {merges} merges")


def main():
    test_snap_and_insert()
    test_overlaps()
    test_gaps()
    test_merge_same_voice()


if __name__ == "__main__":
    main()
