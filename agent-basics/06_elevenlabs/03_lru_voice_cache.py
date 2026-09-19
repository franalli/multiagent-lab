"""LRU voice synthesis cache with TTL + memory budget.

Key = (text, voice_id, model_id). Value = synthesized audio bytes.
  - LRU eviction when over entry count or byte budget.
  - TTL expiration (lazy: reaped on access, not by a sweeper).

OrderedDict is the LRU substrate: insertion order = recency order.
move_to_end(k) marks `k` as recently used; popitem(last=False) evicts LRU.
"""

import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class Entry:
    data: bytes
    expires_at: float


class VoiceCache:
    """LRU cache with TTL and a separate byte budget.

    The byte budget matters for audio caches because one big clip can blow
    a per-entry-count limit's underlying memory assumption. Tracking bytes
    incrementally (not recomputing on each evict) is what keeps put/get O(1).
    """

    def __init__(self, *, capacity=1024, ttl_s=3600.0, max_bytes=100 * 1024 * 1024):
        self.capacity = capacity  # max entry COUNT
        self.ttl_s = ttl_s  # absolute TTL applied at put-time
        self.max_bytes = max_bytes  # max total bytes across all entries
        self.store: OrderedDict[tuple, Entry] = OrderedDict()
        self.total_bytes = 0  # incremental — avoids O(n) scans on evict

    def get(self, key):
        """Return cached bytes, or None on miss/expiry. O(1)."""
        entry = self.store.get(key)
        if entry is None:
            return None
        # Lazy expiry — we don't run a sweeper. Cheaper, but stale entries
        # take memory until they're touched. The byte budget acts as backstop.
        if time.monotonic() >= entry.expires_at:
            self._evict(key)
            return None
        # Mark recently used: move to the right end of the OrderedDict.
        self.store.move_to_end(key)
        return entry.data

    def put(self, key, data):
        """Insert or update an entry, then enforce both budgets. O(1) amortized."""
        # Update-in-place: subtract OLD bytes before adding new ones.
        # Forgetting this is the canonical bug in incremental-bytes LRUs.
        if key in self.store:
            self.total_bytes -= len(self.store[key].data)
            del self.store[key]
        self.store[key] = Entry(data, time.monotonic() + self.ttl_s)
        self.total_bytes += len(data)
        self._enforce_budget()

    def _evict(self, key):
        """Remove a specific key (used by TTL expiry). Keeps total_bytes correct."""
        entry = self.store.pop(key, None)
        if entry:
            self.total_bytes -= len(entry.data)

    def _enforce_budget(self):
        """Drop LRU entries until BOTH the count cap AND the byte cap hold."""
        # `or` in the while: either cap can be violated independently.
        while self.store and (len(self.store) > self.capacity or self.total_bytes > self.max_bytes):
            # popitem(last=False) pops from the LEFT = oldest = LRU. O(1).
            _, oldest = self.store.popitem(last=False)
            self.total_bytes -= len(oldest.data)


# --- tests ------------------------------------------------------------------


def test_lru_ordering():
    c = VoiceCache(capacity=3, max_bytes=10**9)
    c.put(("hi", "v1", "m1"), b"a")
    c.put(("hi", "v2", "m1"), b"b")
    c.put(("hi", "v3", "m1"), b"c")
    c.get(("hi", "v1", "m1"))  # touches v1 -> v2 is LRU now
    c.put(("hi", "v4", "m1"), b"d")  # evicts v2
    assert c.get(("hi", "v2", "m1")) is None
    assert c.get(("hi", "v1", "m1")) == b"a"
    print("LRU ordering OK")


def test_byte_budget():
    c = VoiceCache(capacity=100, max_bytes=30)
    c.put(("a", "v", "m"), b"x" * 10)
    c.put(("b", "v", "m"), b"x" * 10)
    c.put(("c", "v", "m"), b"x" * 10)
    c.put(("d", "v", "m"), b"x" * 10)  # pushes to 40 -> evicts 'a'
    assert c.total_bytes <= 30
    assert c.get(("a", "v", "m")) is None
    print(f"byte budget OK -> total_bytes={c.total_bytes}")


def test_ttl_expiry():
    c = VoiceCache(ttl_s=0.05, max_bytes=10**9)
    c.put(("a", "v", "m"), b"data")
    assert c.get(("a", "v", "m")) == b"data"
    time.sleep(0.06)
    assert c.get(("a", "v", "m")) is None  # lazy expiry on access
    print("TTL expiry OK")


def main():
    test_lru_ordering()
    test_byte_budget()
    test_ttl_expiry()


if __name__ == "__main__":
    main()
