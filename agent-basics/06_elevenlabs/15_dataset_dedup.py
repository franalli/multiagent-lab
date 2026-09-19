"""TTS dataset quality gate: per-clip QA + dedup + versioning.

Maps to RE JD pillar 1 ("data management system... versioning + data quality").

The gate runs every incoming clip through:
  1. Quality checks (duration, SNR, loudness, char/sec)
  2. Perceptual-hash dedup against the existing corpus
  3. Tag with dataset_version

Real systems would use chromaprint for audio fingerprinting and LSH for
1M+ clip scale; here we use a small SHA-derived hash and brute force.
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class RejectReason(Enum):
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    LOW_SNR = "low_snr"
    LOUDNESS_OUT_OF_BAND = "loudness_out_of_band"
    TEXT_AUDIO_MISMATCH = "text_audio_mismatch"
    DUPLICATE = "duplicate"


@dataclass
class RawClip:
    clip_id: str
    text: str
    duration_s: float
    samples: list[float]
    snr_db: float = 30.0
    loudness_lufs: float = -20.0


@dataclass
class AcceptedClip:
    clip_id: str
    text: str
    duration_s: float
    dataset_version: str
    phash: int


@dataclass
class Rejection:
    clip_id: str
    reason: RejectReason
    detail: str = ""


@dataclass
class IngestResult:
    accepted: list[AcceptedClip] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)


# --- perceptual hash stand-in -----------------------------------------------


def perceptual_hash(samples: list[float], n_buckets: int = 64) -> int:
    """64-bit fingerprint: one bit per time bucket, set if energy > median.

    The crucial property of perceptual hashes (vs. SHA): SIMILAR audio
    produces SIMILAR hashes (small Hamming distance), not unrelated ones.
    That's what makes dedup work — a 256kbps and 320kbps re-encode of the
    same clip should land within a few Hamming bits of each other.

    Real systems use Chromaprint or learned embeddings (CLAP). This
    stand-in just bucks bucket-energy against the median — enough to
    demonstrate the algorithm, not robust to real audio noise.
    """
    if not samples:
        return 0
    bucket = max(1, len(samples) // n_buckets)
    # Per-bucket energy (sum of squared amplitudes).
    energies = [sum(s * s for s in samples[i * bucket : (i + 1) * bucket]) for i in range(n_buckets)]
    # Bit i = 1 iff bucket i has above-median energy. Median split gives
    # ~32 bits set on random input -> max Hamming distance is meaningful.
    median = sorted(energies)[len(energies) // 2]
    bits = 0
    for i, e in enumerate(energies):
        if e > median:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    """Number of bit positions where a and b differ. O(64) here."""
    return (a ^ b).bit_count()


# --- the gate ---------------------------------------------------------------


class DatasetGate:
    """Stateful gate: dedups across batches via a running corpus hash list."""

    def __init__(
        self,
        dataset_version: str,
        *,
        min_dur=0.5,
        max_dur=30.0,
        min_snr=20.0,
        loudness_band=(-30.0, -16.0),
        cps_band=(5.0, 25.0),
        dedup_threshold=4,
    ):
        self.dataset_version = dataset_version
        self.min_dur, self.max_dur = min_dur, max_dur
        self.min_snr = min_snr
        self.loudness_band = loudness_band
        self.cps_band = cps_band
        self.dedup_threshold = dedup_threshold
        self.corpus: list[tuple[str, int]] = []  # (clip_id, phash)

    def _check_quality(self, clip: RawClip) -> RejectReason | None:
        """Return the first failure reason, or None if all checks pass.

        Order matters only insofar as we want to surface the most-actionable
        reason first (length issues are easier for upstream to fix than
        text-audio mismatch which requires re-transcribing).
        """
        if clip.duration_s < self.min_dur:
            return RejectReason.TOO_SHORT
        if clip.duration_s > self.max_dur:
            return RejectReason.TOO_LONG
        if clip.snr_db < self.min_snr:
            return RejectReason.LOW_SNR
        low, high = self.loudness_band
        if not (low <= clip.loudness_lufs <= high):
            return RejectReason.LOUDNESS_OUT_OF_BAND
        # Characters per second sanity-checks the transcript-audio alignment
        # without running ASR. Outside [5, 25] cps usually means a wrong
        # transcript glued to the wrong audio.
        cps = len(clip.text) / max(clip.duration_s, 1e-6)
        if not (self.cps_band[0] <= cps <= self.cps_band[1]):
            return RejectReason.TEXT_AUDIO_MISMATCH
        return None

    def _find_dup(self, phash: int, against: list[tuple[str, int]]) -> str | None:
        """Brute-force nearest-neighbour. O(N) per query.

        At 1M+ clips this becomes the bottleneck — swap for MinHash+LSH or
        a vector index (FAISS / pgvector keyed by phash bits).
        """
        for cid, other_hash in against:
            if hamming(phash, other_hash) <= self.dedup_threshold:
                return cid
        return None

    def ingest(self, clips: list[RawClip]) -> IngestResult:
        """Two-pass ingest: quality gate first, then dedup.

        Separating the passes means we don't waste a hash computation on
        a clip the quality gate would have rejected. It also keeps the
        rejection reasons distinct (quality reasons vs. dedup).
        """
        result = IngestResult()
        survivors: list[tuple[RawClip, int]] = []

        # Pass 1: per-clip quality. Quality survivors get hashed for pass 2.
        for clip in clips:
            reason = self._check_quality(clip)
            if reason:
                result.rejected.append(Rejection(clip.clip_id, reason))
            else:
                survivors.append((clip, perceptual_hash(clip.samples)))

        # Pass 2: dedup against (existing corpus + survivors-already-in-batch).
        # Including in-batch survivors catches the case where the SAME clip
        # appears twice in one batch — first occurrence wins, second is a dup.
        new_hashes: list[tuple[str, int]] = []
        for clip, phash in survivors:
            dup = self._find_dup(phash, self.corpus + new_hashes)
            if dup:
                result.rejected.append(Rejection(clip.clip_id, RejectReason.DUPLICATE, f"duplicate of {dup}"))
                continue
            new_hashes.append((clip.clip_id, phash))
            result.accepted.append(
                AcceptedClip(
                    clip_id=clip.clip_id,
                    text=clip.text,
                    duration_s=clip.duration_s,
                    dataset_version=self.dataset_version,
                    phash=phash,
                )
            )

        # Commit so future ingest() calls dedup against THIS batch's accepts.
        # In a DB version this is just an INSERT; the in-memory list is the
        # equivalent of a never-flushed write-behind cache.
        self.corpus.extend(new_hashes)
        return result


# --- tests ------------------------------------------------------------------


def make_samples(seed: str, length: int = 1024) -> list[float]:
    """Reproducible non-cyclic samples — chained SHAs so different seeds differ."""
    out: list[float] = []
    i = 0
    while len(out) < length:
        h = hashlib.sha256(f"{seed}-{i}".encode()).digest()
        out.extend((b - 128) / 128.0 for b in h)
        i += 1
    return out[:length]


def test_quality_rejections() -> None:
    gate = DatasetGate("v1.0")
    bad = [
        RawClip("short", "Hi.", 0.1, [0.0] * 100),
        RawClip("long", "x" * 200, 45.0, [0.0] * 1024),
        RawClip("noisy", "Hello world today.", 2.0, make_samples("a"), snr_db=10.0),
        RawClip("loud", "Hello world today.", 2.0, make_samples("b"), loudness_lufs=-5.0),
        RawClip("mismatch", "Hi.", 10.0, make_samples("c")),  # 0.3 cps
    ]
    r = gate.ingest(bad)
    assert len(r.accepted) == 0 and len(r.rejected) == 5
    print(f"quality rejections OK -> {[(rj.clip_id, rj.reason.value) for rj in r.rejected]}")


def test_dedup_within_batch() -> None:
    gate = DatasetGate("v1.0")
    base = make_samples("identical")
    clips = [
        RawClip("a", "Hello world today.", 2.0, base),
        RawClip("b", "Hello world today.", 2.0, list(base)),  # near-duplicate
        RawClip("c", "Completely different.", 2.0, make_samples("different")),
    ]
    r = gate.ingest(clips)
    accepted_ids = {c.clip_id for c in r.accepted}
    assert accepted_ids == {"a", "c"}
    assert any(rj.reason == RejectReason.DUPLICATE for rj in r.rejected)
    print(f"within-batch dedup OK -> accepted={accepted_ids}")


def test_dedup_across_batches() -> None:
    gate = DatasetGate("v1.0")
    base = make_samples("corpus_resident")
    gate.ingest([RawClip("orig", "Hello world today.", 2.0, base)])
    r2 = gate.ingest([RawClip("repost", "Hello world today.", 2.0, list(base))])
    assert len(r2.accepted) == 0
    assert "duplicate of orig" in r2.rejected[0].detail
    print("cross-batch dedup OK")


def test_versioning() -> None:
    gate = DatasetGate("v2025.10")
    r = gate.ingest([RawClip("c1", "Hello world today.", 2.0, make_samples("x"))])
    assert r.accepted[0].dataset_version == "v2025.10"
    print("versioning OK")


def main() -> None:
    test_quality_rejections()
    test_dedup_within_batch()
    test_dedup_across_batches()
    test_versioning()
    print("\nScale extensions (for the interview):")
    print("  - 1M+ clips: replace brute-force dedup with MinHash+LSH or a vector index.")
    print("  - Real phash: chromaprint (survives re-encoding) or a learned audio embedding.")
    print("  - Speaker balancing: a second gate that rejects over-represented speakers.")


if __name__ == "__main__":
    main()
