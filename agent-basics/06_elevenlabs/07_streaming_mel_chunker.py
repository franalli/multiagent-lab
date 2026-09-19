"""Research Engineer pattern: streaming mel-spectrogram -> audio vocoder.

Problem (verbatim from prep): "Code a function to convert mel-spectrograms
to audio chunks for streaming."

Streaming neural vocoders process fixed-size mel windows. To avoid
audible artifacts at chunk boundaries, you process with an overlap and
trim the overlapping audio before yielding. Two state variables you
must track:

  - `mel_buffer`: accumulating mel frames waiting for enough to vocode.
  - `overlap_frames`: number of frames at the END of the just-processed
    chunk that stay in the buffer to provide context for the next chunk.

The interview version uses torch tensors. This playground version uses
plain Python lists of floats so it runs without the torch dependency —
the buffering/overlap LOGIC is identical, which is what gets graded.
A note at the bottom shows the one-line torch swap.

Trade-off interviewers want to hear: bigger chunk_frames -> better
parallelism per vocoder call, but higher first-audio latency. The
right setting depends on the vocoder's RTF (real-time factor) and the
target time-to-first-audio.
"""

from collections.abc import Callable

MelFrame = list[float]  # one frame of mel bins
AudioChunk = list[float]  # one chunk of audio samples


class StreamingMelToAudio:
    """Wraps a per-chunk vocoder and exposes a streaming push/pop interface.

    Lifecycle:
       upstream model -> push_mels()  (any time, any number of frames)
       pop_audio()                    (when caller wants the next ready chunk)
       flush()                        (once, at end of stream)

    The contract: every sample appears in the output stream EXACTLY ONCE,
    no duplicates at chunk boundaries, no gaps. The overlap-and-trim
    machinery is what guarantees that. See the visual below.
    """

    def __init__(
        self,
        vocoder: Callable[[list[MelFrame]], AudioChunk],
        chunk_frames: int = 32,
        hop_length: int = 256,
        overlap_frames: int = 4,
    ) -> None:
        # Sanity check: if overlap >= chunk, the buffer would grow each call.
        if overlap_frames >= chunk_frames:
            raise ValueError("overlap_frames must be < chunk_frames")
        self.vocoder = vocoder
        self.chunk_frames = chunk_frames  # how many mel frames vocoder processes at once
        self.hop_length = hop_length  # audio samples per mel frame (vocoder-dependent)
        self.overlap_frames = overlap_frames  # mel frames retained for next-chunk context
        # Accumulator of incoming mel frames waiting to be vocoded.
        self.mel_buffer: list[MelFrame] = []

    def push_mels(self, frames: list[MelFrame]) -> None:
        """Append incoming mel frames from the upstream TTS model.

        Cheap append; no vocoding work done here. Call pop_audio() to
        actually drain the buffer when you want the next ready chunk.
        """
        self.mel_buffer.extend(frames)

    def pop_audio(self) -> AudioChunk | None:
        """Vocode one chunk if enough frames are buffered. None if not.

        VISUAL of the overlap-and-trim trick for chunk_frames=8, overlap=2:

            buffer after push_mels:
              [f0 f1 f2 f3 f4 f5 f6 f7  f8 f9 ...]

            pop_audio() processes f0..f7 (chunk_frames=8) through vocoder,
            retains f6, f7 (last `overlap_frames`) in buffer for next call:
              process: [f0 f1 f2 f3 f4 f5 f6 f7]
              retain:                  [f6 f7  f8 f9 ...]

            The vocoder emits ~chunk_frames * hop_length audio samples.
            We TRIM the last `overlap_frames * hop_length` samples, because
            those correspond to f6/f7 which we'll re-vocode next call (with
            f8..f13 for context — giving smoother prosody at the boundary).

            Result: caller gets f0..f5 worth of audio this call, f6..f11
            worth next call. Every sample emitted exactly once.
        """
        # Not enough frames buffered to fill a chunk — caller should push more.
        if len(self.mel_buffer) < self.chunk_frames:
            return None

        # Slice the next chunk to process.
        process_frames = self.mel_buffer[: self.chunk_frames]
        # Retain the trailing `overlap_frames` for next call's context.
        # (Slicing makes a copy; for production-scale buffers a deque would
        # avoid the O(n) slice — interview scale doesn't care.)
        self.mel_buffer = self.mel_buffer[self.chunk_frames - self.overlap_frames :]

        # The actual vocoder call. In a real system this is a torch model
        # forward pass; the playground stand-in just returns hop_length
        # samples per frame so the math is verifiable.
        audio = self.vocoder(process_frames)

        # Trim the OVERLAP region from the audio so we don't emit it now
        # (it'll be re-emitted, with better context, next call).
        trim = self.overlap_frames * self.hop_length
        return audio[:-trim] if trim > 0 else audio

    def flush(self) -> AudioChunk | None:
        """Vocode whatever frames remain at end of stream.

        No trim here — there is no "next call" to re-render the tail with
        better context. The last `overlap_frames` worth of audio is emitted
        as-is. Skipping this is the "trailing silence" bug.
        """
        if not self.mel_buffer:
            return None
        audio = self.vocoder(self.mel_buffer)
        self.mel_buffer = []
        return audio


# --- tests with a fake vocoder ---


def make_fake_vocoder(hop_length: int) -> Callable[[list[MelFrame]], AudioChunk]:
    """Return a vocoder that emits hop_length samples per mel frame."""

    def vocoder(frames: list[MelFrame]) -> AudioChunk:
        audio: AudioChunk = []
        for i, _frame in enumerate(frames):
            audio.extend([float(i)] * hop_length)
        return audio

    return vocoder


def test_buffers_until_chunk_full() -> None:
    s = StreamingMelToAudio(make_fake_vocoder(256), chunk_frames=32, hop_length=256, overlap_frames=4)
    s.push_mels([[0.0] * 80 for _ in range(10)])
    assert s.pop_audio() is None  # not enough yet
    s.push_mels([[0.0] * 80 for _ in range(22)])  # total = 32 -> just enough
    audio = s.pop_audio()
    assert audio is not None
    # 32 frames * 256 hop - 4 frames * 256 trim = (32 - 4) * 256 = 7168 samples.
    assert len(audio) == (32 - 4) * 256, len(audio)
    print(f"buffer-until-full OK -> emitted {len(audio)} samples")


def test_overlap_retained_for_continuity() -> None:
    s = StreamingMelToAudio(make_fake_vocoder(10), chunk_frames=8, hop_length=10, overlap_frames=2)
    s.push_mels([[0.0]] * 8)
    audio_a = s.pop_audio()
    # Buffer should now contain the 2 retained overlap frames.
    assert len(s.mel_buffer) == 2
    s.push_mels([[0.0]] * 6)  # 2 retained + 6 new = 8 -> emit again
    audio_b = s.pop_audio()
    assert audio_a is not None and audio_b is not None
    # Each emit drops `overlap_frames * hop_length` samples from the end.
    assert len(audio_a) == (8 - 2) * 10
    assert len(audio_b) == (8 - 2) * 10
    print("overlap-retained OK")


def test_flush_emits_remainder() -> None:
    s = StreamingMelToAudio(make_fake_vocoder(256), chunk_frames=32, hop_length=256, overlap_frames=4)
    s.push_mels([[0.0]] * 5)  # below chunk threshold
    assert s.pop_audio() is None
    tail = s.flush()
    assert tail is not None and len(tail) == 5 * 256
    print("flush emits remainder OK")


def main() -> None:
    test_buffers_until_chunk_full()
    test_overlap_retained_for_continuity()
    test_flush_emits_remainder()
    print("\nIn torch: same shape, swap list ops for `torch.cat(mel_buffer, dim=0)`,")
    print("the vocoder call becomes `vocoder(frames.unsqueeze(0)).squeeze(0)` under")
    print("`torch.no_grad()`. The buffering/overlap logic above is what the")
    print("interview evaluates — torch is just the runtime substrate.")


if __name__ == "__main__":
    main()
