"""Voice Activity Detection (VAD) — energy-based with hangover.

Problem (interview-shaped): "You're streaming PCM audio frames from the
user's microphone into a voice agent. Decide for each frame whether
the user is currently speaking. Output a stream of (frame_idx, is_speech)
events. False positives during silence waste TTS turns; false negatives
during speech cause the agent to interrupt the user."

Why energy-based: real production VAD uses a small neural model
(Silero, WebRTC VAD). But the *control logic on top* — thresholding,
hangover, hysteresis — is the part interviewers care about. Anyone
can plug in a model; the senior signal is "do you understand why
naive thresholding flaps?"

Concepts demonstrated:
  - RMS energy as a cheap signal proxy.
  - Hysteresis: separate thresholds for entering vs leaving speech
    state, prevents flapping at boundaries.
  - Hangover: stay in "speech" for N frames AFTER energy drops below
    threshold, so brief pauses (between words) don't end the turn.
  - Hangover is the single biggest reason naive VADs ship turn-events
    on every word boundary instead of every sentence boundary.
"""

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum


class VadState(Enum):
    SILENCE = "silence"
    SPEECH = "speech"


@dataclass
class VadEvent:
    frame_idx: int
    state: VadState
    rms: float


def rms(frame: list[float]) -> float:
    """Root mean square — proxy for loudness on a frame of PCM samples in [-1, 1]."""
    if not frame:
        return 0.0
    return math.sqrt(sum(s * s for s in frame) / len(frame))


class EnergyVAD:
    """Streaming VAD: feed frames, get state transitions."""

    def __init__(
        self,
        *,
        enter_speech_rms: float = 0.05,
        exit_speech_rms: float = 0.02,
        hangover_frames: int = 8,
    ) -> None:
        if exit_speech_rms >= enter_speech_rms:
            raise ValueError("exit threshold must be < enter threshold for hysteresis")
        self.enter_speech_rms = enter_speech_rms
        self.exit_speech_rms = exit_speech_rms
        self.hangover_frames = hangover_frames
        self.state = VadState.SILENCE
        self._below_count = 0  # frames spent below exit threshold while in SPEECH

    def push_frame(self, frame_idx: int, frame: list[float]) -> VadEvent | None:
        """Returns a VadEvent only on state CHANGE; None if state is unchanged."""
        energy = rms(frame)

        if self.state == VadState.SILENCE:
            if energy >= self.enter_speech_rms:
                self.state = VadState.SPEECH
                self._below_count = 0
                return VadEvent(frame_idx, VadState.SPEECH, energy)
            return None

        # state == SPEECH
        if energy < self.exit_speech_rms:
            self._below_count += 1
            if self._below_count >= self.hangover_frames:
                self.state = VadState.SILENCE
                self._below_count = 0
                return VadEvent(frame_idx, VadState.SILENCE, energy)
        else:
            self._below_count = 0  # reset on any energetic frame
        return None

    def process_stream(self, frames: Iterable[list[float]]) -> Iterator[VadEvent]:
        for i, frame in enumerate(frames):
            event = self.push_frame(i, frame)
            if event is not None:
                yield event


# --- tests ---


def synth_frame(amplitude: float, n: int = 160) -> list[float]:
    """Generate `n`-sample frame of given amplitude (constant signal stand-in)."""
    return [amplitude] * n


def test_simple_speech_silence() -> None:
    vad = EnergyVAD()
    frames = [synth_frame(0.0)] * 5 + [synth_frame(0.1)] * 20 + [synth_frame(0.0)] * 20
    events = list(vad.process_stream(frames))
    states = [e.state for e in events]
    assert states == [VadState.SPEECH, VadState.SILENCE], states
    # Enters at frame 5 (first energetic frame).
    assert events[0].frame_idx == 5
    # Exits hangover_frames AFTER speech ends (frame 25 = end of speech).
    assert events[1].frame_idx == 25 + vad.hangover_frames - 1, events[1].frame_idx
    print(f"basic OK -> entered@{events[0].frame_idx}, exited@{events[1].frame_idx}")


def test_hangover_swallows_brief_pause() -> None:
    """A 4-frame pause inside hangover=8 should NOT trigger a SILENCE event."""
    vad = EnergyVAD(hangover_frames=8)
    frames = (
        [synth_frame(0.0)] * 3
        + [synth_frame(0.1)] * 10
        + [synth_frame(0.0)] * 4  # brief gap, shorter than hangover
        + [synth_frame(0.1)] * 10
        + [synth_frame(0.0)] * 20
    )
    events = list(vad.process_stream(frames))
    states = [e.state for e in events]
    assert states == [VadState.SPEECH, VadState.SILENCE], states  # not 4 transitions
    print(f"hangover absorbs gap OK -> {len(events)} transitions instead of 4")


def test_hysteresis_prevents_flapping() -> None:
    """Energy oscillating between enter and exit thresholds shouldn't flap."""
    vad = EnergyVAD(enter_speech_rms=0.05, exit_speech_rms=0.02, hangover_frames=2)
    # Trigger speech, then oscillate at 0.03 (between exit and enter).
    frames = [synth_frame(0.1)] * 5 + [synth_frame(0.03)] * 20 + [synth_frame(0.0)] * 10
    events = list(vad.process_stream(frames))
    # Should NOT exit during the 0.03 oscillation; only exits after true silence + hangover.
    assert [e.state for e in events] == [VadState.SPEECH, VadState.SILENCE], events
    print("hysteresis OK")


def test_invalid_thresholds() -> None:
    try:
        EnergyVAD(enter_speech_rms=0.02, exit_speech_rms=0.05)
    except ValueError:
        print("threshold validation OK")
        return
    raise AssertionError("expected ValueError on inverted thresholds")


def main() -> None:
    test_simple_speech_silence()
    test_hangover_swallows_brief_pause()
    test_hysteresis_prevents_flapping()
    test_invalid_thresholds()


if __name__ == "__main__":
    main()
