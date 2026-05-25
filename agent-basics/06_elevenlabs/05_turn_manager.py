"""Practical-round variant 2: conversation turn manager.

Problem: a stream of speech events (speaker_id, ts_start, ts_end, text)
arrives from multiple speakers. Implement turn-taking logic:
  - Detect overlapping speech (interruption).
  - Assign the canonical "current speaker" at each timestamp.
  - Emit a clean transcript with proper turn boundaries.

State machine angle: the "current speaker" mid-conversation depends on
who's been talking continuously. Define a turn as a maximal time window
during which one speaker holds the floor — interruption transfers the
floor only if the new speaker actually keeps talking (we treat very
short interjections as backchannels and don't switch).

This is the kind of small algorithmic system the practical round prizes:
clear state model, named events, edge-case awareness.
"""

from dataclasses import dataclass, field


@dataclass(order=True)
class SpeechEvent:
    ts_start: int  # ms
    ts_end: int
    speaker_id: str = field(compare=False)
    text: str = field(compare=False)


@dataclass
class Turn:
    speaker_id: str
    ts_start: int
    ts_end: int
    text: str
    was_interrupted: bool = False


# Backchannel threshold: how long the new speaker must hold the floor for
# their event to count as an actual takeover (rather than a quick "uh-huh").
# 600ms is roughly the duration of a 2-syllable affirmation. Tuning this is
# product/UX: lower = more responsive, higher = more "let the user finish."
BACKCHANNEL_THRESHOLD_MS = 600


def build_turns(
    events: list[SpeechEvent],
) -> tuple[list[Turn], list[tuple[str, str, int, int]]]:
    """Convert speech events into (turns, interruptions).

    The algorithm is a single pass over time-sorted events maintaining a
    "current turn" (whoever is holding the floor). Each new event either:
      - extends the current turn (same speaker),
      - is ignored as a backchannel (different speaker, too brief),
      - takes over the floor (different speaker, long enough).

    `interruptions` is logged whenever speakers' time-windows actually
    OVERLAP, regardless of whether the new speaker wins the floor. This
    is the analytics signal a product team wants ("how often do users
    interrupt the agent?"); whether the interrupter then KEEPS the floor
    is a separate decision driven by BACKCHANNEL_THRESHOLD_MS.

    Returns:
      turns: list of Turn objects in temporal order
      interruptions: list of (interrupter_id, interrupted_id, overlap_start, overlap_end)
    """
    if not events:
        return [], []

    # SpeechEvent has order=True keyed on (ts_start, ts_end), so sorted()
    # is the canonical chronological ordering.
    sorted_events = sorted(events)
    turns: list[Turn] = []
    interruptions: list[tuple[str, str, int, int]] = []

    current: Turn | None = None
    for ev in sorted_events:
        # CASE 0: first event opens the first turn.
        if current is None:
            current = Turn(ev.speaker_id, ev.ts_start, ev.ts_end, ev.text)
            continue

        # CASE 1: same speaker -> extend the existing turn.
        # This collapses multiple events from one speaker into one Turn,
        # which is what makes the output a clean per-speaker transcript.
        if ev.speaker_id == current.speaker_id:
            current.ts_end = max(current.ts_end, ev.ts_end)
            current.text = f"{current.text} {ev.text}".strip()
            continue

        # CASE 2: different speaker. Compute geometry of the time-window
        # intersection. The classic "intervals overlap" test:
        #   overlap_start = max(starts)
        #   overlap_end   = min(ends)
        #   overlaps iff overlap_end > overlap_start
        overlap_start = max(current.ts_start, ev.ts_start)
        overlap_end = min(current.ts_end, ev.ts_end)
        overlaps = overlap_end > overlap_start
        new_speaker_duration = ev.ts_end - ev.ts_start

        # ALWAYS record an overlap event for analytics, whether or not we
        # actually transfer the floor below.
        if overlaps:
            interruptions.append(
                (ev.speaker_id, current.speaker_id, overlap_start, overlap_end),
            )

        # CASE 2a: backchannel. Too brief to count as a real takeover.
        # We drop this event from the transcript output (the receiver was
        # not really "speaking", just acknowledging) but the interruption
        # is still on record above.
        if new_speaker_duration < BACKCHANNEL_THRESHOLD_MS:
            continue

        # CASE 2b: real takeover. Close the current turn:
        #   - If the takeover OVERLAPPED, truncate the prior turn at the
        #     overlap_start (we lose the part where they were talked over).
        #   - Mark the prior turn as interrupted for downstream highlighting.
        if overlaps:
            current.ts_end = overlap_start
            current.was_interrupted = True
        turns.append(current)
        # Start a new turn for the new speaker.
        current = Turn(ev.speaker_id, ev.ts_start, ev.ts_end, ev.text)

    # End of input: don't forget the open turn (same flush-on-end rule as
    # 01_streaming_chunker.py — most common bug is dropping the final turn).
    if current is not None:
        turns.append(current)

    return turns, interruptions


# --- tests ---


def test_simple_back_and_forth() -> None:
    events = [
        SpeechEvent(0, 1500, "alice", "hello there"),
        SpeechEvent(1500, 3000, "bob", "hi alice"),
        SpeechEvent(3000, 4500, "alice", "how are you"),
    ]
    turns, interruptions = build_turns(events)
    assert [t.speaker_id for t in turns] == ["alice", "bob", "alice"]
    assert interruptions == []
    print("back-and-forth OK")


def test_backchannel_ignored() -> None:
    events = [
        SpeechEvent(0, 3000, "alice", "I think we should consider all the options"),
        SpeechEvent(1000, 1300, "bob", "uh-huh"),  # 300ms backchannel
    ]
    turns, interruptions = build_turns(events)
    assert len(turns) == 1
    assert turns[0].speaker_id == "alice"
    # Interruption is recorded for analytics even though we ignored it.
    assert len(interruptions) == 1
    print("backchannel ignored OK")


def test_real_interruption() -> None:
    events = [
        SpeechEvent(0, 5000, "alice", "I was just saying that we should"),
        SpeechEvent(2000, 4000, "bob", "actually that won't work because"),
    ]
    turns, interruptions = build_turns(events)
    assert [t.speaker_id for t in turns] == ["alice", "bob"]
    # Alice's turn truncated at 2000 (overlap_start), marked interrupted.
    assert turns[0].ts_end == 2000
    assert turns[0].was_interrupted
    assert interruptions[0][:2] == ("bob", "alice")
    print("real interruption OK")


def test_speaker_continuation() -> None:
    events = [
        SpeechEvent(0, 1000, "alice", "hello"),
        SpeechEvent(1200, 2000, "alice", "how are you"),  # same speaker, gap
    ]
    turns, _ = build_turns(events)
    assert len(turns) == 1
    assert turns[0].text == "hello how are you"
    assert turns[0].ts_end == 2000
    print("speaker continuation OK")


def main() -> None:
    test_simple_back_and_forth()
    test_backchannel_ignored()
    test_real_interruption()
    test_speaker_continuation()


if __name__ == "__main__":
    main()
