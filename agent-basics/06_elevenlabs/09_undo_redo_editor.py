"""Practical-round prompt: "Implement undo/redo on edit operations."

Problem: a timeline editor performs edits (insert, delete, move, rename).
Implement a history stack supporting:
  - apply(cmd)  -> execute and record on the undo stack; clear redo
  - undo()      -> invert the most recent command, push to redo
  - redo()      -> re-apply the top redo command, push back to undo
  - bounded history (drop oldest when over capacity)

Pattern: **Command pattern** — each operation is reified as an object
that knows how to do() and undo() itself. Encoding the inverse INSIDE
the command (rather than re-running diffs against a snapshot) is the
senior signal: it scales with command count, not document size.

Subtle invariant: applying a NEW command after some undos must clear
the redo stack (a "branch in history" — the previously redo-able
future is no longer reachable). Forgetting this is the single most
common bug.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Command:
    """One reversible edit. `do` and `undo` are closures that mutate the doc."""

    name: str
    do: Callable[[], None]
    undo: Callable[[], None]


class History:
    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self._undo: deque[Command] = deque()
        self._redo: list[Command] = []

    def apply(self, cmd: Command) -> None:
        cmd.do()
        self._undo.append(cmd)
        if len(self._undo) > self.capacity:
            self._undo.popleft()  # drop oldest — never recoverable past this
        # Branching in history invalidates all redos.
        self._redo.clear()

    def undo(self) -> Command | None:
        if not self._undo:
            return None
        cmd = self._undo.pop()
        cmd.undo()
        self._redo.append(cmd)
        return cmd

    def redo(self) -> Command | None:
        if not self._redo:
            return None
        cmd = self._redo.pop()
        cmd.do()
        self._undo.append(cmd)
        return cmd

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)


# --- domain: a minimal audio segment doc + command factories ---


class AudioDoc:
    """Mutable target. Real version would be the Timeline from 04_audio_segments.py."""

    def __init__(self) -> None:
        self.segments: list[dict[str, Any]] = []

    def __repr__(self) -> str:
        return f"AudioDoc({len(self.segments)} segs)"


def make_insert(doc: AudioDoc, idx: int, seg: dict) -> Command:
    return Command(
        name=f"insert@{idx}",
        do=lambda: doc.segments.insert(idx, seg),
        # Capture by identity isn't strictly safe if other ops touch the list at idx,
        # but `apply` guarantees linear apply/undo ordering on the stack.
        undo=lambda: doc.segments.pop(idx),
    )


def make_delete(doc: AudioDoc, idx: int) -> Command:
    # Capture the segment AT apply time, not at command creation time —
    # otherwise undoing after intervening edits restores a stale snapshot.
    captured: dict[str, Any] = {}

    def do() -> None:
        captured["seg"] = doc.segments.pop(idx)

    def undo() -> None:
        doc.segments.insert(idx, captured["seg"])

    return Command(name=f"delete@{idx}", do=do, undo=undo)


def make_rename(doc: AudioDoc, idx: int, new_name: str) -> Command:
    captured: dict[str, Any] = {}

    def do() -> None:
        captured["old"] = doc.segments[idx]["name"]
        doc.segments[idx]["name"] = new_name

    def undo() -> None:
        doc.segments[idx]["name"] = captured["old"]

    return Command(name=f"rename@{idx}", do=do, undo=undo)


# --- tests ---


def test_basic_undo_redo() -> None:
    doc = AudioDoc()
    h = History()
    h.apply(make_insert(doc, 0, {"name": "intro", "audio": "a.wav"}))
    h.apply(make_insert(doc, 1, {"name": "verse", "audio": "b.wav"}))
    assert [s["name"] for s in doc.segments] == ["intro", "verse"]

    h.undo()
    assert [s["name"] for s in doc.segments] == ["intro"]
    h.undo()
    assert doc.segments == []
    h.redo()
    h.redo()
    assert [s["name"] for s in doc.segments] == ["intro", "verse"]
    print("basic undo/redo OK")


def test_apply_clears_redo() -> None:
    doc = AudioDoc()
    h = History()
    h.apply(make_insert(doc, 0, {"name": "a", "audio": "a.wav"}))
    h.apply(make_insert(doc, 1, {"name": "b", "audio": "b.wav"}))
    h.undo()  # b is now redoable
    assert h.can_redo
    # A new edit creates a new branch — the redo of "b" is no longer reachable.
    h.apply(make_insert(doc, 1, {"name": "c", "audio": "c.wav"}))
    assert not h.can_redo
    assert h.redo() is None
    assert [s["name"] for s in doc.segments] == ["a", "c"]
    print("apply-clears-redo OK")


def test_delete_captures_at_apply_time() -> None:
    """Regression: rename + delete + undo all must round-trip correctly."""
    doc = AudioDoc()
    h = History()
    h.apply(make_insert(doc, 0, {"name": "intro", "audio": "a.wav"}))
    h.apply(make_rename(doc, 0, "INTRO"))
    h.apply(make_delete(doc, 0))
    assert doc.segments == []
    h.undo()  # restore the renamed segment
    assert doc.segments[0]["name"] == "INTRO"
    h.undo()  # rename back
    assert doc.segments[0]["name"] == "intro"
    h.undo()  # remove insert
    assert doc.segments == []
    print("delete captures at apply time OK")


def test_capacity_drops_oldest() -> None:
    doc = AudioDoc()
    h = History(capacity=3)
    for i in range(5):
        h.apply(make_insert(doc, i, {"name": f"s{i}", "audio": "x"}))
    # Only the most recent 3 are undoable.
    assert h.undo() is not None
    assert h.undo() is not None
    assert h.undo() is not None
    assert h.undo() is None  # oldest 2 inserts can't be undone
    print("capacity drops oldest OK")


def main() -> None:
    test_basic_undo_redo()
    test_apply_clears_redo()
    test_delete_captures_at_apply_time()
    test_capacity_drops_oldest()


if __name__ == "__main__":
    main()
