"""Reported ElevenLabs problem: design a voice-dubbing workflow system.

  "Design and implement a system to manage a voice-dubbing workflow,
   replacing an Excel-based process." -- Exponent

State machine: PENDING -> RECORDED -> EDITING -> REVIEW -> APPROVED
                              ^                       |
                              +-- REJECTED <----------+

What interviewers want:
  - Transitions as a TABLE, not if/elif chains.
  - Role-based auth (actor/editor/reviewer).
  - Audit trail of every transition.
  - Per-role queue queries (the editor's main screen).
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class State(Enum):
    PENDING = "pending"
    RECORDED = "recorded"
    EDITING = "editing"
    REVIEW = "review"
    APPROVED = "approved"
    REJECTED = "rejected"


class Role(Enum):
    ACTOR = "actor"
    EDITOR = "editor"
    REVIEWER = "reviewer"


# (from_state, action) -> (to_state, required_role).
# If a transition isn't in this table, it can't happen.
TRANSITIONS = {
    (State.PENDING, "upload"): (State.RECORDED, Role.ACTOR),
    (State.RECORDED, "edit"): (State.EDITING, Role.EDITOR),
    (State.EDITING, "submit"): (State.REVIEW, Role.EDITOR),
    (State.REVIEW, "approve"): (State.APPROVED, Role.REVIEWER),
    (State.REVIEW, "reject"): (State.REJECTED, Role.REVIEWER),
    (State.REJECTED, "reassign"): (State.PENDING, Role.EDITOR),
}


@dataclass
class AuditEntry:
    at: datetime
    user_id: str
    action: str
    from_state: State
    to_state: State


@dataclass
class Segment:
    segment_id: str
    project_id: str
    text: str
    assignee_id: str
    state: State = State.PENDING
    audio_url: str | None = None
    history: list[AuditEntry] = field(default_factory=list)


class Workflow:
    """In-memory implementation; production would back this with Postgres.

    The key design choice: ALL state changes go through `transition()`. No
    code path mutates seg.state directly. This guarantees the audit trail
    is complete — the audit log IS the source of truth, the seg.state is
    just a denormalised view for fast queries.
    """

    def __init__(self):
        self.segments: dict[str, Segment] = {}
        self.user_roles: dict[str, Role] = {}

    def register_user(self, user_id, role):
        self.user_roles[user_id] = role

    def add_segment(self, segment_id, project_id, text, assignee_id):
        seg = Segment(segment_id, project_id, text, assignee_id)
        self.segments[segment_id] = seg
        return seg

    def transition(self, segment_id, user_id, action):
        """The ONE chokepoint for state changes. Validates, mutates, audits.

        Four checks in order, each rejects with a distinct error type:
          1. Segment exists -> KeyError (raised by dict lookup).
          2. User is registered -> PermissionError.
          3. Transition is legal from current state -> ValueError.
          4. User has the role required for this transition -> PermissionError.
        Distinct error types help the caller render meaningful UI messages.
        """
        seg = self.segments[segment_id]
        role = self.user_roles.get(user_id)
        if role is None:
            raise PermissionError(f"unknown user {user_id}")

        # The transitions table is the schema. If a (state, action) pair
        # isn't in it, the transition simply doesn't exist.
        target = TRANSITIONS.get((seg.state, action))
        if target is None:
            raise ValueError(f"action '{action}' not legal from {seg.state.value}")
        next_state, required_role = target

        if role != required_role:
            raise PermissionError(f"action '{action}' requires {required_role.value}, got {role.value}")

        # Audit FIRST, then mutate. If this two-step were a DB transaction,
        # both would commit together; in-memory, the ordering means a
        # stack-trace-killing crash leaves the audit row but a consistent state.
        seg.history.append(
            AuditEntry(
                at=datetime.now(UTC),
                user_id=user_id,
                action=action,
                from_state=seg.state,
                to_state=next_state,
            )
        )
        seg.state = next_state
        return seg

    def upload_recording(self, segment_id, user_id, audio_url):
        """Domain-flavoured wrapper: transition + side effect (set audio_url)."""
        seg = self.transition(segment_id, user_id, "upload")
        seg.audio_url = audio_url
        return seg

    def my_queue(self, user_id) -> list[Segment]:
        """The screen each role lands on after login.

        Each role sees a different filter — that's the whole point of the
        role-based design. In Postgres this is an indexed query on
        (state, assignee_id); here it's just a list-comp scan.
        """
        role = self.user_roles.get(user_id)
        if role == Role.ACTOR:
            # Actors see only their OWN pending assignments.
            return [s for s in self.segments.values() if s.assignee_id == user_id and s.state == State.PENDING]
        if role == Role.EDITOR:
            # Editors see anything ready to edit OR bounced back for re-record.
            return [s for s in self.segments.values() if s.state in (State.RECORDED, State.REJECTED)]
        if role == Role.REVIEWER:
            # Reviewers see anything awaiting their approval verdict.
            return [s for s in self.segments.values() if s.state == State.REVIEW]
        return []

    def project_status(self, project_id) -> dict[State, int]:
        """Roll-up for the project dashboard: how many segments in each state."""
        counts = dict.fromkeys(State, 0)
        for s in self.segments.values():
            if s.project_id == project_id:
                counts[s.state] += 1
        return counts


# --- tests ------------------------------------------------------------------


def setup_workflow() -> Workflow:
    wf = Workflow()
    wf.register_user("actor_a", Role.ACTOR)
    wf.register_user("actor_b", Role.ACTOR)
    wf.register_user("editor_e", Role.EDITOR)
    wf.register_user("reviewer_r", Role.REVIEWER)
    wf.add_segment("seg1", "proj1", "Hello world", "actor_a")
    wf.add_segment("seg2", "proj1", "How are you", "actor_a")
    wf.add_segment("seg3", "proj1", "Goodbye", "actor_b")
    return wf


def test_happy_path() -> None:
    wf = setup_workflow()
    wf.upload_recording("seg1", "actor_a", "s3://seg1.wav")
    wf.transition("seg1", "editor_e", "edit")
    wf.transition("seg1", "editor_e", "submit")
    wf.transition("seg1", "reviewer_r", "approve")
    seg = wf.segments["seg1"]
    assert seg.state == State.APPROVED
    assert len(seg.history) == 4
    print(f"happy path OK -> {len(seg.history)} audit entries")


def test_illegal_transition() -> None:
    wf = setup_workflow()
    try:
        wf.transition("seg1", "reviewer_r", "approve")  # can't approve from PENDING
    except ValueError as e:
        print(f"illegal transition rejected -> {e}")


def test_role_authorization() -> None:
    wf = setup_workflow()
    wf.upload_recording("seg1", "actor_a", "s3://x")
    try:
        wf.transition("seg1", "actor_a", "edit")  # actor can't edit
    except PermissionError as e:
        print(f"role auth enforced -> {e}")


def test_rejection_loop() -> None:
    wf = setup_workflow()
    wf.upload_recording("seg1", "actor_a", "s3://take1")
    wf.transition("seg1", "editor_e", "edit")
    wf.transition("seg1", "editor_e", "submit")
    wf.transition("seg1", "reviewer_r", "reject")
    wf.transition("seg1", "editor_e", "reassign")
    assert wf.segments["seg1"].state == State.PENDING
    assert any(s.segment_id == "seg1" for s in wf.my_queue("actor_a"))
    print("reject -> reassign -> back in actor queue OK")


def test_queues() -> None:
    wf = setup_workflow()
    wf.upload_recording("seg1", "actor_a", "s3://x")
    wf.upload_recording("seg2", "actor_a", "s3://y")
    assert len(wf.my_queue("actor_a")) == 0  # both uploaded
    assert len(wf.my_queue("actor_b")) == 1  # seg3 still pending
    assert len(wf.my_queue("editor_e")) == 2  # seg1, seg2 now RECORDED
    print("queue views OK")


def test_project_status() -> None:
    wf = setup_workflow()
    wf.upload_recording("seg1", "actor_a", "s3://x")
    counts = wf.project_status("proj1")
    assert counts[State.PENDING] == 2 and counts[State.RECORDED] == 1
    nonzero = {k.value: v for k, v in counts.items() if v}
    print(f"project status -> {nonzero}")


def main() -> None:
    test_happy_path()
    test_illegal_transition()
    test_role_authorization()
    test_rejection_loop()
    test_queues()
    test_project_status()
    print("\nExtensions (for the interview):")
    print("  - Postgres: segments table + transitions table (append-only audit).")
    print("  - Concurrent edits: optimistic locking via a version column.")
    print("  - SLA tracking: alert when a segment stalls in any state.")


if __name__ == "__main__":
    main()
