// frontend/components/ThreadView.tsx
//
// The dashboard's main pane.
//
// Two reactive subscriptions, no polling:
//   - messages for the open thread
//   - pending suggestions for the workspace (NOVEL FEATURE 2)
//
// When the agent commits a message mutation from inside the sandbox, this
// component re-renders. Same when the proactive scheduler saves a
// suggestion. The wiring "falls out" of useQuery -- that's the demo line.
//
// Accept/reject feeds the proactive Stage-6 trust loop in
// convex/suggestions.set_status, which updates workspace_proactive_state
// transactionally. Rejection captures a *reason* -- causality tracking
// the prep doc calls out as essential to learn from rejection.

"use client";

import { useMutation, useQuery } from "convex/react";
import { api } from "../../convex/_generated/api";
import type { Doc, Id } from "../../convex/_generated/dataModel";
import {
  REJECTION_REASONS,
  REJECTION_REASON_LABELS,
  type RejectionReason,
} from "../../convex/constants";

type Props = {
  workspace_id: string;
  thread_id: Id<"threads">;
};

export function ThreadView({ workspace_id, thread_id }: Props) {
  // useQuery returns undefined while the first round-trip is in flight; that
  // is the canonical "loading" state.
  const messages = useQuery(api.messages.get_thread_messages, { thread_id });
  const suggestions = useQuery(api.suggestions.get_pending, { workspace_id });
  const state = useQuery(api.proactive.get_state, { workspace_id });

  if (messages === undefined) {
    return <div className="thread-view loading">loading…</div>;
  }

  return (
    <div className="thread-view">
      {state && (
        <div className="proactive-meta">
          <small>
            trust {state.trust_score.toFixed(2)} ·{" "}
            {state.suggestions_this_window}/
            {state.suggestions_per_window_max} this week
            {state.paused ? " · PAUSED" : ""}
          </small>
        </div>
      )}

      <div className="messages">
        {messages.map((m) => (
          <div key={m._id} className={`msg msg-${m.role}`}>
            <span className="who">{m.role === "agent" ? "Agent" : m.user_id}</span>
            <span className="text">{m.content}</span>
          </div>
        ))}
      </div>

      {suggestions && suggestions.length > 0 && (
        <div className="suggestions">
          <h4>Proactive suggestions</h4>
          {suggestions.map((s) => (
            <SuggestionCard key={s._id} suggestion={s} />
          ))}
        </div>
      )}
    </div>
  );
}

type Suggestion = Doc<"suggestions">;

// Surface the breakdown alongside accept + reject-with-reason buttons.
// The reasons mirror the prep doc's causality-tracking buttons; the
// vocabulary lives in convex/constants.ts so schema, mutations, and UI
// can't drift.
function SuggestionCard({ suggestion }: { suggestion: Suggestion }) {
  const setStatus = useMutation(api.suggestions.set_status);
  const sb = suggestion.specificity_breakdown;

  const reject = (reason: RejectionReason) =>
    setStatus({
      suggestion_id: suggestion._id,
      status: "rejected",
      rejection_reason: reason,
    });

  return (
    <div className="suggestion">
      <div className="text">{suggestion.candidate_text}</div>
      <div className="breakdown">
        <small>
          score {suggestion.specificity_score.toFixed(2)} · entities{" "}
          {sb.named_entity_density.toFixed(2)} · recurrence{" "}
          {sb.recurrence_count.toFixed(2)} · attributed{" "}
          {sb.user_attribution ? "yes" : "no"} · timing {sb.timing_pattern}
          {suggestion.genericness_score != null &&
            ` · genericness ${suggestion.genericness_score.toFixed(2)}`}
        </small>
      </div>
      <div className="actions">
        <button
          type="button"
          onClick={() => setStatus({ suggestion_id: suggestion._id, status: "accepted" })}
        >
          Accept
        </button>
        {REJECTION_REASONS.map((reason) => (
          <button key={reason} type="button" onClick={() => reject(reason)}>
            Reject: {REJECTION_REASON_LABELS[reason]}
          </button>
        ))}
      </div>
    </div>
  );
}
