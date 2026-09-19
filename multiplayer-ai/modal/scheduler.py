# modal/scheduler.py
#
# NOVEL FEATURE 2: the granular proactive framework.
#
# Modal crons are deploy-time static and take NO arguments. So the scheduled
# function fans out over active workspaces. Per-workspace timing in
# production lives in Convex; the Modal cron is just the heartbeat.
#
# THE SIX STAGES (mirror the prep doc's "inferred six-stage pipeline"):
#
#   Stage 1 -- Activity-delta gate     cheap, no LLM, exits ~80% of scans
#   Stage 2 -- Candidate extraction    liberal LLM-ish pattern miner
#   Stage 3 -- Specificity scoring     4-dim transparent score (the novel piece)
#   Stage 4 -- Non-invasiveness gate   7 sub-checks reading workspace_proactive_state
#                4a -- workspace paused / kill-switch
#                4b -- weekly frequency cap
#                4c -- channel allow-list
#                4d -- quiet hours (workspace tz)
#                4e -- recent-rejection similarity (negative-space classifier)
#                4f -- user currently active (don't interrupt)
#                4g -- SME deference (don't fire while human expert is in flow)
#   Stage 5 -- Surface in Convex       reactive update + counter bump
#   Stage 6 -- Outcome feedback        accept/reject -> trust score (in suggestions.ts)
#
# The trust score is multiplied into the surfacing threshold: low trust
# means only the strongest candidates survive; trust earned through
# accepted suggestions broadens the funnel.

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from typing import Any

import modal

from common import (  # sibling import; modal/ is intentionally not a package
    DEFAULT_GEMINI_MODEL,
    app,
    control_plane_image,
    convex_post,
    convex_query,
    secrets,
)

# ---------------------------------------------------------------------------
# Stage 1 helpers
# ---------------------------------------------------------------------------


def get_active_workspaces() -> list[str]:
    """Production: query Convex for workspaces with activity in last N days.
    POC: hard-coded to the demo workspace."""
    return ["dev"]


def has_meaningful_activity_since_last_scan(workspace_id: str) -> bool:
    """Stage 1 -- cheap gate that exits most scans without spending tokens.

    POC heuristic: assume activity. Production reads a watermark counter in
    the workspace_proactive_state row and compares against the latest
    message timestamp.
    """
    return True


# ---------------------------------------------------------------------------
# Stage 2 -- candidate extraction
# ---------------------------------------------------------------------------


_SYNTHETIC_MESSAGES = [
    # Used as a cold-start fallback when the workspace has no real history
    # yet. Same shape Convex returns, so the rest of Stage 2 doesn't care
    # which source it got.
    {
        "user_id": "U01A",
        "channel": "C01ENG",
        "text": "the lead's Q3 numbers are ready -- can you recap on Monday?",
        "ts_offset": -100,
    },
    {
        "user_id": "U01A",
        "channel": "C01ENG",
        "text": "Same recap as last week on the lead's Q3 numbers please",
        "ts_offset": -86_400 * 7,
    },
    {
        "user_id": "U01A",
        "channel": "C01ENG",
        "text": "Q3 recap from the lead expected Monday morning",
        "ts_offset": -86_400 * 14,
    },
    {"user_id": "U01B", "channel": "C01ENG", "text": "thanks", "ts_offset": -50},
]


def get_recent_messages(workspace_id: str, *, hours: int = 24) -> list[dict[str, Any]]:
    """Stage 2 input. Try the real Convex `messages.recent_for_workspace`
    query first; fall back to synthetic data when the workspace has no
    history (cold-start) or the route 404s.

    Production lookback uses the by_workspace index; the `since_ms` filter
    happens server-side so we only ship recent rows across the wire. The
    channel field is sourced from the parent thread row -- the messages
    table itself doesn't carry it, so we keep the synthetic channel
    field for compatibility with downstream gates.
    """
    since_ms = int((time.time() - hours * 3600) * 1000)
    real = convex_query(
        "/api/messages/recent",
        {"workspace_id": workspace_id, "since_ms": since_ms, "limit": 200},
    )
    if real:
        # Map Convex rows into the scheduler's expected shape. Channel is
        # not stored on messages -- but the proactive scan only uses it
        # for the channel allow-list gate, which compares against
        # workspace_proactive_state.channel_allow_list. For real-traffic
        # workspaces we leave it as "" and rely on threads.channel via the
        # thread join (production wires this; POC keeps it minimal).
        return [
            {
                "user_id": m.get("user_id", ""),
                "channel": m.get("channel", ""),
                "text": m.get("content", ""),
                "ts": float(m.get("timestamp", 0)) / 1000.0,
            }
            for m in real
        ]
    # Cold-start fallback.
    now = time.time()
    return [{**m, "ts": now + m["ts_offset"]} for m in _SYNTHETIC_MESSAGES]


def _extract_candidates_heuristic(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-user repetition miner -- cheap, no LLM. Acts as the cold-start
    extractor and as a fallback when the LLM call fails / isn't configured."""
    by_user: dict[str, list[dict[str, Any]]] = {}
    for m in messages:
        by_user.setdefault(m["user_id"], []).append(m)

    candidates: list[dict[str, Any]] = []
    for user_id, msgs in by_user.items():
        if len(msgs) < 2:
            continue
        text = msgs[0]["text"]
        channel = msgs[0].get("channel", "")
        candidates.append(
            {
                "user_id": user_id,
                "channel": channel,
                "text": f"Automate: {text}",
                "occurrences": len(msgs),
                "timestamps": [m["ts"] for m in msgs],
            }
        )
    return candidates


def _extract_candidates_via_llm(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Spec line 1710: "LLM call for pattern extraction."

    Asks Gemini to return a JSON list of automation candidates given the
    recent message history. Returns None if the Gemini SDK isn't available
    or the call/parse fails -- the caller then falls back to the heuristic.

    Prompt deliberately asks for the same shape as the heuristic emits so
    Stage 3 (scoring) doesn't care which path produced the candidates.
    """
    if not messages:
        return None
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai  # deferred: optional dep
        from google.genai import types as genai_types
    except ImportError:
        return None

    transcript = "\n".join(f"- [{m['user_id']} @ {m.get('channel', '?')}] {m['text']}" for m in messages[:50])
    system = "You mine recurring workflows from a chat transcript. For each\nautomation candidate, emit ONE JSON object with keys: user_id, channel,\ntext (a short proposal starting with 'Automate:'), occurrences (int >= 2),\ntimestamps (list of unix seconds from the transcript).\nReturn ONLY a JSON array. No prose, no markdown fence."
    user = f"Transcript:\n{transcript}\n\nReturn a JSON array of candidates."

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=DEFAULT_GEMINI_MODEL,
            contents=[genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=user)])],
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=1024,
                # Tell Gemini we want JSON, not prose. The model still
                # occasionally fences; we strip below as belt-and-braces.
                response_mime_type="application/json",
            ),
        )
        text = (resp.text or "").strip()
        # Tolerate a stray code fence around the JSON array.
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n|\n```$", "", text)
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        # LLM output drift -- bad JSON / wrong types. We swallow these because
        # the heuristic fallback already handles "no LLM result". SDK-level
        # exceptions (network, auth) intentionally propagate so we don't
        # mask provider misconfiguration as a quiet "use heuristic" path.
        return None

    if not isinstance(parsed, list):
        return None
    candidates: list[dict[str, Any]] = []
    for c in parsed:
        if not isinstance(c, dict):
            continue
        # Defensive coercion -- LLM output drifts; we accept-or-skip rather
        # than reject the whole batch.
        try:
            candidates.append(
                {
                    "user_id": str(c.get("user_id", "")),
                    "channel": str(c.get("channel", "")),
                    "text": str(c.get("text", "")),
                    "occurrences": max(int(c.get("occurrences", 0)), 1),
                    "timestamps": [float(t) for t in (c.get("timestamps") or [])],
                }
            )
        except (TypeError, ValueError):
            continue
    return candidates or None


def extract_candidate_patterns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stage 2 -- liberal candidate mining. LLM path when configured,
    heuristic otherwise. Both emit the same shape."""
    via_llm = _extract_candidates_via_llm(messages)
    return via_llm if via_llm is not None else _extract_candidates_heuristic(messages)


# ---------------------------------------------------------------------------
# Stage 3 -- specificity scoring (the visible novel piece)
# ---------------------------------------------------------------------------


NAMED_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|Q[1-4]|\d+%|[A-Z]{2,})\b")


def count_named_entities(text: str) -> int:
    # Higher count → candidate references concrete people/things → more
    # specific automation (Stage-3 specificity signal).
    return len(NAMED_ENTITY_RE.findall(text))


def score_timing_regularity(timestamps: list[float]) -> float:
    """1.0 when intervals are near-constant, 0.0 when noisy."""
    if len(timestamps) < 2:
        return 0.0
    ts = sorted(timestamps)
    intervals = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    if not intervals:
        return 0.0
    avg = sum(intervals) / len(intervals)
    if avg <= 0:
        return 0.0
    spread = sum(abs(i - avg) for i in intervals) / len(intervals) / avg
    return max(0.0, 1.0 - spread)


def score_specificity(candidate: dict[str, Any]) -> dict[str, Any]:
    """4-dimensional score in [0, 1] per dimension plus an aggregate `total`.

    Sub-scores are returned alongside the aggregate so the UI can show
    *which* dimension fired (e.g. "named_entity_density=0.8, timing=weekly").
    """
    text = candidate["text"]
    word_count = max(len(text.split()), 1)

    entity_density = min(count_named_entities(text) / word_count, 1.0)
    recurrence = min(candidate["occurrences"] / 5.0, 1.0)
    user_attribution = bool(candidate.get("user_id"))
    timing_regularity = score_timing_regularity(candidate.get("timestamps", []))

    total = (entity_density + recurrence + (1.0 if user_attribution else 0.0) + timing_regularity) / 4.0

    if timing_regularity > 0.7:
        timing_label = "weekly-regular"
    elif timing_regularity > 0.3:
        timing_label = "semi-regular"
    else:
        timing_label = "ad-hoc"

    return {
        "named_entity_density": round(entity_density, 3),
        "recurrence_count": round(recurrence, 3),
        "user_attribution": user_attribution,
        "timing_pattern": timing_label,
        "total": round(total, 3),
    }


# ---------------------------------------------------------------------------
# Stage 4 -- non-invasiveness gates
# ---------------------------------------------------------------------------
#
# Implemented as a chain of pure functions over a `state` dict and the
# candidate. Each returns either `None` (passes through) or a short string
# explaining the rejection -- which we surface in the smoke-test output so
# you can see exactly which gate dropped a candidate during the demo.


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")


def _tokenize(text: str) -> Counter[str]:
    # Intentionally lossy (no stemming, no stopword removal) to stay
    # dependency-free at POC scale.
    return Counter(t.lower() for t in _TOKEN_RE.findall(text))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    """Bag-of-words cosine. Tiny, dependency-free; adequate for "does this
    look like a previous rejection?" without spinning up embeddings."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def gate_paused(state: dict[str, Any]) -> str | None:
    """4a -- workspace kill-switch."""
    return "workspace_paused" if state.get("paused") else None


def gate_frequency_cap(state: dict[str, Any]) -> str | None:
    """4b -- weekly cap. window_resets_at handled by Convex bump_counter."""
    used = state.get("suggestions_this_window", 0)
    cap = state.get("suggestions_per_window_max", 5)
    return "frequency_cap" if used >= cap else None


def gate_channel_allow_list(state: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    """4c -- only fire in channels that opted in.

    Three cases for the channel allow-list:
      - explicit list set: candidate's channel must be in it
      - empty list + candidate has a channel: opt-in mode; accept the
        demo "C01..." synthetic prefix, reject otherwise
      - empty list + candidate has no channel (real Convex doesn't carry
        thread.channel on messages today): treat as DM and allow. Without
        this case the gate fails closed on every real-traffic candidate.
    """
    allow = state.get("channel_allow_list", []) or []
    channel = candidate.get("channel", "")
    if allow:
        return None if channel in allow else "channel_not_allowed"
    if not channel:
        return None
    return None if channel.startswith("C01") else "channel_not_allowed"


def gate_quiet_hours(state: dict[str, Any]) -> str | None:
    """4d -- time-of-day suppression. Uses UTC integer hour offsets."""
    qh = state.get("quiet_hours") or {}
    if not qh.get("enabled", False):
        return None
    hour = time.gmtime().tm_hour
    start = int(qh.get("start_hour_utc", 0))
    end = int(qh.get("end_hour_utc", 24))
    in_window = start <= hour < end if start <= end else (hour >= start or hour < end)
    return None if in_window else "quiet_hours"


REJECTION_SIMILARITY_THRESHOLD = 0.7


def gate_recent_rejection(
    rejection_tokens: list[Counter[str]],
    candidate: dict[str, Any],
) -> tuple[str | None, float]:
    """4e -- negative-space genericness classifier.

    Takes the *pre-tokenised* rejection corpus so the tokeniser doesn't run
    O(N x M) when scoring many candidates against many rejections. Returns
    the similarity score regardless so the suggestion row records
    `genericness_score` for explainability.
    """
    if not rejection_tokens:
        return None, 0.0
    cand_tokens = _tokenize(candidate["text"])
    best = max((_cosine(cand_tokens, rt) for rt in rejection_tokens), default=0.0)
    if best >= REJECTION_SIMILARITY_THRESHOLD:
        return "similar_to_recent_rejection", best
    return None, best


USER_ACTIVE_WINDOW_SECONDS = 5 * 60  # don't interrupt within 5 min of last user message


def gate_user_active(workspace_id: str, candidate: dict[str, Any]) -> str | None:
    """4f -- skip if user is currently active in any thread."""
    last_ts = convex_query(
        "/api/proactive/recent_user_activity",
        {"workspace_id": workspace_id, "user_id": candidate["user_id"]},
    )
    if not last_ts:
        return None
    age = time.time() - (float(last_ts) / 1000.0)
    return "user_currently_active" if age < USER_ACTIVE_WINDOW_SECONDS else None


SME_DEFERENCE_WINDOW_SECONDS = 2 * 60


def gate_sme_deference(workspace_id: str, candidate: dict[str, Any]) -> str | None:
    """4g -- don't fire in a channel where a human expert is currently active."""
    channel = candidate.get("channel", "")
    if not channel:
        return None
    last_ts = convex_query(
        "/api/proactive/channel_last_activity",
        {"workspace_id": workspace_id, "channel": channel},
    )
    if not last_ts:
        return None
    age = time.time() - (float(last_ts) / 1000.0)
    return "human_in_channel" if age < SME_DEFERENCE_WINDOW_SECONDS else None


# ---------------------------------------------------------------------------
# Threshold derivation
# ---------------------------------------------------------------------------


BASE_THRESHOLD = float(os.environ.get("SUGGESTION_SPECIFICITY_THRESHOLD", "0.45"))


def effective_threshold(trust_score: float) -> float:
    """Trust-adjusted surfacing threshold.

    trust 1.0 (high) -> threshold drops 30% (broader funnel, the workspace
                        is engaged and we can suggest more aggressively)
    trust 0.0 (low)  -> threshold rises 30% (start-small, only the very
                        strongest signals survive)
    """
    delta = 0.3 * (0.5 - trust_score)  # +0.15 at trust=0; -0.15 at trust=1
    return max(0.1, min(0.95, BASE_THRESHOLD + delta))


# ---------------------------------------------------------------------------
# Per-workspace scan
# ---------------------------------------------------------------------------


@app.function(image=control_plane_image, secrets=secrets(), timeout=120)
def scan_workspace(workspace_id: str) -> dict[str, Any]:
    """Fan-in target invoked by the cron once per workspace."""

    # Ensure the proactive state row exists (idempotent).
    convex_post("/api/proactive/ensure_state", {"workspace_id": workspace_id})

    # Pull the state row -- POC: do a query call. If unavailable, fall back
    # to in-memory defaults so the demo still runs.
    state = convex_query("/api/proactive/get_state", {"workspace_id": workspace_id}) or {
        "trust_score": 0.3,
        "suggestions_this_window": 0,
        "suggestions_per_window_max": 5,
        "channel_allow_list": [],
        "quiet_hours": {"start_hour_utc": 0, "end_hour_utc": 24, "enabled": False},
        "paused": False,
        "consecutive_rejections": 0,
    }

    if reason := gate_paused(state):
        return {"workspace_id": workspace_id, "exit": reason}
    if reason := gate_quiet_hours(state):
        return {"workspace_id": workspace_id, "exit": reason}

    if not has_meaningful_activity_since_last_scan(workspace_id):
        return {"workspace_id": workspace_id, "exit": "no_activity"}

    recent = get_recent_messages(workspace_id, hours=24)
    raw_candidates = extract_candidate_patterns(recent)

    rejections = (
        convex_query(
            "/api/proactive/recent_rejections",
            {"workspace_id": workspace_id, "limit": 50},
        )
        or []
    )
    # Pre-tokenise the rejection corpus once -- the gate runs per-candidate
    # but the corpus is constant for this scan.
    rejection_tokens = [_tokenize(r["candidate_text"]) for r in rejections]

    threshold = effective_threshold(state["trust_score"])
    surfaced: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for c in raw_candidates:
        scoring = score_specificity(c)
        if scoring["total"] < threshold:
            filtered.append({**c, "scoring": scoring, "reason": "below_threshold"})
            continue

        # Run the non-invasiveness chain. First gate to fire wins.
        gates = [
            ("frequency_cap", gate_frequency_cap(state)),
            ("channel_allow_list", gate_channel_allow_list(state, c)),
            ("user_active", gate_user_active(workspace_id, c)),
            ("sme_deference", gate_sme_deference(workspace_id, c)),
        ]
        rej_reason, genericness = gate_recent_rejection(rejection_tokens, c)
        gates.append(("recent_rejection", rej_reason))

        blocked = next((label for _, label in gates if label), None)
        if blocked:
            filtered.append(
                {
                    **c,
                    "scoring": scoring,
                    "reason": blocked,
                    "genericness_score": genericness,
                }
            )
            continue

        # Surface it. Bump the workspace counter so future candidates in
        # this scan see an updated frequency state.
        convex_post(
            "/api/suggestions/save",
            {
                "workspace_id": workspace_id,
                "user_id": c["user_id"],
                "channel": c.get("channel"),
                "candidate_text": c["text"],
                "specificity_score": scoring["total"],
                "specificity_breakdown": {
                    "named_entity_density": scoring["named_entity_density"],
                    "recurrence_count": scoring["recurrence_count"],
                    "user_attribution": scoring["user_attribution"],
                    "timing_pattern": scoring["timing_pattern"],
                },
                "genericness_score": genericness,
            },
        )
        convex_post("/api/proactive/bump_counter", {"workspace_id": workspace_id})
        # Mirror the bump locally so the next candidate in this same scan
        # sees the updated frequency-cap state and gate_frequency_cap
        # honours the cap across all surfacings, not just the first.
        state["suggestions_this_window"] = state.get("suggestions_this_window", 0) + 1
        surfaced.append({**c, "scoring": scoring, "genericness_score": genericness})

    return {
        "workspace_id": workspace_id,
        "trust_score": state["trust_score"],
        "effective_threshold": threshold,
        "candidates_generated": len(raw_candidates),
        "surfaced": len(surfaced),
        "filtered": [{"text": f["text"], "reason": f["reason"]} for f in filtered],
    }


# ---------------------------------------------------------------------------
# Arg-less cron -- fans out
# ---------------------------------------------------------------------------


@app.function(image=control_plane_image, schedule=modal.Period(hours=6), timeout=120)
def proactive_scan_all() -> dict[str, Any]:
    """Runs ~4x per day. Spawns one scan_workspace per active workspace.

    spawn_map issues all enqueues in a single batched RPC -- at 2k tenants a
    serial `for ws: scan_workspace.spawn(ws)` would do 2k sequential
    control-plane RPCs and block the cron container for the whole walk.

    Modal crons are deploy-time static + arg-less, so the standard pattern
    is "arg-less cron → fan-out to per-tenant scans". POC hardcodes the
    active list to ["dev"]; production reads it from Convex.
    """
    workspaces = get_active_workspaces()
    scan_workspace.spawn_map(workspaces)
    return {"fanned_out": len(workspaces)}


# ---------------------------------------------------------------------------
# Local entrypoint -- smoke test
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def smoke_scheduler() -> None:
    """`modal run modal/scheduler.py` runs ONE scan against the demo workspace.

    Useful for tuning the threshold and seeing which gate dropped which
    candidate. Synthetic messages live in get_recent_messages().
    """
    result = scan_workspace.remote("dev")
    print("[scheduler.smoke]", json.dumps(result, indent=2, default=str))
