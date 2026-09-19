# modal/analytics.py
#
# NOVEL FEATURE 4 (bonus / cuttable): predictive observability.
#
# Pipeline:
#   telemetry  -> features  -> XGBoost ensembles  -> predictions in Convex  -> UI
#
# The trial scope is to demonstrate the PATTERN, not train production models.
# This file therefore:
#   * sketches the feature engineering compute (real, runnable)
#   * leaves the XGBoost model as a placeholder stub with a clear "swap in
#     a trained model here" comment
#   * writes the predictions to Convex in the same shape the UI expects, so
#     the ObservabilityDash.tsx surface has data to render
#
# Why this is bonus, not core: proactive (NF2) is the core demo piece.
# Observability is adjacent. Ship if features 1-3 are solid.

from __future__ import annotations

import math
import statistics
import time
from collections import Counter
from typing import Any

import modal

from common import (  # sibling import
    app,
    control_plane_image,
    convex_post,
    convex_query,
    secrets,
)

# ---------------------------------------------------------------------------
# Helpers (POC stubs -- production reads from Convex / Modal logs)
# ---------------------------------------------------------------------------


def fetch_agent_runs(workspace_id: str, days: int = 7) -> list[dict[str, Any]]:
    """Pull recent agent_runs rows. Reads from Convex first; falls back to
    synthetic data when the workspace has no telemetry yet (cold-start).

    `skill_reads` is included so skill_diversity_index has something to
    chew on. Once sandbox_agent has written a few real rows, this returns
    actual workspace telemetry.
    """
    since_ms = int((time.time() - days * 86400) * 1000)
    rows = convex_query(
        "/api/agent_runs/recent",
        {"workspace_id": workspace_id, "since_ms": since_ms, "limit": 500},
    )
    if rows:
        return rows

    # Cold-start fallback so the pipeline + UI render with believable values
    # before the agent has run enough turns to populate agent_runs.
    now = time.time()
    skill_pool = [
        "company/SKILL.md",
        "team/example/SKILL.md",
        "workflow/weekly_recap/SKILL.md",
        "integration/github/SKILL.md",
    ]
    return [
        {
            "user_id": ["U01A", "U01B", "U01C"][idx % 3],
            "started_at": now - i * 3600,
            "duration_ms": 1500 + i * 50,
            "tool_calls_count": (i % 3),
            "exec_success": (i % 4) != 0,
            "recovery_attempts": (i % 4 == 0),
            # Synthetic cost mirrors the per-call cost in common.py so the
            # cold-start numbers don't lie about what a real run would cost.
            "cost_estimate_usd": 0.012 + 0.002 * (i % 3),
            "skill_reads": [skill_pool[idx % 4]] if idx % 3 else skill_pool[:2],
        }
        for idx, i in enumerate(range(0, days * 24, 6))
    ]


def group_runs_by_user(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """In-memory grouping by user_id. Used by both the workspace classifier
    and any per-user drilldown that operates on the same pre-fetched batch.

    `runs.recent_for_user` exists in Convex for the per-user admin page
    (one user at a time, no workspace pull); this helper is the right
    primitive for the workspace-wide aggregate -- a single pull + group
    is cheaper than N round-trips."""
    by_user: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        by_user.setdefault(row.get("user_id", ""), []).append(row)
    return by_user


# ---------------------------------------------------------------------------
# Per-user classification (power / casual / at_risk)
# ---------------------------------------------------------------------------


# Engagement-score band boundaries. Bands: [0, AT_RISK_TOP) = at_risk,
# [AT_RISK_TOP, POWER_BOTTOM) = casual, [POWER_BOTTOM, 1] = power.
# Anchors are band MIDPOINTS so confidence is "how far inside the band
# we are," not "how far from the lower boundary."
_AT_RISK_TOP = 0.33
_POWER_BOTTOM = 0.66
# Half the width of the casual band -- derived so a future change to the
# band boundaries can't silently drift the confidence scale.
_BAND_HALFWIDTH = (_POWER_BOTTOM - _AT_RISK_TOP) / 2
# Maps midpoint→0, band-edge→±1 distance contribution; clamps at 1.0.
_CONFIDENCE_SCALE = 1.0 / _BAND_HALFWIDTH


def classify_user(runs: list[dict[str, Any]], *, days: int = 7) -> dict[str, Any]:
    """Transparent rule-based classifier. Power = high engagement + clean
    execution; casual = moderate; at_risk = low or failing.

    Production swaps in xgboost.Booster.predict against the same shape;
    keep the {classification, confidence} contract.
    """
    if not runs:
        # No telemetry, no evidence -- confidence is honestly zero. The
        # at_risk label is the conservative default when we don't know.
        return {"classification": "at_risk", "confidence": 0.0}
    runs_per_day = len(runs) / max(days, 1)
    success_rate = sum(1 for r in runs if r.get("exec_success")) / len(runs)

    # Composite engagement in [0, 1]. 60/40 weighting volume vs success.
    engagement = min(runs_per_day / 5.0, 1.0) * 0.6 + success_rate * 0.4

    if engagement >= _POWER_BOTTOM:
        cls, midpoint = "power", (_POWER_BOTTOM + 1.0) / 2.0
    elif engagement >= _AT_RISK_TOP:
        cls, midpoint = "casual", (_AT_RISK_TOP + _POWER_BOTTOM) / 2.0
    else:
        cls, midpoint = "at_risk", _AT_RISK_TOP / 2.0
    confidence = round(min(abs(engagement - midpoint) * _CONFIDENCE_SCALE, 1.0), 3)
    return {"classification": cls, "confidence": confidence}


def classify_all_users(runs: list[dict[str, Any]], *, days: int = 7) -> list[dict[str, Any]]:
    """Per-user segmentation list -- the workspace_predictions.user_classifications
    payload. Operates on a pre-fetched batch of runs (no Convex round-trip)
    so the workspace-level prediction does ONE pull, not two."""
    out: list[dict[str, Any]] = []
    for user_id, user_runs in group_runs_by_user(runs).items():
        if not user_id:
            continue
        cls = classify_user(user_runs, days=days)
        out.append({"user_id": user_id, **cls})
    return out


def count_skills_unused(workspace_id: str, days: int = 7) -> int:
    """Stale skill count = skills declared in the workspace volume that
    received no reads in the lookback window. POC: returns a constant
    against the demo volume; production diffs skill_version_index reads
    vs the skill tree."""
    return 2


def shannon_entropy(items: list[str]) -> float:
    """Shannon entropy in bits over the distribution of `items`. Used as
    skill_diversity_index per spec line 1311. 0.0 = single skill dominates;
    higher = reads spread evenly across many skills."""
    if not items:
        return 0.0
    counts = Counter(items)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def fetch_accept_rate(workspace_id: str, days: int = 7) -> float:
    """Suggestion accept rate over the lookback window. Returns 0.0 on any
    failure -- POC accepts a stale-or-zero rate over a broken feature
    compute. Production would alert."""
    cutoff_ms = int((time.time() - days * 86400) * 1000)
    value = convex_query(
        "/api/suggestions/accept_rate",
        {"workspace_id": workspace_id, "since_ms": cutoff_ms},
    )
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Feature compute
# ---------------------------------------------------------------------------


def compute_features(workspace_id: str, *, runs: list[dict[str, Any]] | None = None, days: int = 7) -> dict[str, Any]:
    """Compute the workspace feature row. Accepts pre-fetched `runs` so
    the caller can share one HTTP pull across feature compute + user
    classification."""
    telemetry = runs if runs is not None else fetch_agent_runs(workspace_id, days=days)
    if not telemetry:
        return {
            "workspace_id": workspace_id,
            "computed_at": time.time(),
            "no_data": True,
        }

    runs_per_day = len(telemetry) / float(days)
    distinct_users = len({r["user_id"] for r in telemetry})
    success_rate = sum(1 for r in telemetry if r["exec_success"]) / len(telemetry)
    cost_per_run = statistics.mean(r["cost_estimate_usd"] for r in telemetry)
    all_skill_reads = [s for r in telemetry for s in r.get("skill_reads", [])]
    diversity = shannon_entropy(all_skill_reads)

    return {
        "workspace_id": workspace_id,
        "computed_at": time.time(),
        "agent_runs_per_day": round(runs_per_day, 3),
        "active_users_count": distinct_users,
        "code_exec_success_rate": round(success_rate, 3),
        "suggestion_accept_rate": round(fetch_accept_rate(workspace_id, days=days), 3),
        "cost_per_run_estimate": round(cost_per_run, 4),
        "skill_diversity_index": round(diversity, 3),
        "stale_skills_count": count_skills_unused(workspace_id, days=days),
    }


# ---------------------------------------------------------------------------
# Predictions (XGBoost placeholder)
# ---------------------------------------------------------------------------


def predict_health(features: dict[str, Any]) -> dict[str, Any]:
    """In production this is an XGBoost ensemble trained on labelled
    workspace-week-outcomes. The DPR-derived pattern from the prep doc.

    For the POC we compose a transparent linear blend so the demo audience
    can see what's driving the score. The `drivers` block is the
    point-by-point contribution of each input -- that's the NF4 demo line
    "explainability matters" made concrete. Swap in `xgboost.Booster.predict`
    when you've got a model artifact; keep the drivers contract.
    """
    adoption = 100.0 * min(features.get("agent_runs_per_day", 0) / 8.0, 1.0)
    # Penalise low success rate and high cost.
    success_penalty = 30.0 * (1.0 - features.get("code_exec_success_rate", 0))
    cost_penalty = 20.0 * min(features.get("cost_per_run_estimate", 0) / 0.05, 1.0)
    health = max(0.0, adoption - success_penalty - cost_penalty)

    return {
        "workspace_id": features["workspace_id"],
        "computed_at": time.time(),
        "adoption_health_score": round(health, 2),
        "churn_risk_30d": round(max(0.0, 1.0 - health / 100.0), 3),
        "predicted_stale_skills": [],  # POC stub; real model would emit a ranked list
        "drivers": {
            "adoption_contribution": round(adoption, 2),
            "success_penalty": round(success_penalty, 2),
            "cost_penalty": round(cost_penalty, 2),
        },
    }


# ---------------------------------------------------------------------------
# Modal Functions
# ---------------------------------------------------------------------------


@app.function(image=control_plane_image, secrets=secrets(), timeout=120)
def compute_workspace_features(workspace_id: str) -> dict[str, Any]:
    features = compute_features(workspace_id)
    # Production would write to `workspace_features`. We skip the explicit
    # Convex write here to keep the smoke test self-contained.
    return features


@app.function(image=control_plane_image, secrets=secrets(), timeout=120)
def predict_workspace_health(workspace_id: str) -> dict[str, Any]:
    # ONE Convex pull feeds both workspace features and the per-user
    # classifier -- avoids two independent fetches of the same data.
    runs = fetch_agent_runs(workspace_id, days=7)
    features = compute_features(workspace_id, runs=runs)
    predictions = predict_health(features)
    user_classifications = classify_all_users(runs)
    convex_post(
        "/api/predictions/save",
        {
            "workspace_id": predictions["workspace_id"],
            "adoption_health_score": predictions["adoption_health_score"],
            "churn_risk_30d": predictions["churn_risk_30d"],
            "predicted_stale_skills": predictions["predicted_stale_skills"],
            "drivers": predictions["drivers"],
            "user_classifications": user_classifications,
        },
    )
    return {**predictions, "user_classifications": user_classifications}


@app.function(image=control_plane_image, schedule=modal.Period(hours=6), timeout=120)
def predict_all_workspaces() -> dict[str, Any]:
    """Arg-less cron. Same fan-out pattern as the proactive scheduler.

    spawn_map batches the enqueues so a 2k-workspace tick isn't 2k serial
    control-plane RPCs.
    """
    # POC: hard-coded list. Production reads active workspaces from Convex.
    workspaces = ["dev"]
    predict_workspace_health.spawn_map(workspaces)
    return {"fanned_out": len(workspaces)}


# ---------------------------------------------------------------------------
# Local entrypoint -- smoke test
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def smoke_analytics() -> None:
    """`modal run modal/analytics.py` runs the feature + prediction pipeline
    once against the demo workspace and prints the result."""
    result = predict_workspace_health.remote("dev")
    print("[analytics.smoke]", result)
