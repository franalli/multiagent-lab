// frontend/components/ObservabilityDash.tsx
//
// NOVEL FEATURE 4 (bonus): the predictive observability dashboard.
//
// Subscribes to `predictions.latest_for_workspace`, which modal/analytics.py
// writes on its scheduled run. If no row exists yet, the component falls
// back to a clearly-labelled placeholder so the surface still renders
// during cold-start.

"use client";

import { useQuery } from "convex/react";
import { api } from "../../convex/_generated/api";

type Props = {
  workspace_id: string;
};

type Classification = "power" | "casual" | "at_risk";
type UserClassification = { user_id: string; classification: Classification; confidence: number };

// Sentence-case display labels. Mirrors the REJECTION_REASON_LABELS pattern
// from convex/constants.ts -- one place to rename without breaking JSX.
const CLASSIFICATION_LABELS: Record<Classification, string> = {
  power: "Power user",
  casual: "Casual",
  at_risk: "At risk",
};

const PLACEHOLDER = {
  adoption_health_score: 0,
  churn_risk_30d: 0,
  predicted_stale_skills: [] as string[],
  drivers: undefined as
    | { adoption_contribution: number; success_penalty: number; cost_penalty: number }
    | undefined,
  user_classifications: undefined as UserClassification[] | undefined,
};

export function ObservabilityDash({ workspace_id }: Props) {
  const latest = useQuery(api.predictions.latest_for_workspace, { workspace_id });

  // undefined = still loading; null = no row yet.
  if (latest === undefined) return <div>loading predictions…</div>;

  const p = latest ?? PLACEHOLDER;
  const empty = !latest;
  const drivers = p.drivers;
  const classifications = p.user_classifications ?? [];

  return (
    <div className="observability-dash">
      <h3>
        Workspace health · {workspace_id}
        {empty && <small style={{ marginLeft: 8 }}>(no predictions written yet)</small>}
      </h3>
      <div className="cards">
        <div className="card">
          <h5>Adoption health</h5>
          <div className="big">{p.adoption_health_score.toFixed(1)}</div>
          <small>0-100 composite from analytics pipeline</small>
        </div>
        <div className="card">
          <h5>Churn risk (30d)</h5>
          <div className="big">{(p.churn_risk_30d * 100).toFixed(1)}%</div>
          <small>probability of disengagement</small>
        </div>
      </div>

      {drivers && (
        <div className="drivers">
          <h5>What's driving the score</h5>
          <table>
            <tbody>
              <tr>
                <td>Adoption contribution</td>
                <td className="pos">+{drivers.adoption_contribution.toFixed(1)}</td>
              </tr>
              <tr>
                <td>Success-rate penalty</td>
                <td className="neg">−{drivers.success_penalty.toFixed(1)}</td>
              </tr>
              <tr>
                <td>Cost penalty</td>
                <td className="neg">−{drivers.cost_penalty.toFixed(1)}</td>
              </tr>
            </tbody>
          </table>
          <small>
            Linear-blend POC: each driver is the raw point contribution. Production swaps in
            an XGBoost ensemble; the explainability contract stays.
          </small>
        </div>
      )}

      <h5>User segmentation</h5>
      <ul>
        {classifications.length === 0 && <li>no telemetry yet</li>}
        {classifications.map((u) => (
          <li key={u.user_id} className={`user-class user-class-${u.classification}`}>
            <span className="user-id">{u.user_id}</span>
            <span className="user-cls">{CLASSIFICATION_LABELS[u.classification]}</span>
            <small>{(u.confidence * 100).toFixed(0)}% confidence</small>
          </li>
        ))}
      </ul>

      <h5>Predicted stale skills (next 7d)</h5>
      <ul>
        {p.predicted_stale_skills.length === 0 && <li>none predicted</li>}
        {p.predicted_stale_skills.map((s) => (
          <li key={s}>{s}</li>
        ))}
      </ul>
    </div>
  );
}
