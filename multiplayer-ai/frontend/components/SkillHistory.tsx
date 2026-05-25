// frontend/components/SkillHistory.tsx
//
// NOVEL FEATURE 3: the skill version admin surface.
//
// Reads from the denormalised skill_version_index. For each skill, lists
// versions newest-first with timestamp / author / change summary. Clicking
// a version exposes the unified diff against the previous version and a
// "Rollback to this version" button.
//
// Framing: git-native versioning of SKILL.md exists at the filesystem
// layer. This is the admin UX *over* that versioning -- a workspace
// admin should not have to `git log` their agent's memory.

"use client";

import { useMutation, useQuery } from "convex/react";
import { api } from "../../convex/_generated/api";
import { useState } from "react";

type Props = {
  workspace_id: string;
  skill_name: string;          // e.g. "company/SKILL.md"
};

export function SkillHistory({ workspace_id, skill_name }: Props) {
  const history = useQuery(api.skills.get_skill_history, { workspace_id, skill_name });
  const rollback = useMutation(api.skills.rollback_to_version);
  const [selected, setSelected] = useState<number | null>(null);
  const [pending, setPending] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  if (history === undefined) return <div>loading history…</div>;
  if (history.length === 0) return <div>no versions yet for {skill_name}</div>;

  const selectedRow = selected != null ? history.find((r) => r.version === selected) : null;
  const latestVersion = history[0].version;
  const isLatest = selectedRow?.version === latestVersion;

  async function onRollback(target_version: number) {
    setPending(true);
    setToast(null);
    try {
      const { new_version } = await rollback({ workspace_id, skill_name, target_version });
      setSelected(new_version);
      setToast(`restored as v${new_version}`);
    } catch (err) {
      setToast(`rollback failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="skill-history">
      <div className="timeline">
        <h4>{skill_name}</h4>
        <ul>
          {history.map((r) => (
            <li
              key={r._id}
              onClick={() => setSelected(r.version)}
              className={selected === r.version ? "selected" : ""}
            >
              <span className="version">v{r.version}</span>
              <span className="actor">{r.modified_by}</span>
              <span className="summary">{r.change_summary}</span>
              <span className="when">
                {new Date(r.modified_at).toLocaleString()}
              </span>
            </li>
          ))}
        </ul>
      </div>

      {selectedRow && (
        <div className="detail">
          <h5>v{selectedRow.version} -- {selectedRow.change_summary}</h5>
          <details open>
            <summary>diff vs previous</summary>
            <pre>{selectedRow.diff || "(no diff stored)"}</pre>
          </details>
          <details>
            <summary>full content</summary>
            <pre>{selectedRow.content}</pre>
          </details>
          <button
            type="button"
            disabled={pending || isLatest}
            onClick={() => onRollback(selectedRow.version)}
            title={isLatest ? "already on this version" : `restore v${selectedRow.version} as a new row`}
          >
            {pending ? "rolling back…" : `Rollback to v${selectedRow.version}`}
          </button>
          {toast && <div className="toast">{toast}</div>}
        </div>
      )}
    </div>
  );
}
