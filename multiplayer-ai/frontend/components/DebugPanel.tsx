// frontend/components/DebugPanel.tsx
//
// NOVEL FEATURE 1: the per-attempt code-execution debug surface.
//
// For any tool_calls row, we list debug_traces in order and render:
//   * the script as run on each attempt
//   * stdout, stderr, exit code
//   * a unified diff between consecutive attempts (so the recovery LLM's
//     changes are visible at a glance)
//
// During the demo this is what you click into after the auto-recovery
// kicks in -- "look, attempt 1 failed, the LLM changed THIS line, attempt
// 2 passed."

"use client";

import { useQuery } from "convex/react";
import { api } from "../../convex/_generated/api";
import type { Id } from "../../convex/_generated/dataModel";

type Props = {
  tool_call_id: Id<"tool_calls">;
};


export function DebugPanel({ tool_call_id }: Props) {
  const traces = useQuery(api.messages.get_debug_traces, { tool_call_id });

  if (traces === undefined) return <div>loading traces…</div>;
  if (traces.length === 0) return <div>no execution attempts yet</div>;

  return (
    <div className="debug-panel">
      {traces.map((t, idx) => {
        const prev = idx > 0 ? traces[idx - 1] : null;
        const diff = prev ? simpleLineDiff(prev.script, t.script) : null;
        return (
          <div key={t._id} className={`attempt attempt-${t.exit_code === 0 ? "ok" : "fail"}`}>
            <h5>
              Attempt {t.attempt} · exit {t.exit_code}
            </h5>
            <pre className="script">{t.script}</pre>
            {diff && (
              <details className="diff" open>
                <summary>diff vs attempt {t.attempt - 1}</summary>
                <pre>{diff}</pre>
              </details>
            )}
            <details className="output">
              <summary>stdout</summary>
              <pre>{t.stdout || "(empty)"}</pre>
            </details>
            <details className="output">
              <summary>stderr</summary>
              <pre>{t.stderr || "(empty)"}</pre>
            </details>
          </div>
        );
      })}
    </div>
  );
}

// Tiny line-level diff that returns unified-diff-shaped text. The intent is
// readability in the panel, not roundtrippable diffs -- swap in `diff` or
// `jsdiff` packages when you want patch fidelity.
function simpleLineDiff(prev: string, next: string): string {
  const a = prev.split("\n");
  const b = next.split("\n");
  const out: string[] = [];
  const max = Math.max(a.length, b.length);
  for (let i = 0; i < max; i++) {
    if (a[i] === b[i]) {
      out.push(`  ${a[i] ?? ""}`);
    } else {
      if (a[i] !== undefined) out.push(`- ${a[i]}`);
      if (b[i] !== undefined) out.push(`+ ${b[i]}`);
    }
  }
  return out.join("\n");
}
