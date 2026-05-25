// convex/llm.ts
//
// ACTION — the only Convex function type that may call external APIs.
// Queries and mutations run inside a transactional, deterministic runtime;
// network egress is disallowed there. Actions run in a separate (Node-like)
// runtime where fetch() works.
//
// In the multiplayer-ai POC, the agent loop usually calls Gemini *directly
// from inside the sandbox* (the sandbox image bundles google-genai).
// This action exists for the alternative path: triggering an LLM call from
// a query/mutation context, e.g. a HTTP webhook that wants to summarise a
// thread and store the result without spinning up a sandbox.
//
// Trial talking point: knowing *when* to reach for an Action vs an in-sandbox
// call is the senior signal. Actions are cheaper (no sandbox spin-up) but
// they don't get the agent loop / progressive disclosure / code-as-tools.

import { v } from "convex/values";
import { action } from "./_generated/server";

const DEFAULT_MODEL = "gemini-3-flash-preview"; // overridable per-call

// Gemini's REST endpoint takes the model name in the URL path. Building it
// here as a function keeps the model + URL coupled.
function geminiUrl(model: string, apiKey: string): string {
  return `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
    model,
  )}:generateContent?key=${encodeURIComponent(apiKey)}`;
}

export const call_gemini = action({
  args: {
    prompt: v.string(),
    model: v.optional(v.string()),
    max_tokens: v.optional(v.number()),
    system: v.optional(v.string()),
  },
  handler: async (_ctx, args) => {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      // Stub when no key -- preserves the offline-demo capability.
      return {
        stub: true,
        content: "[stub] GEMINI_API_KEY not configured in Convex env.",
      };
    }

    const model = args.model ?? DEFAULT_MODEL;
    const body: Record<string, unknown> = {
      contents: [
        { role: "user", parts: [{ text: args.prompt }] },
      ],
      generationConfig: {
        maxOutputTokens: args.max_tokens ?? 1024,
      },
    };
    if (args.system) {
      body.systemInstruction = { parts: [{ text: args.system }] };
    }

    const res = await fetch(geminiUrl(model, apiKey), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Gemini ${res.status}: ${errText}`);
    }

    const data = (await res.json()) as {
      candidates?: Array<{
        content?: { parts?: Array<{ text?: string }> };
      }>;
    };
    // Flatten the response -- POC consumers want a single string. Gemini
    // can return multiple candidates; we always take the first.
    const parts = data.candidates?.[0]?.content?.parts ?? [];
    const text = parts
      .map((p) => p.text)
      .filter((t): t is string => typeof t === "string")
      .join("");

    return { stub: false, content: text };
  },
});
