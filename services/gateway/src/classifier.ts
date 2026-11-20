import type { Message } from "@bae6/ai-core";

export const CLASSIFIER_TIMEOUT_MS = 2_000;
export const CLASSIFIER_MAX_TOKENS = 8;
const EXCERPT_CHARS = 1_500;

/** The gateway's own prompt over an excerpt of the last user message; null if it has no text. */
export function classifierMessages(
  messages: Message[],
  tiers: string[],
): Message[] | null {
  const last = [...messages].reverse().find((m) => m.role === "user");
  const text = (last?.content ?? [])
    .flatMap((p) => (p.type === "text" ? [p.text] : []))
    .join("\n")
    .slice(0, EXCERPT_CHARS);
  if (text.trim() === "") return null;
  return [
    {
      role: "system",
      content: [
        {
          type: "text",
          text: `Rate how demanding the user's message is to answer well. Tiers, easiest first: ${tiers.join(", ")}. Reply with exactly one tier name and nothing else.`,
        },
      ],
    },
    { role: "user", content: [{ type: "text", text }] },
  ];
}

/** Conservative upper estimate, same rule as the request's own estimate. */
export const classifierInputTokens = (messages: Message[]) =>
  messages.reduce(
    (n, m) =>
      n +
      4 +
      m.content.reduce(
        (k, p) =>
          k + (p.type === "text" ? Math.ceil((p.text.length * 4) / 3) : 0),
        0,
      ),
    0,
  );

/** The tier only if the output is exactly one tier name. */
export const parseTier = (output: string, tiers: string[]) =>
  tiers.find((t) => t === output.trim()) ?? null;
