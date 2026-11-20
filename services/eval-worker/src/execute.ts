import { exactMatch, runEval, type Scorer } from "@bae6/ai-evals";
import type { Claimed, Outcome } from "./store.js";

export const SCORERS = ["exact", "contains"] as const;

const contains: Scorer<string, string, string> = {
  name: "contains",
  score: (output, expected) => (output.includes(expected) ? 1 : 0),
};

const scorerFor = (name: string) =>
  name === "exact" ? exactMatch<string, string>() : contains;

export interface ExecuteOptions {
  baseUrl: string;
  token: () => Promise<string>;
  maxOutputTokens: number;
  requestTimeoutMs: number;
  concurrency: number;
  caseTimeoutMs: number;
}

async function complete(
  o: ExecuteOptions,
  model: string,
  input: string,
  signal: AbortSignal,
): Promise<string> {
  const res = await fetch(`${o.baseUrl}/v1/chat/completions`, {
    method: "POST",
    redirect: "error",
    headers: {
      authorization: `Bearer ${await o.token()}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: input }],
      max_tokens: o.maxOutputTokens,
    }),
    signal: AbortSignal.any([signal, AbortSignal.timeout(o.requestTimeoutMs)]),
  });
  if (!res.ok) {
    await res.body?.cancel();
    throw new Error(`gateway returned ${res.status}`);
  }
  const body = (await res.json()) as {
    choices?: { message?: { content?: unknown } }[];
  };
  const content = body.choices?.[0]?.message?.content;
  if (typeof content !== "string")
    throw new Error("gateway returned no content");
  return content.trim();
}

export function createExecutor(o: ExecuteOptions) {
  return async (run: Claimed, signal: AbortSignal): Promise<Outcome> => {
    try {
      const report = await runEval<string, string, string>({
        dataset: {
          name: run.name,
          version: run.datasetVersion,
          cases: run.cases,
        },
        task: async (input, ctx) => ({
          output: await complete(o, run.model, input, ctx.signal),
        }),
        scorers: run.scorers.map(scorerFor),
        seed: 0,
        concurrency: o.concurrency,
        caseTimeoutMs: o.caseTimeoutMs,
        signal,
        runId: `eval_${run.id}`,
      });
      const allFailed =
        report.cases.length > 0 && report.failures === report.cases.length;
      return { stop: allFailed ? "model_error" : "complete", report };
    } catch (err) {
      if (signal.aborted) return { stop: "aborted", report: undefined };
      throw err;
    }
  };
}
