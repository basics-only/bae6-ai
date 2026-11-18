import { randomUUID } from "node:crypto";

export interface Case<I = unknown, E = unknown> {
  id: string;
  input: I;
  expected: E;
}

export interface Dataset<I = unknown, E = unknown> {
  name: string;
  version: string;
  cases: Case<I, E>[];
}

export interface TaskResult<O> {
  output: O;
  cost?: number;
}

export type Task<I, O> = (
  input: I,
  ctx: { signal: AbortSignal; seed: number },
) => Promise<TaskResult<O>>;

export interface Scorer<I, E, O> {
  name: string;
  score(output: O, expected: E, input: I): number;
}

export interface RunOptions<I, E, O> {
  dataset: Dataset<I, E>;
  task: Task<I, O>;
  scorers: Scorer<I, E, O>[];
  seed: number;
  concurrency: number;
  caseTimeoutMs: number;
  signal: AbortSignal;
  runId?: string;
}

export interface CaseResult {
  id: string;
  status: "ok" | "error" | "timeout";
  scores: Record<string, number>;
  latencyMs: number;
  cost: number | null;
  error?: string;
}

export interface Report {
  runId: string;
  dataset: { name: string; version: string };
  seed: number;
  startedAt: string;
  cases: CaseResult[];
  mean: Record<string, number>;
  failures: number;
  costKnown: boolean;
  totalCost: number | null;
}

export interface Regression {
  caseId: string;
  scorer: string;
  before: number;
  after: number | "missing";
}

const MAX_CASES = 10_000;
const MAX_CONCURRENCY = 16;
const MAX_CASE_TIMEOUT_MS = 600_000;
const MAX_ERROR_CHARS = 300;

export const exactMatch = <I, E>(): Scorer<I, E, E> => ({
  name: "exact",
  score: (output, expected) =>
    JSON.stringify(output) === JSON.stringify(expected) ? 1 : 0,
});

export const schemaValid = <I, E, O>(
  validate: (output: unknown) => boolean,
): Scorer<I, E, O> => ({
  name: "schema",
  score: (output) => (validate(output) ? 1 : 0),
});

export const taskSuccess = <I, E, O>(
  succeeded: (output: O, expected: E, input: I) => boolean,
): Scorer<I, E, O> => ({
  name: "success",
  score: (output, expected, input) =>
    succeeded(output, expected, input) ? 1 : 0,
});

function checkOptions<I, E, O>(o: RunOptions<I, E, O>): void {
  const { cases } = o.dataset;
  if (
    cases.length < 1 ||
    cases.length > MAX_CASES ||
    new Set(cases.map((c) => c.id)).size !== cases.length
  ) {
    throw new RangeError("dataset needs 1..10000 cases with unique ids");
  }
  const names = o.scorers.map((s) => s.name);
  if (names.length < 1 || new Set(names).size !== names.length) {
    throw new Error("scorers must be non-empty with unique names");
  }
  if (
    !Number.isInteger(o.concurrency) ||
    o.concurrency < 1 ||
    o.concurrency > MAX_CONCURRENCY ||
    !Number.isInteger(o.caseTimeoutMs) ||
    o.caseTimeoutMs < 1 ||
    o.caseTimeoutMs > MAX_CASE_TIMEOUT_MS ||
    !Number.isSafeInteger(o.seed)
  ) {
    throw new RangeError("concurrency, timeout or seed out of bounds");
  }
}

async function withDeadline<T>(
  work: (signal: AbortSignal) => Promise<T>,
  ms: number,
  parent: AbortSignal,
): Promise<T> {
  const ctl = new AbortController();
  const signal = AbortSignal.any([parent, ctl.signal]);
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctl.abort();
  }, ms);
  let onAbort = () => {};
  const aborted = new Promise<never>((_, reject) => {
    onAbort = () => reject(new Error(timedOut ? "timeout" : "aborted"));
    signal.addEventListener("abort", onAbort, { once: true });
  });
  try {
    return await Promise.race([work(signal), aborted]);
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", onAbort);
  }
}

export async function runEval<I, E, O>(
  o: RunOptions<I, E, O>,
): Promise<Report> {
  checkOptions(o);
  const startedAt = new Date().toISOString();
  const results: CaseResult[] = new Array(o.dataset.cases.length);
  let next = 0;

  const zero = () =>
    Object.fromEntries(o.scorers.map((s) => [s.name, 0])) as Record<
      string,
      number
    >;

  async function runCase(c: Case<I, E>): Promise<CaseResult> {
    const started = Date.now();
    const finish = (r: Partial<CaseResult>): CaseResult => ({
      id: c.id,
      status: "ok",
      scores: zero(),
      latencyMs: Date.now() - started,
      cost: null,
      ...r,
    });
    try {
      const res = await withDeadline(
        (signal) => o.task(c.input, { signal, seed: o.seed }),
        o.caseTimeoutMs,
        o.signal,
      );
      const scores: Record<string, number> = {};
      for (const s of o.scorers) {
        const v = s.score(res.output, c.expected, c.input);
        if (!Number.isFinite(v) || v < 0 || v > 1) {
          throw new Error(`scorer ${s.name} returned an invalid score`);
        }
        scores[s.name] = v;
      }
      const cost =
        typeof res.cost === "number" &&
        Number.isFinite(res.cost) &&
        res.cost >= 0
          ? res.cost
          : null;
      return finish({ scores, cost });
    } catch (e) {
      const msg = e instanceof Error ? e.message : "task failed";
      return finish({
        status: msg === "timeout" ? "timeout" : "error",
        error: msg.slice(0, MAX_ERROR_CHARS),
      });
    }
  }

  async function worker(): Promise<void> {
    while (next < o.dataset.cases.length && !o.signal.aborted) {
      const i = next++;
      const c = o.dataset.cases[i];
      if (c) results[i] = await runCase(c);
    }
  }

  await Promise.all(
    Array.from(
      { length: Math.min(o.concurrency, o.dataset.cases.length) },
      worker,
    ),
  );
  if (o.signal.aborted) throw new Error("eval aborted");

  const mean: Record<string, number> = {};
  for (const s of o.scorers) {
    mean[s.name] =
      results.reduce((a, r) => a + (r.scores[s.name] ?? 0), 0) / results.length;
  }
  const costKnown = results.every((r) => r.cost !== null);
  return {
    runId: o.runId ?? randomUUID(),
    dataset: { name: o.dataset.name, version: o.dataset.version },
    seed: o.seed,
    startedAt,
    cases: results,
    mean,
    failures: results.filter((r) => r.status !== "ok").length,
    costKnown,
    totalCost: costKnown
      ? results.reduce((a, r) => a + (r.cost ?? 0), 0)
      : null,
  };
}

export function compare(
  baseline: Report,
  current: Report,
  tolerance = 0,
): Regression[] {
  if (
    baseline.dataset.name !== current.dataset.name ||
    baseline.dataset.version !== current.dataset.version
  ) {
    throw new Error("reports are for different dataset versions");
  }
  const now = new Map(current.cases.map((c) => [c.id, c]));
  const out: Regression[] = [];
  for (const b of baseline.cases) {
    const c = now.get(b.id);
    for (const [scorer, before] of Object.entries(b.scores)) {
      if (!c) out.push({ caseId: b.id, scorer, before, after: "missing" });
      else {
        const after = c.scores[scorer];
        if (after === undefined) {
          out.push({ caseId: b.id, scorer, before, after: "missing" });
        } else if (after < before - tolerance) {
          out.push({ caseId: b.id, scorer, before, after });
        }
      }
    }
  }
  return out;
}
