import assert from "node:assert/strict";
import { test } from "node:test";
import {
  compare,
  type Dataset,
  exactMatch,
  type Report,
  runEval,
  schemaValid,
  type Task,
  taskSuccess,
} from "./index.js";

const dataset: Dataset<string, string> = {
  name: "upper",
  version: "1",
  cases: [
    { id: "a", input: "a", expected: "A" },
    { id: "b", input: "b", expected: "B" },
    { id: "c", input: "c", expected: "C" },
  ],
};
const upper: Task<string, string> = async (i) => ({
  output: i.toUpperCase(),
  cost: 0.5,
});
const base = {
  dataset,
  task: upper,
  scorers: [exactMatch<string, string>()],
  seed: 1,
  concurrency: 2,
  caseTimeoutMs: 1000,
  signal: new AbortController().signal,
};

test("scores every case and reports means, cost and provenance", async () => {
  const r = await runEval({ ...base, runId: "r1" });
  assert.equal(r.runId, "r1");
  assert.deepEqual(r.dataset, { name: "upper", version: "1" });
  assert.equal(r.mean.exact, 1);
  assert.equal(r.failures, 0);
  assert.equal(r.totalCost, 1.5);
});

test("passes the seed to the task", async () => {
  let seen = -1;
  await runEval({
    ...base,
    seed: 42,
    task: async (i, ctx) => {
      seen = ctx.seed;
      return { output: i.toUpperCase() };
    },
  });
  assert.equal(seen, 42);
});

test("does not treat missing cost as zero", async () => {
  const r = await runEval({
    ...base,
    task: async (i) => ({ output: i.toUpperCase() }),
  });
  assert.equal(r.costKnown, false);
  assert.equal(r.totalCost, null);
});

test("counts a failing or hung case as a zero score and keeps going", async () => {
  const r = await runEval({
    ...base,
    caseTimeoutMs: 30,
    task: async (i, ctx) => {
      if (i === "a") throw new Error("boom");
      if (i === "b") return new Promise<never>(() => {});
      return upper(i, ctx);
    },
  });
  assert.deepEqual(
    r.cases.map((c) => c.status),
    ["error", "timeout", "ok"],
  );
  assert.equal(r.failures, 2);
  assert.equal(r.cases[0]?.scores.exact, 0);
  assert.ok(Math.abs((r.mean.exact ?? 0) - 1 / 3) < 1e-9);
});

test("never runs more than the concurrency limit at once", async () => {
  let live = 0;
  let peak = 0;
  await runEval({
    ...base,
    concurrency: 2,
    task: async (i) => {
      live++;
      peak = Math.max(peak, live);
      await new Promise((r) => setTimeout(r, 10));
      live--;
      return { output: i.toUpperCase() };
    },
  });
  assert.equal(peak, 2);
});

test("rejects an out-of-range scorer result as a case error", async () => {
  const r = await runEval({
    ...base,
    scorers: [{ name: "bad", score: () => 2 }],
  });
  assert.equal(r.failures, 3);
});

test("schema and task-success scorers score validity and outcome", async () => {
  const r = await runEval({
    ...base,
    scorers: [
      schemaValid<string, string, string>((o) => typeof o === "string"),
      taskSuccess<string, string, string>((o, e) => o === e),
    ],
  });
  assert.equal(r.mean.schema, 1);
  assert.equal(r.mean.success, 1);
});

test("rejects bad datasets, scorers and bounds up front", async () => {
  const cases = dataset.cases;
  await assert.rejects(() =>
    runEval({ ...base, dataset: { ...dataset, cases: [] } }),
  );
  await assert.rejects(() =>
    runEval({
      ...base,
      dataset: { ...dataset, cases: [cases[0]!, cases[0]!] },
    }),
  );
  await assert.rejects(() => runEval({ ...base, scorers: [] }));
  await assert.rejects(() => runEval({ ...base, concurrency: 1000 }));
  await assert.rejects(() => runEval({ ...base, caseTimeoutMs: 0 }));
});

test("stops when the caller aborts", async () => {
  const ctl = new AbortController();
  const p = runEval({
    ...base,
    signal: ctl.signal,
    task: () => new Promise<never>(() => {}),
  });
  setTimeout(() => ctl.abort(), 20);
  await assert.rejects(p, /aborted/);
});

function report(scores: Record<string, number>, version = "1"): Report {
  return {
    runId: "x",
    dataset: { name: "upper", version },
    seed: 1,
    startedAt: "",
    cases: Object.entries(scores).map(([id, exact]) => ({
      id,
      status: "ok",
      scores: { exact },
      latencyMs: 0,
      cost: null,
    })),
    mean: {},
    failures: 0,
    costKnown: false,
    totalCost: null,
  };
}

test("compare reports drops and missing cases but not improvements", () => {
  const regressions = compare(
    report({ a: 1, b: 1, c: 0 }),
    report({ a: 0, c: 1 }),
  );
  assert.deepEqual(regressions, [
    { caseId: "a", scorer: "exact", before: 1, after: 0 },
    { caseId: "b", scorer: "exact", before: 1, after: "missing" },
  ]);
});

test("compare applies tolerance and refuses a different dataset version", () => {
  assert.deepEqual(compare(report({ a: 1 }), report({ a: 0.95 }), 0.1), []);
  assert.throws(() => compare(report({ a: 1 }), report({ a: 1 }, "2")));
});
