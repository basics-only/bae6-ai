import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { after, before, test } from "node:test";
import type { Limits, RunResult } from "@bae6/ai-agents";
import pg from "pg";
import { migrate } from "./migrate.js";
import { createRunner, type RunnerOptions } from "./runner.js";
import { type Claimed, createStore, type Store } from "./store.js";

const url = process.env.BAE6_DATABASE_URL;
const skip = url ? false : "BAE6_DATABASE_URL not set";
let pool: pg.Pool;
let store: Store;

before(async () => {
  if (!url) return;
  pool = new pg.Pool({ connectionString: url, max: 5 });
  const c = await pool.connect();
  try {
    await migrate(c);
  } finally {
    c.release();
  }
  store = createStore(pool);
});

after(() => pool?.end());

const limits: Limits = {
  maxSteps: 5,
  maxTotalTokens: 1000,
  timeoutMs: 5000,
  toolTimeoutMs: 1000,
  maxToolOutputChars: 1000,
  maxHistoryChars: 10_000,
};
const finished = (text: string): RunResult => ({
  stop: "final",
  text,
  tokens: 3,
  receipts: [],
});

async function submit() {
  await pool.query("TRUNCATE agent_runs");
  const org = `org-${randomUUID().slice(0, 8)}`;
  const { run } = await store.create(
    { organizationId: org, applicationId: "app", submitterId: "svc" },
    { model: "m", task: "t", tools: [], limits },
    undefined,
    "a",
    10,
  );
  const id = run.id.slice("run_".length);
  return { org, id, get: () => store.get(org, id) };
}

function runner(
  execute: RunnerOptions["execute"],
  extra: Partial<RunnerOptions> = {},
) {
  return createRunner({
    store,
    execute,
    concurrency: 1,
    leaseMs: 2000,
    heartbeatMs: 30,
    pollMs: 10,
    maxAttempts: 3,
    maxResultChars: 1000,
    ...extra,
  });
}

async function until<T>(read: () => Promise<T>, done: (v: T) => boolean) {
  const end = Date.now() + 5000;
  for (;;) {
    const v = await read();
    if (done(v)) return v;
    if (Date.now() > end) assert.fail(`timed out: ${JSON.stringify(v)}`);
    await new Promise((r) => setTimeout(r, 20));
  }
}

test("a queued run is executed and its result recorded", { skip }, async () => {
  const r = await submit();
  const seen: Claimed[] = [];
  const w = runner(async (run) => {
    seen.push(run);
    return finished("hello");
  });
  w.start();
  const done = await until(r.get, (v) => v?.status === "finished");
  await w.stop();
  assert.deepEqual(
    [done?.stop, done?.text, done?.tokens],
    ["final", "hello", 3],
  );
  assert.equal(seen[0]?.organizationId, r.org);
});

test(
  "cancelling a running run aborts its work and ends it as aborted",
  { skip },
  async () => {
    const r = await submit();
    let started = false;
    const w = runner(
      (_run, signal) =>
        new Promise((resolve) => {
          started = true;
          signal.addEventListener("abort", () =>
            resolve({ stop: "aborted", tokens: 1, receipts: [] }),
          );
        }),
    );
    w.start();
    await until(async () => started, Boolean);
    await store.cancel(r.org, r.id);
    const done = await until(r.get, (v) => v?.status === "finished");
    await w.stop();
    assert.equal(done?.stop, "aborted");
  },
);

test("an execution that throws finishes as model_error", { skip }, async () => {
  const r = await submit();
  const w = runner(async () => {
    throw new Error("boom");
  });
  w.start();
  const done = await until(r.get, (v) => v?.status === "finished");
  await w.stop();
  assert.equal(done?.stop, "model_error");
});

test("a result longer than the bound is cut", { skip }, async () => {
  const r = await submit();
  const w = runner(async () => finished("x".repeat(5000)), {
    maxResultChars: 100,
  });
  w.start();
  const done = await until(r.get, (v) => v?.status === "finished");
  await w.stop();
  assert.equal(done?.text?.length, 100);
});

test(
  "a run interrupted by shutdown is left for another replica",
  { skip },
  async () => {
    const r = await submit();
    let started = false;
    const w = runner(
      (_run, signal) =>
        new Promise((resolve) => {
          started = true;
          signal.addEventListener("abort", () =>
            resolve({ stop: "aborted", tokens: 0, receipts: [] }),
          );
        }),
    );
    w.start();
    await until(async () => started, Boolean);
    await w.stop();
    assert.equal((await r.get())?.status, "running");
    await pool.query(
      "UPDATE agent_runs SET lease_until = now() - interval '1 second'",
    );
    await store.reap(3);
    assert.equal((await r.get())?.status, "queued");
  },
);
