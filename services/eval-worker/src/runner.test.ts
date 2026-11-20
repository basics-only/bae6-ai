import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { after, before, test } from "node:test";
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

const aborted = { stop: "aborted", report: undefined };

async function submit() {
  await pool.query("TRUNCATE eval_runs");
  const org = `org-${randomUUID().slice(0, 8)}`;
  const { run } = await store.create(
    { organizationId: org, applicationId: "app", submitterId: "svc" },
    {
      model: "m",
      name: "n",
      datasetVersion: "1",
      cases: [{ id: "a", input: "i", expected: "e" }],
      scorers: ["exact"],
    },
    undefined,
    "a",
    10,
  );
  const id = run.id.slice("eval_".length);
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

const waitForAbort =
  (onAbort?: () => void) => (_run: Claimed, signal: AbortSignal) =>
    new Promise<typeof aborted>((resolve) => {
      signal.addEventListener("abort", () => {
        onAbort?.();
        resolve(aborted);
      });
    });

test("a queued run is executed and its result recorded", { skip }, async () => {
  const r = await submit();
  const seen: Claimed[] = [];
  const w = runner(async (run) => {
    seen.push(run);
    return {
      stop: "complete",
      report: {
        runId: "r",
        dataset: { name: "n", version: "1" },
        seed: 0,
        startedAt: new Date().toISOString(),
        cases: [],
        mean: {},
        failures: 0,
        costKnown: false,
        totalCost: null,
      },
    };
  });
  w.start();
  const done = await until(r.get, (v) => v?.status === "finished");
  await w.stop();
  assert.equal(done?.stop, "complete");
  assert.equal(seen[0]?.organizationId, r.org);
});

test(
  "cancelling a running run aborts its work and ends it as aborted",
  { skip },
  async () => {
    const r = await submit();
    let started = false;
    const wait = waitForAbort();
    const w = runner((run, signal) => {
      started = true;
      return wait(run, signal);
    });
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

test(
  "a run whose heartbeats keep failing is stopped and left for the reaper",
  { skip },
  async () => {
    const r = await submit();
    let stopped = false;
    const failing: Store = {
      ...store,
      heartbeat: async () => {
        throw new Error("db down");
      },
    };
    const w = runner(
      waitForAbort(() => {
        stopped = true;
      }),
      { store: failing, leaseMs: 150 },
    );
    w.start();
    await until(async () => stopped, Boolean);
    await new Promise((resolve) => setTimeout(resolve, 100));
    await w.stop();
    assert.equal((await r.get())?.status, "running");
  },
);

test(
  "a run interrupted by shutdown is left for another replica",
  { skip },
  async () => {
    const r = await submit();
    let started = false;
    const wait = waitForAbort();
    const w = runner((run, signal) => {
      started = true;
      return wait(run, signal);
    });
    w.start();
    await until(async () => started, Boolean);
    await w.stop();
    assert.equal((await r.get())?.status, "running");
    await pool.query(
      "UPDATE eval_runs SET lease_until = now() - interval '1 second'",
    );
    await store.reap(3);
    assert.equal((await r.get())?.status, "queued");
  },
);

test("a paused runner claims nothing new", { skip }, async () => {
  const w = runner(async () => ({ stop: "complete", report: undefined }));
  w.start();
  await new Promise((r) => setTimeout(r, 50));
  w.pause();
  await new Promise((r) => setTimeout(r, 50));
  const r = await submit();
  await new Promise((r) => setTimeout(r, 200));
  assert.equal((await r.get())?.status, "queued");
  await w.stop();
});

test("pausing lets a run already started finish", { skip }, async () => {
  const r = await submit();
  let release: () => void = () => {};
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  let started = false;
  const w = runner(async () => {
    started = true;
    await gate;
    return { stop: "complete", report: undefined };
  });
  w.start();
  await until(async () => started, Boolean);
  w.pause();
  release();
  const done = await until(r.get, (v) => v?.status === "finished");
  await w.stop();
  assert.equal(done?.stop, "complete");
});
