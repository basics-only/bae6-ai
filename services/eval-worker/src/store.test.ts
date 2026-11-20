import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { after, before, test } from "node:test";
import type { Report } from "@bae6/ai-evals";
import pg from "pg";
import { migrate } from "./migrate.js";
import {
  createStore,
  IdempotencyMismatch,
  QueueFull,
  type Store,
} from "./store.js";

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

const run = {
  model: "m",
  name: "smoke",
  datasetVersion: "1",
  cases: [
    { id: "a", input: "1+1", expected: "2" },
    { id: "b", input: "2+2", expected: "4" },
  ],
  scorers: ["exact"],
};
const who = (organizationId: string, submitterId = "svc") => ({
  organizationId,
  applicationId: "app",
  submitterId,
});
const org = () => `org-${randomUUID().slice(0, 8)}`;
const uuid = (id: string) => id.slice("eval_".length);
const report: Report = {
  runId: "r",
  dataset: { name: "smoke", version: "1" },
  seed: 0,
  startedAt: new Date().toISOString(),
  cases: [
    { id: "a", status: "ok", scores: { exact: 1 }, latencyMs: 5, cost: null },
    { id: "b", status: "error", scores: {}, latencyMs: 5, cost: null },
  ],
  mean: { exact: 1 },
  failures: 1,
  costKnown: false,
  totalCost: null,
};
const outcome = { stop: "complete", report };
const clear = () => pool.query("TRUNCATE eval_runs");

test(
  "a repeated idempotency key returns the first run, and another body is refused",
  { skip },
  async () => {
    const o = org();
    const first = await store.create(who(o), run, "key-00001", "a", 10);
    const again = await store.create(who(o), run, "key-00001", "a", 10);
    assert.equal(again.replayed, true);
    assert.equal(again.run.id, first.run.id);
    await assert.rejects(
      store.create(who(o), run, "key-00001", "b", 10),
      IdempotencyMismatch,
    );
    const other = await store.create(
      who(o, "someone"),
      run,
      "key-00001",
      "a",
      10,
    );
    assert.notEqual(other.run.id, first.run.id);
  },
);

test(
  "an organization at its unfinished limit cannot queue more, and others can",
  { skip },
  async () => {
    const o = org();
    await store.create(who(o), run, undefined, "a", 2);
    await store.create(who(o), run, undefined, "a", 2);
    await assert.rejects(
      store.create(who(o), run, undefined, "a", 2),
      QueueFull,
    );
    await store.create(who(org()), run, undefined, "a", 2);
  },
);

test(
  "concurrent submits cannot exceed the unfinished limit",
  { skip },
  async () => {
    const o = org();
    const results = await Promise.allSettled(
      Array.from({ length: 8 }, () =>
        store.create(who(o), run, undefined, "a", 3),
      ),
    );
    assert.equal(results.filter((r) => r.status === "fulfilled").length, 3);
  },
);

test(
  "another organization cannot read, list or cancel a run",
  { skip },
  async () => {
    const a = org();
    const { run: made } = await store.create(who(a), run, undefined, "a", 10);
    const b = org();
    assert.equal(await store.get(b, uuid(made.id)), null);
    assert.equal(await store.cancel(b, uuid(made.id)), null);
    assert.deepEqual((await store.list(b, { limit: 10 })).data, []);
    assert.equal((await store.get(a, uuid(made.id)))?.status, "queued");
  },
);

test("a list pages newest first", { skip }, async () => {
  const o = org();
  const ids = [];
  for (let i = 0; i < 3; i++)
    ids.push((await store.create(who(o), run, undefined, "a", 10)).run.id);
  const first = await store.list(o, { limit: 2 });
  assert.deepEqual(
    first.data.map((r) => r.id),
    [ids[2], ids[1]],
  );
  assert.ok(first.nextCursor);
  const second = await store.list(o, {
    limit: 2,
    cursor: first.nextCursor as string,
  });
  assert.deepEqual(
    second.data.map((r) => r.id),
    [ids[0]],
  );
  assert.equal(second.nextCursor, null);
});

test(
  "a finished run has a summary and a report, and a list has no report",
  { skip },
  async () => {
    await clear();
    const o = org();
    await store.create(who(o), run, undefined, "a", 10);
    const claimed = await store.claim(60_000);
    assert.ok(claimed);
    assert.equal(
      await store.finish(claimed.id, claimed.attempt, outcome),
      true,
    );
    const done = await store.get(o, claimed.id);
    assert.deepEqual(
      [done?.status, done?.stop, done?.caseCount, done?.summary],
      [
        "finished",
        "complete",
        2,
        { cases: 2, failures: 1, mean: { exact: 1 } },
      ],
    );
    assert.deepEqual(done?.report, report);
    const listed = (await store.list(o, { limit: 10 })).data[0];
    assert.equal(listed?.report, null);
    assert.equal(listed?.summary?.failures, 1);
  },
);

test(
  "concurrent claims never hand one run to two workers",
  { skip },
  async () => {
    await clear();
    const o = org();
    for (let i = 0; i < 4; i++)
      await store.create(who(o), run, undefined, "a", 10);
    const claimed = await Promise.all(
      Array.from({ length: 8 }, () => store.claim(60_000)),
    );
    const ids = claimed.filter((c) => c !== null).map((c) => c?.id);
    assert.equal(ids.length, 4);
    assert.equal(new Set(ids).size, 4);
  },
);

test("a cancelled queued run is never claimed", { skip }, async () => {
  await clear();
  const o = org();
  const { run: made } = await store.create(who(o), run, undefined, "a", 10);
  assert.equal((await store.cancel(o, uuid(made.id)))?.status, "cancelled");
  assert.equal(await store.claim(60_000), null);
});

test("cancelling a running run flags it for its worker", { skip }, async () => {
  await clear();
  const o = org();
  const { run: made } = await store.create(who(o), run, undefined, "a", 10);
  const claimed = await store.claim(60_000);
  assert.ok(claimed);
  assert.equal((await store.cancel(o, uuid(made.id)))?.status, "running");
  assert.deepEqual(await store.heartbeat(claimed.id, claimed.attempt, 60_000), {
    cancelRequested: true,
  });
});

test(
  "a result from an attempt that lost the run is discarded",
  { skip },
  async () => {
    await clear();
    const o = org();
    await store.create(who(o), run, undefined, "a", 10);
    const first = await store.claim(1);
    assert.ok(first);
    await new Promise((r) => setTimeout(r, 20));
    assert.equal(await store.reap(3), 1);
    const second = await store.claim(60_000);
    assert.ok(second);
    assert.equal(second.attempt, 2);
    assert.equal(await store.heartbeat(first.id, first.attempt, 60_000), null);
    assert.equal(await store.finish(first.id, first.attempt, outcome), false);
    assert.equal(await store.finish(second.id, second.attempt, outcome), true);
  },
);

test(
  "a run whose attempts are used up finishes as abandoned",
  { skip },
  async () => {
    await clear();
    const o = org();
    const { run: made } = await store.create(who(o), run, undefined, "a", 10);
    for (let attempt = 1; attempt <= 2; attempt++) {
      assert.ok(await store.claim(1));
      await new Promise((r) => setTimeout(r, 20));
      await store.reap(2);
    }
    const done = await store.get(o, uuid(made.id));
    assert.deepEqual([done?.status, done?.stop], ["finished", "abandoned"]);
    assert.equal(await store.claim(60_000), null);
  },
);

test(
  "a cancelled run whose worker died finishes as aborted, not retried",
  { skip },
  async () => {
    await clear();
    const o = org();
    const { run: made } = await store.create(who(o), run, undefined, "a", 10);
    assert.ok(await store.claim(1));
    await store.cancel(o, uuid(made.id));
    await new Promise((r) => setTimeout(r, 20));
    await store.reap(3);
    const done = await store.get(o, uuid(made.id));
    assert.deepEqual([done?.status, done?.stop], ["finished", "aborted"]);
  },
);
