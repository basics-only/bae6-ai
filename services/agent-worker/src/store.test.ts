import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { after, before, test } from "node:test";
import type { Limits } from "@bae6/ai-agents";
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

const limits: Limits = {
  maxSteps: 5,
  maxTotalTokens: 1000,
  timeoutMs: 5000,
  toolTimeoutMs: 1000,
  maxToolOutputChars: 1000,
  maxHistoryChars: 10_000,
};
const run = { model: "m", task: "t", tools: [], limits };
const who = (organizationId: string, submitterId = "svc") => ({
  organizationId,
  applicationId: "app",
  submitterId,
});
const org = () => `org-${randomUUID().slice(0, 8)}`;
const uuid = (id: string) => id.slice("run_".length);
const outcome = { stop: "final", text: "done", tokens: 7, receipts: [] };
const clear = () => pool.query("TRUNCATE agent_runs");

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

test("a list pages newest first without receipts", { skip }, async () => {
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
    const done = await store.get(o, second.id);
    assert.deepEqual(
      [done?.status, done?.stop, done?.text],
      ["finished", "final", "done"],
    );
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
