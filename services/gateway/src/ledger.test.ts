import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { after, before, test } from "node:test";
import pg from "pg";
import { createLedger, type Ledger } from "./ledger.js";
import { migrate } from "./migrate.js";

const url = process.env.BAE6_DATABASE_URL;
const skip = url ? false : "BAE6_DATABASE_URL not set";
let pool: pg.Pool;
let ledger: Ledger;

before(async () => {
  if (!url) return;
  pool = new pg.Pool({ connectionString: url, max: 20 });
  const c = await pool.connect();
  try {
    await migrate(c);
  } finally {
    c.release();
  }
  ledger = createLedger(pool, { priceBookVersion: "test-v1" });
  // 1 per Mtok in, 2 per Mtok out: 1M in + 0 out costs exactly 1.
  await ledger.publishPrices({ m: { inputPerMTok: "1", outputPerMTok: "2" } });
});

after(() => pool?.end());

const caller = () => ({
  organizationId: `org-${randomUUID().slice(0, 8)}`,
  applicationId: "app",
  subjectId: "svc",
});

const reserve = (c: ReturnType<typeof caller>, leaseMs = 60_000) =>
  ledger.reserve({
    requestId: `led_${randomUUID()}`,
    caller: c,
    limit: "5",
    provider: "bedrock",
    modelId: "m",
    inputTokens: 1_000_000,
    maxOutputTokens: 0,
    leaseMs,
  });

test("concurrent reserves never exceed the limit", { skip }, async () => {
  const c = caller();
  const held = await Promise.all(Array.from({ length: 20 }, () => reserve(c)));
  assert.equal(held.filter((e) => e !== null).length, 5);
  const b = await ledger.budget(c.organizationId);
  assert.equal(Number(b.reserved), 5);
});

test(
  "finalize is applied once and uncertain is charged at the estimate",
  { skip },
  async () => {
    const c = caller();
    const id = `led_${randomUUID()}`;
    const estimate = await ledger.reserve({
      requestId: id,
      caller: c,
      limit: "5",
      provider: "bedrock",
      modelId: "m",
      inputTokens: 1_000_000,
      maxOutputTokens: 500_000,
      leaseMs: 60_000,
    });
    assert.equal(Number(estimate), 2);
    assert.equal(await ledger.finalize(id, { kind: "uncertain" }), true);
    assert.equal(
      await ledger.finalize(id, {
        kind: "measured",
        usage: { inputTokens: 1, outputTokens: 1 },
      }),
      false,
    );
    const b = await ledger.budget(c.organizationId);
    assert.deepEqual([Number(b.spent), Number(b.reserved)], [2, 0]);
  },
);

test(
  "measured cost replaces the estimate; failed costs nothing",
  { skip },
  async () => {
    const c = caller();
    const ids = [`led_${randomUUID()}`, `led_${randomUUID()}`];
    for (const id of ids) {
      await ledger.reserve({
        requestId: id,
        caller: c,
        limit: "5",
        provider: "bedrock",
        modelId: "m",
        inputTokens: 1_000_000,
        maxOutputTokens: 0,
        leaseMs: 60_000,
      });
    }
    await ledger.finalize(ids[0] as string, {
      kind: "measured",
      usage: { inputTokens: 500_000, outputTokens: 0 },
    });
    await ledger.finalize(ids[1] as string, { kind: "failed" });
    const b = await ledger.budget(c.organizationId);
    assert.deepEqual([Number(b.spent), Number(b.reserved)], [0.5, 0]);
  },
);

test(
  "the reconciler charges expired leases as uncertain",
  { skip },
  async () => {
    const c = caller();
    assert.notEqual(await reserve(c, -1000), null);
    assert.ok((await ledger.reconcile()) >= 1);
    const b = await ledger.budget(c.organizationId);
    assert.deepEqual([Number(b.spent), Number(b.reserved)], [1, 0]);
  },
);

test(
  "a route's decision survives both finalize and the reconciler, and a replay names the chosen model",
  { skip },
  async () => {
    const c = caller();
    const route = { id: "auto", decision: "rule:0" };
    const settled = `led_${randomUUID()}`;
    const abandoned = `led_${randomUUID()}`;
    for (const [requestId, leaseMs] of [
      [settled, 60_000],
      [abandoned, -1000],
    ] as const) {
      assert.notEqual(
        await ledger.reserve({
          requestId,
          caller: c,
          limit: "5",
          provider: "bedrock",
          modelId: "m",
          inputTokens: 1,
          maxOutputTokens: 0,
          leaseMs,
          route,
        }),
        null,
      );
    }
    await ledger.finalize(settled, { kind: "failed" });
    while ((await ledger.reconcile()) > 0);
    const { rows } = await pool.query(
      "SELECT route_id, decision FROM usage_events WHERE request_id = ANY($1)",
      [[settled, abandoned]],
    );
    assert.deepEqual(rows, [
      { route_id: "auto", decision: "rule:0" },
      { route_id: "auto", decision: "rule:0" },
    ]);

    await ledger.claim(c, "key-route-1", "h1", `led_${randomUUID()}`);
    await ledger.choose(c, "key-route-1", "m");
    await ledger.complete(c, "key-route-1", 200, { ok: true });
    const replay = await ledger.claim(
      c,
      "key-route-1",
      "h1",
      `led_${randomUUID()}`,
    );
    assert.equal(replay.state === "replay" && replay.modelId, "m");
  },
);

test(
  "an idempotency key replays, and refuses a different request",
  { skip },
  async () => {
    const c = caller();
    const rid = () => `led_${randomUUID()}`;
    assert.equal(
      (await ledger.claim(c, "key-00001", "h1", rid())).state,
      "new",
    );
    assert.equal(
      (await ledger.claim(c, "key-00001", "h1", rid())).state,
      "in_progress",
    );
    await ledger.complete(c, "key-00001", 200, { ok: true });
    assert.deepEqual(await ledger.claim(c, "key-00001", "h1", rid()), {
      state: "replay",
      status: 200,
      body: { ok: true },
      modelId: null,
    });
    assert.equal(
      (await ledger.claim(c, "key-00001", "h2", rid())).state,
      "mismatch",
    );
    // Another tenant's identical key is independent.
    assert.equal(
      (await ledger.claim(caller(), "key-00001", "h1", rid())).state,
      "new",
    );
  },
);

test(
  "a changed price under the same version is refused",
  { skip },
  async () => {
    await assert.rejects(
      ledger.publishPrices({ m: { inputPerMTok: "9", outputPerMTok: "2" } }),
      /publish a new version/,
    );
  },
);

test(
  "concurrent reconciles across organizations settle each lease once",
  { skip },
  async () => {
    const orgs = [caller(), caller(), caller()];
    for (const c of orgs) {
      for (let i = 0; i < 3; i++)
        assert.notEqual(await reserve(c, -1000), null);
    }
    await Promise.all(Array.from({ length: 6 }, () => ledger.reconcile()));
    while ((await ledger.reconcile()) > 0);
    for (const c of orgs) {
      const b = await ledger.budget(c.organizationId);
      assert.deepEqual([Number(b.spent), Number(b.reserved)], [3, 0]);
    }
  },
);

test(
  "three replicas sharing a budget never overspend it",
  { skip },
  async () => {
    const pools = Array.from(
      { length: 3 },
      () => new pg.Pool({ connectionString: url, max: 10 }),
    );
    try {
      const replicas = pools.map((p) =>
        createLedger(p, { priceBookVersion: "test-v1" }),
      );
      const c = caller();
      const race = () =>
        Promise.all(
          Array.from({ length: 30 }, async (_, i) => {
            const requestId = `led_${randomUUID()}`;
            const held = await replicas[i % 3]?.reserve({
              requestId,
              caller: c,
              limit: "5",
              provider: "bedrock",
              modelId: "m",
              inputTokens: 1_000_000,
              maxOutputTokens: 0,
              leaseMs: 60_000,
            });
            return held === null ? null : requestId;
          }),
        ).then((ids) => ids.filter((id) => id !== null));

      const first = await race();
      assert.equal(first.length, 5);
      // Settle each on a different replica than reserved it: half the estimate.
      await Promise.all(
        first.map((id, i) =>
          replicas[(i + 1) % 3]?.finalize(id, {
            kind: "measured",
            usage: { inputTokens: 500_000, outputTokens: 0 },
          }),
        ),
      );
      assert.equal((await race()).length, 2);
      const b = await ledger.budget(c.organizationId);
      assert.deepEqual([Number(b.spent), Number(b.reserved)], [2.5, 2]);
    } finally {
      await Promise.all(pools.map((p) => p.end()));
    }
  },
);
