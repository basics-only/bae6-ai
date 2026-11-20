import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { after, before, test } from "node:test";
import pg from "pg";
import {
  type Analytics,
  createAnalytics,
  type GroupBy,
  parseUsageQuery,
  type UsageQuery,
} from "./analytics.js";
import { HttpError } from "./http.js";
import type { Identity } from "./identity.js";
import { createLedger, type Ledger } from "./ledger.js";
import { migrate } from "./migrate.js";

const url = process.env.BAE6_DATABASE_URL;
const skip = url ? false : "BAE6_DATABASE_URL not set";
let pool: pg.Pool;
let analytics: Analytics;
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
  analytics = createAnalytics(pool);
  ledger = createLedger(pool, { priceBookVersion: "test-v1" });
  await ledger.publishPrices({ m: { inputPerMTok: "1", outputPerMTok: "2" } });
});

after(() => pool?.end());

const org = () => `org-${randomUUID().slice(0, 8)}`;
const at = (iso: string) => new Date(iso);

interface Event {
  organization: string;
  time: string;
  subject?: string;
  endUser?: boolean;
  taskType?: string | null;
  runId?: string | null;
  cost?: string;
  latencyMs?: number | null;
  firstTokenMs?: number | null;
  model?: string;
}

const insert = (e: Event) =>
  pool.query(
    `INSERT INTO usage_events (request_id, organization_id, application_id, subject_id,
       provider, model_id, price_book_version, input_tokens, output_tokens, cost, outcome,
       task_type, run_id, end_user, latency_ms, first_token_ms, time)
     VALUES ($1, $2, 'app', $3, 'bedrock', $4, 'test-v1', 10, 5, $5, 'measured',
       $6, $7, $8, $9, $10, $11)`,
    [
      `led_${randomUUID()}`,
      e.organization,
      e.subject ?? "svc",
      e.model ?? "m",
      e.cost ?? "1",
      e.taskType ?? null,
      e.runId ?? null,
      e.endUser ?? false,
      e.latencyMs ?? null,
      e.firstTokenMs ?? null,
      e.time,
    ],
  );

const query = <G extends GroupBy | null>(
  o: Partial<UsageQuery> & { groupBy: G },
): UsageQuery & { groupBy: G } => ({
  from: at("2026-11-01T00:00:00Z"),
  to: at("2026-11-20T00:00:00Z"),
  group: null,
  organization: null,
  runIds: [],
  ...o,
});

test(
  "summary groups by task type with percentiles, spend and active users",
  {
    skip,
  },
  async () => {
    const o = org();
    await insert({
      organization: o,
      time: "2026-11-10T10:00:00Z",
      taskType: "summarization",
      subject: "u1",
      endUser: true,
      latencyMs: 100,
      firstTokenMs: 40,
      cost: "1",
    });
    await insert({
      organization: o,
      time: "2026-11-10T11:00:00Z",
      taskType: "summarization",
      subject: "u2",
      endUser: true,
      latencyMs: 300,
      cost: "2",
    });
    await insert({
      organization: o,
      time: "2026-11-10T12:00:00Z",
      cost: "0.5",
    });

    const r = await analytics.summary(
      query({ groupBy: "taskType", organization: o }),
    );
    assert.equal(r.truncated, false);
    assert.deepEqual(
      r.data.map((g) => g.key),
      ["summarization", "untagged"],
    );
    const [summary, untagged] = r.data;
    assert.equal(summary?.requests, 2);
    assert.equal(Number(summary?.cost), 3);
    assert.equal(summary?.inputTokens, 20);
    assert.deepEqual(summary?.latencyMs, { p50: 200, p95: 290 });
    assert.deepEqual(summary?.firstTokenMs, { p50: 40, p95: 40 });
    assert.equal(summary?.activeUsers.month, 2);
    // A service-token call spends but is no active user, and has no timing to report.
    assert.equal(untagged?.activeUsers.month, 0);
    assert.deepEqual(untagged?.latencyMs, { p50: null, p95: null });
  },
);

test(
  "active users count the trailing windows even when they start before the window",
  {
    skip,
  },
  async () => {
    const o = org();
    const mk = (subject: string, time: string) =>
      insert({ organization: o, time, subject, endUser: true });
    await mk("u1", "2026-11-19T20:00:00Z");
    await mk("u2", "2026-11-17T00:00:00Z");
    await mk("u3", "2026-10-31T00:00:00Z");
    await mk("u1", "2026-11-19T21:00:00Z");

    const r = await analytics.summary(
      query({
        groupBy: "model",
        organization: o,
        from: at("2026-11-19T00:00:00Z"),
      }),
    );
    assert.equal(r.data.length, 1);
    assert.equal(r.data[0]?.requests, 2);
    assert.deepEqual(r.data[0]?.activeUsers, { day: 1, week: 2, month: 3 });
  },
);

test(
  "run grouping lists only the asked runs, and organizations stay apart",
  {
    skip,
  },
  async () => {
    const [a, b] = [org(), org()];
    await insert({
      organization: a,
      time: "2026-11-10T10:00:00Z",
      runId: "r1",
    });
    await insert({
      organization: a,
      time: "2026-11-10T10:00:00Z",
      runId: "r1",
    });
    await insert({
      organization: a,
      time: "2026-11-10T10:00:00Z",
      runId: "r2",
    });
    await insert({
      organization: b,
      time: "2026-11-10T10:00:00Z",
      runId: "r1",
    });

    const runs = await analytics.summary(
      query({ groupBy: "run", organization: a, runIds: ["r1", "missing"] }),
    );
    assert.deepEqual(
      runs.data.map((g) => [g.key, g.requests]),
      [["r1", 2]],
    );

    const orgs = await analytics.summary(query({ groupBy: "organization" }));
    const counts = new Map(orgs.data.map((g) => [g.key, g.requests]));
    assert.equal(counts.get(a), 3);
    assert.equal(counts.get(b), 1);
  },
);

test(
  "series counts the last day's active users over the 24 hours before a mid-day end",
  { skip },
  async () => {
    const o = org();
    const user = (time: string, subject: string) =>
      insert({ organization: o, time, subject, endUser: true, taskType: null });
    await user("2026-11-11T20:00:00Z", "u2");
    await user("2026-11-12T10:00:00Z", "u1");
    await user("2026-11-12T13:00:00Z", "u3");

    const days = await analytics.series(
      query({
        groupBy: null,
        organization: o,
        from: at("2026-11-11T00:00:00Z"),
        to: at("2026-11-12T12:00:00Z"),
      }),
    );
    assert.deepEqual(
      days.map((d) => [d.day, d.requests, d.activeUsers.day]),
      [
        ["2026-11-11", 1, 1],
        ["2026-11-12", 1, 2],
      ],
    );
  },
);

test(
  "series of a 93-day window starting mid-day spans 94 days and fits the contract",
  { skip },
  async () => {
    const o = org();
    const start = Date.parse("2026-06-01T12:00:00Z");
    for (let d = 0; d < 94; d++) {
      await insert({
        organization: o,
        time: new Date(
          start + d * 86_400_000 + (d === 93 ? -11 : 1) * 3_600_000,
        ).toISOString(),
        subject: "u1",
        endUser: true,
      });
    }
    const days = await analytics.series(
      query({
        groupBy: null,
        organization: o,
        from: at("2026-06-01T12:00:00Z"),
        to: at("2026-09-02T12:00:00Z"),
      }),
    );
    assert.equal(days.length, 94);
  },
);

test(
  "series gives one point per UTC day with calls, and clips the first day",
  {
    skip,
  },
  async () => {
    const o = org();
    const mk = (
      time: string,
      subject: string,
      taskType: string | null = null,
    ) => insert({ organization: o, time, subject, endUser: true, taskType });
    await mk("2026-11-10T01:00:00Z", "u1", "proofreading");
    await mk("2026-11-10T13:00:00Z", "u2");
    await mk("2026-11-12T23:30:00Z", "u1");

    const all = await analytics.series(
      query({
        groupBy: null,
        organization: o,
        from: at("2026-11-09T00:00:00Z"),
        to: at("2026-11-13T00:00:00Z"),
      }),
    );
    assert.deepEqual(
      all.map((d) => [d.day, d.requests]),
      [
        ["2026-11-10", 2],
        ["2026-11-12", 1],
      ],
    );
    assert.deepEqual(all[1]?.activeUsers, { day: 1, week: 2, month: 2 });

    const clipped = await analytics.series(
      query({
        groupBy: null,
        organization: o,
        from: at("2026-11-10T12:00:00Z"),
        to: at("2026-11-13T00:00:00Z"),
      }),
    );
    assert.equal(clipped[0]?.requests, 1);

    const one = await analytics.series(
      query({
        groupBy: "taskType",
        group: "proofreading",
        organization: o,
        from: at("2026-11-09T00:00:00Z"),
        to: at("2026-11-13T00:00:00Z"),
      }),
    );
    assert.deepEqual(
      one.map((d) => [d.day, d.requests]),
      [["2026-11-10", 1]],
    );
  },
);

test(
  "tags are listed, renamed and removed per organization",
  {
    skip,
  },
  async () => {
    const [a, b] = [org(), org()];
    await analytics.put(a, "task_type", "summarization", "Summarization");
    await analytics.put(a, "task_type", "summarization", "Summaries");
    await analytics.put(a, "domain", "legal", "Legal");
    assert.deepEqual(
      (await analytics.tags(a, "task_type")).map((t) => [t.id, t.name]),
      [["summarization", "Summaries"]],
    );
    assert.equal(
      await analytics.defined(a, "task_type", "summarization"),
      true,
    );
    assert.equal(
      await analytics.defined(b, "task_type", "summarization"),
      false,
    );
    assert.equal(await analytics.defined(a, "domain", "summarization"), false);
    assert.equal(await analytics.remove(a, "task_type", "summarization"), true);
    assert.equal(
      await analytics.remove(a, "task_type", "summarization"),
      false,
    );
  },
);

test(
  "concurrent creates never pass 50 tags, but a rename at the limit works",
  {
    skip,
  },
  async () => {
    const o = org();
    const results = await Promise.allSettled(
      Array.from({ length: 60 }, (_, i) =>
        analytics.put(o, "task_type", `t${i}`, `Type ${i}`),
      ),
    );
    const refused = results.filter(
      (r) =>
        r.status === "rejected" &&
        r.reason instanceof HttpError &&
        r.reason.status === 409,
    );
    assert.equal(refused.length, 10);
    assert.equal((await analytics.tags(o, "task_type")).length, 50);
    const [first] = await analytics.tags(o, "task_type");
    await analytics.put(o, "task_type", first?.id as string, "Renamed");
  },
);

test(
  "tags and timing survive finalize, and the reconciler keeps the tags",
  {
    skip,
  },
  async () => {
    const o = org();
    const caller = { organizationId: o, applicationId: "app", subjectId: "u1" };
    const reserve = async (runId: string, leaseMs: number) => {
      const requestId = `led_${randomUUID()}`;
      await ledger.reserve({
        requestId,
        caller,
        limit: "5",
        provider: "bedrock",
        modelId: "m",
        inputTokens: 1000,
        maxOutputTokens: 0,
        leaseMs,
        tags: { taskType: "summarization", domain: "legal", runId },
        endUser: true,
      });
      return requestId;
    };
    const done = await reserve("done", 60_000);
    await ledger.finalize(
      done,
      { kind: "measured", usage: { inputTokens: 1000, outputTokens: 0 } },
      { latencyMs: 250, firstTokenMs: 90 },
    );
    await reserve("crashed", -1);
    await ledger.reconcile();

    const { rows } = await pool.query(
      `SELECT run_id, task_type, domain, end_user, latency_ms, first_token_ms
     FROM usage_events WHERE organization_id = $1 ORDER BY run_id DESC`,
      [o],
    );
    assert.deepEqual(rows, [
      {
        run_id: "done",
        task_type: "summarization",
        domain: "legal",
        end_user: true,
        latency_ms: 250,
        first_token_ms: 90,
      },
      {
        run_id: "crashed",
        task_type: "summarization",
        domain: "legal",
        end_user: true,
        latency_ms: null,
        first_token_ms: null,
      },
    ]);
  },
);

const who = (scopes: string[]): Identity => ({
  subjectId: "svc",
  subjectType: "workload",
  organizationId: "acme",
  applicationId: "crm",
  scopes,
});
const parse = (qs: string, scopes: string[], series = false) =>
  parseUsageQuery(
    new URL(`http://x/v1/usage/summary?${qs}`),
    who(scopes),
    series,
  );
const refused = (status: number) => (e: unknown) =>
  e instanceof HttpError && e.status === status;

test("without the platform scope a caller stays in its own organization", () => {
  const read = ["ai:usage:read"];
  assert.throws(() => parse("groupBy=organization", read), refused(403));
  assert.throws(
    () => parse("groupBy=model&organization=globex", read),
    refused(403),
  );
  assert.equal(
    parse("groupBy=model&organization=acme", read).organization,
    "acme",
  );
  assert.equal(parse("groupBy=model", read).organization, "acme");
  const platform = [...read, "ai:usage:platform"];
  assert.equal(parse("groupBy=organization", platform).organization, null);
  assert.equal(
    parse("groupBy=model&organization=globex", platform).organization,
    "globex",
  );
});

test("malformed windows, groups and run lists are refused", () => {
  const read = ["ai:usage:read"];
  for (const bad of [
    "",
    "groupBy=user",
    "groupBy=model&from=yesterday",
    "groupBy=model&from=2026-11-01T00:00:00",
    "groupBy=model&from=2026-01-01T00:00:00Z&to=2026-06-01T00:00:00Z",
    "groupBy=model&from=2026-02-01T00:00:00Z&to=2026-01-01T00:00:00Z",
    "groupBy=run",
    "groupBy=run&runId=has space",
    "groupBy=model&runId=r1",
    "groupBy=model&group=x",
    ...[`groupBy=run${"&runId=r".repeat(51)}`],
  ]) {
    assert.throws(() => parse(bad, read), refused(400), bad);
  }
  for (const bad of ["groupBy=model", "group=x", "groupBy=run&group=a b"]) {
    assert.throws(() => parse(bad, read, true), refused(400), bad);
  }
});
