import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { after, before, test } from "node:test";
import pg from "pg";
import { createFeedback, type Feedback } from "./feedback.js";
import { createLedger, type Ledger } from "./ledger.js";
import { migrate } from "./migrate.js";

const url = process.env.BAE6_DATABASE_URL;
const skip = url ? false : "BAE6_DATABASE_URL not set";
let pool: pg.Pool;
let ledger: Ledger;
let feedback: Feedback;

before(async () => {
  if (!url) return;
  pool = new pg.Pool({ connectionString: url, max: 5 });
  const c = await pool.connect();
  try {
    await migrate(c);
  } finally {
    c.release();
  }
  ledger = createLedger(pool, { priceBookVersion: "test-v1" });
  await ledger.publishPrices({ m: { inputPerMTok: "1", outputPerMTok: "2" } });
  feedback = createFeedback(pool);
});

after(() => pool?.end());

const org = () => `org-${randomUUID().slice(0, 8)}`;

async function request(organizationId: string) {
  const requestId = `led_${randomUUID()}`;
  const caller = { organizationId, applicationId: "app", subjectId: "svc" };
  await ledger.reserve({
    requestId,
    caller,
    limit: "5",
    provider: "bedrock",
    modelId: "m",
    inputTokens: 10,
    maxOutputTokens: 10,
    leaseMs: 60_000,
  });
  await ledger.finalize(requestId, { kind: "failed" });
  return requestId;
}

const verdict = (
  organizationId: string,
  requestId: string,
  extra: Partial<Parameters<Feedback["submit"]>[0]> = {},
) => ({
  organizationId,
  submitterId: "svc",
  requestId,
  verdict: "positive" as const,
  label: null,
  comment: null,
  ...extra,
});

test(
  "feedback on another organization's request records nothing",
  { skip },
  async () => {
    const [a, b] = [org(), org()];
    const id = await request(a);
    assert.equal(await feedback.submit(verdict(b, id)), false);
    assert.equal((await feedback.list(b, 10, null)).data.length, 0);
    assert.equal(
      await feedback.submit(verdict(a, `led_${randomUUID()}`)),
      false,
    );
    assert.equal(await feedback.submit(verdict(a, id)), true);
  },
);

test(
  "resubmitting replaces the entry instead of adding one",
  { skip },
  async () => {
    const a = org();
    const id = await request(a);
    await feedback.submit(verdict(a, id));
    await feedback.submit(
      verdict(a, id, { verdict: "negative", label: "unsafe", comment: "bad" }),
    );
    const { data } = await feedback.list(a, 10, null);
    assert.equal(data.length, 1);
    assert.deepEqual(
      [data[0]?.verdict, data[0]?.label, data[0]?.comment, data[0]?.model],
      ["negative", "unsafe", "bad", "m"],
    );
  },
);

test(
  "listing pages newest first without repeating or skipping",
  { skip },
  async () => {
    const a = org();
    const ids: string[] = [];
    for (let i = 0; i < 5; i++) {
      const id = await request(a);
      ids.push(id);
      await feedback.submit(verdict(a, id));
    }
    const seen: string[] = [];
    let cursor: number | null = null;
    do {
      const page = await feedback.list(a, 2, cursor);
      seen.push(...page.data.map((d) => d.requestId));
      cursor = page.nextCursor === null ? null : Number(page.nextCursor);
    } while (cursor !== null);
    assert.deepEqual(seen, ids.reverse());
  },
);

test("the database refuses a comment over 1000 bytes", { skip }, async () => {
  const a = org();
  const id = await request(a);
  await assert.rejects(
    feedback.submit(verdict(a, id, { comment: "é".repeat(501) })),
  );
});

test("prune deletes only entries past retention", { skip }, async () => {
  const a = org();
  const [old, fresh] = [await request(a), await request(a)];
  await feedback.submit(verdict(a, old));
  await feedback.submit(verdict(a, fresh));
  await pool.query(
    "UPDATE feedback SET created_at = now() - interval '100 days' WHERE request_id = $1",
    [old],
  );
  await feedback.prune(90);
  const { data } = await feedback.list(a, 10, null);
  assert.deepEqual(
    data.map((d) => d.requestId),
    [fresh],
  );
});
