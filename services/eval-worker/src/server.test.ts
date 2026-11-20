import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { join, resolve } from "node:path";
import { after, before, test } from "node:test";
import AjvModule from "ajv/dist/2020.js";
import pg from "pg";
import { parse } from "yaml";
import type { Identity } from "./identity.js";
import { migrate } from "./migrate.js";
import { createWorkerServer } from "./server.js";
import { createStore } from "./store.js";

const url = process.env.BAE6_DATABASE_URL;
const skip = url ? false : "BAE6_DATABASE_URL not set";
let pool: pg.Pool;
let base: string;
let storeDown = false;
const UNFINISHED_LIMIT = 20;
const MAX_CASES = 5;

const identity = (organizationId: string, scopes: string[]): Identity => ({
  subjectId: "svc",
  subjectType: "workload",
  organizationId,
  applicationId: "app",
  scopes,
});
const both = ["eval-runs:write", "eval-runs:read"];
const identities: Record<string, Identity> = {
  acme: identity("acme", both),
  globex: identity("globex", both),
  noscope: identity("acme", []),
  reader: identity("acme", ["eval-runs:read"]),
  writer: identity("acme", ["eval-runs:write"]),
  tiny: identity("tiny", both),
};

let server: ReturnType<typeof createWorkerServer>;
before(async () => {
  if (!url) return;
  pool = new pg.Pool({ connectionString: url, max: 5 });
  const c = await pool.connect();
  try {
    await migrate(c);
  } finally {
    c.release();
  }
  await pool.query("TRUNCATE eval_runs");
  const store = createStore(pool);
  server = createWorkerServer({
    verify: async (t) => (t ? (identities[t] ?? null) : null),
    store: new Proxy(store, {
      get: (target, prop) =>
        storeDown
          ? async () => {
              throw new Error("db down");
            }
          : target[prop as keyof typeof store],
    }),
    maxCases: MAX_CASES,
    maxUnfinishedPerOrg: UNFINISHED_LIMIT,
    ready: async () => true,
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});
after(async () => {
  server?.close();
  await pool?.end();
});

const call = (
  method: string,
  path: string,
  token?: string,
  body?: unknown,
  headers: Record<string, string> = {},
) =>
  fetch(`${base}${path}`, {
    method,
    headers: {
      ...(token ? { authorization: `Bearer ${token}` } : {}),
      ...headers,
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
const submit = (token: string | undefined, body: unknown, key?: string) =>
  call(
    "POST",
    "/v1/eval-runs",
    token,
    body,
    key ? { "idempotency-key": key } : {},
  );
const cases = [
  { id: "a", input: "1+1", expected: "2" },
  { id: "b", input: "2+2", expected: "4" },
];
const evalRun = { name: "smoke", datasetVersion: "1", model: "m", cases };

const contracts = resolve(import.meta.dirname, "../../../contracts");
const spec = parse(readFileSync(join(contracts, "eval-worker.yaml"), "utf8"));
const ajv = new AjvModule.default({ strict: false, validateFormats: false });
for (const f of readdirSync(join(contracts, "vendor/schemas"))) {
  ajv.addSchema(
    JSON.parse(readFileSync(join(contracts, "vendor/schemas", f), "utf8")),
  );
}
ajv.addSchema(spec, "eval-worker");
function conforms(schema: string, body: unknown) {
  const validate = ajv.getSchema(
    schema.startsWith("https:")
      ? schema
      : `eval-worker#/components/schemas/${schema}`,
  );
  assert.ok(validate, schema);
  assert.ok(
    validate(body),
    `${schema}: ${JSON.stringify(validate.errors)} ${JSON.stringify(body)}`,
  );
}
const ERROR =
  "https://github.com/basics-only/bae6-contracts/schemas/error-envelope.json";

test("every documented route is served", { skip }, async () => {
  const routes = Object.entries(spec.paths as Record<string, object>).flatMap(
    ([path, item]) =>
      "$ref" in item
        ? [["GET", path]]
        : Object.keys(item).map((m) => [m.toUpperCase(), path]),
  );
  assert.equal(routes.length, 6);
  for (const [method, path] of routes) {
    const concrete = (path as string).replace(
      "{id}",
      "eval_00000000-0000-0000-0000-000000000000",
    );
    const res = await call(
      method as string,
      concrete,
      "noscope",
      method === "POST" ? {} : undefined,
    );
    assert.notEqual(res.status, 404, `${method} ${path}`);
  }
});

test("responses conform to the contract", { skip }, async () => {
  const res = await submit("acme", {
    ...evalRun,
    scorers: ["exact", "contains"],
  });
  assert.equal(res.status, 202);
  const made = await res.json();
  conforms("EvalRun", made);
  conforms(
    "EvalRun",
    await (await call("GET", `/v1/eval-runs/${made.id}`, "acme")).json(),
  );
  const cancelled = await call(
    "POST",
    `/v1/eval-runs/${made.id}/cancel`,
    "acme",
  );
  assert.equal(cancelled.status, 200);
  conforms("EvalRun", await cancelled.json());
  conforms(
    "EvalRunList",
    await (await call("GET", "/v1/eval-runs?limit=5", "acme")).json(),
  );
  for (const [r, status, code] of [
    [await submit(undefined, evalRun), 401, "unauthenticated"],
    [await submit("noscope", evalRun), 403, "forbidden"],
    [await submit("reader", evalRun), 403, "forbidden"],
    [await call("GET", "/v1/eval-runs", "writer"), 403, "forbidden"],
    [await submit("acme", { ...evalRun, extra: 1 }), 400, "invalid_request"],
    [await call("GET", "/v1/eval-runs/eval_nope", "acme"), 404, "not_found"],
  ] as const) {
    const body = await r.json();
    assert.deepEqual([r.status, body.code], [status, code]);
    conforms(ERROR, body);
  }
});

test("a run belongs to its organization alone", { skip }, async () => {
  const made = await (await submit("acme", evalRun)).json();
  for (const [method, path] of [
    ["GET", `/v1/eval-runs/${made.id}`],
    ["POST", `/v1/eval-runs/${made.id}/cancel`],
  ]) {
    assert.equal(
      (await call(method as string, path as string, "globex")).status,
      404,
    );
  }
  const theirs = await (
    await call("GET", "/v1/eval-runs?limit=100", "globex")
  ).json();
  assert.ok(!theirs.data.some((r: { id: string }) => r.id === made.id));
});

test(
  "an id that is not a uuid is not found, not a storage error",
  { skip },
  async () => {
    const id = `eval_${"-".repeat(36)}`;
    assert.equal(
      (await call("GET", `/v1/eval-runs/${id}`, "acme")).status,
      404,
    );
    assert.equal(
      (await call("POST", `/v1/eval-runs/${id}/cancel`, "acme")).status,
      404,
    );
  },
);

test("malformed requests are refused", { skip }, async () => {
  const tooMany = Array.from({ length: MAX_CASES + 1 }, (_, i) => ({
    id: `c${i}`,
    input: "i",
    expected: "e",
  }));
  for (const body of [
    {},
    { ...evalRun, cases: [] },
    { ...evalRun, cases: tooMany },
    { ...evalRun, cases: [cases[0], cases[0]] },
    { ...evalRun, cases: [{ ...cases[0], extra: 1 }] },
    { ...evalRun, cases: [{ id: "a", input: "", expected: "2" }] },
    {
      ...evalRun,
      cases: [{ id: "a", input: "x".repeat(4001), expected: "2" }],
    },
    { ...evalRun, cases: [{ id: "a", input: "x", expected: "  " }] },
    { ...evalRun, cases: [{ id: "bad id", input: "x", expected: "2" }] },
    { ...evalRun, model: "Bad Model" },
    { ...evalRun, name: "has space" },
    { ...evalRun, scorers: ["unknown"] },
    { ...evalRun, scorers: ["exact", "exact"] },
    { ...evalRun, scorers: [] },
    [],
  ]) {
    assert.equal(
      (await submit("acme", body)).status,
      400,
      JSON.stringify(body).slice(0, 80),
    );
  }
  assert.equal((await submit("acme", evalRun, "short")).status, 400);
});

test(
  "a repeated idempotency key replays the run and a changed body conflicts",
  { skip },
  async () => {
    const first = await submit("acme", evalRun, "key-replay-1");
    const again = await submit("acme", evalRun, "key-replay-1");
    assert.equal(again.status, 200);
    assert.equal(again.headers.get("idempotent-replayed"), "true");
    assert.equal((await again.json()).id, (await first.json()).id);
    const changed = await submit(
      "acme",
      { ...evalRun, name: "other" },
      "key-replay-1",
    );
    assert.equal(changed.status, 409);
    assert.equal((await changed.json()).code, "idempotency_key_reused");
  },
);

test(
  "an organization over its unfinished limit gets 429 with a retry hint",
  { skip },
  async () => {
    for (let i = 0; i < UNFINISHED_LIMIT; i++)
      assert.equal((await submit("tiny", evalRun)).status, 202);
    const res = await submit("tiny", evalRun);
    assert.equal(res.status, 429);
    assert.ok(res.headers.get("retry-after"));
    assert.equal((await submit("acme", evalRun)).status, 202);
  },
);

test(
  "a list takes only a bounded page and a known status",
  { skip },
  async () => {
    for (const q of ["limit=0", "limit=101", "cursor=abc", "status=bogus"]) {
      assert.equal(
        (await call("GET", `/v1/eval-runs?${q}`, "acme")).status,
        400,
        q,
      );
    }
  },
);

test(
  "storage failures answer 503 instead of leaking details",
  { skip },
  async () => {
    storeDown = true;
    try {
      const res = await submit("acme", evalRun);
      assert.equal(res.status, 503);
      assert.equal((await res.json()).code, "unavailable");
      assert.equal((await call("GET", "/v1/eval-runs", "acme")).status, 503);
    } finally {
      storeDown = false;
    }
  },
);
