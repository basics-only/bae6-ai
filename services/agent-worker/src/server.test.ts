import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { join, resolve } from "node:path";
import { after, before, test } from "node:test";
import type { Limits } from "@bae6/ai-agents";
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
const UNFINISHED_LIMIT = 50;

const ceilings: Limits = {
  maxSteps: 10,
  maxTotalTokens: 5000,
  timeoutMs: 60_000,
  toolTimeoutMs: 5000,
  maxToolOutputChars: 2000,
  maxHistoryChars: 50_000,
};

const identity = (organizationId: string, scopes: string[]): Identity => ({
  subjectId: "svc",
  subjectType: "workload",
  organizationId,
  applicationId: "app",
  scopes,
});
const both = ["agent-runs:write", "agent-runs:read"];
const identities: Record<string, Identity> = {
  acme: identity("acme", both),
  globex: identity("globex", both),
  noscope: identity("acme", []),
  reader: identity("acme", ["agent-runs:read"]),
  writer: identity("acme", ["agent-runs:write"]),
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
  await pool.query("TRUNCATE agent_runs");
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
    catalog: ["search"],
    ceilings,
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
    "/v1/agent-runs",
    token,
    body,
    key ? { "idempotency-key": key } : {},
  );
const task = { task: "say hi", model: "m" };

const contracts = resolve(import.meta.dirname, "../../../contracts");
const spec = parse(readFileSync(join(contracts, "agent-worker.yaml"), "utf8"));
const ajv = new AjvModule.default({ strict: false, validateFormats: false });
for (const f of readdirSync(join(contracts, "vendor/schemas"))) {
  ajv.addSchema(
    JSON.parse(readFileSync(join(contracts, "vendor/schemas", f), "utf8")),
  );
}
ajv.addSchema(spec, "agent-worker");
function conforms(schema: string, body: unknown) {
  const validate = ajv.getSchema(
    schema.startsWith("https:")
      ? schema
      : `agent-worker#/components/schemas/${schema}`,
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
      "run_00000000-0000-0000-0000-000000000000",
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
    ...task,
    tools: ["search"],
    limits: { maxSteps: 3 },
  });
  assert.equal(res.status, 202);
  const made = await res.json();
  conforms("AgentRun", made);
  conforms(
    "AgentRun",
    await (await call("GET", `/v1/agent-runs/${made.id}`, "acme")).json(),
  );
  const cancelled = await call(
    "POST",
    `/v1/agent-runs/${made.id}/cancel`,
    "acme",
  );
  assert.equal(cancelled.status, 200);
  conforms("AgentRun", await cancelled.json());
  conforms(
    "AgentRunList",
    await (await call("GET", "/v1/agent-runs?limit=5", "acme")).json(),
  );
  for (const [r, status, code] of [
    [await submit(undefined, task), 401, "unauthenticated"],
    [await submit("noscope", task), 403, "forbidden"],
    [await submit("reader", task), 403, "forbidden"],
    [await call("GET", "/v1/agent-runs", "writer"), 403, "forbidden"],
    [await submit("acme", { ...task, extra: 1 }), 400, "invalid_request"],
    [await call("GET", "/v1/agent-runs/run_nope", "acme"), 404, "not_found"],
  ] as const) {
    const body = await r.json();
    assert.deepEqual([r.status, body.code], [status, code]);
    conforms(ERROR, body);
  }
});

test("a run belongs to its organization alone", { skip }, async () => {
  const made = await (await submit("acme", task)).json();
  for (const [method, path] of [
    ["GET", `/v1/agent-runs/${made.id}`],
    ["POST", `/v1/agent-runs/${made.id}/cancel`],
  ]) {
    assert.equal(
      (await call(method as string, path as string, "globex")).status,
      404,
    );
  }
  const theirs = await (
    await call("GET", "/v1/agent-runs?limit=100", "globex")
  ).json();
  assert.ok(!theirs.data.some((r: { id: string }) => r.id === made.id));
});

test(
  "limits above a ceiling are refused, never clamped",
  { skip },
  async () => {
    for (const limits of [
      { maxSteps: 11 },
      { timeoutMs: 60_001 },
      { maxSteps: 0 },
      { maxSteps: 1.5 },
      { bogus: 1 },
    ]) {
      const res = await submit("acme", { ...task, limits });
      assert.equal(res.status, 400, JSON.stringify(limits));
    }
    const ok = await (
      await submit("acme", { ...task, limits: { maxSteps: 2 } })
    ).json();
    assert.equal(ok.status, "queued");
  },
);

test(
  "an id that is not a uuid is not found, not a storage error",
  { skip },
  async () => {
    const id = `run_${"-".repeat(36)}`;
    assert.equal(
      (await call("GET", `/v1/agent-runs/${id}`, "acme")).status,
      404,
    );
    assert.equal(
      (await call("POST", `/v1/agent-runs/${id}/cancel`, "acme")).status,
      404,
    );
  },
);

test("malformed requests are refused", { skip }, async () => {
  for (const body of [
    {},
    { task: "", model: "m" },
    { task: "x".repeat(32_769), model: "m" },
    { task: "x", model: "Bad Model" },
    { task: "x", model: "m", tools: ["unknown"] },
    { task: "x", model: "m", tools: ["search", "search"] },
    { task: "x", model: "m", tools: "search" },
    [],
  ]) {
    assert.equal(
      (await submit("acme", body)).status,
      400,
      JSON.stringify(body).slice(0, 60),
    );
  }
  assert.equal((await submit("acme", task, "short")).status, 400);
});

test(
  "a repeated idempotency key replays the run and a changed body conflicts",
  { skip },
  async () => {
    const first = await submit("acme", task, "key-replay-1");
    const again = await submit("acme", task, "key-replay-1");
    assert.equal(again.headers.get("idempotent-replayed"), "true");
    assert.equal((await again.json()).id, (await first.json()).id);
    const changed = await submit(
      "acme",
      { ...task, task: "other" },
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
      assert.equal((await submit("tiny", task)).status, 202);
    const res = await submit("tiny", task);
    assert.equal(res.status, 429);
    assert.ok(res.headers.get("retry-after"));
    assert.equal((await submit("acme", task)).status, 202);
  },
);

test(
  "a list takes only a bounded page and a known status",
  { skip },
  async () => {
    for (const q of ["limit=0", "limit=101", "cursor=abc", "status=bogus"]) {
      assert.equal(
        (await call("GET", `/v1/agent-runs?${q}`, "acme")).status,
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
      const res = await submit("acme", task);
      assert.equal(res.status, 503);
      assert.equal((await res.json()).code, "unavailable");
      assert.equal((await call("GET", "/v1/agent-runs", "acme")).status, 503);
    } finally {
      storeDown = false;
    }
  },
);
