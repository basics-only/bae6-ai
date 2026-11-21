import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { join, resolve } from "node:path";
import { after, before, test } from "node:test";
import { type ModelProvider, ProviderError } from "@bae6/ai-core";
import AjvModule from "ajv/dist/2020.js";
import { parse } from "yaml";
import type { Analytics, UsageQuery } from "./analytics.js";
import type { ModelConfig } from "./config.js";
import type { Feedback } from "./feedback.js";
import type { Identity } from "./identity.js";
import type { Ledger, Outcome, Tags, Timing } from "./ledger.js";
import { createGateway } from "./server.js";

const finalized: Outcome[] = [];
const reserved: { tags: Tags; endUser: boolean }[] = [];
const timings: (Timing | undefined)[] = [];
const keys: string[] = [];
const claims = new Map<string, string>();
let fail: ProviderError | undefined;
let ledgerDown = false;
let hold: Promise<void> | undefined;
let entered = 0;
let held = 0;

const ledger = {
  reserve: async (i: { tags: Tags; endUser: boolean }) => {
    if (ledgerDown) throw new Error("db down");
    reserved.push({ tags: i.tags, endUser: i.endUser });
    return "0.01";
  },
  finalize: async (_: string, o: Outcome, t?: Timing) => {
    finalized.push(o);
    timings.push(t);
    return true;
  },
  claim: async (_c: unknown, key: string, hash: string) => {
    const seen = claims.get(key);
    if (seen === undefined) {
      claims.set(key, hash);
      return { state: "new" };
    }
    return seen === hash
      ? {
          state: "replay",
          status: 200,
          body: { replayed: true },
          modelId: null,
        }
      : { state: "mismatch" };
  },
  complete: async (_: unknown, _k: string, status: number) => {
    keys.push(`complete ${status}`);
  },
  release: async (_c: unknown, key: string) => {
    claims.delete(key);
    keys.push("release");
  },
  budget: async () => {
    if (ledgerDown) throw new Error("db down");
    return { period: "2026-11", limit: "10", spent: "1.5", reserved: "0" };
  },
  usage: async () => {
    if (ledgerDown) throw new Error("db down");
    return [
      {
        model: "m",
        outcome: "measured",
        requests: 2,
        inputTokens: 6,
        outputTokens: 4,
        cost: "0.0001",
      },
    ];
  },
} as unknown as Ledger;

const measures = {
  requests: 2,
  inputTokens: 6,
  outputTokens: 4,
  cost: "0.0001",
  latencyMs: { p50: 120, p95: 300 },
  firstTokenMs: { p50: null, p95: null },
  activeUsers: { day: 1, week: 2, month: 3 },
};
const asked: UsageQuery[] = [];
const tagTime = "2026-11-20T00:00:00.000Z";
let analyticsDown = false;
const live = <T>(v: T) => {
  if (analyticsDown) throw new Error("db down");
  return v;
};
let tagLookupMs = 0;
let summarizationDeleted = false;
const analytics = {
  defined: async (_o: string, kind: string, id: string) => {
    await new Promise((r) => setTimeout(r, tagLookupMs));
    return live(
      kind === "task_type" && id === "summarization" && !summarizationDeleted,
    );
  },
  tags: async () =>
    live([{ id: "summarization", name: "Summarization", updatedAt: tagTime }]),
  put: async (_o: string, _k: string, id: string, name: string) =>
    live({ id, name, updatedAt: tagTime }),
  remove: async (_o: string, _k: string, id: string) =>
    live(id === "summarization"),
  summary: async (q: UsageQuery) => {
    asked.push(q);
    return live({
      truncated: false,
      data: [{ key: "summarization", ...measures }],
    });
  },
  series: async (q: UsageQuery) => {
    asked.push(q);
    return live([{ day: "2026-11-19", ...measures }]);
  },
  concentration: async (q: UsageQuery) => {
    asked.push(q);
    return live([
      { month: "2026-11", users: 5, top5: 0.4, top10: 0.5, top20: 0.7 },
    ]);
  },
  distribution: async (q: UsageQuery) => {
    asked.push(q);
    return live({
      users: 2,
      total: "100",
      data: [
        { topUsersPercent: 50, share: 0.9 },
        { topUsersPercent: 100, share: 1 },
      ],
    });
  },
} as unknown as Analytics;

const ratable = new Map<string, string>();
const rated: unknown[] = [];
let feedbackDown = false;
const feedback = {
  submit: async (f: { requestId: string; organizationId: string }) => {
    if (feedbackDown) throw new Error("db down");
    if (ratable.get(f.requestId) !== f.organizationId) return false;
    rated.push(f);
    return true;
  },
  list: async () => {
    if (feedbackDown) throw new Error("db down");
    return {
      data: [
        {
          requestId: "chatcmpl-led_00000000-0000-0000-0000-000000000000",
          submitterId: "svc",
          model: "m",
          verdict: "negative",
          label: null,
          comment: "wrong",
          updatedAt: new Date().toISOString(),
        },
      ],
      nextCursor: null,
    };
  },
} as unknown as Feedback;

const config: ModelConfig = {
  id: "m",
  provider: "bedrock",
  providerModel: "x",
  region: "us-east-1",
  operations: ["chat", "chat_stream"],
  inputModalities: ["text"],
  maxInputTokens: 1000,
  maxOutputTokens: 100,
  organizations: ["acme"],
};

const usage = { inputTokens: 3, outputTokens: 2 };
const provider = {
  descriptor: {
    kind: "model",
    id: "m",
    operations: ["chat", "chat_stream"],
    limits: { maxInputTokens: 1000, maxOutputTokens: 100 },
    attributes: { provider: "bedrock", inputModalities: ["text"] },
  },
  chat: async () => {
    if (++entered <= held) await hold;
    if (fail) throw fail;
    return { text: "hello", finishReason: "stop", usage };
  },
  async *stream() {
    yield { type: "delta", text: "hel" };
    yield { type: "delta", text: "lo" };
    yield { type: "done", finishReason: "stop", usage };
  },
} as unknown as ModelProvider;

const identities: Record<string, Identity> = {
  acme: {
    subjectId: "svc",
    subjectType: "workload",
    organizationId: "acme",
    applicationId: "crm",
    scopes: ["ai:invoke"],
  },
  globex: {
    subjectId: "svc",
    subjectType: "workload",
    organizationId: "globex",
    applicationId: "crm",
    scopes: ["ai:invoke"],
  },
  noscope: {
    subjectId: "svc",
    subjectType: "workload",
    organizationId: "acme",
    applicationId: "crm",
    scopes: [],
  },
  reader: {
    subjectId: "svc",
    subjectType: "workload",
    organizationId: "acme",
    applicationId: "crm",
    scopes: ["ai:invoke", "ai:usage:read"],
  },
  rater: {
    subjectId: "svc",
    subjectType: "workload",
    organizationId: "acme",
    applicationId: "crm",
    scopes: ["feedback:write"],
  },
  rater2: {
    subjectId: "svc",
    subjectType: "workload",
    organizationId: "globex",
    applicationId: "crm",
    scopes: ["feedback:write"],
  },
  spammer: {
    subjectId: "svc",
    subjectType: "workload",
    organizationId: "umbrella",
    applicationId: "crm",
    scopes: ["feedback:write"],
  },
  auditor: {
    subjectId: "svc",
    subjectType: "workload",
    organizationId: "acme",
    applicationId: "crm",
    scopes: ["feedback:read"],
  },
  analyst: {
    subjectId: "svc",
    subjectType: "workload",
    organizationId: "acme",
    applicationId: "crm",
    scopes: ["ai:usage:read"],
  },
  admin: {
    subjectId: "svc",
    subjectType: "workload",
    organizationId: "acme",
    applicationId: "crm",
    scopes: ["ai:usage:read", "ai:usage:admin"],
  },
  operator: {
    subjectId: "svc",
    subjectType: "workload",
    organizationId: "acme",
    applicationId: "crm",
    scopes: ["ai:usage:read", "ai:usage:platform"],
  },
  person: {
    subjectId: "ada",
    subjectType: "user",
    organizationId: "acme",
    applicationId: "crm",
    scopes: ["ai:invoke"],
  },
  nobudget: {
    subjectId: "svc",
    subjectType: "workload",
    organizationId: "initech",
    applicationId: "crm",
    scopes: ["ai:invoke"],
  },
};

const server = createGateway({
  credentials: () => [
    { provider: "bedrock", source: "pod-identity", lastRefreshAgeSeconds: 12 },
    { provider: "vertex", source: "metadata", lastRefreshAgeSeconds: null },
  ],
  verify: async (t) => (t ? (identities[t] ?? null) : null),
  ledger,
  feedback,
  analytics,
  models: new Map([
    ["m", { config, provider }],
    [
      "open",
      { config: { ...config, id: "open", organizations: undefined }, provider },
    ],
  ] as [string, { config: ModelConfig; provider: ModelProvider }][]),
  budgets: { acme: "10", globex: "10" },
  timeoutMs: 5000,
  maxInflight: 8,
  maxInflightPerOrg: 2,
  ready: async () => true,
});
let base: string;
before(async () => {
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});
after(() => server.close());

const post = (token: string | undefined, body: unknown, key?: string) =>
  fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(token ? { authorization: `Bearer ${token}` } : {}),
      ...(key ? { "idempotency-key": key } : {}),
    },
    body: JSON.stringify(body),
  });
const hi = { model: "m", messages: [{ role: "user", content: "hi" }] };

test("refuses missing credentials, missing scope and another tenant's model", async () => {
  assert.equal((await post(undefined, hi)).status, 401);
  assert.equal((await post("noscope", hi)).status, 403);
  assert.equal((await post("globex", hi)).status, 404);
  const none = await post("nobudget", { ...hi, model: "open" });
  assert.equal(none.status, 403);
});

test("answers a metered chat and finalizes it as measured", async () => {
  finalized.length = 0;
  const res = await post("acme", hi);
  const body = (await res.json()) as {
    choices: { message: { content: string } }[];
    usage: { total_tokens: number };
  };
  assert.equal(res.status, 200);
  assert.equal(body.choices[0]?.message.content, "hello");
  assert.equal(body.usage.total_tokens, 5);
  assert.deepEqual(finalized, [{ kind: "measured", usage }]);
});

test("a provider error answer is not billed; a lost answer is charged as uncertain", async () => {
  finalized.length = 0;
  fail = new ProviderError("x", "no", 400);
  assert.equal((await post("acme", hi)).status, 400);
  fail = new ProviderError("x", "unknown");
  assert.equal((await post("acme", hi)).status, 502);
  fail = undefined;
  assert.deepEqual(finalized, [{ kind: "failed" }, { kind: "uncertain" }]);
});

test("a provider throttle tells the client when to retry", async () => {
  fail = new ProviderError("x", "no", 429);
  const res = await post("acme", hi);
  fail = undefined;
  assert.equal(res.status, 429);
  assert.equal(res.headers.get("retry-after"), "2");
});

test("a model that is still starting is a 503 the client retries, not a provider failure", async () => {
  fail = new ProviderError("x", "no", 503);
  const res = await post("acme", hi);
  fail = undefined;
  assert.equal(res.status, 503);
  assert.equal(res.headers.get("retry-after"), "5");
});

test("streams OpenAI chunks with usage and finalizes once", async () => {
  finalized.length = 0;
  const res = await post("acme", {
    ...hi,
    stream: true,
    stream_options: { include_usage: true },
  });
  const text = await res.text();
  assert.equal(res.headers.get("content-type"), "text/event-stream");
  assert.match(text, /"content":"hel"/);
  assert.match(text, /"total_tokens":5/);
  assert.ok(text.trimEnd().endsWith("data: [DONE]"));
  assert.deepEqual(finalized, [{ kind: "measured", usage }]);
});

test("rejects unsupported parameters before any reservation", async () => {
  finalized.length = 0;
  const res = await post("acme", { ...hi, tools: [] });
  assert.equal(res.status, 400);
  assert.equal(
    ((await res.json()) as { code: string }).code,
    "unsupported_parameter",
  );
  assert.deepEqual(finalized, []);
});

test("an idempotency key is freed when nothing was billed, and kept when it may have been", async () => {
  keys.length = 0;
  fail = new ProviderError("x", "no", 429);
  assert.equal((await post("acme", hi, "key-00001")).status, 429);
  fail = new ProviderError("x", "unknown");
  assert.equal((await post("acme", hi, "key-00002")).status, 502);
  fail = undefined;
  ledgerDown = true;
  assert.equal((await post("acme", hi, "key-00003")).status, 503);
  ledgerDown = false;
  assert.deepEqual(keys, ["release", "complete 502", "release"]);
});

const contracts = resolve(import.meta.dirname, "../../../contracts");
const spec = parse(
  readFileSync(join(contracts, "openapi.yaml"), "utf8")
    // The vendored schemas are the release published under these $ids.
    .replaceAll(
      "./vendor/schemas/",
      "https://github.com/basics-only/bae6-contracts/schemas/",
    ),
);
const ajv = new AjvModule.default({ strict: false, validateFormats: false });
for (const f of readdirSync(join(contracts, "vendor/schemas"))) {
  ajv.addSchema(
    JSON.parse(readFileSync(join(contracts, "vendor/schemas", f), "utf8")),
  );
}
ajv.addSchema(spec, "openapi");
function conforms(schema: string, body: unknown) {
  const validate = ajv.getSchema(`openapi#/components/schemas/${schema}`);
  assert.ok(validate, schema);
  assert.ok(
    validate(body),
    `${schema}: ${JSON.stringify(validate.errors)} ${JSON.stringify(body)}`,
  );
}
const call = (method: string, path: string, token: string, body?: unknown) =>
  fetch(`${base}${path}`, {
    method,
    headers: { authorization: `Bearer ${token}` },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });

test("every documented route is served", async () => {
  const routes = Object.entries(spec.paths as Record<string, object>).flatMap(
    ([path, item]) =>
      "$ref" in item
        ? [["GET", path]]
        : Object.keys(item)
            .filter((m) => m !== "parameters")
            .map((m) => [m.toUpperCase(), path]),
  );
  assert.equal(routes.length, 20);
  for (const [method, path] of routes) {
    const body = method === "POST" ? {} : undefined;
    const res = await call(method as string, path as string, "noscope", body);
    assert.notEqual(res.status, 404, `${method} ${path}`);
  }
});

test("responses conform to the contract", async () => {
  conforms("ChatCompletion", await (await post("acme", hi)).json());
  const stream = await (
    await post("acme", {
      ...hi,
      stream: true,
      stream_options: { include_usage: true },
    })
  ).text();
  const events = stream
    .split("\n\n")
    .filter((e) => e.startsWith("data: {"))
    .map((e) => JSON.parse(e.slice(6)));
  assert.equal(events.length, 5);
  for (const e of events) conforms("ChatCompletionChunk", e);
  conforms("ModelList", await (await call("GET", "/v1/models", "acme")).json());
  conforms("Budget", await (await call("GET", "/v1/budgets", "reader")).json());
  conforms("Usage", await (await call("GET", "/v1/usage", "reader")).json());
  conforms(
    "CredentialStatusList",
    await (await call("GET", "/v1/credentials", "reader")).json(),
  );
  for (const [res, status, code] of [
    [await post(undefined, hi), 401, "unauthenticated"],
    [await post("noscope", hi), 403, "forbidden"],
    [await call("GET", "/v1/credentials", "acme"), 403, "forbidden"],
    [await call("GET", "/v1/credentials", "unknown"), 401, "unauthenticated"],
    [await post("globex", hi), 404, "not_found"],
    [await post("acme", { ...hi, tools: [] }), 400, "unsupported_parameter"],
    [
      await call("GET", "/v1/usage?from=nonsense", "reader"),
      400,
      "invalid_request",
    ],
  ] as const) {
    const body = await res.json();
    assert.deepEqual([res.status, body.code], [status, code]);
    conforms("ErrorEnvelope", body);
  }
});

test("budget and usage reads fail closed with 503 when metering is down", async () => {
  ledgerDown = true;
  const res = [
    await call("GET", "/v1/budgets", "reader"),
    await call("GET", "/v1/usage", "reader"),
  ];
  ledgerDown = false;
  for (const r of res) {
    assert.equal(r.status, 503);
    assert.equal(((await r.json()) as { code: string }).code, "unavailable");
  }
});

const tagged = (token: string, headers: Record<string, string>, body = hi) =>
  fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: { authorization: `Bearer ${token}`, ...headers },
    body: JSON.stringify(body),
  });

test("a tag the organization has not defined is refused before any spend", async () => {
  reserved.length = 0;
  for (const [headers, code] of [
    [{ "bae6-task-type": "gossip" }, "unknown_tag"],
    [{ "bae6-domain": "summarization" }, "unknown_tag"],
    [{ "bae6-task-type": "Bad Tag" }, "invalid_request"],
    [{ "bae6-run-id": "has space" }, "invalid_request"],
  ] as const) {
    const res = await tagged("acme", headers);
    const body = await res.json();
    assert.deepEqual(
      [res.status, body.code],
      [400, code],
      JSON.stringify(headers),
    );
    conforms("ErrorEnvelope", body);
  }
  assert.equal(reserved.length, 0);
});

const keyed = (key: string, headers: Record<string, string>) =>
  tagged("acme", { "idempotency-key": key, ...headers });

test("a retry replays after its tag was deleted, and a new key is refused", async () => {
  const headers = { "bae6-task-type": "summarization" };
  assert.equal((await keyed("key-tag-001", headers)).status, 200);
  summarizationDeleted = true;
  try {
    const retry = await keyed("key-tag-001", headers);
    assert.equal(retry.status, 200);
    assert.equal(retry.headers.get("idempotent-replayed"), "true");
    const fresh = await keyed("key-tag-002", headers);
    assert.equal(fresh.status, 400);
    assert.equal((await fresh.json()).code, "unknown_tag");
    assert.equal(claims.has("key-tag-002"), false);
  } finally {
    summarizationDeleted = false;
  }
});

test("a key reused with different tags is a conflict, not a replay", async () => {
  assert.equal(
    (await keyed("key-tag-003", { "bae6-task-type": "summarization" })).status,
    200,
  );
  const other = await keyed("key-tag-003", { "bae6-run-id": "run-9" });
  assert.equal(other.status, 409);
  assert.equal((await other.json()).code, "idempotency_key_reused");
});

test("latency leaves out the time spent before the provider call", async () => {
  timings.length = 0;
  tagLookupMs = 300;
  try {
    const res = await tagged("acme", { "bae6-task-type": "summarization" });
    assert.equal(res.status, 200);
  } finally {
    tagLookupMs = 0;
  }
  assert.ok((timings[0]?.latencyMs ?? 1000) < 200);
});

test("a defined tag and run id reach the ledger with the end-user flag and timing", async () => {
  reserved.length = 0;
  timings.length = 0;
  const headers = { "bae6-task-type": "summarization", "bae6-run-id": "run-1" };
  assert.equal((await tagged("acme", headers)).status, 200);
  assert.equal((await tagged("person", {})).status, 200);
  await (
    await tagged("acme", headers, { ...hi, stream: true } as never)
  ).text();
  assert.deepEqual(reserved[0], {
    tags: { taskType: "summarization", domain: null, runId: "run-1" },
    endUser: false,
  });
  assert.deepEqual(reserved[1], {
    tags: { taskType: null, domain: null, runId: null },
    endUser: true,
  });
  const [plain, , streamed] = timings;
  assert.ok(
    Number.isInteger(plain?.latencyMs) && (plain?.latencyMs ?? -1) >= 0,
  );
  assert.equal(plain?.firstTokenMs, null);
  assert.ok(Number.isInteger(streamed?.firstTokenMs));
  assert.ok((streamed?.latencyMs ?? 0) >= (streamed?.firstTokenMs ?? 1));
});

test("usage summary and series are confined to the caller's organization unless it is a platform operator", async () => {
  asked.length = 0;
  const summary = await call(
    "GET",
    "/v1/usage/summary?groupBy=taskType",
    "analyst",
  );
  assert.equal(summary.status, 200);
  conforms("UsageSummary", await summary.json());
  conforms(
    "UsageSeries",
    await (await call("GET", "/v1/usage/series", "analyst")).json(),
  );
  assert.deepEqual(
    asked.map((q) => q.organization),
    ["acme", "acme"],
  );
  for (const path of [
    "/v1/usage/summary?groupBy=organization",
    "/v1/usage/summary?groupBy=model&organization=globex",
    "/v1/usage/series?organization=globex",
  ]) {
    assert.equal((await call("GET", path, "analyst")).status, 403, path);
  }
  assert.equal(
    (await call("GET", "/v1/usage/summary?groupBy=model", "noscope")).status,
    403,
  );
  assert.equal((await call("GET", "/v1/usage/summary", "analyst")).status, 400);
  asked.length = 0;
  const all = await call(
    "GET",
    "/v1/usage/summary?groupBy=organization",
    "operator",
  );
  assert.equal(all.status, 200);
  assert.equal(asked[0]?.organization, null);
});

test("usage distribution is confined to the caller's organization and refuses an unknown measure", async () => {
  asked.length = 0;
  const ok = await call(
    "GET",
    "/v1/usage/distribution?measure=cost",
    "analyst",
  );
  assert.equal(ok.status, 200);
  conforms("UsageDistribution", await ok.json());
  assert.equal(asked[0]?.organization, "acme");
  assert.equal(
    (await call("GET", "/v1/usage/distribution?organization=globex", "analyst"))
      .status,
    403,
  );
  assert.equal(
    (await call("GET", "/v1/usage/distribution", "noscope")).status,
    403,
  );
  assert.equal(
    (await call("GET", "/v1/usage/distribution?measure=drop", "analyst"))
      .status,
    400,
  );
});

test("usage concentration is confined to the caller's organization", async () => {
  asked.length = 0;
  const ok = await call("GET", "/v1/usage/concentration", "analyst");
  assert.equal(ok.status, 200);
  conforms("UsageConcentration", await ok.json());
  assert.equal(asked[0]?.organization, "acme");
  assert.equal(
    (
      await call(
        "GET",
        "/v1/usage/concentration?organization=globex",
        "analyst",
      )
    ).status,
    403,
  );
  assert.equal(
    (await call("GET", "/v1/usage/concentration", "noscope")).status,
    403,
  );
});

test("task types and domains are readable with usage read and writable only by an admin", async () => {
  conforms(
    "TagList",
    await (await call("GET", "/v1/task-types", "analyst")).json(),
  );
  conforms(
    "TagList",
    await (await call("GET", "/v1/domains", "analyst")).json(),
  );
  assert.equal((await call("GET", "/v1/task-types", "noscope")).status, 403);

  const name = { name: "Summarization" };
  assert.equal(
    (await call("PUT", "/v1/task-types/summarization", "analyst", name)).status,
    403,
  );
  const put = await call("PUT", "/v1/domains/legal", "admin", name);
  assert.equal(put.status, 200);
  conforms("Tag", await put.json());
  for (const [id, body] of [
    ["untagged", name],
    ["Bad_Id", name],
    ["legal", {}],
    ["legal", { name: "" }],
    ["legal", { name: "x".repeat(65) }],
    ["legal", { name: "ok", id: "other" }],
  ] as const) {
    const res = await call("PUT", `/v1/domains/${id}`, "admin", body);
    assert.equal(res.status, 400, `${id} ${JSON.stringify(body)}`);
  }

  assert.equal(
    (await call("DELETE", "/v1/task-types/summarization", "analyst")).status,
    403,
  );
  assert.equal(
    (await call("DELETE", "/v1/task-types/summarization", "admin")).status,
    204,
  );
  assert.equal(
    (await call("DELETE", "/v1/task-types/gossip", "admin")).status,
    404,
  );
});

test("tag and analytics reads fail closed with 503 when their store is down", async () => {
  analyticsDown = true;
  const res = [
    await call("GET", "/v1/task-types", "analyst"),
    await call("GET", "/v1/usage/summary?groupBy=model", "analyst"),
    await tagged("acme", { "bae6-task-type": "summarization" }),
  ];
  analyticsDown = false;
  for (const r of res) assert.equal(r.status, 503);
});

test("one organization at its concurrency limit is refused while others are served", async () => {
  let release!: () => void;
  hold = new Promise((r) => {
    release = r;
  });
  entered = 0;
  held = 2;
  const open = { ...hi, model: "open" };
  const calls = [post("acme", open), post("acme", open)];
  while (entered < 2) await new Promise((r) => setTimeout(r, 5));
  const over = await post("acme", open);
  assert.equal(over.status, 429);
  assert.equal(over.headers.get("retry-after"), "1");
  const other = await post("globex", open);
  assert.equal(other.status, 200);
  held = 0;
  release();
  assert.deepEqual(
    (await Promise.all(calls)).map((r) => r.status),
    [200, 200],
  );
  assert.equal((await post("acme", open)).status, 200);
});

const ratedId = `chatcmpl-led_${"0".repeat(8)}-0000-0000-0000-${"1".repeat(12)}`;
const rate = (token: string, body: unknown) =>
  call("POST", "/v1/feedback", token, body);

test("feedback needs its own scopes", async () => {
  const v = { requestId: ratedId, verdict: "positive" };
  assert.equal((await rate("noscope", v)).status, 403);
  assert.equal((await rate("reader", v)).status, 403);
  assert.equal((await rate("auditor", v)).status, 403);
  assert.equal((await call("GET", "/v1/feedback", "rater")).status, 403);
});

test("feedback on another organization's request is the same 404 as an unknown one", async () => {
  ratable.set("led_00000000-0000-0000-0000-111111111111", "acme");
  rated.length = 0;
  const body = { requestId: ratedId, verdict: "negative", label: "unsafe" };
  const own = await rate("rater", body);
  assert.equal(own.status, 200);
  conforms("FeedbackAck", await own.json());
  const other = await rate("rater2", body);
  const unknown = await rate("rater", {
    ...body,
    requestId: ratedId.replace("1111", "2222"),
  });
  assert.equal(other.status, 404);
  assert.deepEqual(
    [(await other.json()).message, unknown.status],
    [(await unknown.json()).message, 404],
  );
  assert.equal(rated.length, 1);
});

test("feedback refuses malformed, oversized and unknown fields", async () => {
  const ok = { requestId: ratedId, verdict: "positive" };
  for (const bad of [
    { ...ok, requestId: "led_00000000-0000-0000-0000-111111111111" },
    { ...ok, requestId: "chatcmpl-req_x" },
    { ...ok, verdict: "meh" },
    { ...ok, label: "spicy" },
    { ...ok, comment: 5 },
    { ...ok, comment: "é".repeat(501) },
    { ...ok, comment: "a\u0000b" },
    [],
  ]) {
    assert.equal((await rate("rater", bad)).status, 400, JSON.stringify(bad));
  }
  const extra = await rate("rater", { ...ok, organizationId: "globex" });
  assert.equal(extra.status, 400);
  assert.equal((await extra.json()).code, "unsupported_parameter");
  assert.equal(
    (await rate("rater", { ...ok, comment: "é".repeat(500) })).status,
    200,
  );
});

test("feedback listing is paged within bounds and conforms to the contract", async () => {
  const res = await call("GET", "/v1/feedback", "auditor");
  assert.equal(res.status, 200);
  conforms("FeedbackList", await res.json());
  for (const q of ["limit=0", "limit=101", "limit=x", "cursor=abc"]) {
    const bad = await call("GET", `/v1/feedback?${q}`, "auditor");
    assert.equal(bad.status, 400, q);
  }
});

test("a feedback outage fails feedback calls but not inference", async () => {
  feedbackDown = true;
  const res = [
    await rate("rater", { requestId: ratedId, verdict: "positive" }),
    await call("GET", "/v1/feedback", "auditor"),
  ];
  const chat = await post("acme", hi);
  feedbackDown = false;
  for (const r of res) assert.equal(r.status, 503);
  assert.equal(chat.status, 200);
});

test("one organization's feedback writes are limited without affecting another", async () => {
  ratable.set("led_00000000-0000-0000-0000-111111111111", "umbrella");
  const v = { requestId: ratedId, verdict: "positive" };
  const statuses: number[] = [];
  for (let i = 0; i < 121; i++)
    statuses.push((await rate("spammer", v)).status);
  assert.equal(statuses.filter((s) => s === 200).length, 120);
  assert.equal(statuses[120], 429);
  ratable.set("led_00000000-0000-0000-0000-111111111111", "acme");
  assert.equal((await rate("rater", v)).status, 200);
});

test("credential status lists only providers the caller's organization has a model on", async () => {
  const res = await call("GET", "/v1/credentials", "reader");
  const { data } = (await res.json()) as { data: { provider: string }[] };
  assert.deepEqual(
    data.map((c) => c.provider),
    ["bedrock"],
  );
});
