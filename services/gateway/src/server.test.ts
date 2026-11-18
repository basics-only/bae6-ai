import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { join, resolve } from "node:path";
import { after, before, test } from "node:test";
import { type ModelProvider, ProviderError } from "@bae6/ai-core";
import AjvModule from "ajv/dist/2020.js";
import { parse } from "yaml";
import type { ModelConfig } from "./config.js";
import type { Identity } from "./identity.js";
import type { Ledger, Outcome } from "./ledger.js";
import { createGateway } from "./server.js";

const finalized: Outcome[] = [];
const keys: string[] = [];
let fail: ProviderError | undefined;
let ledgerDown = false;

const ledger = {
  reserve: async () => {
    if (ledgerDown) throw new Error("db down");
    return "0.01";
  },
  finalize: async (_: string, o: Outcome) => {
    finalized.push(o);
    return true;
  },
  claim: async () => ({ state: "new" }),
  complete: async (_: unknown, _k: string, status: number) => {
    keys.push(`complete ${status}`);
  },
  release: async () => {
    keys.push("release");
  },
  budget: async () => ({
    period: "2026-11",
    limit: "10",
    spent: "1.5",
    reserved: "0",
  }),
  usage: async () => [
    {
      model: "m",
      outcome: "measured",
      requests: 2,
      inputTokens: 6,
      outputTokens: 4,
      cost: "0.0001",
    },
  ],
} as unknown as Ledger;

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
  nobudget: {
    subjectId: "svc",
    subjectType: "workload",
    organizationId: "initech",
    applicationId: "crm",
    scopes: ["ai:invoke"],
  },
};

const server = createGateway({
  verify: async (t) => (t ? (identities[t] ?? null) : null),
  ledger,
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
        : Object.keys(item).map((m) => [m.toUpperCase(), path]),
  );
  assert.equal(routes.length, 7);
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
  for (const [res, status, code] of [
    [await post(undefined, hi), 401, "unauthenticated"],
    [await post("noscope", hi), 403, "forbidden"],
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
