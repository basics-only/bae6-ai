import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { after, before, test } from "node:test";
import { type ModelProvider, ProviderError } from "@bae6/ai-core";
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
  descriptor: {},
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
