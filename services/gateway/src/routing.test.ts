import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { after, before, test } from "node:test";
import type { ModelProvider } from "@bae6/ai-core";
import type { ModelConfig, RouteConfig } from "./config.js";
import type { Identity } from "./identity.js";
import type { Claim, Ledger } from "./ledger.js";
import { pick } from "./routing.js";
import { createGateway } from "./server.js";

const route: RouteConfig = {
  id: "auto",
  default: "cheap",
  rules: [
    { images: true, model: "vision" },
    { minInputTokens: 500, model: "vision" },
  ],
};
const text = { inputTokens: 10, hasImages: false };

test("the first matching rule whose model is usable wins", () => {
  assert.deepEqual(
    pick(route, { ...text, hasImages: true }, () => true),
    { modelId: "vision", decision: "rule:0" },
  );
  assert.deepEqual(
    pick(route, { ...text, inputTokens: 600 }, () => true),
    { modelId: "vision", decision: "rule:1" },
  );
  assert.deepEqual(
    pick(route, text, () => true),
    { modelId: "cheap", decision: "default" },
  );
});

test("a matching rule whose model is unusable is skipped, never forced", () => {
  const onlyCheap = (id: string) => id === "cheap";
  assert.deepEqual(pick(route, { ...text, hasImages: true }, onlyCheap), {
    modelId: "cheap",
    decision: "default:skipped",
  });
  assert.equal(
    pick(route, text, () => false),
    null,
  );
});

const cheap: ModelConfig = {
  id: "cheap",
  provider: "bedrock",
  providerModel: "x",
  region: "us-east-1",
  operations: ["chat", "chat_stream", "embed"],
  inputModalities: ["text"],
  maxInputTokens: 1000,
  maxOutputTokens: 100,
};
const vision: ModelConfig = {
  ...cheap,
  id: "vision",
  operations: ["chat", "chat_stream"],
  inputModalities: ["text", "image"],
  maxInputTokens: 100_000,
  maxOutputTokens: 1000,
  organizations: ["acme"],
};

const reserved: { modelId: string; route?: unknown }[] = [];
const chosen: string[] = [];
const keys: string[] = [];
let nextClaim: Claim = { state: "new", requestId: "x" };
const ledger = {
  reserve: async (r: { modelId: string; route?: unknown }) => {
    reserved.push(r);
    return "0.01";
  },
  finalize: async () => true,
  claim: async () => nextClaim,
  choose: async (_: unknown, _k: string, modelId: string) => {
    chosen.push(modelId);
  },
  complete: async () => {},
  release: async () => {
    keys.push("release");
  },
} as unknown as Ledger;

const provider = {
  descriptor: { kind: "model", id: "x", operations: [], attributes: {} },
  chat: async () => ({
    text: "hello",
    finishReason: "stop",
    usage: { inputTokens: 3, outputTokens: 2 },
  }),
} as unknown as ModelProvider;

const who = (organizationId: string): Identity => ({
  subjectId: "svc",
  subjectType: "workload",
  organizationId,
  applicationId: "crm",
  scopes: ["ai:invoke"],
});
const identities: Record<string, Identity> = {
  acme: who("acme"),
  globex: who("globex"),
};

const server = createGateway({
  credentials: () => [],
  verify: async (t) => (t ? (identities[t] ?? null) : null),
  ledger,
  feedback: {} as never,
  analytics: {} as never,
  models: new Map(
    [cheap, vision].map((config) => [config.id, { config, provider }]),
  ),
  routes: [route, { ...route, id: "internal", organizations: ["acme"] }],
  budgets: { acme: "10", globex: "10" },
  timeoutMs: 5000,
  maxInflight: 8,
  maxInflightPerOrg: 8,
  ready: async () => true,
});
let base: string;
before(async () => {
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});
after(() => server.close());

const call = (path: string, token: string, body: unknown, key?: string) =>
  fetch(`${base}${path}`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${token}`,
      ...(key ? { "idempotency-key": key } : {}),
    },
    body: JSON.stringify(body),
  });
const chat = (token: string, content: unknown, model = "auto", key?: string) =>
  call(
    "/v1/chat/completions",
    token,
    { model, messages: [{ role: "user", content }] },
    key,
  );
const picture = [
  { type: "text", text: "what is this" },
  { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
];
const modelOf = async (res: Response) =>
  ((await res.json()) as { model: string }).model;

test("a route sends short text to the default and long text or images to the vision model", async () => {
  reserved.length = 0;
  assert.equal(await modelOf(await chat("acme", "hi")), "cheap");
  assert.equal(await modelOf(await chat("acme", "x".repeat(1800))), "vision");
  assert.equal(await modelOf(await chat("acme", picture)), "vision");
  assert.deepEqual(
    reserved.map((r) => [r.modelId, r.route]),
    [
      ["cheap", { id: "auto", decision: "default" }],
      ["vision", { id: "auto", decision: "rule:1" }],
      ["vision", { id: "auto", decision: "rule:0" }],
    ],
  );
});

test("a route never uses a model the organization may not use", async () => {
  reserved.length = 0;
  const res = await chat("globex", "x".repeat(1800));
  assert.equal(res.status, 200);
  assert.equal(reserved[0]?.modelId, "cheap");
  assert.deepEqual(reserved[0]?.route, {
    id: "auto",
    decision: "default:skipped",
  });
});

test("a request no candidate can serve is a 4xx that frees the idempotency key", async () => {
  keys.length = 0;
  reserved.length = 0;
  const res = await chat("globex", picture, "auto", "route-key-1");
  assert.equal(res.status, 400);
  assert.equal(
    ((await res.json()) as { code: string }).code,
    "no_eligible_model",
  );
  assert.deepEqual(keys, ["release"]);
  assert.deepEqual(reserved, []);
});

test("a route hidden from the organization looks like a missing model", async () => {
  assert.equal((await chat("globex", "hi", "internal")).status, 404);
  assert.equal((await chat("acme", "hi", "internal")).status, 200);
});

test("an embed on a route is refused", async () => {
  const res = await call("/v1/embeddings", "acme", {
    model: "auto",
    input: "hi",
  });
  assert.equal(res.status, 400);
  assert.equal(
    ((await res.json()) as { code: string }).code,
    "unsupported_operation",
  );
});

test("a keyed route request records the model it chose on the claim", async () => {
  chosen.length = 0;
  await chat("acme", picture, "auto", "route-key-2");
  assert.deepEqual(chosen, ["vision"]);
});

test("a replay is served without routing again, unless the chosen model is no longer allowed", async () => {
  reserved.length = 0;
  nextClaim = {
    state: "replay",
    status: 200,
    body: { model: "vision" },
    modelId: "vision",
  };
  const ok = await chat("acme", picture, "auto", "route-key-3");
  assert.equal(ok.status, 200);
  assert.equal(ok.headers.get("idempotent-replayed"), "true");
  assert.deepEqual(reserved, []);
  const lost = await chat("globex", "hi", "auto", "route-key-3");
  assert.equal(lost.status, 409);
  nextClaim = { state: "new", requestId: "x" };
});

test("/v1/models lists a route with only the limits of models the caller may use", async () => {
  const list = async (token: string) =>
    (
      (await (
        await fetch(`${base}/v1/models`, {
          headers: { authorization: `Bearer ${token}` },
        })
      ).json()) as {
        data: {
          id: string;
          object: string;
          owned_by?: string;
          capabilities: { limits: { maxInputTokens: number } };
        }[];
      }
    ).data.filter((m) => m.object === "route");
  const [acme] = await list("acme");
  assert.equal(acme?.capabilities.limits.maxInputTokens, 100_000);
  assert.equal(acme?.owned_by, undefined);
  const globex = await list("globex");
  assert.deepEqual(
    globex.map((r) => [r.id, r.capabilities.limits.maxInputTokens]),
    [["auto", 1000]],
  );
});
