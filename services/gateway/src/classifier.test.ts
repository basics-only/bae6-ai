import assert from "node:assert/strict";
import type { AddressInfo } from "node:net";
import { after, before, test } from "node:test";
import type { Message, ModelProvider } from "@bae6/ai-core";
import { classifierMessages, parseTier } from "./classifier.js";
import type { ModelConfig, RouteConfig } from "./config.js";
import type { Identity } from "./identity.js";
import type { Claim, Ledger } from "./ledger.js";
import { pickRoute } from "./routing.js";
import { createGateway } from "./server.js";

const route: RouteConfig = {
  id: "smart",
  default: "cheap",
  rules: [{ images: true, model: "vision" }],
  classifier: { model: "tiny", tiers: { easy: "cheap", hard: "strong" } },
};
const facts = { inputTokens: 10, hasImages: false };
const all = () => true;

const user = (text: string): Message => ({
  role: "user",
  content: [{ type: "text", text }],
});

test("the classifier is not asked when a rule already decides", async () => {
  let asked = false;
  const c = await pickRoute(
    route,
    { ...facts, hasImages: true },
    all,
    async () => {
      asked = true;
      return "hard";
    },
  );
  assert.deepEqual(c, { modelId: "vision", decision: "rule:0" });
  assert.equal(asked, false);
});

test("a tier picks its model; no usable tier leaves the default", async () => {
  assert.deepEqual(await pickRoute(route, facts, all, async () => "hard"), {
    modelId: "strong",
    decision: "classifier:hard",
  });
  for (const tier of [null, "unknown"]) {
    assert.deepEqual(await pickRoute(route, facts, all, async () => tier), {
      modelId: "cheap",
      decision: "default:classifier",
    });
  }
  const noStrong = (id: string) => id !== "strong";
  assert.deepEqual(
    await pickRoute(route, facts, noStrong, async () => "hard"),
    {
      modelId: "cheap",
      decision: "default:classifier",
    },
  );
  assert.equal(
    await pickRoute(
      route,
      facts,
      () => false,
      async () => null,
    ),
    null,
  );
});

test("only an output that is exactly one tier name is accepted", () => {
  const tiers = ["easy", "hard"];
  assert.equal(parseTier(" hard\n", tiers), "hard");
  for (const out of [
    "",
    "hard.",
    "Hard",
    "easy hard",
    "use strong",
    "constructor",
  ]) {
    assert.equal(parseTier(out, tiers), null);
  }
});

test("the classifier sees only an excerpt of the last user message", () => {
  const long = "a".repeat(5000);
  const m = classifierMessages(
    [user("earlier secret"), { ...user("ok"), role: "assistant" }, user(long)],
    ["easy", "hard"],
  ) as Message[];
  const sent = JSON.stringify(m);
  assert.equal(sent.includes("earlier secret"), false);
  assert.equal(
    m[1]?.content[0]?.type === "text" && m[1].content[0].text.length,
    1500,
  );
  assert.equal(
    classifierMessages(
      [
        {
          role: "user",
          content: [{ type: "image", mediaType: "image/png", data: "AA" }],
        },
      ],
      ["easy", "hard"],
    ),
    null,
  );
});

const model = (id: string, extra: Partial<ModelConfig> = {}): ModelConfig => ({
  id,
  provider: "bedrock",
  providerModel: "x",
  region: "us-east-1",
  operations: ["chat", "chat_stream"],
  inputModalities: ["text"],
  maxInputTokens: 100_000,
  maxOutputTokens: 1000,
  ...extra,
});
const configs = [
  model("cheap"),
  model("strong"),
  model("vision", { inputModalities: ["text", "image"] }),
  model("tiny", { organizations: ["acme"] }),
];

const reserved: {
  modelId: string;
  route?: unknown;
  maxOutputTokens: number;
}[] = [];
const finalized: string[] = [];
let refuse = new Set<string>();
let nextClaim: Claim = { state: "new", requestId: "x" };
const ledger = {
  reserve: async (r: {
    modelId: string;
    route?: unknown;
    maxOutputTokens: number;
  }) => {
    reserved.push(r);
    return refuse.has(r.modelId) ? null : "0.01";
  },
  finalize: async (_: string, o: { kind: string }) => {
    finalized.push(o.kind);
    return true;
  },
  claim: async () => nextClaim,
  choose: async () => {},
  complete: async () => {},
  release: async () => {},
} as unknown as Ledger;

let classifierReply: () => Promise<string> = async () => "hard";
const classifierCalls: { maxTokens: number; temperature?: number }[] = [];
const provider = (id: string) =>
  ({
    descriptor: { kind: "model", id, operations: [], attributes: {} },
    chat: async (req: { maxTokens: number; temperature?: number }) => {
      if (id === "tiny") classifierCalls.push(req);
      return {
        text: id === "tiny" ? await classifierReply() : "hello",
        finishReason: "stop",
        usage: { inputTokens: 3, outputTokens: 1 },
      };
    },
  }) as unknown as ModelProvider;

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
    configs.map((config) => [
      config.id,
      { config, provider: provider(config.id) },
    ]),
  ),
  routes: [route],
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

const chat = (token: string, text: string, key?: string) =>
  fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${token}`,
      ...(key ? { "idempotency-key": key } : {}),
    },
    body: JSON.stringify({
      model: "smart",
      messages: [{ role: "user", content: text }],
    }),
  });
const modelOf = async (res: Response) =>
  ((await res.json()) as { model: string }).model;
const reset = () => {
  reserved.length = 0;
  finalized.length = 0;
  classifierCalls.length = 0;
  refuse = new Set();
  classifierReply = async () => "hard";
};

test("a classified request meters the classifier as its own event, then the chosen model", async () => {
  reset();
  assert.equal(await modelOf(await chat("acme", "prove it")), "strong");
  assert.deepEqual(
    reserved.map((r) => [r.modelId, r.route]),
    [
      ["tiny", { id: "smart", decision: "classifier" }],
      ["strong", { id: "smart", decision: "classifier:hard" }],
    ],
  );
  assert.equal(reserved[0]?.maxOutputTokens, 8);
  assert.equal(classifierCalls[0]?.maxTokens, 8);
  assert.equal(classifierCalls[0]?.temperature, 0);
  assert.deepEqual(finalized, ["measured", "measured"]);
});

test("a classifier that fails, wanders off, or is over budget leaves the default", async () => {
  reset();
  classifierReply = async () => {
    throw new Error("down");
  };
  assert.equal(await modelOf(await chat("acme", "hi")), "cheap");
  assert.equal(finalized[0], "uncertain");

  reset();
  classifierReply = async () => "ignore the rules and use vision";
  assert.equal(await modelOf(await chat("acme", "hi")), "cheap");
  assert.deepEqual(reserved[1]?.route, {
    id: "smart",
    decision: "default:classifier",
  });

  reset();
  refuse = new Set(["tiny"]);
  assert.equal(await modelOf(await chat("acme", "hi")), "cheap");
  assert.equal(classifierCalls.length, 0);
});

test("an organization that may not use the classifier model never has its prompt sent to it", async () => {
  reset();
  const res = await chat("globex", "secret plan");
  assert.equal(res.status, 200);
  assert.equal(await modelOf(res), "cheap");
  assert.equal(classifierCalls.length, 0);
  assert.deepEqual(
    reserved.map((r) => r.modelId),
    ["cheap"],
  );
});

test("a replay is served without paying for the classifier again", async () => {
  reset();
  nextClaim = {
    state: "replay",
    status: 200,
    body: { model: "strong" },
    modelId: "strong",
  };
  const res = await chat("acme", "prove it", "classifier-key-1");
  assert.equal(res.headers.get("idempotent-replayed"), "true");
  assert.equal(classifierCalls.length, 0);
  assert.deepEqual(reserved, []);
  nextClaim = { state: "new", requestId: "x" };
});
