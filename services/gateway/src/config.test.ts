import assert from "node:assert/strict";
import { test } from "node:test";
import { parseConfig } from "./config.js";

const config = (model: Record<string, unknown>) => ({
  priceBook: {
    version: "t",
    prices: { m: { inputPerMTok: "1", outputPerMTok: "1" } },
  },
  models: [
    {
      id: "m",
      provider: "bedrock",
      providerModel: "x",
      region: "us-east-1",
      operations: ["chat"],
      inputModalities: ["text"],
      maxInputTokens: 10,
      maxOutputTokens: 10,
      ...model,
    },
  ],
  budgets: {},
});

const withRoutes = (routes: unknown) => ({ ...config({}), routes });
const route = {
  id: "auto",
  default: "m",
  rules: [{ model: "m", images: true }],
};

test("accepts a route over configured chat models", () => {
  assert.equal(parseConfig(withRoutes([route])).routes[0]?.id, "auto");
  assert.deepEqual(parseConfig(config({})).routes, []);
});

test("rejects routes that name unknown models, collide with a model, or carry unsupported keys", () => {
  for (const bad of [
    { ...route, id: "m" },
    { ...route, default: "nope" },
    { ...route, rules: [{ model: "nope" }] },
    { ...route, rules: [{ model: "m", minInputTokens: 9, maxInputTokens: 1 }] },
    { ...route, rules: [{ model: "m", when: "always" }] },
    { ...route, classifier: { model: "m" } },
    { ...route, organizations: ["Not A Slug!"] },
  ]) {
    assert.throws(() => parseConfig(withRoutes([bad])), /route|invalid/);
  }
  assert.throws(() => parseConfig(withRoutes([route, route])), /collides/);
});

const withClassifier = (
  classifier: unknown,
  extra: Record<string, unknown> = {},
  tiny: Record<string, unknown> = {},
) => ({
  ...config({}),
  models: [
    ...config({}).models,
    { ...config({}).models[0], id: "tiny", ...tiny },
  ],
  priceBook: {
    version: "t",
    prices: {
      m: { inputPerMTok: "1", outputPerMTok: "1" },
      tiny: { inputPerMTok: "1", outputPerMTok: "1" },
    },
  },
  routes: [{ ...route, classifier, ...extra }],
});
const tiers = { easy: "tiny", hard: "m" };

test("accepts a classifier whose tiers map to chat models", () => {
  const parsed = parseConfig(withClassifier({ model: "tiny", tiers }));
  assert.equal(parsed.routes[0]?.classifier?.model, "tiny");
});

test("rejects classifiers that are malformed, name unknown models, or could expose prompts", () => {
  for (const [classifier, extra, tiny] of [
    [{ model: "nope", tiers }, {}, {}],
    [{ model: "tiny", tiers: { only: "m" } }, {}, {}],
    [{ model: "tiny", tiers: { Easy: "tiny", hard: "m" } }, {}, {}],
    [{ model: "tiny", tiers: { easy: "tiny", hard: "nope" } }, {}, {}],
    [{ model: "tiny", tiers, prompt: "be nice" }, {}, {}],
    [{ model: "tiny", tiers }, {}, { operations: ["embed"] }],
    [{ model: "tiny", tiers }, {}, { organizations: ["acme"] }],
    [
      { model: "tiny", tiers },
      { organizations: ["acme", "globex"] },
      { organizations: ["acme"] },
    ],
  ] as const) {
    assert.throws(
      () => parseConfig(withClassifier(classifier, extra, tiny)),
      /route|invalid/,
    );
  }
  const scoped = withClassifier(
    { model: "tiny", tiers },
    { organizations: ["acme"] },
    { organizations: ["acme", "globex"] },
  );
  assert.equal(parseConfig(scoped).routes.length, 1);
});

test("rejects a route to a model that cannot chat", () => {
  const embedOnly = { ...config({ operations: ["embed"] }), routes: [route] };
  assert.throws(() => parseConfig(embedOnly), /invalid default/);
});

const hf = (extra: Record<string, unknown> = {}) =>
  config({
    provider: "huggingface",
    providerModel: "meta-llama/Llama-3.1-8B-Instruct",
    region: undefined,
    hosting: "dedicated",
    endpoint: "https://abc.endpoints.huggingface.cloud",
    tokenEnv: "BAE6_HF_TOKEN_LLAMA",
    ...extra,
  });

test("accepts Hugging Face dedicated endpoints and org-limited Inference Providers", () => {
  assert.equal(parseConfig(hf()).models[0]?.hosting, "dedicated");
  const shared = hf({
    hosting: "inference-providers",
    endpoint: undefined,
    organizations: ["acme"],
  });
  assert.equal(parseConfig(shared).models[0]?.hosting, "inference-providers");
});

test("refuses Hugging Face configs that could send the token or prompts somewhere unintended", () => {
  for (const bad of [
    { tokenEnv: "BAE6_DATABASE_URL" },
    { tokenEnv: "BAE6_HF_TOKEN_" },
    { tokenEnv: undefined },
    { endpoint: "https://evil.example" },
    { endpoint: "https://abc.endpoints.huggingface.cloud.evil.example" },
    { endpoint: "http://abc.endpoints.huggingface.cloud" },
    { endpoint: undefined },
    { hosting: "elsewhere" },
    { hosting: undefined },
    { hosting: "inference-providers", endpoint: undefined },
    {
      hosting: "inference-providers",
      endpoint: "https://abc.endpoints.huggingface.cloud",
      organizations: ["acme"],
    },
    { hosting: "inference-providers", endpoint: undefined, organizations: [] },
    { region: "us-east-1" },
    { operations: ["chat", "embed"] },
  ]) {
    assert.throws(() => parseConfig(hf(bad)), /invalid|gateway config/);
  }
  assert.throws(
    () => parseConfig(config({ tokenEnv: "BAE6_HF_TOKEN_X" })),
    /invalid hosting or tokenEnv/,
  );
});

test("refuses a Bedrock or Vertex model whose region is missing or not a string", () => {
  for (const region of [undefined, null, 7]) {
    assert.throws(() => parseConfig(config({ region })), /invalid region/);
  }
});

test("accepts a bedrock endpoint that is a bare origin", () => {
  const parsed = parseConfig(config({ endpoint: "http://mock-bedrock:8080" }));
  assert.equal(parsed.models[0]?.endpoint, "http://mock-bedrock:8080");
});

test("stores the normalized origin so a gate on it cannot be bypassed by case or whitespace", () => {
  const parsed = parseConfig(config({ endpoint: " HTTP://Mock:8080/ " }));
  assert.equal(parsed.models[0]?.endpoint, "http://mock:8080");
});

test("rejects endpoints that carry a path, credentials or the wrong provider", () => {
  for (const endpoint of [
    "https://host/path",
    "https://user:pw@host",
    "https://host?a=1",
    "ftp://host",
    "not a url",
    42,
  ]) {
    assert.throws(() => parseConfig(config({ endpoint })), /invalid endpoint/);
  }
  assert.throws(
    () =>
      parseConfig(
        config({
          provider: "vertex",
          project: "my-project",
          endpoint: "https://host",
        }),
      ),
    /invalid endpoint/,
  );
});
