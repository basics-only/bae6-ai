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

test("rejects a route to a model that cannot chat", () => {
  const embedOnly = { ...config({ operations: ["embed"] }), routes: [route] };
  assert.throws(() => parseConfig(embedOnly), /invalid default/);
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
