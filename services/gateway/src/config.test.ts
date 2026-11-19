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
