import assert from "node:assert/strict";
import { test } from "node:test";
import { parseConfig } from "./config.js";
import { HttpError } from "./http.js";
import { parseChat, parseEmbeddings } from "./openai.js";

const code = (fn: () => unknown) => {
  try {
    fn();
  } catch (err) {
    assert.ok(err instanceof HttpError);
    return err.code;
  }
  assert.fail("expected an HttpError");
};

const chat = (extra: Record<string, unknown> = {}) => ({
  model: "m",
  messages: [{ role: "user", content: "hi" }],
  ...extra,
});

test("parses the supported chat subset, with images inline", () => {
  const input = parseChat(
    chat({
      messages: [
        { role: "developer", content: "Be brief." },
        {
          role: "user",
          content: [
            { type: "text", text: "What is this?" },
            {
              type: "image_url",
              image_url: { url: "data:image/png;base64,aGk=" },
            },
          ],
        },
      ],
      max_tokens: 50,
      stream: true,
      stream_options: { include_usage: true },
    }),
  );
  assert.equal(input.messages[0]?.role, "system");
  assert.deepEqual(input.messages[1]?.content[1], {
    type: "image",
    mediaType: "image/png",
    data: "aGk=",
  });
  assert.equal(input.maxTokens, 50);
  assert.equal(input.includeUsage, true);
  // The estimate over-reserves: text bytes / 3, a flat cost per image, per-message overhead.
  assert.ok(input.inputTokens >= 1600);
});

test("rejects what it does not support instead of ignoring it", () => {
  assert.equal(
    code(() => parseChat(chat({ tools: [] }))),
    "unsupported_parameter",
  );
  assert.equal(
    code(() => parseChat(chat({ n: 2 }))),
    "unsupported_parameter",
  );
  assert.equal(
    code(() => parseChat(chat({ messages: [{ role: "tool", content: "x" }] }))),
    "unsupported_parameter",
  );
  assert.equal(
    code(() =>
      parseChat(
        chat({
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "image_url",
                  image_url: { url: "http://169.254.169.254/x.png" },
                },
              ],
            },
          ],
        }),
      ),
    ),
    "invalid_request",
  );
  assert.equal(
    code(() => parseChat(chat({ max_tokens: 0 }))),
    "invalid_request",
  );
  assert.equal(
    code(() => parseChat(chat({ messages: [] }))),
    "invalid_request",
  );
  assert.equal(
    code(() => parseEmbeddings({ model: "e", input: Array(17).fill("x") })),
    "invalid_request",
  );
  assert.equal(
    code(() =>
      parseEmbeddings({ model: "e", input: "x", encoding_format: "base64" }),
    ),
    "unsupported_parameter",
  );
});

test("config refuses a model without a price, and float-looking money", () => {
  const base = {
    priceBook: {
      version: "v1",
      prices: { m: { inputPerMTok: "1", outputPerMTok: "2" } },
    },
    models: [
      {
        id: "m",
        provider: "bedrock",
        providerModel: "anthropic.x",
        region: "us-east-1",
        operations: ["chat"],
        inputModalities: ["text"],
        maxInputTokens: 1000,
        maxOutputTokens: 100,
      },
    ],
    budgets: { acme: "10.50" },
  };
  assert.equal(parseConfig(base).models.length, 1);
  assert.throws(
    () => parseConfig({ ...base, priceBook: { version: "v1", prices: {} } }),
    /no price/,
  );
  assert.throws(
    () => parseConfig({ ...base, budgets: { acme: "1e3" } }),
    /decimal/,
  );
  assert.throws(
    () => parseConfig({ ...base, budgets: { acme: 10 } }),
    /decimal/,
  );
});
