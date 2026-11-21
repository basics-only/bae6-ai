import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { after, before, test } from "node:test";
import { gatewayModel } from "./model.js";

let reply: { status: number; body: unknown } = { status: 200, body: {} };
let seen: {
  auth?: string | undefined;
  runId?: string | undefined;
  body?: { max_tokens: number; messages: { role: string; content: string }[] };
} = {};
let base: string;

const server = createServer((req, res) => {
  let raw = "";
  req.on("data", (c) => {
    raw += c;
  });
  req.on("end", () => {
    seen = {
      auth: req.headers.authorization,
      runId: req.headers["bae6-run-id"] as string | undefined,
      body: JSON.parse(raw),
    };
    res.writeHead(reply.status, { "content-type": "application/json" });
    res.end(JSON.stringify(reply.body));
  });
});
before(async () => {
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});
after(() => server.close());

const completion = (
  content: string,
  usage: object = { prompt_tokens: 4, completion_tokens: 2 },
) => ({
  status: 200,
  body: { choices: [{ message: { content } }], usage },
});
const model = () =>
  gatewayModel({
    baseUrl: base,
    token: async () => "tok",
    model: "m",
    runId: "run_abc",
    maxOutputTokens: 100,
    requestTimeoutMs: 2000,
  });
const search = [{ name: "search", description: "find things" }];
const ask = [{ role: "user" as const, text: "hi" }];
const next = (tools: typeof search, remaining = 1000) =>
  model().next(ask, tools, new AbortController().signal, remaining);

test("every model call carries the run id so its spend can be attributed", async () => {
  reply = completion('{"final": "x"}');
  await next([]);
  assert.equal(seen.runId, "run_abc");
});

test("without tools the reply is the final answer, as the model wrote it", async () => {
  reply = completion('{"final": "not a protocol"}');
  const turn = await next([]);
  assert.deepEqual(turn, {
    type: "final",
    text: '{"final": "not a protocol"}',
    usage: { inputTokens: 4, outputTokens: 2 },
  });
  assert.equal(seen.auth, "Bearer tok");
});

test("with tools a JSON reply is read as a tool call or a final answer", async () => {
  reply = completion('{"tool": "search", "input": {"q": "x"}}');
  const call = await next(search);
  assert.deepEqual(
    [call.type, call.type === "tool" && call.call],
    ["tool", { name: "search", input: { q: "x" } }],
  );
  reply = completion('{"final": "done"}');
  const done = await next(search);
  assert.deepEqual(
    [done.type, done.type === "final" && done.text],
    ["final", "done"],
  );
});

test("a reply that is neither a call nor an answer is an error, not a guess", async () => {
  for (const content of [
    "plain prose",
    '{"tool": "a", "final": "b"}',
    "[]",
    "null",
    "{}",
  ]) {
    reply = completion(content);
    await assert.rejects(next(search), Error, content);
  }
});

test("the tool list and the remaining budget bound what the model is asked", async () => {
  reply = completion('{"final": "x"}');
  await next(search, 7);
  assert.equal(seen.body?.max_tokens, 7);
  assert.match(seen.body?.messages[0]?.content ?? "", /search: find things/);
  await next(search, 5000);
  assert.equal(seen.body?.max_tokens, 100);
});

test("a gateway error is raised rather than turned into an answer", async () => {
  reply = { status: 402, body: { code: "budget_exceeded" } };
  await assert.rejects(next([]), /402/);
});

test("missing token counts stay missing so the agent can reject them", async () => {
  reply = completion("hi", {});
  const turn = await next([]);
  assert.equal(turn.usage.inputTokens, undefined);
});
