import assert from "node:assert/strict";
import { createServer, type IncomingHttpHeaders } from "node:http";
import type { AddressInfo } from "node:net";
import { after, before, test } from "node:test";
import { inspect } from "node:util";
import {
  Ai,
  AiApiError,
  AiAuthError,
  AiBudgetError,
  AiTimeoutError,
  AiTransportError,
} from "./index.js";

let seen: IncomingHttpHeaders = {};
let reply: (res: import("node:http").ServerResponse) => void;
const server = createServer((req, res) => {
  seen = req.headers;
  req.resume();
  req.on("end", () => reply(res));
});
let ai: Ai;

before(async () => {
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  ai = new Ai({
    baseUrl: `http://127.0.0.1:${(server.address() as AddressInfo).port}/`,
    credentials: "svc-token",
  });
});
after(() => server.close());

const json =
  (status: number, body: unknown) =>
  (res: import("node:http").ServerResponse) => {
    res.writeHead(status, { "content-type": "application/json" });
    res.end(JSON.stringify(body));
  };
const sse = (events: string[]) => (res: import("node:http").ServerResponse) => {
  res.writeHead(200, { "content-type": "text/event-stream" });
  res.end(events.map((e) => `data: ${e}\n\n`).join(""));
};
const req = {
  model: "m",
  messages: [{ role: "user" as const, content: "hi" }],
};
const usage = { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 };

test("sends credentials, the end-user token and the idempotency key", async () => {
  reply = json(200, { id: "c", model: "m", choices: [], usage });
  await ai.chat(req, {
    endUserToken: "user-token",
    idempotencyKey: "key-00001",
  });
  assert.equal(seen.authorization, "Bearer svc-token");
  assert.equal(seen["bae6-end-user-token"], "user-token");
  assert.equal(seen["idempotency-key"], "key-00001");
});

test("streams chunks and requires the stream to finish", async () => {
  const c = (content: string) =>
    JSON.stringify({
      id: "c",
      model: "m",
      choices: [{ index: 0, delta: { content }, finish_reason: null }],
    });
  reply = sse([
    c("hel"),
    c("lo"),
    JSON.stringify({ id: "c", model: "m", choices: [], usage }),
    "[DONE]",
  ]);
  const chunks = [];
  for await (const chunk of ai.chatStream(req)) chunks.push(chunk);
  assert.equal(chunks.length, 3);
  assert.deepEqual(chunks[2]?.usage, usage);

  reply = sse([c("hel")]);
  await assert.rejects(async () => {
    for await (const _ of ai.chatStream(req));
  }, AiTransportError);
});

test("a mid-stream error surfaces as an API error", async () => {
  reply = sse([
    JSON.stringify({
      error: { code: "timeout", message: "late", requestId: "r1" },
    }),
  ]);
  const err = await (async () => {
    for await (const _ of ai.chatStream(req));
  })().catch((e: unknown) => e);
  assert.ok(err instanceof AiApiError);
  assert.equal(err.code, "timeout");
  assert.equal(err.requestId, "r1");
});

test("maps budget, auth and throttling errors", async () => {
  reply = json(402, { code: "budget_exceeded", message: "no", requestId: "r" });
  await assert.rejects(ai.chat(req), AiBudgetError);
  reply = json(401, { code: "unauthenticated", message: "no", requestId: "r" });
  await assert.rejects(ai.chat(req), AiAuthError);
  reply = json(429, {
    code: "rate_limited",
    message: "slow",
    requestId: "r",
    retryAfterSeconds: 2,
  });
  const err = await ai.chat(req).catch((e: unknown) => e);
  assert.ok(err instanceof AiApiError);
  assert.equal(err.retryAfterSeconds, 2);
});

test("a transport error does not carry a token, even in its cause chain", async () => {
  const down = new Ai({ baseUrl: "http://127.0.0.1:9", credentials: "svc" });
  const err = await down
    .chat(req, { endUserToken: "SECRET\nX" })
    .catch((e: unknown) => e);
  assert.ok(err instanceof AiTransportError);
  assert.ok(!inspect(err, { depth: 10 }).includes("SECRET"));
});

test("reads a last stream line sent without a newline", async () => {
  reply = (res) => {
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.end(
      `data: ${JSON.stringify({ id: "c", model: "m", choices: [], usage })}\n\ndata: [DONE]`,
    );
  };
  const chunks = [];
  for await (const chunk of ai.chatStream(req)) chunks.push(chunk);
  assert.deepEqual(chunks[0]?.usage, usage);
});

test("a stream that stalls times out instead of hanging", async () => {
  const stalled = new Ai({
    baseUrl: `http://127.0.0.1:${(server.address() as AddressInfo).port}`,
    credentials: "svc",
    timeoutMs: 200,
  });
  reply = (res) => {
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.write(": open\n\n");
  };
  await assert.rejects(async () => {
    for await (const _ of stalled.chatStream(req));
  }, AiTimeoutError);
});

test("an oversized response is reported as too large", async () => {
  reply = (res) => {
    res.writeHead(200, { "content-type": "application/json" });
    const mb = Buffer.alloc(1024 * 1024, 0x20);
    for (let i = 0; i < 65; i++) res.write(mb);
    res.end("{}");
  };
  await assert.rejects(ai.chat(req), /too large/);
});
