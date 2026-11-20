import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { after, before, test } from "node:test";
import { createExecutor } from "./execute.js";
import type { Claimed } from "./store.js";

let answer: (input: string) => { status: number; content?: unknown };
const seen: { auth?: string; model?: string; maxTokens?: number }[] = [];
const gateway = createServer((req, res) => {
  let raw = "";
  req.on("data", (c) => {
    raw += c;
  });
  req.on("end", () => {
    const body = JSON.parse(raw);
    seen.push({
      auth: req.headers.authorization,
      model: body.model,
      maxTokens: body.max_tokens,
    });
    const out = answer(body.messages[0].content);
    res.writeHead(out.status, { "content-type": "application/json" });
    res.end(
      JSON.stringify({ choices: [{ message: { content: out.content } }] }),
    );
  });
});
let baseUrl: string;

before(async () => {
  await new Promise<void>((r) => gateway.listen(0, "127.0.0.1", r));
  baseUrl = `http://127.0.0.1:${(gateway.address() as AddressInfo).port}`;
});
after(() => {
  gateway.close();
});

const executor = () =>
  createExecutor({
    baseUrl,
    token: async () => "worker-token",
    maxOutputTokens: 64,
    requestTimeoutMs: 2000,
    concurrency: 2,
    caseTimeoutMs: 2000,
  });
const claimed = (scorers: string[]): Claimed => ({
  id: "00000000-0000-0000-0000-000000000000",
  organizationId: "acme",
  attempt: 1,
  model: "m1",
  name: "smoke",
  datasetVersion: "1",
  scorers,
  cases: [
    { id: "a", input: "1+1", expected: "2" },
    { id: "b", input: "2+2", expected: "4" },
  ],
});

test("cases are sent to the gateway with the worker credential and scored", async () => {
  answer = (input) => ({
    status: 200,
    content: input === "1+1" ? " 2 " : "the answer is 4",
  });
  seen.length = 0;
  const out = await executor()(
    claimed(["exact", "contains"]),
    new AbortController().signal,
  );
  assert.equal(out.stop, "complete");
  const byId = Object.fromEntries(
    out.report?.cases.map((c) => [c.id, c.scores]) ?? [],
  );
  assert.deepEqual(byId, {
    a: { exact: 1, contains: 1 },
    b: { exact: 0, contains: 1 },
  });
  assert.ok(
    seen.every(
      (s) =>
        s.auth === "Bearer worker-token" &&
        s.model === "m1" &&
        s.maxTokens === 64,
    ),
  );
});

test("a gateway failure is recorded against its case, not the whole run", async () => {
  answer = (input) =>
    input === "1+1" ? { status: 500 } : { status: 200, content: "4" };
  const out = await executor()(
    claimed(["exact"]),
    new AbortController().signal,
  );
  assert.equal(out.stop, "complete");
  assert.equal(out.report?.failures, 1);
  assert.equal(out.report?.cases.find((c) => c.id === "a")?.status, "error");
});

test("an aborted run reports aborted without a report", async () => {
  answer = () => ({ status: 200, content: "x" });
  const ctl = new AbortController();
  ctl.abort();
  const out = await executor()(claimed(["exact"]), ctl.signal);
  assert.deepEqual(out, { stop: "aborted", report: undefined });
});
