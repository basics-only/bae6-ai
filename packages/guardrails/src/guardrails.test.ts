import assert from "node:assert/strict";
import { test } from "node:test";
import {
  guardTools,
  type Policy,
  piiDetectors,
  redact,
  secretDetectors,
} from "./index.js";

const policy: Policy = {
  maxInputChars: 100,
  maxOutputChars: 200,
  maxCalls: 2,
  input: secretDetectors,
  output: [...secretDetectors, ...piiDetectors],
  onOutput: "redact",
};

const echo = (reply: string) => ({
  name: "echo",
  parse: (i: unknown) => i as { q: string },
  run: async (_: { q: string }, _signal: AbortSignal) => reply,
});

const signal = new AbortController().signal;

function tool(reply: string, p: Policy = policy) {
  const [t] = guardTools([echo(reply)], { echo: p });
  if (!t) throw new Error("no tool");
  return t;
}

test("redacts secrets and PII from tool output", async () => {
  const out = await tool(
    "key AKIAABCDEFGHIJKLMNOP mail a.b@example.com ssn 123-45-6789",
  ).run({ q: "x" }, signal);
  assert.equal(
    out,
    "key [REDACTED:aws-key] mail [REDACTED:email] ssn [REDACTED:ssn]",
  );
});

test("redacts only card numbers that pass the Luhn check", () => {
  assert.equal(
    redact("4111 1111 1111 1111 and 4111 1111 1111 1112", piiDetectors),
    "[REDACTED:card] and 4111 1111 1111 1112",
  );
});

test("deny mode rejects output that contains a secret", async () => {
  await assert.rejects(
    tool("sk-aaaaaaaaaaaaaaaaaaaaaaaa", { ...policy, onOutput: "deny" }).run(
      { q: "x" },
      signal,
    ),
    /output blocked: api-key/,
  );
});

test("blocks inputs that carry a secret or exceed the size bound", () => {
  const t = tool("ok");
  assert.throws(
    () => t.parse({ q: "AKIAABCDEFGHIJKLMNOP" }),
    /input blocked: aws-key/,
  );
  assert.throws(() => t.parse({ q: "a".repeat(200) }), /size bound/);
  assert.deepEqual(t.parse({ q: "fine" }), { q: "fine" });
});

test("rejects oversized output and spends the call budget", async () => {
  await assert.rejects(
    tool("a".repeat(300)).run({ q: "x" }, signal),
    /size bound/,
  );
  const t = tool("ok", { ...policy, maxCalls: 1 });
  await t.run({ q: "x" }, signal);
  await assert.rejects(t.run({ q: "x" }, signal), /budget spent/);
});

test("fails closed on missing, unknown or invalid policies", () => {
  assert.throws(() => guardTools([echo("x")], {}), /no policy/);
  assert.throws(
    () => guardTools([echo("x")], { echo: policy, other: policy }),
    /unknown tool/,
  );
  assert.throws(
    () => guardTools([echo("x")], { echo: { ...policy, maxCalls: 0 } }),
    RangeError,
  );
  assert.throws(
    () => guardTools([echo("x")], { echo: { ...policy, maxCalls: 1e9 } }),
    RangeError,
  );
});

test("a tool named like an object prototype key has no policy", () => {
  const t = { ...echo("x"), name: "constructor" };
  assert.throws(() => guardTools([t], {}), /no policy/);
});

test("finds secrets after JSON-escaped whitespace and in keys", () => {
  const t = tool("ok");
  assert.throws(() => t.parse({ q: "x\nAKIAABCDEFGHIJKLMNOP" }), /aws-key/);
  assert.throws(
    () => t.parse({ q: "x\tsk-aaaaaaaaaaaaaaaaaaaaaaaa" }),
    /api-key/,
  );
  assert.throws(() => t.parse({ AKIAABCDEFGHIJKLMNOP: 1 }), /aws-key/);
});

test("rejects circular, bigint and too deep input", () => {
  const t = tool("ok");
  const loop: Record<string, unknown> = {};
  loop.self = loop;
  assert.throws(() => t.parse(loop), /circular/);
  assert.throws(() => t.parse({ n: 1n }), /plain data/);
  let deep: unknown = "x";
  for (let i = 0; i < 40; i++) deep = [deep];
  assert.throws(() => t.parse(deep), /too deep/);
});

test("detects a card number next to other digits", () => {
  assert.equal(
    redact("order 12 4111111111111111", piiDetectors),
    "order [REDACTED:card]",
  );
});

test("redacts the whole private key block", () => {
  const key =
    "-----BEGIN RSA PRIVATE KEY-----\nMIIEvQ\n-----END RSA PRIVATE KEY-----";
  assert.equal(
    redact(`k ${key} z`, secretDetectors),
    "k [REDACTED:private-key] z",
  );
});

test("rejects non-global detectors and non-array detector lists", () => {
  const bad = { name: "n", pattern: /x/ };
  assert.throws(
    () => guardTools([echo("x")], { echo: { ...policy, output: [bad] } }),
    RangeError,
  );
  assert.throws(
    () =>
      guardTools([echo("x")], {
        echo: { ...policy, input: undefined as never },
      }),
    RangeError,
  );
});

test("keeps requiresApproval and description from class-based tools", () => {
  class T {
    name = "echo";
    get requiresApproval() {
      return true;
    }
    get description() {
      return "d";
    }
    parse(i: unknown) {
      return i;
    }
    async run(_i: unknown, _s: AbortSignal) {
      return "ok";
    }
  }
  const [g] = guardTools([new T()], { echo: policy });
  assert.equal(g?.requiresApproval, true);
  assert.equal(g?.description, "d");
});

test("rejects redacted output that grows past the bound", async () => {
  const t = tool("a@b.co ".repeat(12), { ...policy, maxOutputChars: 90 });
  await assert.rejects(t.run({ q: "x" }, signal), /size bound/);
});

test("later changes to the policy or shared detectors do not weaken a guard", async () => {
  const p: Policy = { ...policy, input: [...secretDetectors], maxCalls: 1 };
  const t = tool("ok", p);
  p.maxCalls = 1000;
  p.input.length = 0;
  await t.run({ q: "x" }, signal);
  await assert.rejects(t.run({ q: "x" }, signal), /budget spent/);
  assert.throws(() => t.parse({ q: "AKIAABCDEFGHIJKLMNOP" }), /aws-key/);
});
