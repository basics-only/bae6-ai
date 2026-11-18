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
