import assert from "node:assert/strict";
import { test } from "node:test";
import {
  type AgentMessage,
  type AgentModel,
  createAgent,
  type Limits,
  type ModelTurn,
  type Tool,
} from "./index.js";

const limits: Limits = {
  maxSteps: 5,
  maxTotalTokens: 1000,
  timeoutMs: 5000,
  toolTimeoutMs: 1000,
  maxToolOutputChars: 20,
  maxHistoryChars: 10_000,
};
const usage = { inputTokens: 10, outputTokens: 5 };

function script(turns: ModelTurn[], seen: AgentMessage[][] = []): AgentModel {
  let i = 0;
  return {
    async next(messages) {
      seen.push(structuredClone(messages));
      const turn = turns[Math.min(i++, turns.length - 1)];
      if (!turn) throw new Error("script is empty");
      return turn;
    },
  };
}

const echo: Tool<{ q: string }> = {
  name: "echo",
  description: "echo",
  parse(input) {
    if (
      typeof input !== "object" ||
      input === null ||
      typeof (input as any).q !== "string"
    ) {
      throw new Error("q must be a string");
    }
    return { q: (input as any).q };
  },
  async run(i) {
    return i.q;
  },
};

const call = (name: string, input: unknown): ModelTurn => ({
  type: "tool",
  call: { name, input },
  usage,
});
const final: ModelTurn = { type: "final", text: "done", usage };
const signal = () => new AbortController().signal;

test("runs a tool then returns the final answer with a receipt", async () => {
  const agent = createAgent({
    model: script([call("echo", { q: "hi" }), final]),
    tools: [echo],
    allow: ["echo"],
    limits,
  });
  const r = await agent.run("task", signal());
  assert.equal(r.stop, "final");
  assert.equal(r.text, "done");
  assert.equal(r.tokens, 30);
  assert.equal(r.receipts.length, 1);
  assert.equal(r.receipts[0]?.outcome, "ok");
  assert.equal(r.receipts[0]?.detail, "hi");
});

test("rejects invalid tool input without running the tool", async () => {
  let ran = false;
  const tool: Tool = {
    ...echo,
    async run() {
      ran = true;
      return "";
    },
  };
  const agent = createAgent({
    model: script([call("echo", { q: 1 }), final]),
    tools: [tool],
    allow: ["echo"],
    limits,
  });
  const r = await agent.run("task", signal());
  assert.equal(r.receipts[0]?.outcome, "invalid_input");
  assert.equal(ran, false);
});

test("does not run a registered tool that is not allowlisted", async () => {
  const seen: AgentMessage[][] = [];
  const agent = createAgent({
    model: script([call("echo", { q: "hi" }), final], seen),
    tools: [echo],
    allow: [],
    limits,
  });
  const r = await agent.run("task", signal());
  assert.equal(r.receipts[0]?.outcome, "not_allowed");
});

test("requires approval and fails closed without a hook", async () => {
  const guarded: Tool = { ...echo, requiresApproval: true };
  const make = (approve?: (c: unknown) => Promise<boolean>) =>
    createAgent({
      model: script([call("echo", { q: "hi" }), final]),
      tools: [guarded],
      allow: ["echo"],
      limits,
      approve: approve as any,
    });
  assert.equal(
    (await make().run("t", signal())).receipts[0]?.outcome,
    "denied",
  );
  assert.equal(
    (await make(async () => false).run("t", signal())).receipts[0]?.outcome,
    "denied",
  );
  assert.equal(
    (
      await make(async () => {
        throw new Error("boom");
      }).run("t", signal())
    ).receipts[0]?.outcome,
    "denied",
  );
  assert.equal(
    (await make(async () => true).run("t", signal())).receipts[0]?.outcome,
    "ok",
  );
});

test("stops at the step ceiling when the model keeps calling tools", async () => {
  const agent = createAgent({
    model: script([call("echo", { q: "x" })]),
    tools: [echo],
    allow: ["echo"],
    limits: { ...limits, maxSteps: 3 },
  });
  const r = await agent.run("task", signal());
  assert.equal(r.stop, "max_steps");
  assert.equal(r.receipts.length, 3);
});

test("stops at the token budget before running another tool", async () => {
  const agent = createAgent({
    model: script([call("echo", { q: "x" })]),
    tools: [echo],
    allow: ["echo"],
    limits: { ...limits, maxTotalTokens: 20 },
  });
  const r = await agent.run("task", signal());
  assert.equal(r.stop, "token_budget");
  assert.equal(r.receipts.length, 1);
});

test("fails the run when the model reports no usage", async () => {
  const agent = createAgent({
    model: script([{ type: "final", text: "x" } as any]),
    tools: [],
    allow: [],
    limits,
  });
  assert.equal((await agent.run("t", signal())).stop, "model_error");
});

test("truncates tool output and marks it untrusted for the model", async () => {
  const seen: AgentMessage[][] = [];
  const long: Tool = { ...echo, run: async () => "x".repeat(100) };
  const agent = createAgent({
    model: script([call("echo", { q: "a" }), final], seen),
    tools: [long],
    allow: ["echo"],
    limits,
  });
  const r = await agent.run("task", signal());
  assert.equal(r.receipts[0]?.truncated, true);
  assert.equal(r.receipts[0]?.detail.length, 20);
  const tool = seen[1]?.find((m) => m.role === "tool");
  assert.equal(tool?.role === "tool" && tool.untrusted, true);
});

test("aborts mid-tool when the caller cancels", async () => {
  const slow: Tool = {
    ...echo,
    run: (_i, s) =>
      new Promise((_, reject) =>
        s.addEventListener("abort", () => reject(new Error("aborted"))),
      ),
  };
  const ctl = new AbortController();
  const agent = createAgent({
    model: script([call("echo", { q: "a" }), final]),
    tools: [slow],
    allow: ["echo"],
    limits,
  });
  const p = agent.run("task", ctl.signal);
  setTimeout(() => ctl.abort(), 20);
  assert.equal((await p).stop, "aborted");
});

test("times out a hung tool and continues the run", async () => {
  const hung: Tool = {
    ...echo,
    run: (_i, s) =>
      new Promise((_, reject) =>
        s.addEventListener("abort", () => reject(new Error("tool timed out"))),
      ),
  };
  const agent = createAgent({
    model: script([call("echo", { q: "a" }), final]),
    tools: [hung],
    allow: ["echo"],
    limits: { ...limits, toolTimeoutMs: 30 },
  });
  const r = await agent.run("task", signal());
  assert.equal(r.receipts[0]?.outcome, "error");
  assert.equal(r.stop, "final");
});

test("rejects out-of-bound limits and duplicate tool names", () => {
  const base = { model: script([final]), tools: [echo], allow: [] };
  assert.throws(() =>
    createAgent({ ...base, limits: { ...limits, maxSteps: 0 } }),
  );
  assert.throws(() =>
    createAgent({ ...base, limits: { ...limits, maxSteps: 1e9 } }),
  );
  assert.throws(() => createAgent({ ...base, tools: [echo, echo], limits }));
});

const never = () => new Promise<never>(() => {});

test("ends the run when the model ignores the abort signal", async () => {
  const agent = createAgent({
    model: { next: never },
    tools: [],
    allow: [],
    limits: { ...limits, timeoutMs: 30 },
  });
  assert.equal((await agent.run("t", signal())).stop, "timeout");
});

test("ends a tool call that ignores its abort signal", async () => {
  const stuck: Tool = { ...echo, run: never };
  const agent = createAgent({
    model: script([call("echo", { q: "a" }), final]),
    tools: [stuck],
    allow: ["echo"],
    limits: { ...limits, toolTimeoutMs: 30 },
  });
  const r = await agent.run("t", signal());
  assert.equal(r.receipts[0]?.outcome, "error");
  assert.equal(r.stop, "final");
});

test("does not wait forever for an approver that never answers", async () => {
  const guarded: Tool = { ...echo, requiresApproval: true };
  const agent = createAgent({
    model: script([call("echo", { q: "a" }), final]),
    tools: [guarded],
    allow: ["echo"],
    limits: { ...limits, timeoutMs: 50 },
    approve: never,
  });
  assert.equal((await agent.run("t", signal())).stop, "timeout");
});

test("rejects a final turn that overshoots the token budget", async () => {
  const agent = createAgent({
    model: script([
      call("echo", { q: "a" }),
      {
        type: "final",
        text: "x",
        usage: { inputTokens: 100, outputTokens: 0 },
      },
    ]),
    tools: [echo],
    allow: ["echo"],
    limits: { ...limits, maxTotalTokens: 50 },
  });
  const r = await agent.run("t", signal());
  assert.equal(r.stop, "token_budget");
  assert.equal(r.text, undefined);
});

test("tells the model how many tokens remain", async () => {
  const remaining: number[] = [];
  const agent = createAgent({
    model: {
      async next(_m, _t, _s, left) {
        remaining.push(left);
        return remaining.length === 1 ? call("echo", { q: "a" }) : final;
      },
    },
    tools: [echo],
    allow: ["echo"],
    limits,
  });
  await agent.run("t", signal());
  assert.deepEqual(remaining, [1000, 985]);
});

test("approves and runs the same parsed input", async () => {
  let approved: unknown;
  let ran: unknown;
  const normalizing: Tool<{ q: string }> = {
    ...echo,
    requiresApproval: true,
    parse: (i) => ({ q: String((i as any).q).trim() }),
    async run(i) {
      ran = i;
      return "";
    },
  };
  const agent = createAgent({
    model: script([call("echo", { q: "  x  " }), final]),
    tools: [normalizing],
    allow: ["echo"],
    limits,
    approve: async (c) => {
      approved = c.input;
      (c.input as any).q = "tampered";
      return true;
    },
  });
  await agent.run("t", signal());
  assert.deepEqual(approved, { q: "tampered" });
  assert.deepEqual(ran, { q: "x" });
});

test("fails closed on malformed model turns instead of throwing", async () => {
  for (const bad of [
    { type: "tool", usage },
    { type: "other", usage },
    { type: "final", text: 1, usage },
    null,
  ]) {
    const agent = createAgent({
      model: script([bad as any]),
      tools: [],
      allow: [],
      limits,
    });
    assert.equal((await agent.run("t", signal())).stop, "model_error");
  }
});

test("bounds error detail, tool name and recorded input", async () => {
  const leaky: Tool = {
    ...echo,
    run: async () => {
      throw new Error("x".repeat(500));
    },
  };
  const agent = createAgent({
    model: script([call("echo", { q: "y".repeat(5000) }), final]),
    tools: [leaky],
    allow: ["echo"],
    limits,
  });
  const r = await agent.run("t", signal());
  assert.equal(r.receipts[0]?.detail.length, 20);
  assert.equal(r.receipts[0]?.truncated, true);
  assert.equal(r.receipts[0]?.input.length, 1000);
});

test("stops immediately when the signal is already aborted", async () => {
  const ctl = new AbortController();
  ctl.abort();
  const agent = createAgent({
    model: script([final]),
    tools: [],
    allow: [],
    limits,
  });
  assert.equal((await agent.run("t", ctl.signal)).stop, "aborted");
});

test("treats prototype-key tool names as unavailable", async () => {
  const agent = createAgent({
    model: script([call("__proto__", {}), call("constructor", {}), final]),
    tools: [echo],
    allow: ["echo"],
    limits,
  });
  const r = await agent.run("t", signal());
  assert.deepEqual(
    r.receipts.map((x) => x.outcome),
    ["not_allowed", "not_allowed"],
  );
});

test("does not record or replay input the tool refused", async () => {
  const secret = "AKIAABCDEFGHIJKLMNOP";
  const seen: AgentMessage[][] = [];
  const guarded: Tool = {
    ...echo,
    parse() {
      throw new Error("input blocked: aws-key");
    },
  };
  const agent = createAgent({
    model: script(
      [call("echo", { q: secret }), call("other", { q: secret }), final],
      seen,
    ),
    tools: [guarded],
    allow: ["echo"],
    limits,
  });
  const r = await agent.run("t", signal());
  assert.deepEqual(
    r.receipts.map((x) => x.outcome),
    ["invalid_input", "not_allowed"],
  );
  assert.equal(JSON.stringify([r.receipts, seen]).includes(secret), false);
});

test("stops when the conversation outgrows the history bound", async () => {
  const big: Tool = { ...echo, run: async () => "x".repeat(20) };
  const agent = createAgent({
    model: script([call("echo", { q: "a" })]),
    tools: [big],
    allow: ["echo"],
    limits: { ...limits, maxSteps: 100, maxHistoryChars: 200 },
  });
  const r = await agent.run("t", signal());
  assert.equal(r.stop, "history_limit");
  assert.ok(r.receipts.length < 100);
});

test("rejects a task larger than the history bound", async () => {
  const agent = createAgent({
    model: script([final]),
    tools: [],
    allow: [],
    limits: { ...limits, maxHistoryChars: 5 },
  });
  assert.equal(
    (await agent.run("x".repeat(6), signal())).stop,
    "history_limit",
  );
});
