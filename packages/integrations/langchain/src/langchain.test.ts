import assert from "node:assert/strict";
import { test } from "node:test";
import { fromLangChainTool, type LangChainTool } from "./index.js";

const echo: LangChainTool = {
  name: "echo",
  description: "echoes",
  schema: {
    parse(input) {
      if (typeof input !== "object" || input === null || !("text" in input)) {
        throw new Error("text required");
      }
      return input;
    },
  },
  invoke: async (input) => (input as { text: string }).text,
};

test("keeps name and description and runs the wrapped tool", async () => {
  const tool = fromLangChainTool(echo);
  assert.equal(tool.name, "echo");
  const input = tool.parse({ text: "hi" });
  assert.equal(await tool.run(input, new AbortController().signal), "hi");
});

test("rejects input the langchain schema refuses", () => {
  assert.throws(() => fromLangChainTool(echo).parse({}), /text required/);
});

test("serializes non-string output and forwards the abort signal", async () => {
  let seen: AbortSignal | undefined;
  const tool = fromLangChainTool({
    name: "n",
    description: "d",
    invoke: async (_input, config) => {
      seen = config?.signal;
      return { ok: true };
    },
  });
  const controller = new AbortController();
  assert.equal(await tool.run({}, controller.signal), '{"ok":true}');
  assert.equal(seen, controller.signal);
});

test("carries the approval requirement through", () => {
  assert.equal(
    fromLangChainTool(echo, { requiresApproval: true }).requiresApproval,
    true,
  );
});
