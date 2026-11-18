export interface Usage {
  inputTokens: number;
  outputTokens: number;
}

export interface ToolCall {
  name: string;
  input: unknown;
}

export type AgentMessage =
  | { role: "user" | "assistant"; text: string }
  | { role: "tool"; name: string; text: string; untrusted: true };

export type ModelTurn =
  | { type: "final"; text: string; usage: Usage }
  | { type: "tool"; call: ToolCall; usage: Usage };

export interface AgentModel {
  next(
    messages: AgentMessage[],
    tools: { name: string; description: string }[],
    signal: AbortSignal,
  ): Promise<ModelTurn>;
}

export interface Tool<I = any> {
  name: string;
  description: string;
  parse(input: unknown): I;
  run(input: I, signal: AbortSignal): Promise<string>;
  requiresApproval?: boolean;
}

export interface Limits {
  maxSteps: number;
  maxTotalTokens: number;
  timeoutMs: number;
  toolTimeoutMs: number;
  maxToolOutputChars: number;
}

export interface Receipt {
  step: number;
  tool: string;
  input: unknown;
  outcome: "ok" | "invalid_input" | "not_allowed" | "denied" | "error";
  detail: string;
  truncated: boolean;
  durationMs: number;
}

export type StopReason =
  | "final"
  | "max_steps"
  | "token_budget"
  | "timeout"
  | "aborted"
  | "model_error";

export interface RunResult {
  stop: StopReason;
  text?: string;
  tokens: number;
  receipts: Receipt[];
}

export interface AgentConfig {
  model: AgentModel;
  tools: Tool[];
  allow: readonly string[];
  limits: Limits;
  approve?: (call: ToolCall) => Promise<boolean>;
}

const HARD_MAX = {
  steps: 100,
  tokens: 10_000_000,
  timeoutMs: 3_600_000,
  outputChars: 1_000_000,
};

export function validateLimits(l: Limits): void {
  const ok = (v: number, max: number) =>
    Number.isInteger(v) && v >= 1 && v <= max;
  if (
    !ok(l.maxSteps, HARD_MAX.steps) ||
    !ok(l.maxTotalTokens, HARD_MAX.tokens) ||
    !ok(l.timeoutMs, HARD_MAX.timeoutMs) ||
    !ok(l.toolTimeoutMs, HARD_MAX.timeoutMs) ||
    !ok(l.maxToolOutputChars, HARD_MAX.outputChars)
  ) {
    throw new RangeError("agent limits must be integers within hard bounds");
  }
}

// AbortSignal.timeout is unref'd, so a hung tool could let the process exit without ever timing out.
function withTimeout(
  parent: AbortSignal,
  ms: number,
): { signal: AbortSignal; clear: () => void } {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  return {
    signal: AbortSignal.any([parent, ctl.signal]),
    clear: () => clearTimeout(timer),
  };
}

function validUsage(u: Usage | undefined): u is Usage {
  return (
    !!u &&
    Number.isSafeInteger(u.inputTokens) &&
    Number.isSafeInteger(u.outputTokens) &&
    u.inputTokens >= 0 &&
    u.outputTokens >= 0
  );
}

export function createAgent(config: AgentConfig) {
  validateLimits(config.limits);
  const byName = new Map(config.tools.map((t) => [t.name, t]));
  if (byName.size !== config.tools.length) {
    throw new Error("duplicate tool name");
  }
  const allowed = new Set(config.allow);
  const offered = config.tools
    .filter((t) => allowed.has(t.name))
    .map((t) => ({ name: t.name, description: t.description }));

  return {
    async run(task: string, outer: AbortSignal): Promise<RunResult> {
      const { limits } = config;
      const { signal, clear } = withTimeout(outer, limits.timeoutMs);
      try {
        return await loop(task, outer, signal);
      } finally {
        clear();
      }
    },
  };

  async function loop(
    task: string,
    outer: AbortSignal,
    signal: AbortSignal,
  ): Promise<RunResult> {
    const { limits } = config;
    const messages: AgentMessage[] = [{ role: "user", text: task }];
    const receipts: Receipt[] = [];
    let tokens = 0;
    const done = (stop: StopReason, text?: string): RunResult =>
      text === undefined
        ? { stop, tokens, receipts }
        : { stop, text, tokens, receipts };
    const interrupted = (): RunResult =>
      done(outer.aborted ? "aborted" : "timeout");

    for (let step = 1; step <= limits.maxSteps; step++) {
      if (signal.aborted) return interrupted();
      let turn: ModelTurn;
      try {
        turn = await config.model.next(messages, offered, signal);
      } catch {
        return signal.aborted ? interrupted() : done("model_error");
      }
      if (!validUsage(turn.usage)) return done("model_error");
      tokens += turn.usage.inputTokens + turn.usage.outputTokens;
      if (turn.type === "final") return done("final", turn.text);
      if (tokens >= limits.maxTotalTokens) return done("token_budget");

      const receipt = await execute(step, turn.call, signal);
      receipts.push(receipt);
      if (signal.aborted) return interrupted();
      messages.push({ role: "assistant", text: `call ${turn.call.name}` });
      messages.push({
        role: "tool",
        name: turn.call.name,
        text: receipt.detail,
        untrusted: true,
      });
    }
    return done("max_steps");
  }

  async function execute(
    step: number,
    call: ToolCall,
    signal: AbortSignal,
  ): Promise<Receipt> {
    const started = Date.now();
    const receipt = (
      outcome: Receipt["outcome"],
      detail: string,
      truncated = false,
    ): Receipt => ({
      step,
      tool: call.name,
      input: call.input,
      outcome,
      detail,
      truncated,
      durationMs: Date.now() - started,
    });
    const tool = byName.get(call.name);
    if (!tool || !allowed.has(call.name)) {
      return receipt("not_allowed", "tool is not available");
    }
    let input: unknown;
    try {
      input = tool.parse(call.input);
    } catch (e) {
      return receipt("invalid_input", message(e));
    }
    if (tool.requiresApproval) {
      let ok = false;
      try {
        ok = (await config.approve?.(call)) === true;
      } catch {
        ok = false;
      }
      if (!ok) return receipt("denied", "approval was not granted");
    }
    const bounded = withTimeout(signal, config.limits.toolTimeoutMs);
    try {
      const out = await tool.run(input, bounded.signal);
      const max = config.limits.maxToolOutputChars;
      return out.length > max
        ? receipt("ok", out.slice(0, max), true)
        : receipt("ok", out);
    } catch (e) {
      return receipt("error", message(e));
    } finally {
      bounded.clear();
    }
  }
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : "tool failed";
}
