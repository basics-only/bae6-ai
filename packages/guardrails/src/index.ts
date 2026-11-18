export interface Detector {
  name: string;
  pattern: RegExp;
  accept?: (match: string) => boolean;
}

export interface Policy {
  maxInputChars: number;
  maxOutputChars: number;
  maxCalls: number;
  input: Detector[];
  output: Detector[];
  onOutput: "redact" | "deny";
}

interface GuardableTool {
  name: string;
  parse(input: unknown): unknown;
  run(input: never, signal: AbortSignal): Promise<string>;
}

const MAX_CHARS = 1_000_000;
const MAX_CALLS = 1000;

const luhn = (m: string): boolean => {
  const digits = m.replace(/\D/g, "");
  if (digits.length < 13 || digits.length > 19) return false;
  let sum = 0;
  for (let i = 0; i < digits.length; i++) {
    let d = Number(digits[digits.length - 1 - i]);
    if (i % 2 === 1) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
  }
  return sum % 10 === 0;
};

export const secretDetectors: Detector[] = [
  { name: "aws-key", pattern: /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g },
  { name: "private-key", pattern: /-----BEGIN [A-Z ]{0,20}PRIVATE KEY-----/g },
  {
    name: "jwt",
    pattern: /\beyJ[\w-]{10,2000}\.[\w-]{10,2000}\.[\w-]{5,2000}/g,
  },
  { name: "api-key", pattern: /\bsk-[A-Za-z0-9_-]{20,200}/g },
];

export const piiDetectors: Detector[] = [
  {
    name: "email",
    pattern: /[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,8}/g,
  },
  { name: "ssn", pattern: /\b\d{3}-\d{2}-\d{4}\b/g },
  { name: "card", pattern: /\b\d(?:[ -]?\d){12,18}\b/g, accept: luhn },
];

function findings(text: string, detectors: Detector[]): string[] {
  const hit = new Set<string>();
  for (const d of detectors) {
    for (const m of text.matchAll(d.pattern)) {
      if (!d.accept || d.accept(m[0])) hit.add(d.name);
    }
  }
  return [...hit];
}

export const scan = findings;

export function redact(text: string, detectors: Detector[]): string {
  let out = text;
  for (const d of detectors) {
    out = out.replace(d.pattern, (m) =>
      !d.accept || d.accept(m) ? `[REDACTED:${d.name}]` : m,
    );
  }
  return out;
}

function checkPolicy(name: string, p: Policy): void {
  const ok = (n: number, max: number) =>
    Number.isInteger(n) && n >= 1 && n <= max;
  if (
    !ok(p.maxInputChars, MAX_CHARS) ||
    !ok(p.maxOutputChars, MAX_CHARS) ||
    !ok(p.maxCalls, MAX_CALLS) ||
    (p.onOutput !== "redact" && p.onOutput !== "deny")
  ) {
    throw new RangeError(`invalid policy for tool ${name}`);
  }
}

function guardTool<T extends GuardableTool>(tool: T, policy: Policy): T {
  let calls = 0;
  return {
    ...tool,
    parse(input: unknown) {
      const text = JSON.stringify(input) ?? "";
      if (text.length > policy.maxInputChars) {
        throw new Error("input exceeds the size bound");
      }
      const hit = findings(text, policy.input);
      if (hit.length > 0) throw new Error(`input blocked: ${hit.join(",")}`);
      return tool.parse(input);
    },
    async run(input: never, signal: AbortSignal) {
      if (++calls > policy.maxCalls) throw new Error("tool call budget spent");
      const out = await tool.run(input, signal);
      if (out.length > policy.maxOutputChars) {
        throw new Error("output exceeds the size bound");
      }
      if (policy.onOutput === "redact") return redact(out, policy.output);
      const hit = findings(out, policy.output);
      if (hit.length > 0) throw new Error(`output blocked: ${hit.join(",")}`);
      return out;
    },
  };
}

export function guardTools<T extends GuardableTool>(
  tools: T[],
  policies: Record<string, Policy>,
): T[] {
  const names = new Set(tools.map((t) => t.name));
  for (const key of Object.keys(policies)) {
    if (!names.has(key)) throw new Error(`policy for unknown tool ${key}`);
  }
  return tools.map((t) => {
    const policy = Object.hasOwn(policies, t.name)
      ? policies[t.name]
      : undefined;
    if (!policy) throw new Error(`no policy for tool ${t.name}`);
    checkPolicy(t.name, policy);
    return guardTool(t, policy);
  });
}
