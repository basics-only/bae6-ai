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
const MAX_DEPTH = 32;
const MAX_NODES = 10_000;

const luhnValid = (digits: string): boolean => {
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

const cardInRun = (m: string): boolean => {
  const digits = m.replace(/\D/g, "");
  const candidates = [digits];
  for (const len of [15, 16]) {
    if (digits.length > len) {
      candidates.push(digits.slice(0, len), digits.slice(-len));
    }
  }
  return candidates.some(
    (c) => c.length >= 13 && c.length <= 19 && /^[3-6]/.test(c) && luhnValid(c),
  );
};

export const secretDetectors: Detector[] = [
  { name: "aws-key", pattern: /(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}/g },
  {
    name: "private-key",
    pattern:
      /-----BEGIN [A-Z ]{0,20}PRIVATE KEY-----(?:[\s\S]{0,8000}?-----END [A-Z ]{0,20}PRIVATE KEY-----)?/g,
  },
  {
    name: "jwt",
    pattern:
      /(?<![A-Za-z0-9])eyJ[\w-]{10,2000}\.[\w-]{10,2000}\.[\w-]{5,2000}/g,
  },
  { name: "api-key", pattern: /(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,200}/g },
];

export const piiDetectors: Detector[] = [
  {
    name: "email",
    pattern: /[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,8}/g,
  },
  { name: "ssn", pattern: /(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)/g },
  { name: "card", pattern: /\d(?:[ -]?\d){12,60}/g, accept: cardInRun },
];

function compile(name: string, detectors: Detector[]): Detector[] {
  if (!Array.isArray(detectors)) {
    throw new RangeError(`invalid detectors for tool ${name}`);
  }
  return Object.freeze(
    detectors.map((d) => {
      if (
        !(d.pattern instanceof RegExp) ||
        !d.pattern.global ||
        d.pattern.sticky
      ) {
        throw new RangeError(`detector ${d.name} needs a global regex`);
      }
      return Object.freeze({
        name: d.name,
        pattern: new RegExp(d.pattern.source, d.pattern.flags),
        ...(d.accept ? { accept: d.accept } : {}),
      });
    }),
  ) as Detector[];
}

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

function strings(input: unknown, maxChars: number): string[] {
  const out: string[] = [];
  let chars = 0;
  let nodes = 0;
  const seen = new Set<object>();
  const add = (s: string) => {
    chars += s.length;
    if (chars > maxChars) throw new Error("input exceeds the size bound");
    out.push(s);
  };
  const walk = (v: unknown, depth: number): void => {
    if (++nodes > MAX_NODES || depth > MAX_DEPTH) {
      throw new Error("input is too large or too deep");
    }
    if (typeof v === "string" || typeof v === "number") {
      add(String(v));
      return;
    }
    if (v === null || typeof v === "boolean" || v === undefined) return;
    if (typeof v !== "object") throw new Error("input is not plain data");
    if (seen.has(v)) throw new Error("input is circular");
    seen.add(v);
    if (Array.isArray(v)) {
      for (const item of v) walk(item, depth + 1);
    } else {
      for (const [k, item] of Object.entries(v)) {
        add(k);
        walk(item, depth + 1);
      }
    }
    seen.delete(v);
  };
  walk(input, 0);
  return out;
}

function checkPolicy(name: string, p: Policy): Policy {
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
  return Object.freeze({
    maxInputChars: p.maxInputChars,
    maxOutputChars: p.maxOutputChars,
    maxCalls: p.maxCalls,
    onOutput: p.onOutput,
    input: compile(name, p.input),
    output: compile(name, p.output),
  });
}

function guardTool<T extends GuardableTool>(tool: T, policy: Policy): T {
  let calls = 0;
  return Object.create(tool, {
    parse: {
      value(input: unknown) {
        for (const s of strings(input, policy.maxInputChars)) {
          const hit = findings(s, policy.input);
          if (hit.length > 0) {
            throw new Error(`input blocked: ${hit.join(",")}`);
          }
        }
        return tool.parse(input);
      },
    },
    run: {
      async value(input: never, signal: AbortSignal) {
        if (++calls > policy.maxCalls) {
          throw new Error("tool call budget spent");
        }
        const out = await tool.run(input, signal);
        if (out.length > policy.maxOutputChars) {
          throw new Error("output exceeds the size bound");
        }
        if (policy.onOutput === "deny") {
          const hit = findings(out, policy.output);
          if (hit.length > 0) {
            throw new Error(`output blocked: ${hit.join(",")}`);
          }
          return out;
        }
        const clean = redact(out, policy.output);
        if (clean.length > policy.maxOutputChars) {
          throw new Error("output exceeds the size bound");
        }
        return clean;
      },
    },
  });
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
    return guardTool(t, checkPolicy(t.name, policy));
  });
}
