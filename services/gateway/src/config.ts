import type { Modality, Operation } from "@bae6/ai-core";
import { SLUG } from "./identity.js";

export interface ModelConfig {
  id: string;
  provider: "bedrock" | "vertex";
  providerModel: string;
  region: string;
  /** Vertex only. */
  project?: string;
  /** Bedrock only. Replaces the AWS host; for mocks and private endpoints. */
  endpoint?: string;
  operations: Operation[];
  inputModalities: Modality[];
  maxInputTokens: number;
  maxOutputTokens: number;
  /** Organizations allowed to use the model; absent means all. */
  organizations?: string[];
}

export interface RouteRule {
  model: string;
  minInputTokens?: number;
  maxInputTokens?: number;
  images?: boolean;
}

export interface RouteConfig {
  id: string;
  default: string;
  /** First match wins. */
  rules: RouteRule[];
  /** Organizations allowed to use the route; absent means all. */
  organizations?: string[];
}

export interface GatewayConfig {
  priceBook: {
    version: string;
    prices: Record<string, { inputPerMTok: string; outputPerMTok: string }>;
  };
  models: ModelConfig[];
  routes: RouteConfig[];
  /** Monthly spend limits, decimal strings in the price book's currency. */
  budgets: Record<string, string>;
}

const MODEL_ID = /^[a-z0-9][a-z0-9._-]{0,63}$/;
const DECIMAL = /^\d{1,12}(\.\d{1,9})?$/;
const REGION = /^[a-z0-9-]{1,32}$/;
const PROJECT = /^[a-z][a-z0-9-]{4,29}$/;
const PROVIDER_MODEL = /^[A-Za-z0-9._:@/-]{1,128}$/;
const OPERATIONS = new Set(["chat", "chat_stream", "embed"]);
const MODALITIES = new Set(["text", "image"]);

const isObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/** An origin with no path, credentials or query, so requests cannot be steered elsewhere. */
export function bedrockEndpoint(v: unknown): URL | null {
  if (typeof v !== "string") return null;
  try {
    const u = new URL(v);
    const plain =
      (u.protocol === "https:" || u.protocol === "http:") &&
      u.username === "" &&
      u.password === "" &&
      u.search === "" &&
      u.hash === "" &&
      u.pathname === "/";
    return plain ? u : null;
  } catch {
    return null;
  }
}

/** Validates the whole file up front so a bad config stops start-up, not a request. */
export function parseConfig(raw: unknown): GatewayConfig {
  const fail = (msg: string): never => {
    throw new Error(`gateway config: ${msg}`);
  };
  if (!isObject(raw)) fail("must be an object");
  const c = raw as Record<string, unknown>;
  const pb = c.priceBook;
  if (
    !isObject(pb) ||
    typeof pb.version !== "string" ||
    !/^[A-Za-z0-9._-]{1,64}$/.test(pb.version) ||
    !isObject(pb.prices)
  ) {
    fail("priceBook must be {version, prices}");
  }
  const priceBook = pb as GatewayConfig["priceBook"];
  for (const [model, p] of Object.entries(priceBook.prices)) {
    if (
      !isObject(p) ||
      typeof p.inputPerMTok !== "string" ||
      !DECIMAL.test(p.inputPerMTok) ||
      typeof p.outputPerMTok !== "string" ||
      !DECIMAL.test(p.outputPerMTok)
    ) {
      fail(
        `price for ${model} must be decimal strings inputPerMTok, outputPerMTok`,
      );
    }
  }

  if (
    !Array.isArray(c.models) ||
    c.models.length === 0 ||
    c.models.length > 64
  ) {
    fail("models must be a list of 1 to 64 models");
  }
  const seen = new Set<string>();
  const models = (c.models as unknown[]).map((m): ModelConfig => {
    if (!isObject(m)) return fail("each model must be an object");
    const id = String(m.id);
    if (!MODEL_ID.test(id) || seen.has(id))
      fail(`model id ${id} is invalid or repeated`);
    seen.add(id);
    const bad = (field: string): never => fail(`model ${id}: invalid ${field}`);
    if (m.provider !== "bedrock" && m.provider !== "vertex") bad("provider");
    if (
      typeof m.providerModel !== "string" ||
      !PROVIDER_MODEL.test(m.providerModel)
    )
      bad("providerModel");
    if (typeof m.region !== "string" || !REGION.test(m.region)) bad("region");
    if (
      m.provider === "vertex" &&
      (typeof m.project !== "string" || !PROJECT.test(m.project))
    )
      bad("project");
    if (m.endpoint !== undefined) {
      const url = bedrockEndpoint(m.endpoint);
      if (m.provider !== "bedrock" || url === null) bad("endpoint");
      // Keep the normalized origin so later checks see what will really be called.
      m.endpoint = (url as URL).origin;
    }
    const ops = m.operations;
    if (
      !Array.isArray(ops) ||
      ops.length === 0 ||
      !ops.every((o) => OPERATIONS.has(o))
    )
      bad("operations");
    const mods = m.inputModalities;
    if (
      !Array.isArray(mods) ||
      !mods.includes("text") ||
      !mods.every((x) => MODALITIES.has(x))
    )
      bad("inputModalities");
    for (const f of ["maxInputTokens", "maxOutputTokens"] as const) {
      if (
        !Number.isInteger(m[f]) ||
        (m[f] as number) < 1 ||
        (m[f] as number) > 2_000_000
      )
        bad(f);
    }
    if (
      m.organizations !== undefined &&
      (!Array.isArray(m.organizations) ||
        !m.organizations.every((o) => typeof o === "string" && SLUG.test(o)))
    ) {
      bad("organizations");
    }
    if (!priceBook.prices[id])
      fail(`model ${id} has no price in ${priceBook.version}`);
    return m as unknown as ModelConfig;
  });

  const routes = parseRoutes(c.routes, models, fail);

  if (!isObject(c.budgets))
    fail("budgets must map organization to a monthly limit");
  const budgets = c.budgets as Record<string, unknown>;
  for (const [org, limit] of Object.entries(budgets)) {
    if (!SLUG.test(org) || typeof limit !== "string" || !DECIMAL.test(limit)) {
      fail(`budget for ${org} must be a decimal string`);
    }
  }
  return {
    priceBook,
    models,
    routes,
    budgets: budgets as Record<string, string>,
  };
}

const MAX_ROUTES = 32;
const MAX_RULES = 16;
const ROUTE_KEYS = new Set(["id", "default", "rules", "organizations"]);
const RULE_KEYS = new Set([
  "model",
  "minInputTokens",
  "maxInputTokens",
  "images",
]);

function parseRoutes(
  raw: unknown,
  models: ModelConfig[],
  fail: (msg: string) => never,
): RouteConfig[] {
  if (raw === undefined) return [];
  if (!Array.isArray(raw) || raw.length > MAX_ROUTES) {
    fail(`routes must be a list of at most ${MAX_ROUTES} routes`);
  }
  const byId = new Map(models.map((m) => [m.id, m]));
  const ids = new Set(byId.keys());
  return (raw as unknown[]).map((r): RouteConfig => {
    if (!isObject(r)) return fail("each route must be an object");
    const id = String(r.id);
    if (!MODEL_ID.test(id) || ids.has(id))
      fail(`route id ${id} is invalid or collides with a model or route`);
    ids.add(id);
    const bad = (field: string): never => fail(`route ${id}: invalid ${field}`);
    for (const k of Object.keys(r)) {
      if (!ROUTE_KEYS.has(k)) bad(k);
    }
    const chatModel = (name: unknown, field: string) => {
      const m = typeof name === "string" ? byId.get(name) : undefined;
      if (!m?.operations.includes("chat")) bad(field);
    };
    chatModel(r.default, "default");
    if (!Array.isArray(r.rules) || r.rules.length > MAX_RULES) bad("rules");
    for (const rule of r.rules as unknown[]) {
      if (!isObject(rule)) return bad("rules");
      for (const k of Object.keys(rule)) {
        if (!RULE_KEYS.has(k)) bad(`rules.${k}`);
      }
      chatModel(rule.model, "rules.model");
      for (const f of ["minInputTokens", "maxInputTokens"] as const) {
        const v = rule[f];
        if (
          v !== undefined &&
          (!Number.isInteger(v) ||
            (v as number) < 0 ||
            (v as number) > 2_000_000)
        )
          bad(`rules.${f}`);
      }
      if (rule.images !== undefined && typeof rule.images !== "boolean")
        bad("rules.images");
      const lo = (rule.minInputTokens as number | undefined) ?? 0;
      const hi = (rule.maxInputTokens as number | undefined) ?? Infinity;
      if (lo > hi) bad("rules (minInputTokens is above maxInputTokens)");
    }
    if (
      r.organizations !== undefined &&
      (!Array.isArray(r.organizations) ||
        !r.organizations.every((o) => typeof o === "string" && SLUG.test(o)))
    )
      bad("organizations");
    return r as unknown as RouteConfig;
  });
}
