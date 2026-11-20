import {
  huggingfaceOrigin,
  type Modality,
  type Operation,
} from "@bae6/ai-core";
import { SLUG } from "./identity.js";

export interface ModelConfig {
  id: string;
  provider: "bedrock" | "vertex" | "huggingface";
  providerModel: string;
  /** Bedrock and Vertex only. */
  region?: string;
  /** Vertex only. */
  project?: string;
  /**
   * Bedrock: replaces the AWS host; for mocks and private endpoints.
   * Hugging Face: the dedicated endpoint's origin.
   */
  endpoint?: string;
  /** Hugging Face only. Inference Providers sends prompts to third-party hosts. */
  hosting?: "inference-providers" | "dedicated";
  /** Hugging Face only. The environment variable holding the access token. */
  tokenEnv?: string;
  operations: Operation[];
  inputModalities: Modality[];
  maxInputTokens: number;
  maxOutputTokens: number;
  /** Organizations allowed to use the model; absent means all. */
  organizations?: string[];
}

export interface GatewayConfig {
  priceBook: {
    version: string;
    prices: Record<string, { inputPerMTok: string; outputPerMTok: string }>;
  };
  models: ModelConfig[];
  /** Monthly spend limits, decimal strings in the price book's currency. */
  budgets: Record<string, string>;
}

const MODEL_ID = /^[a-z0-9][a-z0-9._-]{0,63}$/;
const DECIMAL = /^\d{1,12}(\.\d{1,9})?$/;
const REGION = /^[a-z0-9-]{1,32}$/;
const PROJECT = /^[a-z][a-z0-9-]{4,29}$/;
const HF_TOKEN_ENV = /^BAE6_HF_TOKEN_[A-Z0-9_]{1,32}$/;
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
    if (
      m.provider !== "bedrock" &&
      m.provider !== "vertex" &&
      m.provider !== "huggingface"
    )
      bad("provider");
    const hf = m.provider === "huggingface";
    if (
      typeof m.providerModel !== "string" ||
      !PROVIDER_MODEL.test(m.providerModel)
    )
      bad("providerModel");
    if (
      hf
        ? m.region !== undefined
        : typeof m.region !== "string" || !REGION.test(m.region)
    )
      bad("region");
    if (
      m.provider === "vertex" &&
      (typeof m.project !== "string" || !PROJECT.test(m.project))
    )
      bad("project");
    if (!hf && (m.hosting !== undefined || m.tokenEnv !== undefined))
      bad("hosting or tokenEnv");
    if (hf) {
      if (m.hosting !== "inference-providers" && m.hosting !== "dedicated")
        bad("hosting");
      if (typeof m.tokenEnv !== "string" || !HF_TOKEN_ENV.test(m.tokenEnv))
        bad("tokenEnv");
      // Only a dedicated endpoint names a host; Inference Providers is always the router.
      const origin = huggingfaceOrigin(m.endpoint);
      if ((m.hosting === "dedicated") !== (m.endpoint !== undefined)) {
        bad("endpoint");
      }
      if (m.endpoint !== undefined) {
        if (origin === null || origin === "https://router.huggingface.co")
          bad("endpoint");
        m.endpoint = origin;
      }
      // Prompts leave for third-party hosts, so an unlisted model must not be open to everyone.
      if (
        m.hosting === "inference-providers" &&
        (!Array.isArray(m.organizations) || m.organizations.length === 0)
      )
        bad("organizations (required for inference-providers)");
    } else if (m.endpoint !== undefined) {
      const url = bedrockEndpoint(m.endpoint);
      if (m.provider !== "bedrock" || url === null) bad("endpoint");
      // Keep the normalized origin so later checks see what will really be called.
      m.endpoint = (url as URL).origin;
    }
    const ops = m.operations;
    if (
      !Array.isArray(ops) ||
      ops.length === 0 ||
      !ops.every((o) => OPERATIONS.has(o)) ||
      (hf && ops.includes("embed"))
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

  if (!isObject(c.budgets))
    fail("budgets must map organization to a monthly limit");
  const budgets = c.budgets as Record<string, unknown>;
  for (const [org, limit] of Object.entries(budgets)) {
    if (!SLUG.test(org) || typeof limit !== "string" || !DECIMAL.test(limit)) {
      fail(`budget for ${org} must be a decimal string`);
    }
  }
  return { priceBook, models, budgets: budgets as Record<string, string> };
}
