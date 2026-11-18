import { readFileSync } from "node:fs";
import {
  bedrockProvider,
  type ModelDescriptor,
  type ModelProvider,
  vertexProvider,
} from "@bae6/ai-core";
import pg from "pg";
import { type ModelConfig, parseConfig } from "./config.js";
import { awsCredentials, gcpToken } from "./credentials.js";
import { createVerifier, type Issuer } from "./identity.js";
import { createLedger } from "./ledger.js";
import { migrate } from "./migrate.js";
import { createGateway, type GatewayModel } from "./server.js";

const env = process.env;
const log = (e: Record<string, unknown>) => console.log(JSON.stringify(e));

function int(name: string, fallback: number, min: number, max: number) {
  const v = Number(env[name] ?? fallback);
  if (!Number.isInteger(v) || v < min || v > max) {
    throw new Error(`${name} must be an integer from ${min} to ${max}`);
  }
  return v;
}

const databaseUrl = env.BAE6_DATABASE_URL;
if (!databaseUrl) throw new Error("BAE6_DATABASE_URL is required");
const allowHttpJwks = env.BAE6_ALLOW_HTTP_JWKS === "true";
if (allowHttpJwks && env.NODE_ENV === "production") {
  throw new Error("BAE6_ALLOW_HTTP_JWKS is for development only");
}
const issuers = JSON.parse(env.BAE6_AI_ISSUERS ?? "null") as Issuer[] | null;
if (!Array.isArray(issuers)) {
  throw new Error(
    "BAE6_AI_ISSUERS must be a JSON array of {issuer,audience,jwksUri,organization,application}",
  );
}
const configFile = env.BAE6_AI_CONFIG_FILE;
if (!configFile) throw new Error("BAE6_AI_CONFIG_FILE is required");
const config = parseConfig(JSON.parse(readFileSync(configFile, "utf8")));

const port = int("PORT", 8080, 1, 65535);
const timeoutMs = int("BAE6_PROVIDER_TIMEOUT_MS", 60_000, 1000, 300_000);
const maxInflight = int("BAE6_MAX_INFLIGHT", 64, 1, 10_000);
// Size so pool max x max replicas fits the database's share of connections (ADR-0007).
const poolMax = int("BAE6_DB_POOL_MAX", 10, 1, 100);
const shutdownDelayMs = int("BAE6_SHUTDOWN_DELAY_MS", 0, 0, 15_000);

const descriptor = (m: ModelConfig): ModelDescriptor => ({
  kind: "model",
  id: m.id,
  operations: m.operations,
  limits: {
    maxInputTokens: m.maxInputTokens,
    maxOutputTokens: m.maxOutputTokens,
  },
  attributes: { provider: m.provider, inputModalities: m.inputModalities },
});

const aws = awsCredentials(env);
const gcp = gcpToken(env);
const models = new Map<string, GatewayModel>(
  config.models.map((m) => {
    const provider: ModelProvider =
      m.provider === "bedrock"
        ? bedrockProvider({
            descriptor: descriptor(m),
            modelId: m.providerModel,
            region: m.region,
            credentials: aws,
          })
        : vertexProvider({
            descriptor: descriptor(m),
            model: m.providerModel,
            project: m.project as string,
            region: m.region,
            token: gcp,
          });
    return [m.id, { config: m, provider }];
  }),
);

const pool = new pg.Pool({
  connectionString: databaseUrl,
  max: poolMax,
  connectionTimeoutMillis: 5000,
  statement_timeout: 10_000,
});
const client = await pool.connect();
try {
  await migrate(client);
} finally {
  client.release();
}
const ledger = createLedger(pool, {
  priceBookVersion: config.priceBook.version,
});
await ledger.publishPrices(config.priceBook.prices);

const server = createGateway({
  verify: createVerifier(issuers, { allowHttpJwks }),
  ledger,
  models,
  budgets: config.budgets,
  timeoutMs,
  maxInflight,
  log,
  ready: async () => {
    await pool.query("SELECT 1");
    return true;
  },
});
server.listen(port, "0.0.0.0", () => log({ msg: "listening", port }));

const reconciler = setInterval(() => {
  ledger
    .reconcile()
    .then((n) => n > 0 && log({ msg: "reconciled", reservations: n }))
    .catch((err) =>
      log({ msg: "reconcile_failed", error: (err as Error).name }),
    );
}, 30_000);
reconciler.unref();

process.once("SIGTERM", () => {
  server.drain();
  setTimeout(() => {
    clearInterval(reconciler);
    server.close(() => {
      pool.end().finally(() => process.exit(0));
    });
    server.closeIdleConnections();
    // Hard stop if a stream outlives the grace period; its lease is reconciled.
    setTimeout(() => process.exit(1), 10_000).unref();
  }, shutdownDelayMs).unref();
});
