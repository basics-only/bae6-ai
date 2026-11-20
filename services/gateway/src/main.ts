import { readFileSync } from "node:fs";
import {
  bedrockProvider,
  huggingfaceProvider,
  type ModelDescriptor,
  type ModelProvider,
  vertexProvider,
} from "@bae6/ai-core";
import pg from "pg";
import { createAnalytics } from "./analytics.js";
import {
  type ModelConfig,
  nonAwsEndpointModels,
  parseConfig,
} from "./config.js";
import {
  awsCredentials,
  awsSource,
  gcpToken,
  hasCloudIdentity,
  type RefreshState,
} from "./credentials.js";
import { createFeedback } from "./feedback.js";
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
function devOnly(name: string) {
  const on = env[name] === "true";
  if (on && env.NODE_ENV !== "development") {
    throw new Error(`${name} requires NODE_ENV=development`);
  }
  return on;
}
const allowHttpJwks = devOnly("BAE6_ALLOW_HTTP_JWKS");
const issuers = JSON.parse(env.BAE6_AI_ISSUERS ?? "null") as Issuer[] | null;
if (!Array.isArray(issuers)) {
  throw new Error(
    "BAE6_AI_ISSUERS must be a JSON array of {issuer,audience,jwksUri,organization,application}",
  );
}
const configFile = env.BAE6_AI_CONFIG_FILE;
const rawConfig =
  env.BAE6_AI_CONFIG ?? (configFile && readFileSync(configFile, "utf8"));
if (!rawConfig) {
  throw new Error("BAE6_AI_CONFIG or BAE6_AI_CONFIG_FILE is required");
}
const config = parseConfig(JSON.parse(rawConfig));

// Signed requests carry real AWS credentials, so only AWS hosts are allowed outside development.
const allowCustomEndpoints = devOnly("BAE6_ALLOW_CUSTOM_PROVIDER_ENDPOINTS");
const foreignEndpoint = nonAwsEndpointModels(config)[0];
if (foreignEndpoint !== undefined && !allowCustomEndpoints) {
  throw new Error(
    `model ${foreignEndpoint}: endpoint is not an AWS host; needs BAE6_ALLOW_CUSTOM_PROVIDER_ENDPOINTS`,
  );
}

if (
  config.models.some((m) => m.provider === "bedrock") &&
  !hasCloudIdentity(env) &&
  env.NODE_ENV !== "development"
) {
  throw new Error(
    "Bedrock models need EKS Pod Identity (AWS_CONTAINER_CREDENTIALS_FULL_URI) or web identity (AWS_ROLE_ARN and AWS_WEB_IDENTITY_TOKEN_FILE); static AWS keys require NODE_ENV=development",
  );
}

const port = int("PORT", 8080, 1, 65535);
const timeoutMs = int("BAE6_PROVIDER_TIMEOUT_MS", 60_000, 1000, 300_000);
const maxInflight = int("BAE6_MAX_INFLIGHT", 64, 1, 10_000);
const maxInflightPerOrg = int(
  "BAE6_MAX_INFLIGHT_PER_ORG",
  maxInflight,
  1,
  maxInflight,
);
// Size so pool max x max replicas fits the database's share of connections (ADR-0007).
const poolMax = int("BAE6_DB_POOL_MAX", 10, 1, 100);
const analyticsPoolMax = int("BAE6_ANALYTICS_POOL_MAX", 3, 1, 20);
const shutdownDelayMs = int("BAE6_SHUTDOWN_DELAY_MS", 0, 0, 15_000);

const descriptor = (m: ModelConfig): ModelDescriptor => ({
  kind: "model",
  id: m.id,
  operations: m.operations,
  limits: {
    maxInputTokens: m.maxInputTokens,
    maxOutputTokens: m.maxOutputTokens,
  },
  attributes: {
    provider: m.provider,
    inputModalities: m.inputModalities,
    ...(m.hosting === undefined ? {} : { hosting: m.hosting }),
  },
});

function huggingfaceToken(m: ModelConfig): string {
  const token = env[m.tokenEnv as string];
  if (!token) throw new Error(`model ${m.id}: ${m.tokenEnv} is not set`);
  return token;
}

const awsState: RefreshState = {};
const gcpState: RefreshState = {};
const aws = awsCredentials(env, awsState);
const gcp = gcpToken(env, gcpState);
const ageSeconds = (s: RefreshState) =>
  s.lastRefresh === undefined
    ? null
    : Math.floor((Date.now() - s.lastRefresh) / 1000);
const credentialStatus = () => [
  ...(config.models.some((m) => m.provider === "bedrock")
    ? [
        {
          provider: "bedrock" as const,
          source: awsSource(env),
          lastRefreshAgeSeconds: ageSeconds(awsState),
        },
      ]
    : []),
  ...(config.models.some((m) => m.provider === "vertex")
    ? [
        {
          provider: "vertex" as const,
          source: "metadata" as const,
          lastRefreshAgeSeconds: ageSeconds(gcpState),
        },
      ]
    : []),
];
const models = new Map<string, GatewayModel>(
  config.models.map((m) => {
    const provider: ModelProvider =
      m.provider === "bedrock"
        ? bedrockProvider({
            descriptor: descriptor(m),
            modelId: m.providerModel,
            region: m.region as string,
            credentials: aws,
            ...(m.endpoint === undefined ? {} : { endpoint: m.endpoint }),
          })
        : m.provider === "huggingface"
          ? huggingfaceProvider({
              descriptor: descriptor(m),
              model: m.providerModel,
              token: huggingfaceToken(m),
              ...(m.endpoint === undefined ? {} : { endpoint: m.endpoint }),
            })
          : vertexProvider({
              descriptor: descriptor(m),
              model: m.providerModel,
              project: m.project as string,
              region: m.region as string,
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
// Apart from the ledger's pool so slow reports cannot starve reserve and finalize.
const analyticsPool = new pg.Pool({
  connectionString: databaseUrl,
  max: analyticsPoolMax,
  connectionTimeoutMillis: 5000,
  statement_timeout: 8_000,
});
for (const p of [pool, analyticsPool]) {
  p.on("error", (err) => log({ msg: "pool_error", error: err.name }));
}
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
const feedback = createFeedback(pool);
const analytics = createAnalytics(analyticsPool, pool);
const feedbackRetentionDays = int("BAE6_FEEDBACK_RETENTION_DAYS", 90, 1, 3650);

const server = createGateway({
  credentials: credentialStatus,
  verify: createVerifier(issuers, { allowHttpJwks }),
  ledger,
  feedback,
  analytics,
  models,
  routes: config.routes,
  budgets: config.budgets,
  timeoutMs,
  maxInflight,
  maxInflightPerOrg,
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
  feedback
    .prune(feedbackRetentionDays)
    .catch((err) => log({ msg: "prune_failed", error: (err as Error).name }));
}, 30_000);
reconciler.unref();

process.once("SIGTERM", () => {
  server.drain();
  setTimeout(() => {
    clearInterval(reconciler);
    server.close(() => {
      Promise.allSettled([pool.end(), analyticsPool.end()]).then(() =>
        process.exit(0),
      );
    });
    server.closeIdleConnections();
    // In-flight calls end within timeoutMs; set terminationGracePeriodSeconds above this.
    setTimeout(() => process.exit(1), timeoutMs + 5000).unref();
  }, shutdownDelayMs).unref();
});
