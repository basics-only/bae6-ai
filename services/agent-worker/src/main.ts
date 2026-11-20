import { readFile } from "node:fs/promises";
import { createAgent, type Limits, type Tool } from "@bae6/ai-agents";
import pg from "pg";
import { createVerifier, type Issuer } from "./identity.js";
import { migrate } from "./migrate.js";
import { gatewayModel } from "./model.js";
import { createRunner } from "./runner.js";
import { createWorkerServer } from "./server.js";
import { createStore } from "./store.js";

const env = process.env;
const log = (e: Record<string, unknown>) => console.log(JSON.stringify(e));

function int(name: string, fallback: number, min: number, max: number) {
  const v = Number(env[name] ?? fallback);
  if (!Number.isInteger(v) || v < min || v > max) {
    throw new Error(`${name} must be an integer from ${min} to ${max}`);
  }
  return v;
}

function required(name: string) {
  const v = env[name];
  if (!v) throw new Error(`${name} is required`);
  return v;
}

const allowHttpJwks = env.BAE6_ALLOW_HTTP_JWKS === "true";
if (allowHttpJwks && env.NODE_ENV !== "development") {
  throw new Error("BAE6_ALLOW_HTTP_JWKS requires NODE_ENV=development");
}
const issuers = JSON.parse(env.BAE6_AI_ISSUERS ?? "null") as Issuer[] | null;
if (!Array.isArray(issuers) || issuers.length === 0) {
  throw new Error("BAE6_AI_ISSUERS must be a non-empty JSON array of issuers");
}
const organizations = new Set(issuers.map((i) => i.organization));
if (organizations.size !== 1) {
  throw new Error("every issuer must map to the same organization");
}

const gatewayUrl = required("BAE6_AI_GATEWAY_URL").replace(/\/+$/, "");
const tokenFile = required("BAE6_WORKER_TOKEN_FILE");
const ceilings: Limits = {
  maxSteps: int("BAE6_MAX_STEPS", 20, 1, 100),
  maxTotalTokens: int("BAE6_MAX_TOTAL_TOKENS", 100_000, 1, 10_000_000),
  timeoutMs: int("BAE6_RUN_TIMEOUT_MS", 300_000, 1000, 3_600_000),
  toolTimeoutMs: int("BAE6_TOOL_TIMEOUT_MS", 30_000, 1, 600_000),
  maxToolOutputChars: int("BAE6_MAX_TOOL_OUTPUT_CHARS", 20_000, 1, 1_000_000),
  maxHistoryChars: int("BAE6_MAX_HISTORY_CHARS", 200_000, 1, 10_000_000),
};
const port = int("PORT", 8081, 1, 65535);
const concurrency = int("BAE6_WORKER_CONCURRENCY", 4, 1, 64);
const leaseMs = int("BAE6_LEASE_MS", 60_000, 1000, 600_000);
const maxAttempts = int("BAE6_MAX_ATTEMPTS", 3, 1, 10);
const maxUnfinished = int("BAE6_MAX_UNFINISHED_PER_ORG", 100, 1, 10_000);
const modelTimeoutMs = int("BAE6_MODEL_TIMEOUT_MS", 60_000, 1000, 300_000);
const maxOutputTokens = int("BAE6_MODEL_MAX_OUTPUT_TOKENS", 1024, 1, 100_000);
const poolMax = int("BAE6_DB_POOL_MAX", 10, 1, 100);
const shutdownDelayMs = int("BAE6_SHUTDOWN_DELAY_MS", 0, 0, 15_000);
const databaseUrl = required("BAE6_DATABASE_URL");

// shortcut: no tools ship yet, so runs are model-only; register Tool objects here to offer them.
const catalog: Tool[] = [];

const pool = new pg.Pool({
  connectionString: databaseUrl,
  max: poolMax,
  connectionTimeoutMillis: 5000,
  statement_timeout: 10_000,
});
pool.on("error", (err) => log({ msg: "pool_error", error: err.name }));
const client = await pool.connect();
try {
  await migrate(client);
} finally {
  client.release();
}
const store = createStore(pool);

const runner = createRunner({
  store,
  concurrency,
  leaseMs,
  heartbeatMs: Math.floor(leaseMs / 3),
  pollMs: 1000,
  maxAttempts,
  maxResultChars: 100_000,
  log,
  // Agent and guards are built per run: guards count calls for the life of the tool object.
  execute: (run, signal) =>
    createAgent({
      model: gatewayModel({
        baseUrl: gatewayUrl,
        token: async () => (await readFile(tokenFile, "utf8")).trim(),
        model: run.model,
        maxOutputTokens,
        requestTimeoutMs: modelTimeoutMs,
      }),
      tools: catalog.filter((t) => run.tools.includes(t.name)),
      allow: run.tools,
      limits: run.limits,
    }).run(run.task, signal),
});

const server = createWorkerServer({
  verify: createVerifier(issuers, { allowHttpJwks }),
  store,
  catalog: catalog.map((t) => t.name),
  ceilings,
  maxUnfinishedPerOrg: maxUnfinished,
  log,
  ready: async () => {
    await pool.query("SELECT 1");
    return true;
  },
});
server.listen(port, "0.0.0.0", () => log({ msg: "listening", port }));
runner.start();

process.once("SIGTERM", () => {
  server.drain();
  setTimeout(() => {
    server.close();
    server.closeIdleConnections();
    // Interrupted runs keep their lease and are retried by another replica.
    runner
      .stop()
      .finally(() => pool.end())
      .finally(() => process.exit(0));
    setTimeout(() => process.exit(1), 10_000).unref();
  }, shutdownDelayMs).unref();
});
