import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { test } from "node:test";

const config = (endpoint?: string) =>
  JSON.stringify({
    priceBook: {
      version: "t",
      prices: { m: { inputPerMTok: "1", outputPerMTok: "1" } },
    },
    models: [
      {
        id: "m",
        provider: "bedrock",
        providerModel: "x",
        region: "us-east-1",
        endpoint,
        operations: ["chat"],
        inputModalities: ["text"],
        maxInputTokens: 10,
        maxOutputTokens: 10,
      },
    ],
    budgets: {},
  });

function start(env: Record<string, string>) {
  return spawnSync(
    process.execPath,
    ["--import", "tsx", "--conditions=source", "src/main.ts"],
    {
      env: {
        PATH: process.env.PATH ?? "",
        BAE6_DATABASE_URL: "postgres://x:x@127.0.0.1:1/x",
        BAE6_AI_ISSUERS: "[]",
        ...env,
      },
      cwd: new URL("..", import.meta.url),
      encoding: "utf8",
      timeout: 20_000,
    },
  );
}

function boot(
  flag: string | undefined,
  nodeEnv: string | undefined,
  extra: Record<string, string> = {},
) {
  const env: Record<string, string> = {
    BAE6_AI_CONFIG: config(
      flag === "BAE6_ALLOW_CUSTOM_PROVIDER_ENDPOINTS"
        ? "http://elsewhere:8080"
        : undefined,
    ),
    ...(flag ? { [flag]: "true" } : {}),
    ...extra,
  };
  if (nodeEnv !== undefined) env.NODE_ENV = nodeEnv;
  return start(env);
}

test("a Vertex-only config starts in production without any AWS settings", () => {
  const vertex = JSON.parse(config());
  vertex.models[0].provider = "vertex";
  vertex.models[0].project = "bae6-test";
  const run = start({
    NODE_ENV: "production",
    BAE6_AI_CONFIG: JSON.stringify(vertex),
  });
  assert.match(run.stderr, /ECONNREFUSED/);
});

test("refuses static AWS keys for Bedrock models unless NODE_ENV is development", () => {
  for (const nodeEnv of [undefined, "", "production", "staging"]) {
    const run = boot(undefined, nodeEnv, {
      AWS_ACCESS_KEY_ID: "k",
      AWS_SECRET_ACCESS_KEY: "s",
    });
    assert.notEqual(run.status, 0, `NODE_ENV=${nodeEnv}`);
    assert.match(run.stderr, /need EKS Pod Identity/);
  }
});

test("accepts static AWS keys when NODE_ENV is development", () => {
  const run = boot(undefined, "development", {
    AWS_ACCESS_KEY_ID: "k",
    AWS_SECRET_ACCESS_KEY: "s",
  });
  assert.doesNotMatch(run.stderr, /need EKS Pod Identity/);
  assert.match(run.stderr, /ECONNREFUSED/);
});

test("accepts Pod Identity outside development", () => {
  const run = boot(undefined, "production", {
    AWS_CONTAINER_CREDENTIALS_FULL_URI: "http://169.254.170.23/v1/credentials",
  });
  assert.doesNotMatch(run.stderr, /need EKS Pod Identity/);
  assert.match(run.stderr, /ECONNREFUSED/);
});

test("accepts web identity outside development", () => {
  const run = boot(undefined, "production", {
    AWS_ROLE_ARN: "arn:aws:iam::123456789012:role/gateway",
    AWS_WEB_IDENTITY_TOKEN_FILE: "/var/run/secrets/token",
  });
  assert.doesNotMatch(run.stderr, /need EKS Pod Identity/);
  assert.match(run.stderr, /ECONNREFUSED/);
});

for (const flag of [
  "BAE6_ALLOW_HTTP_JWKS",
  "BAE6_ALLOW_CUSTOM_PROVIDER_ENDPOINTS",
]) {
  test(`refuses ${flag} unless NODE_ENV is development`, () => {
    for (const nodeEnv of [undefined, "", "production", "prod", "staging"]) {
      const run = boot(flag, nodeEnv);
      assert.notEqual(run.status, 0, `NODE_ENV=${nodeEnv}`);
      assert.match(run.stderr, /requires NODE_ENV=development/);
    }
  });

  test(`accepts ${flag} when NODE_ENV is development`, () => {
    const run = boot(flag, "development");
    assert.doesNotMatch(run.stderr, /requires NODE_ENV=development/);
    assert.match(run.stderr, /ECONNREFUSED/);
  });
}
