import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { parse } from "yaml";

const config = (model: Record<string, unknown> = {}) => ({
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
      operations: ["chat"],
      inputModalities: ["text"],
      maxInputTokens: 10,
      maxOutputTokens: 10,
      ...model,
    },
  ],
  budgets: {},
});

const dir = mkdtempSync(join(tmpdir(), "check-config-"));

function check(contents: string | undefined) {
  const args = [
    "--import",
    "tsx",
    "--conditions=source",
    "src/check-config.ts",
  ];
  if (contents !== undefined) {
    const file = join(dir, `${Math.random().toString(36).slice(2)}.json`);
    writeFileSync(file, contents);
    args.push(file);
  }
  return spawnSync(process.execPath, args, {
    cwd: new URL("..", import.meta.url),
    encoding: "utf8",
    timeout: 20_000,
  });
}

test("exits 0 for a valid config", () => {
  const run = check(JSON.stringify(config()));
  assert.equal(run.status, 0);
  assert.match(run.stdout, /ok: 1 models, price book t/);
});

test("exits 1 and names the problem for a config the gateway would refuse", () => {
  const run = check(JSON.stringify(config({ region: "Not A Region" })));
  assert.equal(run.status, 1);
  assert.match(run.stderr, /model m: invalid region/);
});

test("exits 0 for a private endpoint on an AWS host", () => {
  const run = check(
    JSON.stringify(
      config({
        endpoint:
          "https://vpce-0abc.bedrock-runtime.us-east-1.vpce.amazonaws.com",
      }),
    ),
  );
  assert.equal(run.status, 0);
});

test("exits 1 for a custom endpoint that is not an AWS host", () => {
  const run = check(
    JSON.stringify(config({ endpoint: "http://elsewhere:8080" })),
  );
  assert.equal(run.status, 1);
  assert.match(run.stderr, /endpoint is not an AWS host/);
});

test("exits 1 for a file that is not JSON", () => {
  const run = check("{");
  assert.equal(run.status, 1);
});

test("exits 2 when no file is given", () => {
  const run = check(undefined);
  assert.equal(run.status, 2);
  assert.match(run.stderr, /usage: check-config/);
});

test("the config shipped in deploy/values.yaml is valid", () => {
  const values = parse(
    readFileSync(
      new URL("../../../deploy/values.yaml", import.meta.url),
      "utf8",
    ),
  ) as { env: { BAE6_AI_CONFIG: string } };
  const run = check(values.env.BAE6_AI_CONFIG);
  assert.equal(run.status, 0, run.stderr);
});
