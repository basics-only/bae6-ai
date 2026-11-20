import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import {
  awsCredentials,
  awsSource,
  hasCloudIdentity,
  type RefreshState,
  stsEndpoint,
} from "./credentials.js";

const dir = mkdtempSync(join(tmpdir(), "credentials-"));
const tokenFile = join(dir, "token");
writeFileSync(tokenFile, "token-1\n");

const requests: URLSearchParams[] = [];
let status = 200;
let expiresInSeconds = 3600;
const server: Server = createServer((req, res) => {
  let body = "";
  req.on("data", (c) => {
    body += c;
  });
  req.on("end", () => {
    requests.push(new URLSearchParams(body));
    res.statusCode = status;
    res.end(
      JSON.stringify({
        AssumeRoleWithWebIdentityResponse: {
          AssumeRoleWithWebIdentityResult: {
            Credentials: {
              AccessKeyId: "ASIA",
              SecretAccessKey: "secret",
              SessionToken: "session",
              Expiration: Date.now() / 1000 + expiresInSeconds,
            },
          },
        },
      }),
    );
  });
});
await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
after(() => server.close());

const env = (extra: Record<string, string> = {}): NodeJS.ProcessEnv => ({
  NODE_ENV: "development",
  AWS_ROLE_ARN: "arn:aws:iam::123456789012:role/gateway",
  AWS_WEB_IDENTITY_TOKEN_FILE: tokenFile,
  AWS_ENDPOINT_URL_STS: `http://127.0.0.1:${(server.address() as AddressInfo).port}`,
  ...extra,
});

test("exchanges the projected token for temporary credentials", async () => {
  requests.length = 0;
  const creds = await awsCredentials(env())();
  assert.deepEqual(creds, {
    accessKeyId: "ASIA",
    secretAccessKey: "secret",
    sessionToken: "session",
  });
  const sent = requests[0];
  assert.equal(sent?.get("Action"), "AssumeRoleWithWebIdentity");
  assert.equal(sent?.get("RoleArn"), "arn:aws:iam::123456789012:role/gateway");
  assert.equal(sent?.get("WebIdentityToken"), "token-1");
});

test("reuses credentials until shortly before they expire", async () => {
  requests.length = 0;
  const get = awsCredentials(env());
  await get();
  await get();
  assert.equal(requests.length, 1);
});

test("reads the token file again on each refresh because projected tokens rotate", async () => {
  requests.length = 0;
  expiresInSeconds = 1;
  const get = awsCredentials(env());
  await get();
  writeFileSync(tokenFile, "token-2\n");
  await get();
  expiresInSeconds = 3600;
  assert.deepEqual(
    requests.map((r) => r.get("WebIdentityToken")),
    ["token-1", "token-2"],
  );
});

test("fails loudly when STS refuses the token", async () => {
  status = 403;
  await assert.rejects(awsCredentials(env())(), /sts returned 403/);
  status = 200;
});

test("a Pod Identity URI counts as a cloud identity", () => {
  assert.equal(
    hasCloudIdentity(
      env({ AWS_CONTAINER_CREDENTIALS_FULL_URI: "http://169.254.170.23/v1" }),
    ),
    true,
  );
});

test("static keys are not a cloud identity", () => {
  assert.equal(
    hasCloudIdentity({ AWS_ACCESS_KEY_ID: "k", AWS_SECRET_ACCESS_KEY: "s" }),
    false,
  );
  assert.equal(hasCloudIdentity({ AWS_ROLE_ARN: "arn" }), false);
});

test("the STS endpoint cannot be overridden outside development", () => {
  const override = { AWS_ENDPOINT_URL_STS: "http://attacker.invalid" };
  for (const nodeEnv of [undefined, "production", "staging"]) {
    assert.equal(
      stsEndpoint({
        ...override,
        ...(nodeEnv ? { NODE_ENV: nodeEnv } : {}),
        AWS_REGION: "eu-west-1",
      }),
      "https://sts.eu-west-1.amazonaws.com",
    );
  }
});

test("rejects a region that could redirect the token to another host", () => {
  assert.throws(
    () => stsEndpoint({ AWS_REGION: "x.attacker.invalid/" }),
    /not a valid region/,
  );
});

test("records when credentials were last refreshed, not when they were last served", async () => {
  const state: RefreshState = {};
  const get = awsCredentials(env(), state);
  assert.equal(state.lastRefresh, undefined);
  await get();
  const first = state.lastRefresh;
  assert.notEqual(first, undefined);
  await new Promise((r) => setTimeout(r, 5));
  await get();
  assert.equal(state.lastRefresh, first);
});

test("names the source from the environment", () => {
  assert.equal(awsSource(env()), "web-identity");
  assert.equal(
    awsSource(env({ AWS_CONTAINER_CREDENTIALS_FULL_URI: "http://x" })),
    "pod-identity",
  );
  assert.equal(awsSource({ AWS_ACCESS_KEY_ID: "k" }), "development-static");
});
