import { readFile } from "node:fs/promises";
import type { AwsCredentials } from "@bae6/ai-core";

const REFRESH_EARLY_MS = 5 * 60_000;

function cached<T>(fetchOne: () => Promise<{ value: T; expires: number }>) {
  let current: { value: T; expires: number } | undefined;
  let inflight: Promise<{ value: T; expires: number }> | undefined;
  return async (): Promise<T> => {
    if (current && current.expires - REFRESH_EARLY_MS > Date.now()) {
      return current.value;
    }
    inflight ??= fetchOne().finally(() => {
      inflight = undefined;
    });
    current = await inflight;
    return current.value;
  };
}

const REGION = /^[a-z0-9-]{1,32}$/;
const SESSION_NAME = /^[\w+=,.@-]{2,64}$/;

export const hasCloudIdentity = (env: NodeJS.ProcessEnv): boolean =>
  Boolean(
    env.AWS_CONTAINER_CREDENTIALS_FULL_URI ||
      (env.AWS_ROLE_ARN && env.AWS_WEB_IDENTITY_TOKEN_FILE),
  );

/** The STS endpoint can be overridden only in development, so a token is never sent elsewhere. */
export function stsEndpoint(env: NodeJS.ProcessEnv): string {
  if (env.NODE_ENV === "development" && env.AWS_ENDPOINT_URL_STS) {
    return env.AWS_ENDPOINT_URL_STS;
  }
  const region = env.AWS_REGION ?? env.AWS_DEFAULT_REGION;
  if (region === undefined) return "https://sts.amazonaws.com";
  if (!REGION.test(region)) throw new Error("AWS_REGION is not a valid region");
  return `https://sts.${region}.amazonaws.com`;
}

function webIdentityCredentials(
  env: NodeJS.ProcessEnv,
  roleArn: string,
  tokenFile: string,
): () => Promise<AwsCredentials> {
  const endpoint = stsEndpoint(env);
  const sessionName = env.AWS_ROLE_SESSION_NAME ?? "bae6-gateway";
  if (!SESSION_NAME.test(sessionName)) {
    throw new Error("AWS_ROLE_SESSION_NAME is not a valid session name");
  }
  return cached(async () => {
    const token = (await readFile(tokenFile, "utf8")).trim();
    const res = await fetch(endpoint, {
      method: "POST",
      headers: {
        "content-type": "application/x-www-form-urlencoded",
        accept: "application/json",
      },
      body: new URLSearchParams({
        Action: "AssumeRoleWithWebIdentity",
        Version: "2011-06-15",
        RoleArn: roleArn,
        RoleSessionName: sessionName,
        WebIdentityToken: token,
      }),
      redirect: "error",
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) throw new Error(`sts returned ${res.status}`);
    const c = (
      (await res.json()) as {
        AssumeRoleWithWebIdentityResponse?: {
          AssumeRoleWithWebIdentityResult?: {
            Credentials?: Record<string, string | number>;
          };
        };
      }
    ).AssumeRoleWithWebIdentityResponse?.AssumeRoleWithWebIdentityResult
      ?.Credentials;
    if (
      typeof c?.AccessKeyId !== "string" ||
      typeof c.SecretAccessKey !== "string" ||
      typeof c.SessionToken !== "string" ||
      typeof c.Expiration !== "number"
    ) {
      throw new Error("sts returned no credentials");
    }
    return {
      value: {
        accessKeyId: c.AccessKeyId,
        secretAccessKey: c.SecretAccessKey,
        sessionToken: c.SessionToken,
      },
      expires: c.Expiration * 1000,
    };
  });
}

/**
 * EKS Pod Identity (container credentials endpoint) when configured, then web identity
 * (IRSA, or any cluster whose OIDC provider IAM trusts), else static environment keys.
 */
export function awsCredentials(
  env: NodeJS.ProcessEnv,
): () => Promise<AwsCredentials> {
  const uri = env.AWS_CONTAINER_CREDENTIALS_FULL_URI;
  if (!uri && env.AWS_ROLE_ARN && env.AWS_WEB_IDENTITY_TOKEN_FILE) {
    return webIdentityCredentials(
      env,
      env.AWS_ROLE_ARN,
      env.AWS_WEB_IDENTITY_TOKEN_FILE,
    );
  }
  if (uri) {
    return cached(async () => {
      const tokenFile = env.AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE;
      const token = tokenFile
        ? (await readFile(tokenFile, "utf8")).trim()
        : env.AWS_CONTAINER_AUTHORIZATION_TOKEN;
      const res = await fetch(uri, {
        headers: token ? { authorization: token } : {},
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok)
        throw new Error(`credentials endpoint returned ${res.status}`);
      const b = (await res.json()) as Record<string, string>;
      if (!b.AccessKeyId || !b.SecretAccessKey) {
        throw new Error("credentials endpoint returned no keys");
      }
      return {
        value: {
          accessKeyId: b.AccessKeyId,
          secretAccessKey: b.SecretAccessKey,
          ...(b.Token ? { sessionToken: b.Token } : {}),
        },
        expires: b.Expiration
          ? Date.parse(b.Expiration)
          : Date.now() + REFRESH_EARLY_MS * 2,
      };
    });
  }
  return async () => {
    const { AWS_ACCESS_KEY_ID: id, AWS_SECRET_ACCESS_KEY: secret } = env;
    if (!id || !secret) throw new Error("no AWS credentials configured");
    return {
      accessKeyId: id,
      secretAccessKey: secret,
      ...(env.AWS_SESSION_TOKEN ? { sessionToken: env.AWS_SESSION_TOKEN } : {}),
    };
  };
}

/** An access token from the GCE/GKE metadata server (GKE Workload Identity). */
export function gcpToken(env: NodeJS.ProcessEnv): () => Promise<string> {
  const host = env.GCE_METADATA_HOST ?? "metadata.google.internal";
  return cached(async () => {
    const res = await fetch(
      `http://${host}/computeMetadata/v1/instance/service-accounts/default/token`,
      {
        headers: { "metadata-flavor": "Google" },
        signal: AbortSignal.timeout(5000),
      },
    );
    if (!res.ok) throw new Error(`metadata server returned ${res.status}`);
    const b = (await res.json()) as {
      access_token?: string;
      expires_in?: number;
    };
    if (!b.access_token || typeof b.expires_in !== "number") {
      throw new Error("metadata server returned no token");
    }
    return { value: b.access_token, expires: Date.now() + b.expires_in * 1000 };
  });
}
