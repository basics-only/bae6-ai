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

/**
 * EKS Pod Identity (container credentials endpoint) when configured, else static
 * environment keys. shortcut: no IRSA web-identity exchange; add it if a cluster needs it.
 */
export function awsCredentials(
  env: NodeJS.ProcessEnv,
): () => Promise<AwsCredentials> {
  const uri = env.AWS_CONTAINER_CREDENTIALS_FULL_URI;
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
