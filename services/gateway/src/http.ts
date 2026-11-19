import { createHash } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";

export class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly retryAfterSeconds?: number,
  ) {
    super(message);
  }
}

export function send(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

export function envelope(err: HttpError, requestId: string) {
  return {
    code: err.code,
    message: err.message,
    requestId,
    ...(err.retryAfterSeconds === undefined
      ? {}
      : { retryAfterSeconds: err.retryAfterSeconds }),
  };
}

/** The raw body (for idempotency hashing) and its JSON, refusing anything over `maxBytes`. */
export async function readBody(
  req: IncomingMessage,
  maxBytes: number,
  deadlineMs = 15_000,
): Promise<{ json: unknown; sha256: string }> {
  const chunks: Buffer[] = [];
  let size = 0;
  const timer = setTimeout(() => req.destroy(), deadlineMs);
  try {
    for await (const chunk of req) {
      size += chunk.length;
      if (size > maxBytes) {
        throw new HttpError(
          413,
          "payload_too_large",
          `Body over ${maxBytes} bytes.`,
        );
      }
      chunks.push(chunk);
    }
  } finally {
    clearTimeout(timer);
  }
  const raw = Buffer.concat(chunks);
  try {
    return {
      json: JSON.parse(raw.toString("utf8")),
      sha256: createHash("sha256").update(raw).digest("hex"),
    };
  } catch {
    throw new HttpError(400, "invalid_request", "Body must be valid JSON.");
  }
}
