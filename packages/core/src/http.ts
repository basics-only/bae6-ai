import { ProviderError } from "./types.js";

const MAX_RESPONSE_BYTES = 16 * 1024 * 1024;

/** POSTs to a provider. An HTTP error answer was not billed; a transport failure may have been. */
export async function post(
  url: string,
  init: { headers: Record<string, string>; body: string; signal: AbortSignal },
): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(url, { method: "POST", redirect: "error", ...init });
  } catch (err) {
    throw new ProviderError(
      `provider request failed (${(err as Error).name})`,
      "unknown",
    );
  }
  if (!res.ok) {
    // The error body can echo the prompt, so it is not read or kept.
    await res.body?.cancel();
    throw new ProviderError(
      `provider returned HTTP ${res.status}`,
      "no",
      res.status,
    );
  }
  return res;
}

/** The response body as chunks; a failure while reading may follow a billed call. */
export async function* chunks(res: Response): AsyncGenerator<Uint8Array> {
  if (!res.body) throw new ProviderError("provider sent no body", "unknown");
  let size = 0;
  try {
    for await (const chunk of res.body) {
      size += chunk.length;
      if (size > MAX_RESPONSE_BYTES) {
        throw new ProviderError("provider response too large", "unknown");
      }
      yield chunk;
    }
  } catch (err) {
    if (err instanceof ProviderError) throw err;
    throw new ProviderError(
      `provider response failed (${(err as Error).name})`,
      "unknown",
    );
  }
}

export async function json(res: Response): Promise<Record<string, unknown>> {
  const parts: Uint8Array[] = [];
  for await (const c of chunks(res)) parts.push(c);
  let parsed: unknown;
  try {
    parsed = JSON.parse(Buffer.concat(parts).toString("utf8"));
  } catch {
    throw new ProviderError("provider sent invalid JSON", "unknown");
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new ProviderError("provider sent an unexpected body", "unknown");
  }
  return parsed as Record<string, unknown>;
}

/** Token counts the provider must report; missing usage is never read as zero. */
export function count(v: unknown): number {
  if (typeof v !== "number" || !Number.isInteger(v) || v < 0) {
    throw new ProviderError("provider reported no usage", "unknown");
  }
  return v;
}

const MAX_SSE_LINE = 1024 * 1024;

/** The body's lines without line endings, including a last unterminated one. */
export async function* sseLines(res: Response): AsyncGenerator<string> {
  const tooLong = () =>
    new ProviderError("provider stream line too long", "unknown");
  let pending = "";
  const decoder = new TextDecoder();
  for await (const c of chunks(res)) {
    pending += decoder.decode(c, { stream: true });
    const lines = pending.split("\n");
    pending = lines.pop() ?? "";
    if (pending.length > MAX_SSE_LINE) throw tooLong();
    for (const line of lines) {
      if (line.length > MAX_SSE_LINE) throw tooLong();
      yield line.replace(/\r$/, "");
    }
  }
  pending += decoder.decode();
  if (pending) yield pending.replace(/\r$/, "");
}
