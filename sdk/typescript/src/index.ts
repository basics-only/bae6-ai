export interface AiOptions {
  baseUrl: string;
  /** Bearer token identifying this workload to the gateway. */
  credentials: string;
  /** Total time allowed per non-streaming request. Defaults to 120000. */
  timeoutMs?: number;
}

export interface CallOptions {
  /** Attribute usage to this user; must belong to the workload's organization and application. */
  endUserToken?: string;
  /** Retrying with the same key returns the first answer instead of calling the model again. */
  idempotencyKey?: string;
  signal?: AbortSignal;
}

export type ContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

export interface ChatMessage {
  role: "system" | "developer" | "user" | "assistant";
  content: string | ContentPart[];
}

export interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  max_completion_tokens?: number;
  temperature?: number;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletion {
  id: string;
  model: string;
  choices: {
    index: number;
    message: { role: "assistant"; content: string };
    finish_reason: string;
  }[];
  usage: TokenUsage;
}

export interface ChatChunk {
  id: string;
  model: string;
  choices: {
    index: number;
    delta: { role?: string; content?: string };
    finish_reason: string | null;
  }[];
  usage?: TokenUsage;
}

export interface Embeddings {
  model: string;
  data: { index: number; embedding: number[] }[];
  usage: { prompt_tokens: number; total_tokens: number };
}

export interface UsageRow {
  model: string;
  outcome: "measured" | "uncertain" | "failed";
  requests: number;
  inputTokens: number;
  outputTokens: number;
  /** Decimal string in the price book's currency. */
  cost: string;
}

export interface Budget {
  period: string;
  limit: string;
  spent: string;
  reserved: string;
}

export class AiError extends Error {
  readonly requestId: string | undefined;
  constructor(message: string, requestId?: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
    this.requestId = requestId;
  }
}

/** The gateway answered with an error. */
export class AiApiError extends AiError {
  constructor(
    readonly status: number,
    readonly code: string | undefined,
    message: string,
    requestId?: string,
    readonly retryAfterSeconds?: number,
  ) {
    super(message, requestId);
  }
}

/** 401 or 403: bad workload or end-user credentials, a missing scope, or no budget. */
export class AiAuthError extends AiApiError {}

/** 400, 409 or 413: the gateway rejected the request; retrying it unchanged will not help. */
export class AiValidationError extends AiApiError {}

/** 402: the organization's budget cannot cover this request. */
export class AiBudgetError extends AiApiError {}

/** The gateway was not reached, did not answer, or a stream ended early. */
export class AiTransportError extends AiError {}

export class AiTimeoutError extends AiTransportError {}

type Json = Record<string, unknown>;

const MAX_RESPONSE_BYTES = 64 * 1024 * 1024;
const MAX_SSE_LINE = 1024 * 1024;

const isObject = (v: unknown): v is Json =>
  typeof v === "object" && v !== null && !Array.isArray(v);
const str = (v: unknown) => (typeof v === "string" ? v : undefined);

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function httpError(status: number, parsed: unknown): AiApiError {
  const body = isObject(parsed) ? parsed : {};
  const Err =
    status === 402
      ? AiBudgetError
      : status === 401 || status === 403
        ? AiAuthError
        : status === 400 || status === 409 || status === 413
          ? AiValidationError
          : AiApiError;
  const retry = body.retryAfterSeconds;
  return new Err(
    status,
    str(body.code),
    str(body.message) ?? `the gateway returned HTTP ${status}`,
    str(body.requestId),
    typeof retry === "number" ? retry : undefined,
  );
}

const invalidResponse = () =>
  new AiApiError(
    200,
    "invalid_response",
    "the gateway returned an unusable response",
  );

async function* lines(body: ReadableStream<Uint8Array>) {
  const decoder = new TextDecoder();
  let buffered = "";
  for await (const bytes of body) {
    buffered += decoder.decode(bytes, { stream: true });
    let nl = buffered.indexOf("\n");
    while (nl >= 0) {
      yield buffered.slice(0, nl).replace(/\r$/, "");
      buffered = buffered.slice(nl + 1);
      nl = buffered.indexOf("\n");
    }
    if (buffered.length > MAX_SSE_LINE) {
      throw new AiTransportError("the gateway sent an oversized stream event");
    }
  }
}

export class Ai {
  readonly #baseUrl: string;
  readonly #credentials: string;
  readonly #timeoutMs: number;

  constructor(options: AiOptions) {
    this.#baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.#credentials = options.credentials;
    this.#timeoutMs = options.timeoutMs ?? 120_000;
  }

  async chat(
    request: ChatRequest,
    options: CallOptions = {},
  ): Promise<ChatCompletion> {
    const body = await this.#call(
      "POST",
      "/v1/chat/completions",
      options,
      request,
    );
    if (!Array.isArray(body.choices) || !isObject(body.usage))
      throw invalidResponse();
    return body as unknown as ChatCompletion;
  }

  /**
   * Yields chunks as the model produces them; the last carries `usage`. Breaking out of
   * the loop cancels the model call. A stream that ends early raises AiTransportError.
   */
  async *chatStream(
    request: ChatRequest,
    options: Omit<CallOptions, "idempotencyKey"> = {},
  ): AsyncGenerator<ChatChunk> {
    const res = await this.#fetch("POST", "/v1/chat/completions", options, {
      ...request,
      stream: true,
      stream_options: { include_usage: true },
    });
    if (res.status !== 200 || !res.body) {
      throw httpError(res.status, parseJson(await this.#text(res)));
    }
    try {
      for await (const line of lines(res.body)) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") return;
        const event = parseJson(data);
        if (!isObject(event)) throw invalidResponse();
        if (isObject(event.error)) {
          throw httpError(502, event.error);
        }
        yield event as unknown as ChatChunk;
      }
    } catch (err) {
      if (err instanceof AiError) throw err;
      throw new AiTransportError(
        `the stream failed (${err instanceof Error ? err.name : "error"})`,
      );
    } finally {
      await res.body.cancel().catch(() => {});
    }
    throw new AiTransportError("the stream ended before completion");
  }

  async embeddings(
    request: { model: string; input: string | string[] },
    options: CallOptions = {},
  ): Promise<Embeddings> {
    const body = await this.#call("POST", "/v1/embeddings", options, request);
    if (!Array.isArray(body.data)) throw invalidResponse();
    return body as unknown as Embeddings;
  }

  async models(options: CallOptions = {}) {
    const body = await this.#call("GET", "/v1/models", options);
    if (!Array.isArray(body.data)) throw invalidResponse();
    return body.data as { id: string; owned_by: string; capabilities: Json }[];
  }

  /** Usage of this application, by model and outcome, in [from, to) (at most 93 days). */
  async usage(
    range: { from?: Date; to?: Date } = {},
    options: CallOptions = {},
  ) {
    const q = new URLSearchParams();
    if (range.from) q.set("from", range.from.toISOString());
    if (range.to) q.set("to", range.to.toISOString());
    const body = await this.#call("GET", `/v1/usage?${q}`, options);
    if (!Array.isArray(body.data)) throw invalidResponse();
    return body.data as UsageRow[];
  }

  async budget(options: CallOptions = {}): Promise<Budget> {
    const body = await this.#call("GET", "/v1/budgets", options);
    if (typeof body.limit !== "string" || typeof body.spent !== "string") {
      throw invalidResponse();
    }
    return body as unknown as Budget;
  }

  async #call(
    method: string,
    path: string,
    options: CallOptions,
    payload?: unknown,
  ) {
    const res = await this.#fetch(method, path, options, payload);
    const parsed = parseJson(await this.#text(res));
    if (res.status !== 200) throw httpError(res.status, parsed);
    if (!isObject(parsed)) throw invalidResponse();
    return parsed;
  }

  async #fetch(
    method: string,
    path: string,
    options: CallOptions,
    payload?: unknown,
  ) {
    const streaming = isObject(payload) && payload.stream === true;
    const signals = [
      options.signal,
      streaming ? undefined : AbortSignal.timeout(this.#timeoutMs),
    ];
    try {
      return await fetch(this.#baseUrl + path, {
        method,
        // Credentials must not follow a redirect to another host.
        redirect: "manual",
        signal: AbortSignal.any(
          signals.filter((s): s is AbortSignal => s !== undefined),
        ),
        headers: {
          authorization: `Bearer ${this.#credentials}`,
          ...(options.endUserToken
            ? { "bae6-end-user-token": options.endUserToken }
            : {}),
          ...(options.idempotencyKey
            ? { "idempotency-key": options.idempotencyKey }
            : {}),
          ...(payload === undefined
            ? {}
            : { "content-type": "application/json" }),
        },
        ...(payload === undefined ? {} : { body: JSON.stringify(payload) }),
      });
    } catch (cause) {
      throw this.#transport(cause);
    }
  }

  async #text(res: Response): Promise<string> {
    if (!res.body) return "";
    const chunks: Uint8Array[] = [];
    let size = 0;
    try {
      for await (const chunk of res.body) {
        size += chunk.byteLength;
        if (size > MAX_RESPONSE_BYTES) {
          await res.body.cancel();
          throw new AiTransportError("the gateway response is too large");
        }
        chunks.push(chunk);
      }
    } catch (cause) {
      if (cause instanceof AiError) throw cause;
      throw this.#transport(cause);
    }
    return Buffer.concat(chunks).toString("utf8");
  }

  #transport(cause: unknown) {
    if (cause instanceof Error && cause.name === "TimeoutError") {
      return new AiTimeoutError(
        `the gateway did not answer within ${this.#timeoutMs} ms`,
        undefined,
        { cause },
      );
    }
    // No `cause`: fetch's error can quote a header value, i.e. a token.
    return new AiTransportError(
      `could not reach the gateway (${cause instanceof Error ? cause.name : "error"})`,
    );
  }
}
