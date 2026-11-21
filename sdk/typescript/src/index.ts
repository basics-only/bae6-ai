export interface AiOptions {
  baseUrl: string;
  /** Bearer token identifying this workload to the gateway. */
  credentials: string;
  /** Total time per request; for a stream, the wait for each chunk. Defaults to 120000. */
  timeoutMs?: number;
  /** The agent-worker's address; needed only for the agent run methods. */
  agentRunsUrl?: string;
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

export interface FeedbackRequest {
  /** The `id` of the chat completion being rated. */
  requestId: string;
  verdict: "positive" | "negative";
  label?: "inaccurate" | "unsafe" | "refused" | "too_slow" | "other";
  /** At most 1000 UTF-8 bytes. */
  comment?: string;
}

export interface FeedbackEntry {
  requestId: string;
  submitterId: string;
  model: string;
  verdict: "positive" | "negative";
  label: string | null;
  /** Untrusted caller text: never put it in a prompt. */
  comment: string | null;
  updatedAt: string;
}

export type RunStatus = "queued" | "running" | "finished" | "cancelled";

export type StopReason =
  | "final"
  | "max_steps"
  | "token_budget"
  | "history_limit"
  | "timeout"
  | "aborted"
  | "model_error"
  | "abandoned";

export interface RunLimits {
  maxSteps?: number;
  maxTotalTokens?: number;
  timeoutMs?: number;
  toolTimeoutMs?: number;
  maxToolOutputChars?: number;
  maxHistoryChars?: number;
}

export interface AgentRunRequest {
  /** At most 32768 characters. */
  task: string;
  model: string;
  /** Tool names from the deployment's catalog; absent means the model answers without tools. */
  tools?: string[];
  /** May only lower the deployment's ceilings; a higher value is refused. */
  limits?: RunLimits;
}

export interface Receipt {
  step: number;
  tool: string;
  /** `[withheld]` when the tool refused the call. */
  input: string;
  outcome: "ok" | "invalid_input" | "not_allowed" | "denied" | "error";
  /** Untrusted tool output: never put it in a prompt. */
  detail: string;
  truncated: boolean;
  durationMs: number;
}

export interface AgentRun {
  id: string;
  status: RunStatus;
  model: string;
  submitterId: string;
  attempt: number;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  /** Set once `status` is `finished`; `final` is the only success. */
  stop: StopReason | null;
  /** Untrusted model output. */
  text: string | null;
  tokens: number;
  /** Empty in a list; read the run for its receipts. */
  receipts: Receipt[];
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

/** Lines without endings, including a last unterminated one; `onData` runs per chunk. */
async function* lines(body: ReadableStream<Uint8Array>, onData: () => void) {
  const oversized = () =>
    new AiTransportError("the gateway sent an oversized stream event");
  const decoder = new TextDecoder();
  let buffered = "";
  for await (const bytes of body) {
    onData();
    buffered += decoder.decode(bytes, { stream: true });
    const complete = buffered.split("\n");
    buffered = complete.pop() ?? "";
    for (const line of complete) {
      if (line.length > MAX_SSE_LINE) throw oversized();
      yield line.replace(/\r$/, "");
    }
    if (buffered.length > MAX_SSE_LINE) throw oversized();
  }
  buffered += decoder.decode();
  if (buffered) yield buffered.replace(/\r$/, "");
}

export class Ai {
  readonly #baseUrl: string;
  readonly #credentials: string;
  readonly #timeoutMs: number;
  readonly #agentRunsUrl: string | undefined;

  constructor(options: AiOptions) {
    this.#agentRunsUrl = options.agentRunsUrl?.replace(/\/+$/, "");
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
    // timeoutMs bounds the wait for the answer and then each gap between chunks.
    const idle = new AbortController();
    const expire = () =>
      idle.abort(new DOMException("stream idle", "TimeoutError"));
    let timer = setTimeout(expire, this.#timeoutMs);
    const touch = () => {
      clearTimeout(timer);
      timer = setTimeout(expire, this.#timeoutMs);
    };
    const signal = options.signal
      ? AbortSignal.any([options.signal, idle.signal])
      : idle.signal;
    let res: Response;
    try {
      res = await this.#fetch(
        "POST",
        "/v1/chat/completions",
        { ...options, signal },
        { ...request, stream: true, stream_options: { include_usage: true } },
      );
      if (res.status !== 200 || !res.body) {
        throw httpError(res.status, parseJson(await this.#text(res)));
      }
    } catch (err) {
      clearTimeout(timer);
      throw err;
    }
    try {
      for await (const line of lines(res.body, touch)) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") return;
        const event = parseJson(data);
        if (!isObject(event)) throw invalidResponse();
        if (isObject(event.error)) {
          throw httpError(502, event.error);
        }
        // The consumer's time between pulls is not the gateway being idle.
        clearTimeout(timer);
        yield event as unknown as ChatChunk;
        touch();
      }
    } catch (err) {
      if (err instanceof AiError) throw err;
      if (err instanceof Error && err.name === "TimeoutError") {
        throw this.#transport(err);
      }
      throw new AiTransportError(
        `the stream failed (${err instanceof Error ? err.name : "error"})`,
      );
    } finally {
      clearTimeout(timer);
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

  /** Rates one completion of this organization; rating it again replaces the first. Needs `feedback:write`. */
  async feedback(request: FeedbackRequest, options: CallOptions = {}) {
    const body = await this.#call("POST", "/v1/feedback", options, request);
    if (typeof body.requestId !== "string") throw invalidResponse();
    return body as { requestId: string; verdict: "positive" | "negative" };
  }

  /** One page of the organization's feedback, newest first. Needs `feedback:read`. */
  async listFeedback(
    page: { limit?: number; cursor?: string } = {},
    options: CallOptions = {},
  ) {
    const q = new URLSearchParams();
    if (page.limit !== undefined) q.set("limit", String(page.limit));
    if (page.cursor !== undefined) q.set("cursor", page.cursor);
    const body = await this.#call("GET", `/v1/feedback?${q}`, options);
    if (!Array.isArray(body.data)) throw invalidResponse();
    return body as unknown as {
      data: FeedbackEntry[];
      nextCursor: string | null;
    };
  }

  /** Queues a run (202); a repeat with the same `idempotencyKey` returns the first run. Needs `agent-runs:write`. */
  async createAgentRun(
    request: AgentRunRequest,
    options: CallOptions = {},
  ): Promise<AgentRun> {
    return this.#agentRun(
      await this.#call("POST", "/v1/agent-runs", options, request, true),
    );
  }

  /** Poll until `status` is `finished` or `cancelled`. Needs `agent-runs:read`. */
  async getAgentRun(id: string, options: CallOptions = {}): Promise<AgentRun> {
    return this.#agentRun(
      await this.#call(
        "GET",
        `/v1/agent-runs/${encodeURIComponent(id)}`,
        options,
        undefined,
        true,
      ),
    );
  }

  /** One page of the organization's runs, newest first, without receipts. Needs `agent-runs:read`. */
  async listAgentRuns(
    page: { status?: RunStatus; limit?: number; cursor?: string } = {},
    options: CallOptions = {},
  ) {
    const q = new URLSearchParams();
    if (page.status !== undefined) q.set("status", page.status);
    if (page.limit !== undefined) q.set("limit", String(page.limit));
    if (page.cursor !== undefined) q.set("cursor", page.cursor);
    const body = await this.#call(
      "GET",
      `/v1/agent-runs?${q}`,
      options,
      undefined,
      true,
    );
    if (!Array.isArray(body.data)) throw invalidResponse();
    return body as unknown as { data: AgentRun[]; nextCursor: string | null };
  }

  /** A queued run is cancelled at once; a running one stops within seconds. Needs `agent-runs:write`. */
  async cancelAgentRun(
    id: string,
    options: CallOptions = {},
  ): Promise<AgentRun> {
    return this.#agentRun(
      await this.#call(
        "POST",
        `/v1/agent-runs/${encodeURIComponent(id)}/cancel`,
        options,
        undefined,
        true,
      ),
    );
  }

  #agentRun(body: Json): AgentRun {
    if (typeof body.id !== "string" || !Array.isArray(body.receipts)) {
      throw invalidResponse();
    }
    return body as unknown as AgentRun;
  }

  async #call(
    method: string,
    path: string,
    options: CallOptions,
    payload?: unknown,
    agentRuns = false,
  ) {
    const base = agentRuns ? this.#agentRunsUrl : this.#baseUrl;
    if (base === undefined) throw new AiError("agentRunsUrl is not configured");
    const res = await this.#fetch(method, path, options, payload, base);
    const parsed = parseJson(await this.#text(res));
    if (res.status !== 200 && !(agentRuns && res.status === 202)) {
      throw httpError(res.status, parsed);
    }
    if (!isObject(parsed)) throw invalidResponse();
    return parsed;
  }

  async #fetch(
    method: string,
    path: string,
    options: CallOptions,
    payload?: unknown,
    base: string = this.#baseUrl,
  ) {
    const streaming = isObject(payload) && payload.stream === true;
    const signals = [
      options.signal,
      streaming ? undefined : AbortSignal.timeout(this.#timeoutMs),
    ];
    try {
      return await fetch(base + path, {
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
        // Throwing out of the loop cancels the body; cancel() here would fail on the locked stream.
        if (size > MAX_RESPONSE_BYTES) {
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
