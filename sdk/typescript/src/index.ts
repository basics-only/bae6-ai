export interface AiOptions {
  baseUrl: string;
  /** Bearer token identifying this workload to the gateway. */
  credentials: string;
  /** Total time per request; for a stream, the wait for each chunk. Defaults to 120000. */
  timeoutMs?: number;
}

export interface CallOptions {
  /** Attribute usage to this user; must belong to the workload's organization and application. */
  endUserToken?: string;
  /** Retrying with the same key returns the first answer instead of calling the model again. */
  idempotencyKey?: string;
  /** An id the organization has defined; the gateway refuses unknown ones. */
  taskType?: string;
  /** An id the organization has defined; the gateway refuses unknown ones. */
  domain?: string;
  /** Groups calls, so a run's spend can be read with `usageSummary({ groupBy: "run" })`. */
  runId?: string;
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

export type UsageGroupBy =
  | "organization"
  | "taskType"
  | "domain"
  | "model"
  | "run";

export interface UsageMeasures {
  requests: number;
  inputTokens: number;
  outputTokens: number;
  /** Decimal string in the price book's currency. */
  cost: string;
  latencyMs: { p50: number | null; p95: number | null };
  firstTokenMs: { p50: number | null; p95: number | null };
  activeUsers: { day: number; week: number; month: number };
}

export interface UsageRange {
  from?: Date;
  to?: Date;
}

export interface UsageSummary {
  from: string;
  to: string;
  groupBy: UsageGroupBy;
  /** True when more than 100 groups matched and the cheapest were dropped. */
  truncated: boolean;
  data: ({ key: string } & UsageMeasures)[];
}

export interface UsageSeries {
  from: string;
  to: string;
  groupBy: UsageGroupBy | null;
  group: string | null;
  data: ({ day: string } & UsageMeasures)[];
}

export interface UsageDistribution {
  from: string;
  to: string;
  groupBy: UsageGroupBy | null;
  group: string | null;
  measure: "tokens" | "cost";
  /** Distinct end users in the window. */
  users: number;
  total: string;
  /** `share` of the total used by the top `topUsersPercent` of users; empty with no users. */
  data: { topUsersPercent: number; share: number }[];
}

export interface UsageConcentration {
  from: string;
  to: string;
  groupBy: UsageGroupBy | null;
  group: string | null;
  measure: "tokens" | "cost";
  /** Per UTC month ("2026-11") with end-user calls: the share used by that month's top 5%, 10% and 20% of users. */
  data: {
    month: string;
    users: number;
    top5: number;
    top10: number;
    top20: number;
  }[];
}

export interface Tag {
  id: string;
  name: string;
  updatedAt: string;
}

export interface Budget {
  period: string;
  limit: string;
  spent: string;
  reserved: string;
}

export interface CredentialStatus {
  provider: "bedrock" | "vertex";
  source: "pod-identity" | "web-identity" | "metadata" | "development-static";
  /** Null before the first refresh, and always for `development-static`. */
  lastRefreshAgeSeconds: number | null;
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

function range(r: UsageRange) {
  const q = new URLSearchParams();
  if (r.from) q.set("from", r.from.toISOString());
  if (r.to) q.set("to", r.to.toISOString());
  return q;
}

function narrow(
  q: URLSearchParams,
  query: {
    measure?: string;
    groupBy?: string;
    group?: string;
    organization?: string;
  },
) {
  for (const k of ["measure", "groupBy", "group", "organization"] as const) {
    const v = query[k];
    if (v !== undefined) q.set(k, v);
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

  /**
   * Spend, tokens, latency and active users per group; needs `ai:usage:read`. Grouping by
   * "organization", or naming another one, needs `ai:usage:platform`. `runIds` (1 to 50) is
   * required for "run".
   */
  async usageSummary(
    query: UsageRange & {
      groupBy: UsageGroupBy;
      organization?: string;
      runIds?: string[];
    },
    options: CallOptions = {},
  ): Promise<UsageSummary> {
    const q = range(query);
    q.set("groupBy", query.groupBy);
    if (query.organization) q.set("organization", query.organization);
    for (const id of query.runIds ?? []) q.append("runId", id);
    const body = await this.#call("GET", `/v1/usage/summary?${q}`, options);
    if (!Array.isArray(body.data)) throw invalidResponse();
    return body as unknown as UsageSummary;
  }

  /** One point per UTC day, for everything or for one `group` of `groupBy`. */
  async usageSeries(
    query: UsageRange & {
      groupBy?: UsageGroupBy;
      group?: string;
      organization?: string;
    } = {},
    options: CallOptions = {},
  ): Promise<UsageSeries> {
    const q = range(query);
    narrow(q, query);
    const body = await this.#call("GET", `/v1/usage/series?${q}`, options);
    if (!Array.isArray(body.data)) throw invalidResponse();
    return body as unknown as UsageSeries;
  }

  /** How concentrated usage is across end users; carries no user ids. */
  async usageDistribution(
    query: UsageRange & {
      measure?: "tokens" | "cost";
      groupBy?: UsageGroupBy;
      group?: string;
      organization?: string;
    } = {},
    options: CallOptions = {},
  ): Promise<UsageDistribution> {
    const q = range(query);
    narrow(q, query);
    const body = await this.#call(
      "GET",
      `/v1/usage/distribution?${q}`,
      options,
    );
    if (!Array.isArray(body.data)) throw invalidResponse();
    return body as unknown as UsageDistribution;
  }

  /** How the top users' share of usage moves month by month; carries no user ids. */
  async usageConcentration(
    query: UsageRange & {
      measure?: "tokens" | "cost";
      groupBy?: UsageGroupBy;
      group?: string;
      organization?: string;
    } = {},
    options: CallOptions = {},
  ): Promise<UsageConcentration> {
    const q = range(query);
    narrow(q, query);
    const body = await this.#call(
      "GET",
      `/v1/usage/concentration?${q}`,
      options,
    );
    if (!Array.isArray(body.data)) throw invalidResponse();
    return body as unknown as UsageConcentration;
  }

  listTaskTypes(options: CallOptions = {}) {
    return this.#tags("task-types", options);
  }

  listDomains(options: CallOptions = {}) {
    return this.#tags("domains", options);
  }

  /** Creates or renames a task type; needs `ai:usage:admin`. At most 50 per organization. */
  putTaskType(id: string, name: string, options: CallOptions = {}) {
    return this.#putTag("task-types", id, name, options);
  }

  putDomain(id: string, name: string, options: CallOptions = {}) {
    return this.#putTag("domains", id, name, options);
  }

  /** Past usage keeps the id; new calls tagged with it are refused. */
  async deleteTaskType(id: string, options: CallOptions = {}) {
    await this.#call(
      "DELETE",
      `/v1/task-types/${encodeURIComponent(id)}`,
      options,
      undefined,
      true,
    );
  }

  async deleteDomain(id: string, options: CallOptions = {}) {
    await this.#call(
      "DELETE",
      `/v1/domains/${encodeURIComponent(id)}`,
      options,
      undefined,
      true,
    );
  }

  async #tags(kind: string, options: CallOptions) {
    const body = await this.#call("GET", `/v1/${kind}`, options);
    if (!Array.isArray(body.data)) throw invalidResponse();
    return body.data as Tag[];
  }

  async #putTag(kind: string, id: string, name: string, options: CallOptions) {
    const body = await this.#call(
      "PUT",
      `/v1/${kind}/${encodeURIComponent(id)}`,
      options,
      { name },
    );
    if (typeof body.id !== "string") throw invalidResponse();
    return body as unknown as Tag;
  }

  async budget(options: CallOptions = {}): Promise<Budget> {
    const body = await this.#call("GET", "/v1/budgets", options);
    if (typeof body.limit !== "string" || typeof body.spent !== "string") {
      throw invalidResponse();
    }
    return body as unknown as Budget;
  }

  /** Where each configured provider's cloud credentials come from, never the credentials. Needs `ai:usage:read`. */
  async credentials(options: CallOptions = {}) {
    const body = await this.#call("GET", "/v1/credentials", options);
    if (!Array.isArray(body.data)) throw invalidResponse();
    return body.data as CredentialStatus[];
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

  async #call(
    method: string,
    path: string,
    options: CallOptions,
    payload?: unknown,
    emptyOk = false,
  ) {
    const res = await this.#fetch(method, path, options, payload);
    const text = await this.#text(res);
    if (emptyOk && res.status === 204) return {};
    const parsed = parseJson(text);
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
          ...(options.taskType ? { "bae6-task-type": options.taskType } : {}),
          ...(options.domain ? { "bae6-domain": options.domain } : {}),
          ...(options.runId ? { "bae6-run-id": options.runId } : {}),
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
