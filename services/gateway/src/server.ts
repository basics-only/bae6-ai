import { createHash, randomUUID } from "node:crypto";
import { once } from "node:events";
import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import {
  type Message,
  type ModelProvider,
  type Operation,
  ProviderError,
  type StreamEvent,
  UnsupportedOperation,
  type Usage,
} from "@bae6/ai-core";
import {
  type Analytics,
  type GroupBy,
  KIND_OF_PATH,
  type Kind,
  parseUsageQuery,
  RUN_ID,
  TAG_ID,
} from "./analytics.js";
import type { ModelConfig } from "./config.js";
import { type Feedback, parseFeedback } from "./feedback.js";
import { envelope, HttpError, readBody, send } from "./http.js";
import { type Identity, identify, type Verifier } from "./identity.js";
import type { Caller, Ledger, Outcome, Tags, Timing } from "./ledger.js";
import {
  chunk,
  completion,
  parseChat,
  parseEmbeddings,
  usageChunk,
} from "./openai.js";

export interface GatewayModel {
  config: ModelConfig;
  provider: ModelProvider;
}

export interface GatewayDeps {
  verify: Verifier;
  ledger: Ledger;
  feedback: Feedback;
  analytics: Analytics;
  models: Map<string, GatewayModel>;
  /** Monthly limit per organization; an organization without one cannot spend. */
  budgets: Record<string, string>;
  timeoutMs: number;
  maxInflight: number;
  /** In-flight invocations one organization may hold, so it cannot starve the others. */
  maxInflightPerOrg: number;
  ready: () => Promise<boolean>;
  log?: (event: Record<string, unknown>) => void;
}

const MAX_BODY_BYTES = 8 * 1024 * 1024;
const DEFAULT_MAX_TOKENS = 1024;
const MAX_USAGE_WINDOW_MS = 93 * 24 * 3600_000;
const MAX_FEEDBACK_BODY_BYTES = 16 * 1024;
const MAX_TAG_BODY_BYTES = 4 * 1024;
const MAX_TAG_NAME = 64;
const MAX_FEEDBACK_PAGE = 100;
const FEEDBACK_PER_WINDOW = 120;
const FEEDBACK_WINDOW_MS = 60_000;
const REQUEST_ID = /^[A-Za-z0-9._-]{1,64}$/;
const IDEMPOTENCY_KEY = /^[A-Za-z0-9._-]{8,128}$/;

type Who = Identity & { workloadId?: string };

/** What a call was tagged with and when its provider call started, for the usage event. */
interface Meta {
  tags: Tags;
  startedAt: number;
}

const callerOf = (who: Who): Caller => ({
  organizationId: who.organizationId,
  applicationId: who.applicationId,
  subjectId: who.subjectId,
});

function requireScope(who: Who, scope: string) {
  if (!who.scopes.includes(scope)) {
    throw new HttpError(403, "forbidden", `Missing scope ${scope}.`);
  }
}

const outcomeOf = (err: unknown): Outcome =>
  err instanceof UnsupportedOperation ||
  (err instanceof ProviderError && err.billed === "no")
    ? { kind: "failed" }
    : { kind: "uncertain" };

function providerHttpError(err: unknown, timedOut: boolean): HttpError {
  if (err instanceof UnsupportedOperation) {
    return new HttpError(400, "unsupported_operation", err.message);
  }
  if (err instanceof ProviderError && err.billed === "no") {
    if (err.status === 429) {
      return new HttpError(
        429,
        "rate_limited",
        "The provider is throttling.",
        2,
      );
    }
    if (err.status === 400) {
      return new HttpError(
        400,
        "provider_rejected",
        "The provider rejected the request.",
      );
    }
  }
  if (timedOut)
    return new HttpError(
      504,
      "timeout",
      "The provider did not answer in time.",
    );
  return new HttpError(502, "provider_error", "The provider call failed.");
}

export function createGateway(deps: GatewayDeps) {
  const { ledger, feedback, analytics, models, log = () => {} } = deps;
  // shortcut: fixed window per replica, so N replicas allow N times the limit; share it if that matters.
  const feedbackWindows = new Map<string, { start: number; count: number }>();
  let draining = false;
  let inflight = 0;
  const inflightByOrg = new Map<string, number>();

  const timing = (m: Meta, firstTokenAt: number | null): Timing => ({
    latencyMs: Math.round(performance.now() - m.startedAt),
    firstTokenMs:
      firstTokenAt === null ? null : Math.round(firstTokenAt - m.startedAt),
  });

  async function finalize(id: string, outcome: Outcome, t: Timing) {
    try {
      await ledger.finalize(id, outcome, t);
    } catch (err) {
      // The reconciler finalizes it as uncertain once the lease expires.
      log({ msg: "finalize_failed", ledgerId: id, error: (err as Error).name });
    }
  }

  const metering = <T>(read: Promise<T>) =>
    read.catch((err: Error) => {
      log({ msg: "ledger_unavailable", error: err.name });
      throw new HttpError(503, "unavailable", "Metering is unavailable.", 1);
    });

  // Feedback is off the inference path: its outage fails only feedback calls.
  const feedbackStore = <T>(read: Promise<T>) =>
    read.catch((err: Error) => {
      log({ msg: "feedback_unavailable", error: err.name });
      throw new HttpError(503, "unavailable", "Feedback is unavailable.", 1);
    });

  // Tags and tag lists sit beside the ledger, so their outage reads as metering's.
  const analyticsStore = <T>(read: Promise<T>) =>
    read.catch((err: unknown) => {
      if (err instanceof HttpError) throw err;
      log({ msg: "analytics_unavailable", error: (err as Error).name });
      throw new HttpError(
        503,
        "unavailable",
        "Usage analytics is unavailable.",
        1,
      );
    });

  function tagsOf(req: IncomingMessage): Tags {
    const defined = (header: string, name: string) => {
      const v = req.headers[header];
      if (v === undefined) return null;
      if (typeof v !== "string" || !TAG_ID.test(v)) {
        throw new HttpError(
          400,
          "invalid_request",
          `${name} must match ${TAG_ID}.`,
        );
      }
      return v;
    };
    const run = req.headers["bae6-run-id"];
    if (run !== undefined && (typeof run !== "string" || !RUN_ID.test(run))) {
      throw new HttpError(
        400,
        "invalid_request",
        `Bae6-Run-Id must match ${RUN_ID}.`,
      );
    }
    return {
      taskType: defined("bae6-task-type", "Bae6-Task-Type"),
      domain: defined("bae6-domain", "Bae6-Domain"),
      runId: run ?? null,
    };
  }

  /** Each tag must be one the organization defined, else nothing is spent. */
  async function checkTags(who: Who, tags: Tags) {
    const named: [Kind, string, string | null][] = [
      ["task_type", "Bae6-Task-Type", tags.taskType],
      ["domain", "Bae6-Domain", tags.domain],
    ];
    for (const [kind, name, id] of named) {
      if (
        id !== null &&
        !(await analyticsStore(analytics.defined(who.organizationId, kind, id)))
      ) {
        throw new HttpError(
          400,
          "unknown_tag",
          `${name} names a tag this organization has not defined.`,
        );
      }
    }
  }

  function resolve(id: string, who: Who, op: Operation): GatewayModel {
    const m = models.get(id);
    if (
      !m ||
      (m.config.organizations &&
        !m.config.organizations.includes(who.organizationId))
    ) {
      throw new HttpError(404, "not_found", "Model not found.");
    }
    if (!m.config.operations.includes(op)) {
      throw new HttpError(
        400,
        "unsupported_operation",
        `${id} does not support ${op}.`,
      );
    }
    return m;
  }

  /** Claims the idempotency key (if any) and reserves budget; returns null after a replay. */
  async function begin(
    res: ServerResponse,
    who: Who,
    model: GatewayModel,
    key: string | undefined,
    sha256: string,
    inputTokens: number,
    maxOutputTokens: number,
    tags: Tags,
  ): Promise<string | null> {
    const caller = callerOf(who);
    const ledgerId = `led_${randomUUID()}`;
    try {
      if (key) {
        const requestHash = createHash("sha256")
          .update(JSON.stringify([sha256, tags]))
          .digest("hex");
        const claim = await ledger.claim(caller, key, requestHash, ledgerId);
        if (claim.state === "replay") {
          res.setHeader("idempotent-replayed", "true");
          send(res, claim.status, claim.body);
          return null;
        }
        if (claim.state === "in_progress") {
          throw new HttpError(
            409,
            "conflict",
            "A request with this Idempotency-Key is in progress.",
            1,
          );
        }
        if (claim.state === "mismatch") {
          throw new HttpError(
            409,
            "idempotency_key_reused",
            "This Idempotency-Key was used for a different request.",
          );
        }
      }
      try {
        await checkTags(who, tags);
      } catch (err) {
        if (key) await ledger.release(caller, key).catch(() => {});
        throw err;
      }
      const limit = deps.budgets[who.organizationId];
      const estimate =
        limit === undefined
          ? null
          : await ledger.reserve({
              requestId: ledgerId,
              caller,
              limit,
              provider: model.config.provider,
              modelId: model.config.id,
              inputTokens,
              maxOutputTokens,
              leaseMs: deps.timeoutMs + 60_000,
              tags,
              endUser: who.subjectType === "user",
            });
      if (estimate === null) {
        if (key) await ledger.release(caller, key);
        throw limit === undefined
          ? new HttpError(
              403,
              "forbidden",
              "No budget is configured for this organization.",
            )
          : new HttpError(
              402,
              "budget_exceeded",
              "This request would exceed the organization's budget.",
            );
      }
      return ledgerId;
    } catch (err) {
      if (err instanceof HttpError) throw err;
      log({ msg: "ledger_unavailable", error: (err as Error).name });
      // Nothing reached a provider: free the key so the client's retry can proceed.
      if (key) await ledger.release(caller, key).catch(() => {});
      throw new HttpError(503, "unavailable", "Metering is unavailable.", 1);
    }
  }

  /** Runs one metered, non-streaming provider call and answers (and records it for replay). */
  async function metered(
    res: ServerResponse,
    requestId: string,
    who: Who,
    ledgerId: string,
    key: string | undefined,
    meta: Meta,
    call: (signal: AbortSignal) => Promise<{ body: unknown; usage: Usage }>,
  ) {
    const closed = new AbortController();
    // `req` closes once its body is read; only `res` closes on a client disconnect.
    res.once("close", () => {
      if (!res.writableEnded) closed.abort();
    });
    const timeout = AbortSignal.timeout(deps.timeoutMs);
    let status = 200;
    let body: unknown;
    let outcome: Outcome;
    try {
      const r = await call(AbortSignal.any([timeout, closed.signal]));
      outcome = { kind: "measured", usage: r.usage };
      body = r.body;
    } catch (err) {
      outcome = outcomeOf(err);
      const e = providerHttpError(err, timeout.aborted);
      status = e.status;
      body = envelope(e, requestId);
      if (e.retryAfterSeconds !== undefined)
        res.setHeader("retry-after", String(e.retryAfterSeconds));
    }
    await finalize(ledgerId, outcome, timing(meta, null));
    if (key) {
      const caller = callerOf(who);
      // A call the provider did not bill is safe to retry, so it is not replayed.
      // A failed store leaves the key in progress: a 409 beats a second billed call.
      await (outcome.kind === "failed"
        ? ledger.release(caller, key)
        : ledger.complete(caller, key, status, body)
      ).catch((err) =>
        log({ msg: "idempotency_store_failed", error: (err as Error).name }),
      );
    }
    send(res, status, body);
  }

  async function chat(
    req: IncomingMessage,
    res: ServerResponse,
    requestId: string,
    who: Who,
    key: string | undefined,
    meta: Meta,
  ) {
    const { json, sha256 } = await readBody(req, MAX_BODY_BYTES);
    const input = parseChat(json);
    const model = resolve(
      input.model,
      who,
      input.stream ? "chat_stream" : "chat",
    );
    const { config } = model;
    const maxTokens =
      input.maxTokens ?? Math.min(DEFAULT_MAX_TOKENS, config.maxOutputTokens);
    if (maxTokens > config.maxOutputTokens) {
      throw new HttpError(
        400,
        "invalid_request",
        `max_tokens is at most ${config.maxOutputTokens} for ${config.id}.`,
      );
    }
    if (input.inputTokens > config.maxInputTokens) {
      throw new HttpError(
        400,
        "context_too_large",
        `The input is over ${config.maxInputTokens} tokens for ${config.id}.`,
      );
    }
    if (
      input.messages.some((m) =>
        m.content.some((p) => !config.inputModalities.includes(p.type)),
      )
    ) {
      throw new HttpError(
        400,
        "unsupported_modality",
        `${config.id} accepts only ${config.inputModalities.join(", ")}.`,
      );
    }
    if (key && input.stream) {
      throw new HttpError(
        400,
        "unsupported_parameter",
        "Idempotency-Key is not supported with stream.",
      );
    }
    const ledgerId = await begin(
      res,
      who,
      model,
      key,
      sha256,
      input.inputTokens,
      maxTokens,
      meta.tags,
    );
    if (ledgerId === null) return;
    meta.startedAt = performance.now();
    const chatRequest = (signal: AbortSignal) => ({
      messages: input.messages as Message[],
      maxTokens,
      ...(input.temperature === undefined
        ? {}
        : { temperature: input.temperature }),
      signal,
    });

    if (!input.stream) {
      return metered(
        res,
        requestId,
        who,
        ledgerId,
        key,
        meta,
        async (signal) => {
          const r = await model.provider.chat(chatRequest(signal));
          return {
            body: completion(
              ledgerId,
              config.id,
              r.text,
              r.finishReason,
              r.usage,
            ),
            usage: r.usage,
          };
        },
      );
    }

    const closed = new AbortController();
    res.once("close", () => {
      if (!res.writableEnded) closed.abort();
    });
    const timeout = AbortSignal.timeout(deps.timeoutMs);
    const stop = AbortSignal.any([timeout, closed.signal]);
    let events: AsyncIterator<StreamEvent>;
    let next: IteratorResult<StreamEvent>;
    try {
      events = model.provider.stream(chatRequest(stop))[Symbol.asyncIterator]();
      next = await events.next();
    } catch (err) {
      await finalize(ledgerId, outcomeOf(err), timing(meta, null));
      throw providerHttpError(err, timeout.aborted);
    }
    // Stamped before any client write, which can block on a slow reader.
    let firstTokenAt: number | null =
      !next.done && next.value.type === "delta" ? performance.now() : null;
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "x-accel-buffering": "no",
    });
    const write = async (data: unknown) => {
      // Backpressure: wait for a slow client instead of buffering the stream in memory.
      if (
        !res.write(
          `data: ${typeof data === "string" ? data : JSON.stringify(data)}\n\n`,
        )
      ) {
        await once(res, "drain", { signal: stop });
      }
    };
    let outcome: Outcome = { kind: "uncertain" };
    try {
      await write(
        chunk(ledgerId, config.id, { role: "assistant", content: "" }),
      );
      while (!next.done) {
        const e = next.value;
        if (e.type === "delta") {
          firstTokenAt ??= performance.now();
          await write(chunk(ledgerId, config.id, { content: e.text }));
        } else {
          outcome = { kind: "measured", usage: e.usage };
          await write(chunk(ledgerId, config.id, {}, e.finishReason));
          if (input.includeUsage)
            await write(usageChunk(ledgerId, config.id, e.usage));
        }
        next = await events.next();
      }
      await write("[DONE]");
    } catch (err) {
      if (outcome.kind !== "measured") outcome = { kind: "uncertain" };
      if (!closed.signal.aborted) {
        res.write(
          `data: ${JSON.stringify({ error: envelope(providerHttpError(err, timeout.aborted), requestId) })}\n\n`,
        );
      }
    } finally {
      await finalize(ledgerId, outcome, timing(meta, firstTokenAt));
      res.end();
      // A client that stopped reading would otherwise hold the socket indefinitely.
      if (timeout.aborted && res.writableNeedDrain) res.destroy();
    }
  }

  async function embeddings(
    req: IncomingMessage,
    res: ServerResponse,
    requestId: string,
    who: Who,
    key: string | undefined,
    meta: Meta,
  ) {
    const { json, sha256 } = await readBody(req, MAX_BODY_BYTES);
    const input = parseEmbeddings(json);
    const model = resolve(input.model, who, "embed");
    if (input.inputTokens > model.config.maxInputTokens * input.input.length) {
      throw new HttpError(
        400,
        "context_too_large",
        "An input is too long for this model.",
      );
    }
    const ledgerId = await begin(
      res,
      who,
      model,
      key,
      sha256,
      input.inputTokens,
      0,
      meta.tags,
    );
    if (ledgerId === null) return;
    meta.startedAt = performance.now();
    return metered(res, requestId, who, ledgerId, key, meta, async (signal) => {
      const r = await model.provider.embed(input.input, signal);
      const usage = { inputTokens: r.usage.inputTokens, outputTokens: 0 };
      return {
        usage,
        body: {
          object: "list",
          data: r.vectors.map((embedding, index) => ({
            object: "embedding",
            index,
            embedding,
          })),
          model: model.config.id,
          usage: {
            prompt_tokens: usage.inputTokens,
            total_tokens: usage.inputTokens,
          },
        },
      };
    });
  }

  async function route(
    req: IncomingMessage,
    res: ServerResponse,
    url: URL,
    requestId: string,
  ) {
    const { pathname: path } = url;
    const started = performance.now();
    if (req.method === "GET" && path === "/health/live")
      return send(res, 200, { status: "ok" });
    if (req.method === "GET" && path === "/health/ready") {
      const ok = !draining && (await deps.ready().catch(() => false));
      return ok
        ? send(res, 200, { status: "ok" })
        : send(
            res,
            503,
            envelope(
              new HttpError(503, "unavailable", "Not ready."),
              requestId,
            ),
          );
    }
    if (!path.startsWith("/v1/"))
      throw new HttpError(404, "not_found", "Not found.");

    let who: Who | null;
    try {
      const endUser = req.headers["bae6-end-user-token"];
      who = await identify(
        deps.verify,
        req.headers.authorization,
        typeof endUser === "string" ? endUser : undefined,
      );
    } catch (err) {
      log({ msg: "identity_unavailable", error: (err as Error).name });
      throw new HttpError(
        503,
        "unavailable",
        "Identity provider unavailable.",
        5,
      );
    }
    if (!who)
      throw new HttpError(
        401,
        "unauthenticated",
        "Missing or invalid credentials.",
      );

    const endpoint = `${req.method} ${path}`;
    if (endpoint === "GET /v1/models") {
      requireScope(who, "ai:invoke");
      const org = who.organizationId;
      return send(res, 200, {
        object: "list",
        data: [...models.values()]
          .filter(
            (m) =>
              !m.config.organizations || m.config.organizations.includes(org),
          )
          .map((m) => ({
            id: m.config.id,
            object: "model",
            owned_by: m.config.provider,
            capabilities: m.provider.descriptor,
          })),
      });
    }
    if (endpoint === "GET /v1/budgets") {
      requireScope(who, "ai:usage:read");
      const b = await metering(ledger.budget(who.organizationId));
      return send(res, 200, {
        period: b.period,
        limit: deps.budgets[who.organizationId] ?? "0",
        spent: b.spent ?? "0",
        reserved: b.reserved ?? "0",
      });
    }
    if (endpoint === "GET /v1/usage") {
      requireScope(who, "ai:usage:read");
      const now = new Date();
      const from = new Date(
        url.searchParams.get("from") ??
          `${now.toISOString().slice(0, 7)}-01T00:00:00Z`,
      );
      const to = new Date(url.searchParams.get("to") ?? now.toISOString());
      const span = to.getTime() - from.getTime();
      if (Number.isNaN(span) || span < 0 || span > MAX_USAGE_WINDOW_MS) {
        throw new HttpError(
          400,
          "invalid_request",
          "from and to must be ISO times at most 93 days apart.",
        );
      }
      return send(res, 200, {
        from: from.toISOString(),
        to: to.toISOString(),
        data: await metering(ledger.usage(who, from, to)),
      });
    }

    if (
      endpoint === "GET /v1/usage/summary" ||
      endpoint === "GET /v1/usage/series"
    ) {
      requireScope(who, "ai:usage:read");
      const series = path.endsWith("/series");
      const q = parseUsageQuery(url, who, series);
      const head = {
        from: q.from.toISOString(),
        to: q.to.toISOString(),
        groupBy: q.groupBy,
      };
      if (series) {
        return send(res, 200, {
          ...head,
          group: q.group,
          data: await analyticsStore(analytics.series(q)),
        });
      }
      const r = await analyticsStore(
        analytics.summary({ ...q, groupBy: q.groupBy as GroupBy }),
      );
      return send(res, 200, { ...head, ...r });
    }
    const tagRoute = /^\/v1\/(task-types|domains)(?:\/([^/]+))?$/.exec(path);
    if (tagRoute) {
      const kind = KIND_OF_PATH[tagRoute[1] as string] as Kind;
      const id = tagRoute[2];
      if (id === undefined && req.method === "GET") {
        requireScope(who, "ai:usage:read");
        return send(res, 200, {
          data: await analyticsStore(analytics.tags(who.organizationId, kind)),
        });
      }
      if (id !== undefined && req.method === "PUT") {
        requireScope(who, "ai:usage:admin");
        if (!TAG_ID.test(id) || id === "untagged") {
          throw new HttpError(
            400,
            "invalid_request",
            `The id must match ${TAG_ID} and not be untagged.`,
          );
        }
        const { json } = await readBody(req, MAX_TAG_BODY_BYTES);
        const name = (json as { name?: unknown } | null)?.name;
        if (
          typeof json !== "object" ||
          json === null ||
          Object.keys(json).some((k) => k !== "name") ||
          typeof name !== "string" ||
          name.length < 1 ||
          name.length > MAX_TAG_NAME ||
          name.includes("\u0000")
        ) {
          throw new HttpError(
            400,
            "invalid_request",
            `Body must be {"name": 1 to ${MAX_TAG_NAME} characters}.`,
          );
        }
        return send(
          res,
          200,
          await analyticsStore(
            analytics.put(who.organizationId, kind, id, name),
          ),
        );
      }
      if (id !== undefined && req.method === "DELETE") {
        requireScope(who, "ai:usage:admin");
        const removed =
          TAG_ID.test(id) &&
          (await analyticsStore(
            analytics.remove(who.organizationId, kind, id),
          ));
        if (!removed) throw new HttpError(404, "not_found", "Tag not found.");
        res.writeHead(204).end();
        return;
      }
    }

    if (endpoint === "POST /v1/feedback") {
      requireScope(who, "feedback:write");
      const now = Date.now();
      const w = feedbackWindows.get(who.organizationId);
      if (!w || now - w.start >= FEEDBACK_WINDOW_MS) {
        for (const [org, old] of feedbackWindows) {
          if (now - old.start >= FEEDBACK_WINDOW_MS)
            feedbackWindows.delete(org);
        }
        feedbackWindows.set(who.organizationId, { start: now, count: 1 });
      } else if (++w.count > FEEDBACK_PER_WINDOW) {
        throw new HttpError(
          429,
          "rate_limited",
          "This organization is submitting feedback too fast.",
          Math.ceil((w.start + FEEDBACK_WINDOW_MS - now) / 1000),
        );
      }
      const { json } = await readBody(req, MAX_FEEDBACK_BODY_BYTES);
      const input = parseFeedback(json);
      const found = await feedbackStore(
        feedback.submit({
          ...input,
          organizationId: who.organizationId,
          submitterId: who.subjectId,
        }),
      );
      if (!found) throw new HttpError(404, "not_found", "Request not found.");
      return send(res, 200, {
        requestId: `chatcmpl-${input.requestId}`,
        verdict: input.verdict,
      });
    }
    if (endpoint === "GET /v1/feedback") {
      requireScope(who, "feedback:read");
      const limit = Number(url.searchParams.get("limit") ?? 50);
      const cursor = url.searchParams.get("cursor");
      if (
        !Number.isInteger(limit) ||
        limit < 1 ||
        limit > MAX_FEEDBACK_PAGE ||
        (cursor !== null && !/^[0-9]{1,15}$/.test(cursor))
      ) {
        throw new HttpError(
          400,
          "invalid_request",
          `limit is 1 to ${MAX_FEEDBACK_PAGE} and cursor is one returned by a previous page.`,
        );
      }
      return send(
        res,
        200,
        await feedbackStore(
          feedback.list(
            who.organizationId,
            limit,
            cursor === null ? null : Number(cursor),
          ),
        ),
      );
    }

    const invoke =
      endpoint === "POST /v1/chat/completions"
        ? chat
        : endpoint === "POST /v1/embeddings"
          ? embeddings
          : undefined;
    if (!invoke) throw new HttpError(404, "not_found", "Not found.");
    requireScope(who, "ai:invoke");
    const keyHeader = req.headers["idempotency-key"];
    if (
      keyHeader !== undefined &&
      (typeof keyHeader !== "string" || !IDEMPOTENCY_KEY.test(keyHeader))
    ) {
      throw new HttpError(
        400,
        "invalid_request",
        "Idempotency-Key must match ^[A-Za-z0-9._-]{8,128}$.",
      );
    }
    if (inflight >= deps.maxInflight) {
      throw new HttpError(
        429,
        "rate_limited",
        "This replica is at its concurrency limit.",
        1,
      );
    }
    const org = who.organizationId;
    const orgInflight = inflightByOrg.get(org) ?? 0;
    if (orgInflight >= deps.maxInflightPerOrg) {
      throw new HttpError(
        429,
        "rate_limited",
        "This organization is at its concurrency limit.",
        1,
      );
    }
    inflight++;
    inflightByOrg.set(org, orgInflight + 1);
    try {
      const tags = tagsOf(req);
      await invoke(req, res, requestId, who, keyHeader, {
        tags,
        startedAt: started,
      });
    } finally {
      inflight--;
      const left = (inflightByOrg.get(org) ?? 1) - 1;
      if (left === 0) inflightByOrg.delete(org);
      else inflightByOrg.set(org, left);
    }
  }

  const server = createServer(async (req, res) => {
    const supplied = req.headers["x-request-id"];
    const requestId =
      typeof supplied === "string" && REQUEST_ID.test(supplied)
        ? supplied
        : `req_${randomUUID()}`;
    res.setHeader("x-request-id", requestId);
    const url = new URL(req.url ?? "/", "http://gateway");
    const started = performance.now();
    res.once("finish", () =>
      log({
        msg: "request",
        requestId,
        method: req.method,
        path: url.pathname,
        status: res.statusCode,
        durationMs: Math.round(performance.now() - started),
      }),
    );
    try {
      await route(req, res, url, requestId);
    } catch (err) {
      let e: HttpError;
      if (err instanceof HttpError) {
        e = err;
      } else {
        log({ msg: "internal_error", requestId, error: (err as Error).name });
        e = new HttpError(500, "internal", "Internal error.");
      }
      if (e.retryAfterSeconds !== undefined)
        res.setHeader("retry-after", String(e.retryAfterSeconds));
      if (!res.headersSent) send(res, e.status, envelope(e, requestId));
      else res.end();
    }
  });
  server.headersTimeout = 10_000;
  // Covers the longest stream; the provider timeout ends it first.
  server.requestTimeout = deps.timeoutMs + 30_000;

  return Object.assign(server, {
    /** Fail readiness and close idle sockets; in-flight requests finish. */
    drain() {
      draining = true;
      server.closeIdleConnections();
    },
  });
}
