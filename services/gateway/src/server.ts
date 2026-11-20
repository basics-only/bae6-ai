import { randomUUID } from "node:crypto";
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
  CLASSIFIER_MAX_TOKENS,
  CLASSIFIER_TIMEOUT_MS,
  classifierInputTokens,
  classifierMessages,
  parseTier,
} from "./classifier.js";
import type { ClassifierConfig, ModelConfig, RouteConfig } from "./config.js";
import { type Feedback, parseFeedback } from "./feedback.js";
import { envelope, HttpError, readBody, send } from "./http.js";
import { type Identity, identify, type Verifier } from "./identity.js";
import type { Caller, Ledger, Outcome } from "./ledger.js";
import {
  type ChatInput,
  chunk,
  completion,
  parseChat,
  parseEmbeddings,
  usageChunk,
} from "./openai.js";
import { pickRoute } from "./routing.js";

export interface GatewayModel {
  config: ModelConfig;
  provider: ModelProvider;
}

/** The one concrete model a request runs on, and the route that chose it, if any. */
interface Chosen {
  model: GatewayModel;
  maxTokens: number;
  route?: { id: string; decision: string };
}

export interface GatewayDeps {
  verify: Verifier;
  ledger: Ledger;
  feedback: Feedback;
  models: Map<string, GatewayModel>;
  routes?: RouteConfig[];
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
const MAX_FEEDBACK_PAGE = 100;
const FEEDBACK_PER_WINDOW = 120;
const FEEDBACK_WINDOW_MS = 60_000;
const REQUEST_ID = /^[A-Za-z0-9._-]{1,64}$/;
const IDEMPOTENCY_KEY = /^[A-Za-z0-9._-]{8,128}$/;

type Who = Identity & { workloadId?: string };

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

const mayUse = (c: { organizations?: string[] }, org: string) =>
  !c.organizations || c.organizations.includes(org);

const maxTokensFor = (config: ModelConfig, input: ChatInput) =>
  input.maxTokens ?? Math.min(DEFAULT_MAX_TOKENS, config.maxOutputTokens);

/** Why this model cannot take the request, or null if it can. */
function fitError(config: ModelConfig, input: ChatInput): HttpError | null {
  if (maxTokensFor(config, input) > config.maxOutputTokens) {
    return new HttpError(
      400,
      "invalid_request",
      `max_tokens is at most ${config.maxOutputTokens} for ${config.id}.`,
    );
  }
  if (input.inputTokens > config.maxInputTokens) {
    return new HttpError(
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
    return new HttpError(
      400,
      "unsupported_modality",
      `${config.id} accepts only ${config.inputModalities.join(", ")}.`,
    );
  }
  return null;
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
    // A dedicated endpoint scaled to zero answers 503 while it starts; the client retries, we do not.
    if (err.status === 503) {
      return new HttpError(
        503,
        "unavailable",
        "The model is not available yet.",
        5,
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
  const { ledger, feedback, models, log = () => {} } = deps;
  const routes = new Map((deps.routes ?? []).map((r) => [r.id, r]));
  // shortcut: fixed window per replica, so N replicas allow N times the limit; share it if that matters.
  const feedbackWindows = new Map<string, { start: number; count: number }>();
  let draining = false;
  let inflight = 0;
  const inflightByOrg = new Map<string, number>();

  async function finalize(id: string, outcome: Outcome) {
    try {
      await ledger.finalize(id, outcome);
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

  function resolve(id: string, who: Who, op: Operation): GatewayModel {
    const m = models.get(id);
    if (!m || !mayUse(m.config, who.organizationId)) {
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

  /**
   * Claims the idempotency key (if any), picks the model and reserves budget; returns null
   * after a replay. `choose` runs only for a new request, so a replay never pays for routing.
   */
  async function begin(
    res: ServerResponse,
    who: Who,
    key: string | undefined,
    sha256: string,
    inputTokens: number,
    choose: () => Promise<Chosen>,
  ): Promise<{ ledgerId: string; chosen: Chosen } | null> {
    const caller = callerOf(who);
    const ledgerId = `led_${randomUUID()}`;
    try {
      if (key) {
        const claim = await ledger.claim(caller, key, sha256, ledgerId);
        if (claim.state === "replay") {
          const first =
            claim.modelId === null ? undefined : models.get(claim.modelId);
          if (
            claim.modelId !== null &&
            (!first || !mayUse(first.config, who.organizationId))
          ) {
            throw new HttpError(
              409,
              "conflict",
              "The model that answered this request is no longer available to this organization.",
            );
          }
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
      let chosen: Chosen;
      try {
        chosen = await choose();
      } catch (err) {
        // Anything else is released once, by the outer handler.
        if (key && err instanceof HttpError) await ledger.release(caller, key);
        throw err;
      }
      const { model } = chosen;
      if (key && chosen.route)
        await ledger.choose(caller, key, model.config.id);
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
              maxOutputTokens: chosen.maxTokens,
              leaseMs: deps.timeoutMs + 60_000,
              ...(chosen.route ? { route: chosen.route } : {}),
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
      if (chosen.route)
        log({
          msg: "route",
          ledgerId,
          route: chosen.route.id,
          model: model.config.id,
          decision: chosen.route.decision,
        });
      return { ledgerId, chosen };
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
    await finalize(ledgerId, outcome);
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
  ) {
    const { json, sha256 } = await readBody(req, MAX_BODY_BYTES);
    const input = parseChat(json);
    const op = input.stream ? "chat_stream" : "chat";
    const route = routes.get(input.model);
    if (route && !mayUse(route, who.organizationId)) {
      throw new HttpError(404, "not_found", "Model not found.");
    }
    const named = (): Chosen => {
      const model = resolve(input.model, who, op);
      const err = fitError(model.config, input);
      if (err) throw err;
      return { model, maxTokens: maxTokensFor(model.config, input) };
    };
    const classify = async (
      r: RouteConfig,
      c: ClassifierConfig,
    ): Promise<string | null> => {
      const m = models.get(c.model);
      const limit = deps.budgets[who.organizationId];
      const tiers = Object.keys(c.tiers);
      const messages = classifierMessages(input.messages, tiers);
      if (
        !m ||
        !mayUse(m.config, who.organizationId) ||
        limit === undefined ||
        !messages
      )
        return null;
      const requestId = `led_${randomUUID()}`;
      const estimate = await ledger.reserve({
        requestId,
        caller: callerOf(who),
        limit,
        provider: m.config.provider,
        modelId: m.config.id,
        inputTokens: classifierInputTokens(messages),
        maxOutputTokens: CLASSIFIER_MAX_TOKENS,
        leaseMs: CLASSIFIER_TIMEOUT_MS + 60_000,
        route: { id: r.id, decision: "classifier" },
      });
      if (estimate === null) return null;
      let text: string | null = null;
      let outcome: Outcome;
      try {
        const out = await m.provider.chat({
          messages,
          maxTokens: CLASSIFIER_MAX_TOKENS,
          temperature: 0,
          signal: AbortSignal.timeout(
            Math.min(CLASSIFIER_TIMEOUT_MS, deps.timeoutMs),
          ),
        });
        text = out.text;
        outcome = { kind: "measured", usage: out.usage };
      } catch (err) {
        outcome = outcomeOf(err);
        log({
          msg: "classifier_failed",
          route: r.id,
          error: (err as Error).name,
        });
      }
      await finalize(requestId, outcome);
      return text === null ? null : parseTier(text, tiers);
    };
    const routed = async (r: RouteConfig): Promise<Chosen> => {
      const usable = (id: string) => {
        const m = models.get(id);
        return (
          m !== undefined &&
          mayUse(m.config, who.organizationId) &&
          m.config.operations.includes(op) &&
          fitError(m.config, input) === null
        );
      };
      const hasImages = input.messages.some((m) =>
        m.content.some((p) => p.type === "image"),
      );
      const c = await pickRoute(
        r,
        { inputTokens: input.inputTokens, hasImages },
        usable,
        (cl) => classify(r, cl),
      );
      if (!c) {
        throw new HttpError(
          400,
          "no_eligible_model",
          `No model on ${r.id} can serve this request.`,
        );
      }
      const model = models.get(c.modelId) as GatewayModel;
      return {
        model,
        maxTokens: maxTokensFor(model.config, input),
        route: { id: r.id, decision: c.decision },
      };
    };
    // A named model is checked before anything else; a route is resolved after the claim.
    const first = route ? undefined : named();
    if (key && input.stream) {
      throw new HttpError(
        400,
        "unsupported_parameter",
        "Idempotency-Key is not supported with stream.",
      );
    }
    const started = await begin(
      res,
      who,
      key,
      sha256,
      input.inputTokens,
      first ? async () => first : () => routed(route as RouteConfig),
    );
    if (started === null) return;
    const { ledgerId, chosen } = started;
    const { model, maxTokens } = chosen;
    const { config } = model;
    const chatRequest = (signal: AbortSignal) => ({
      messages: input.messages as Message[],
      maxTokens,
      ...(input.temperature === undefined
        ? {}
        : { temperature: input.temperature }),
      signal,
    });

    if (!input.stream) {
      return metered(res, requestId, who, ledgerId, key, async (signal) => {
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
      });
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
      await finalize(ledgerId, outcomeOf(err));
      throw providerHttpError(err, timeout.aborted);
    }
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
      await finalize(ledgerId, outcome);
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
  ) {
    const { json, sha256 } = await readBody(req, MAX_BODY_BYTES);
    const input = parseEmbeddings(json);
    const route = routes.get(input.model);
    if (route && mayUse(route, who.organizationId)) {
      throw new HttpError(
        400,
        "unsupported_operation",
        `${route.id} is a route and does not support embed.`,
      );
    }
    const model = resolve(input.model, who, "embed");
    if (input.inputTokens > model.config.maxInputTokens * input.input.length) {
      throw new HttpError(
        400,
        "context_too_large",
        "An input is too long for this model.",
      );
    }
    const started = await begin(
      res,
      who,
      key,
      sha256,
      input.inputTokens,
      async () => ({ model, maxTokens: 0 }),
    );
    if (started === null) return;
    const { ledgerId } = started;
    return metered(res, requestId, who, ledgerId, key, async (signal) => {
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
      const visible = [...models.values()].filter((m) => mayUse(m.config, org));
      const listed = [...routes.values()]
        .filter((r) => mayUse(r, org))
        .flatMap((r) => {
          const ids = new Set([r.default, ...r.rules.map((x) => x.model)]);
          const c = visible
            .filter((m) => ids.has(m.config.id))
            .map((m) => m.config);
          if (c.length === 0) return [];
          return [
            {
              id: r.id,
              object: "route",
              capabilities: {
                kind: "model",
                id: r.id,
                operations: ["chat", "chat_stream"].filter((o) =>
                  c.some((m) => m.operations.includes(o as Operation)),
                ),
                limits: {
                  maxInputTokens: Math.max(...c.map((m) => m.maxInputTokens)),
                  maxOutputTokens: Math.max(...c.map((m) => m.maxOutputTokens)),
                },
                attributes: {
                  inputModalities: [
                    ...new Set(c.flatMap((m) => m.inputModalities)),
                  ],
                },
              },
            },
          ];
        });
      return send(res, 200, {
        object: "list",
        data: [
          ...visible.map((m) => ({
            id: m.config.id,
            object: "model",
            owned_by: m.config.provider,
            capabilities: m.provider.descriptor,
          })),
          ...listed,
        ],
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
      await invoke(req, res, requestId, who, keyHeader);
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
