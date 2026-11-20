import { randomUUID } from "node:crypto";
import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import type { Limits } from "@bae6/ai-agents";
import { envelope, HttpError, readBody, send } from "./http.js";
import { identify, type Verifier } from "./identity.js";
import {
  IdempotencyMismatch,
  QueueFull,
  type RunStatus,
  type Store,
} from "./store.js";

export interface WorkerDeps {
  verify: Verifier;
  store: Store;
  /** Tool names a run may request. */
  catalog: string[];
  /** Upper bound for every limit; a request may only lower them. */
  ceilings: Limits;
  maxUnfinishedPerOrg: number;
  ready: () => Promise<boolean>;
  log?: (event: Record<string, unknown>) => void;
}

const MAX_BODY_BYTES = 64 * 1024;
const MAX_TASK_CHARS = 32_768;
const MAX_TOOLS = 32;
const MAX_PAGE = 100;
const REQUEST_ID = /^[A-Za-z0-9._-]{1,64}$/;
const IDEMPOTENCY_KEY = /^[A-Za-z0-9._-]{8,128}$/;
const MODEL = /^[a-z0-9][a-z0-9._-]{0,63}$/;
const TOOL = /^[A-Za-z0-9_.-]{1,64}$/;
const RUN_ID =
  /^run_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/;
const STATUSES: RunStatus[] = ["queued", "running", "finished", "cancelled"];
const LIMIT_KEYS = [
  "maxSteps",
  "maxTotalTokens",
  "timeoutMs",
  "toolTimeoutMs",
  "maxToolOutputChars",
  "maxHistoryChars",
] as const;

const invalid = (message: string) =>
  new HttpError(400, "invalid_request", message);

function requireScope(scopes: string[], scope: string) {
  if (!scopes.includes(scope)) {
    throw new HttpError(403, "forbidden", `Missing scope ${scope}.`);
  }
}

export function createWorkerServer(deps: WorkerDeps) {
  const { store, log = () => {} } = deps;
  let draining = false;

  const storage = <T>(read: Promise<T>) =>
    read.catch((err: Error) => {
      if (err instanceof QueueFull || err instanceof IdempotencyMismatch)
        throw err;
      log({ msg: "store_unavailable", error: err.name });
      throw new HttpError(503, "unavailable", "Storage is unavailable.", 1);
    });

  function parseCreate(json: unknown) {
    const o = json as Record<string, unknown>;
    if (typeof o !== "object" || o === null || Array.isArray(o)) {
      throw invalid("The body must be a JSON object.");
    }
    for (const k of Object.keys(o)) {
      if (!["task", "model", "tools", "limits"].includes(k))
        throw invalid(`Unknown field ${k}.`);
    }
    if (
      typeof o.task !== "string" ||
      o.task.length < 1 ||
      o.task.length > MAX_TASK_CHARS
    ) {
      throw invalid(`task is 1 to ${MAX_TASK_CHARS} characters.`);
    }
    if (typeof o.model !== "string" || !MODEL.test(o.model)) {
      throw invalid("model is not a valid model id.");
    }
    const tools = o.tools ?? [];
    if (
      !Array.isArray(tools) ||
      tools.length > MAX_TOOLS ||
      tools.some((t) => typeof t !== "string" || !TOOL.test(t)) ||
      new Set(tools).size !== tools.length
    ) {
      throw invalid(`tools is at most ${MAX_TOOLS} unique tool names.`);
    }
    const unknown = (tools as string[]).find((t) => !deps.catalog.includes(t));
    if (unknown !== undefined) throw invalid(`Unknown tool ${unknown}.`);

    const asked = o.limits ?? {};
    if (typeof asked !== "object" || asked === null || Array.isArray(asked)) {
      throw invalid("limits must be an object.");
    }
    const limits = { ...deps.ceilings };
    for (const [k, v] of Object.entries(asked)) {
      if (!(LIMIT_KEYS as readonly string[]).includes(k))
        throw invalid(`Unknown limit ${k}.`);
      const key = k as (typeof LIMIT_KEYS)[number];
      if (!Number.isInteger(v) || (v as number) < 1) {
        throw invalid(`${k} must be a positive integer.`);
      }
      if ((v as number) > deps.ceilings[key]) {
        throw invalid(`${k} is at most ${deps.ceilings[key]} here.`);
      }
      limits[key] = v as number;
    }
    return {
      task: o.task,
      model: o.model,
      tools: tools as string[],
      limits,
    };
  }

  const runId = (raw: string) => {
    const m = RUN_ID.exec(raw);
    if (!m) throw new HttpError(404, "not_found", "Run not found.");
    return m[1] as string;
  };

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

    let who: Awaited<ReturnType<typeof identify>>;
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
    const org = who.organizationId;

    if (req.method === "POST" && path === "/v1/agent-runs") {
      requireScope(who.scopes, "agent-runs:write");
      const keyHeader = req.headers["idempotency-key"];
      if (
        keyHeader !== undefined &&
        (typeof keyHeader !== "string" || !IDEMPOTENCY_KEY.test(keyHeader))
      ) {
        throw invalid("Idempotency-Key must match ^[A-Za-z0-9._-]{8,128}$.");
      }
      const { json, sha256 } = await readBody(req, MAX_BODY_BYTES);
      const run = parseCreate(json);
      try {
        const out = await storage(
          store.create(
            {
              organizationId: org,
              applicationId: who.applicationId,
              submitterId: who.subjectId,
            },
            run,
            keyHeader,
            sha256,
            deps.maxUnfinishedPerOrg,
          ),
        );
        if (out.replayed) res.setHeader("idempotent-replayed", "true");
        return send(res, out.replayed ? 200 : 202, out.run);
      } catch (err) {
        if (err instanceof QueueFull) {
          throw new HttpError(
            429,
            "rate_limited",
            "This organization has too many unfinished runs.",
            5,
          );
        }
        if (err instanceof IdempotencyMismatch) {
          throw new HttpError(
            409,
            "idempotency_key_reused",
            "This Idempotency-Key was used for a different request.",
          );
        }
        throw err;
      }
    }

    if (req.method === "GET" && path === "/v1/agent-runs") {
      requireScope(who.scopes, "agent-runs:read");
      const limit = Number(url.searchParams.get("limit") ?? 50);
      const cursor = url.searchParams.get("cursor");
      const status = url.searchParams.get("status");
      if (
        !Number.isInteger(limit) ||
        limit < 1 ||
        limit > MAX_PAGE ||
        (cursor !== null && !/^[0-9]{1,15}$/.test(cursor)) ||
        (status !== null && !STATUSES.includes(status as RunStatus))
      ) {
        throw invalid(
          `limit is 1 to ${MAX_PAGE}, cursor is one returned by a previous page, status is one of ${STATUSES.join(", ")}.`,
        );
      }
      const page = await storage(
        store.list(org, {
          limit,
          ...(cursor === null ? {} : { cursor }),
          ...(status === null ? {} : { status: status as RunStatus }),
        }),
      );
      return send(res, 200, page);
    }

    const one = /^\/v1\/agent-runs\/([^/]+)(\/cancel)?$/.exec(path);
    if (one && req.method === "GET" && !one[2]) {
      requireScope(who.scopes, "agent-runs:read");
      const run = await storage(store.get(org, runId(one[1] as string)));
      if (!run) throw new HttpError(404, "not_found", "Run not found.");
      return send(res, 200, run);
    }
    if (one && req.method === "POST" && one[2]) {
      requireScope(who.scopes, "agent-runs:write");
      const run = await storage(store.cancel(org, runId(one[1] as string)));
      if (!run) throw new HttpError(404, "not_found", "Run not found.");
      return send(res, 200, run);
    }
    throw new HttpError(404, "not_found", "Not found.");
  }

  const server = createServer(async (req, res) => {
    const supplied = req.headers["x-request-id"];
    const requestId =
      typeof supplied === "string" && REQUEST_ID.test(supplied)
        ? supplied
        : `req_${randomUUID()}`;
    res.setHeader("x-request-id", requestId);
    const url = new URL(req.url ?? "/", "http://agent-worker");
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
  server.requestTimeout = 30_000;

  return Object.assign(server, {
    /** Fail readiness and close idle sockets; in-flight requests finish. */
    drain() {
      draining = true;
      server.closeIdleConnections();
    },
  });
}
