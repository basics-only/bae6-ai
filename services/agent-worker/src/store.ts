import { randomUUID } from "node:crypto";
import type { Limits, Receipt } from "@bae6/ai-agents";
import type pg from "pg";

export type RunStatus = "queued" | "running" | "finished" | "cancelled";

export interface Submitter {
  organizationId: string;
  applicationId: string;
  submitterId: string;
}

export interface NewRun {
  model: string;
  task: string;
  tools: string[];
  limits: Limits;
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
  stop: string | null;
  text: string | null;
  tokens: number;
  receipts: Receipt[];
}

export interface Claimed extends NewRun {
  id: string;
  organizationId: string;
  attempt: number;
}

export interface Outcome {
  stop: string;
  text: string | undefined;
  tokens: number;
  receipts: Receipt[];
}

export class QueueFull extends Error {}
export class IdempotencyMismatch extends Error {}

interface Row {
  id: string;
  status: RunStatus;
  model: string;
  submitter_id: string;
  attempt: number;
  created_at: Date;
  started_at: Date | null;
  finished_at: Date | null;
  stop: string | null;
  result_text: string | null;
  tokens: string;
  receipts: Receipt[];
  request_sha256?: string | null;
  seq: string;
}

const iso = (d: Date | null) => (d ? d.toISOString() : null);

function toRun(r: Row): AgentRun {
  return {
    id: `run_${r.id}`,
    status: r.status,
    model: r.model,
    submitterId: r.submitter_id,
    attempt: r.attempt,
    createdAt: r.created_at.toISOString(),
    startedAt: iso(r.started_at),
    finishedAt: iso(r.finished_at),
    stop: r.stop,
    text: r.result_text,
    tokens: Number(r.tokens),
    receipts: r.receipts,
  };
}

const COLUMNS = `id, status, model, submitter_id, attempt, created_at, started_at,
  finished_at, stop, result_text, tokens, receipts, request_sha256, seq`;

export function createStore(pool: pg.Pool) {
  return {
    /** Queues a run, or returns the one an earlier request with the same key created. */
    async create(
      who: Submitter,
      run: NewRun,
      key: string | undefined,
      sha256: string,
      maxUnfinished: number,
    ): Promise<{ run: AgentRun; replayed: boolean }> {
      const client = await pool.connect();
      try {
        await client.query("BEGIN");
        // Serializes this organization's submits so the queue bound and the key claim cannot race.
        await client.query("SELECT pg_advisory_xact_lock(hashtext($1))", [
          `agent_runs:${who.organizationId}`,
        ]);
        if (key) {
          const found = await client.query<Row>(
            `SELECT ${COLUMNS} FROM agent_runs
             WHERE organization_id = $1 AND submitter_id = $2 AND idempotency_key = $3`,
            [who.organizationId, who.submitterId, key],
          );
          const existing = found.rows[0];
          if (existing) {
            await client.query("COMMIT");
            if (existing.request_sha256 !== sha256)
              throw new IdempotencyMismatch();
            return { run: toRun(existing), replayed: true };
          }
        }
        const open = await client.query<{ n: string }>(
          `SELECT count(*) AS n FROM agent_runs
           WHERE organization_id = $1 AND status IN ('queued', 'running')`,
          [who.organizationId],
        );
        if (Number(open.rows[0]?.n) >= maxUnfinished) {
          await client.query("ROLLBACK");
          throw new QueueFull();
        }
        const inserted = await client.query<Row>(
          `INSERT INTO agent_runs
             (id, organization_id, application_id, submitter_id, model, task, tools,
              limits, status, idempotency_key, request_sha256)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'queued', $9, $10)
           RETURNING ${COLUMNS}`,
          [
            randomUUID(),
            who.organizationId,
            who.applicationId,
            who.submitterId,
            run.model,
            run.task,
            run.tools,
            JSON.stringify(run.limits),
            key ?? null,
            sha256,
          ],
        );
        await client.query("COMMIT");
        return { run: toRun(inserted.rows[0] as Row), replayed: false };
      } catch (err) {
        await client.query("ROLLBACK").catch(() => {});
        throw err;
      } finally {
        client.release();
      }
    },

    async get(organizationId: string, id: string): Promise<AgentRun | null> {
      const r = await pool.query<Row>(
        `SELECT ${COLUMNS} FROM agent_runs WHERE organization_id = $1 AND id = $2`,
        [organizationId, id],
      );
      return r.rows[0] ? toRun(r.rows[0]) : null;
    },

    async list(
      organizationId: string,
      opts: { status?: RunStatus; limit: number; cursor?: string },
    ): Promise<{ data: AgentRun[]; nextCursor: string | null }> {
      const r = await pool.query<Row>(
        `SELECT ${COLUMNS.replace("receipts", "'[]'::jsonb AS receipts")}
         FROM agent_runs
         WHERE organization_id = $1
           AND ($2::text IS NULL OR status = $2)
           AND ($3::bigint IS NULL OR seq < $3)
         ORDER BY seq DESC LIMIT $4`,
        [
          organizationId,
          opts.status ?? null,
          opts.cursor ?? null,
          opts.limit + 1,
        ],
      );
      const page = r.rows.slice(0, opts.limit);
      return {
        data: page.map(toRun),
        nextCursor:
          r.rows.length > opts.limit
            ? (page[page.length - 1]?.seq ?? null)
            : null,
      };
    },

    /** A queued run becomes cancelled; a running one is flagged for its worker to stop. */
    async cancel(organizationId: string, id: string): Promise<AgentRun | null> {
      const r = await pool.query<Row>(
        `UPDATE agent_runs SET
           status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END,
           finished_at = CASE WHEN status = 'queued' THEN now() ELSE finished_at END,
           cancel_requested = cancel_requested OR status IN ('queued', 'running')
         WHERE organization_id = $1 AND id = $2
         RETURNING ${COLUMNS}`,
        [organizationId, id],
      );
      return r.rows[0] ? toRun(r.rows[0]) : null;
    },

    async claim(leaseMs: number): Promise<Claimed | null> {
      const r = await pool.query<{
        id: string;
        organization_id: string;
        model: string;
        task: string;
        tools: string[];
        limits: Limits;
        attempt: number;
      }>(
        `UPDATE agent_runs SET
           status = 'running',
           attempt = attempt + 1,
           lease_until = now() + $1 * interval '1 millisecond',
           started_at = COALESCE(started_at, now())
         WHERE id = (
           SELECT id FROM agent_runs WHERE status = 'queued'
           ORDER BY seq LIMIT 1 FOR UPDATE SKIP LOCKED)
         RETURNING id, organization_id, model, task, tools, limits, attempt`,
        [leaseMs],
      );
      const row = r.rows[0];
      return row
        ? {
            id: row.id,
            organizationId: row.organization_id,
            model: row.model,
            task: row.task,
            tools: row.tools,
            limits: row.limits,
            attempt: row.attempt,
          }
        : null;
    },

    /** Extends the lease. Null means this attempt no longer owns the run. */
    async heartbeat(
      id: string,
      attempt: number,
      leaseMs: number,
    ): Promise<{ cancelRequested: boolean } | null> {
      const r = await pool.query<{ cancel_requested: boolean }>(
        `UPDATE agent_runs SET lease_until = now() + $3 * interval '1 millisecond'
         WHERE id = $1 AND attempt = $2 AND status = 'running'
         RETURNING cancel_requested`,
        [id, attempt, leaseMs],
      );
      return r.rows[0] ? { cancelRequested: r.rows[0].cancel_requested } : null;
    },

    /** False when this attempt lost the run, so its result is discarded. */
    async finish(id: string, attempt: number, o: Outcome): Promise<boolean> {
      const r = await pool.query(
        `UPDATE agent_runs SET
           status = 'finished', finished_at = now(), lease_until = NULL,
           stop = $3, result_text = $4, tokens = $5, receipts = $6
         WHERE id = $1 AND attempt = $2 AND status = 'running'`,
        [
          id,
          attempt,
          o.stop,
          o.text ?? null,
          o.tokens,
          JSON.stringify(o.receipts),
        ],
      );
      return r.rowCount === 1;
    },

    /** Requeues runs whose worker stopped renewing; ends cancelled or exhausted ones. */
    async reap(maxAttempts: number): Promise<number> {
      const r = await pool.query(
        `UPDATE agent_runs SET
           status = CASE WHEN cancel_requested OR attempt >= $1 THEN 'finished' ELSE 'queued' END,
           stop = CASE WHEN cancel_requested THEN 'aborted'
                       WHEN attempt >= $1 THEN 'abandoned' END,
           finished_at = CASE WHEN cancel_requested OR attempt >= $1 THEN now() END,
           lease_until = NULL
         WHERE status = 'running' AND lease_until < now()`,
        [maxAttempts],
      );
      return r.rowCount ?? 0;
    },
  };
}

export type Store = ReturnType<typeof createStore>;
