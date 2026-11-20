import { randomUUID } from "node:crypto";
import type { Report } from "@bae6/ai-evals";
import type pg from "pg";

export type RunStatus = "queued" | "running" | "finished" | "cancelled";

export interface Submitter {
  organizationId: string;
  applicationId: string;
  submitterId: string;
}

export interface EvalCase {
  id: string;
  input: string;
  expected: string;
}

export interface NewRun {
  model: string;
  name: string;
  datasetVersion: string;
  cases: EvalCase[];
  scorers: string[];
}

export interface Summary {
  cases: number;
  failures: number;
  mean: Record<string, number>;
}

export interface EvalRun {
  id: string;
  status: RunStatus;
  model: string;
  name: string;
  datasetVersion: string;
  caseCount: number;
  scorers: string[];
  submitterId: string;
  attempt: number;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  stop: string | null;
  summary: Summary | null;
  report: Report | null;
}

export interface Claimed extends NewRun {
  id: string;
  organizationId: string;
  attempt: number;
}

export interface Outcome {
  stop: string;
  report: Report | undefined;
}

export class QueueFull extends Error {}
export class IdempotencyMismatch extends Error {}

interface Row {
  id: string;
  status: RunStatus;
  model: string;
  name: string;
  dataset_version: string;
  case_count: number;
  scorers: string[];
  submitter_id: string;
  attempt: number;
  created_at: Date;
  started_at: Date | null;
  finished_at: Date | null;
  stop: string | null;
  summary: Summary | null;
  report: Report | null;
  request_sha256?: string | null;
  seq: string;
}

const iso = (d: Date | null) => (d ? d.toISOString() : null);

function toRun(r: Row): EvalRun {
  return {
    id: `eval_${r.id}`,
    status: r.status,
    model: r.model,
    name: r.name,
    datasetVersion: r.dataset_version,
    caseCount: r.case_count,
    scorers: r.scorers,
    submitterId: r.submitter_id,
    attempt: r.attempt,
    createdAt: r.created_at.toISOString(),
    startedAt: iso(r.started_at),
    finishedAt: iso(r.finished_at),
    stop: r.stop,
    summary: r.summary,
    report: r.report,
  };
}

const COLUMNS = `id, status, model, name, dataset_version, jsonb_array_length(cases) AS case_count,
  scorers, submitter_id, attempt, created_at, started_at, finished_at, stop, summary,
  report, request_sha256, seq`;

export function createStore(pool: pg.Pool) {
  return {
    async create(
      who: Submitter,
      run: NewRun,
      key: string | undefined,
      sha256: string,
      maxUnfinished: number,
    ): Promise<{ run: EvalRun; replayed: boolean }> {
      const client = await pool.connect();
      try {
        await client.query("BEGIN");
        await client.query("SELECT pg_advisory_xact_lock(hashtext($1))", [
          `eval_runs:${who.organizationId}`,
        ]);
        if (key) {
          const found = await client.query<Row>(
            `SELECT ${COLUMNS} FROM eval_runs
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
          `SELECT count(*) AS n FROM eval_runs
           WHERE organization_id = $1 AND status IN ('queued', 'running')`,
          [who.organizationId],
        );
        if (Number(open.rows[0]?.n) >= maxUnfinished) {
          await client.query("ROLLBACK");
          throw new QueueFull();
        }
        const inserted = await client.query<Row>(
          `INSERT INTO eval_runs
             (id, organization_id, application_id, submitter_id, model, name,
              dataset_version, cases, scorers, status, idempotency_key, request_sha256)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'queued', $10, $11)
           RETURNING ${COLUMNS}`,
          [
            randomUUID(),
            who.organizationId,
            who.applicationId,
            who.submitterId,
            run.model,
            run.name,
            run.datasetVersion,
            JSON.stringify(run.cases),
            run.scorers,
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

    async get(organizationId: string, id: string): Promise<EvalRun | null> {
      const r = await pool.query<Row>(
        `SELECT ${COLUMNS} FROM eval_runs WHERE organization_id = $1 AND id = $2`,
        [organizationId, id],
      );
      return r.rows[0] ? toRun(r.rows[0]) : null;
    },

    async list(
      organizationId: string,
      opts: { status?: RunStatus; limit: number; cursor?: string },
    ): Promise<{ data: EvalRun[]; nextCursor: string | null }> {
      const r = await pool.query<Row>(
        `SELECT ${COLUMNS.replace("report,", "NULL::jsonb AS report,")}
         FROM eval_runs
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

    async cancel(organizationId: string, id: string): Promise<EvalRun | null> {
      const r = await pool.query<Row>(
        `UPDATE eval_runs SET
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
        name: string;
        dataset_version: string;
        cases: EvalCase[];
        scorers: string[];
        attempt: number;
      }>(
        `UPDATE eval_runs SET
           status = 'running',
           attempt = attempt + 1,
           lease_until = now() + $1 * interval '1 millisecond',
           started_at = COALESCE(started_at, now())
         WHERE id = (
           SELECT id FROM eval_runs WHERE status = 'queued'
           ORDER BY seq LIMIT 1 FOR UPDATE SKIP LOCKED)
         RETURNING id, organization_id, model, name, dataset_version, cases, scorers, attempt`,
        [leaseMs],
      );
      const row = r.rows[0];
      return row
        ? {
            id: row.id,
            organizationId: row.organization_id,
            model: row.model,
            name: row.name,
            datasetVersion: row.dataset_version,
            cases: row.cases,
            scorers: row.scorers,
            attempt: row.attempt,
          }
        : null;
    },

    async heartbeat(
      id: string,
      attempt: number,
      leaseMs: number,
    ): Promise<{ cancelRequested: boolean } | null> {
      const r = await pool.query<{ cancel_requested: boolean }>(
        `UPDATE eval_runs SET lease_until = now() + $3 * interval '1 millisecond'
         WHERE id = $1 AND attempt = $2 AND status = 'running'
         RETURNING cancel_requested`,
        [id, attempt, leaseMs],
      );
      return r.rows[0] ? { cancelRequested: r.rows[0].cancel_requested } : null;
    },

    async finish(id: string, attempt: number, o: Outcome): Promise<boolean> {
      const summary: Summary | null = o.report
        ? {
            cases: o.report.cases.length,
            failures: o.report.failures,
            mean: o.report.mean,
          }
        : null;
      const r = await pool.query(
        `UPDATE eval_runs SET
           status = 'finished', finished_at = now(), lease_until = NULL,
           stop = $3, summary = $4, report = $5
         WHERE id = $1 AND attempt = $2 AND status = 'running'`,
        [
          id,
          attempt,
          o.stop,
          summary && JSON.stringify(summary),
          o.report && JSON.stringify(o.report),
        ],
      );
      return r.rowCount === 1;
    },

    async reap(maxAttempts: number): Promise<number> {
      const r = await pool.query(
        `UPDATE eval_runs SET
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
