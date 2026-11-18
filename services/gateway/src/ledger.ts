import type { Usage } from "@bae6/ai-core";
import type pg from "pg";

export interface Caller {
  organizationId: string;
  applicationId: string;
  subjectId: string;
}

export type Outcome =
  | { kind: "measured"; usage: Usage }
  /** The provider may have billed; recorded at the estimate, never as zero. */
  | { kind: "uncertain" }
  /** The provider answered with an error: nothing was billed. */
  | { kind: "failed" };

export type Claim =
  | { state: "new"; requestId: string }
  | { state: "replay"; status: number; body: unknown }
  | { state: "in_progress" }
  | { state: "mismatch" };

const MAX_RECONCILE_BATCH = 100;

export const periodOf = (now = new Date()) => now.toISOString().slice(0, 7);

/** The metering ledger of ADR-0004: reserve before a provider call, finalize after it. */
export function createLedger(
  pool: pg.Pool,
  opts: { priceBookVersion: string },
) {
  const { priceBookVersion } = opts;

  return {
    /**
     * Holds the estimate against the organization's budget for this month, or returns
     * null (no provider call) when it would not fit.
     */
    async reserve(input: {
      requestId: string;
      caller: Caller;
      limit: string;
      provider: string;
      modelId: string;
      inputTokens: number;
      maxOutputTokens: number;
      leaseMs: number;
    }): Promise<string | null> {
      const { caller } = input;
      const period = periodOf();
      const client = await pool.connect();
      try {
        await client.query("BEGIN");
        await client.query(
          `INSERT INTO budgets (organization_id, period, limit_amount) VALUES ($1, $2, $3)
           ON CONFLICT (organization_id, period) DO UPDATE SET limit_amount = EXCLUDED.limit_amount
           WHERE budgets.limit_amount <> EXCLUDED.limit_amount`,
          [caller.organizationId, period, input.limit],
        );
        const held = await client.query<{ estimate: string }>(
          `UPDATE budgets b SET reserved = b.reserved + p.estimate
           FROM (SELECT ($3::int * input_per_mtok + $4::int * output_per_mtok) / 1000000 AS estimate
                 FROM prices WHERE price_book_version = $5 AND model_id = $6) p
           WHERE b.organization_id = $1 AND b.period = $2
             AND b.spent + b.reserved + p.estimate <= b.limit_amount
           RETURNING p.estimate::text AS estimate`,
          [
            caller.organizationId,
            period,
            input.inputTokens,
            input.maxOutputTokens,
            priceBookVersion,
            input.modelId,
          ],
        );
        const estimate = held.rows[0]?.estimate;
        if (estimate === undefined) {
          await client.query("ROLLBACK");
          return null;
        }
        await client.query(
          `INSERT INTO reservations (request_id, organization_id, period, application_id,
             subject_id, provider, model_id, price_book_version, estimate, lease_expires)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, now() + $10 * interval '1 millisecond')`,
          [
            input.requestId,
            caller.organizationId,
            period,
            caller.applicationId,
            caller.subjectId,
            input.provider,
            input.modelId,
            priceBookVersion,
            estimate,
            input.leaseMs,
          ],
        );
        await client.query("COMMIT");
        return estimate;
      } catch (err) {
        await client.query("ROLLBACK");
        throw err;
      } finally {
        client.release();
      }
    },

    /**
     * Moves the reservation into spend and writes the usage event, in one statement. A
     * second finalize of the same request finds no reservation and changes nothing.
     */
    async finalize(requestId: string, outcome: Outcome): Promise<boolean> {
      const usage = outcome.kind === "measured" ? outcome.usage : undefined;
      const result = await pool.query(
        `WITH r AS (DELETE FROM reservations WHERE request_id = $1 RETURNING *),
         c AS (
           SELECT r.*, CASE $2
               WHEN 'measured' THEN ($3::int * p.input_per_mtok + $4::int * p.output_per_mtok) / 1000000
               WHEN 'uncertain' THEN r.estimate
               ELSE 0 END AS cost
           FROM r JOIN prices p
             ON p.price_book_version = r.price_book_version AND p.model_id = r.model_id),
         e AS (
           INSERT INTO usage_events (request_id, organization_id, application_id, subject_id,
             provider, model_id, price_book_version, input_tokens, output_tokens, cost, outcome)
           SELECT request_id, organization_id, application_id, subject_id, provider, model_id,
             price_book_version, $3::int, $4::int, cost, $2 FROM c)
         UPDATE budgets b SET reserved = b.reserved - c.estimate, spent = b.spent + c.cost
         FROM c WHERE b.organization_id = c.organization_id AND b.period = c.period`,
        [
          requestId,
          outcome.kind,
          usage?.inputTokens ?? null,
          usage?.outputTokens ?? null,
        ],
      );
      return result.rowCount === 1;
    },

    /**
     * Finalizes reservations whose lease expired (their replica crashed or hung) as uncertain
     * at the estimate. Safe on every replica: claimed rows are skipped by the others.
     */
    async reconcile(): Promise<number> {
      const result = await pool.query<{ n: number }>(
        `WITH r AS (
           DELETE FROM reservations WHERE request_id IN (
             SELECT request_id FROM reservations WHERE lease_expires < now()
             ORDER BY lease_expires LIMIT $1 FOR UPDATE SKIP LOCKED)
           RETURNING *),
         e AS (
           INSERT INTO usage_events (request_id, organization_id, application_id, subject_id,
             provider, model_id, price_book_version, cost, outcome)
           SELECT request_id, organization_id, application_id, subject_id, provider, model_id,
             price_book_version, estimate, 'uncertain' FROM r),
         t AS (SELECT organization_id, period, sum(estimate) AS total FROM r GROUP BY 1, 2),
         u AS (
           UPDATE budgets b SET reserved = b.reserved - t.total, spent = b.spent + t.total
           FROM t WHERE b.organization_id = t.organization_id AND b.period = t.period)
         SELECT count(*)::int AS n FROM r`,
        [MAX_RECONCILE_BATCH],
      );
      await pool.query(
        `DELETE FROM idempotency_keys WHERE ctid IN (
           SELECT ctid FROM idempotency_keys
           WHERE created_at < now() - interval '24 hours' LIMIT 1000)`,
      );
      return result.rows[0]?.n ?? 0;
    },

    /** One request ID per caller and key; a retry replays the stored answer. */
    async claim(
      caller: Caller,
      key: string,
      requestHash: string,
      requestId: string,
    ): Promise<Claim> {
      const args = [
        caller.organizationId,
        caller.applicationId,
        caller.subjectId,
        key,
      ];
      const inserted = await pool.query(
        `INSERT INTO idempotency_keys (organization_id, application_id, subject_id, key,
           request_hash, request_id) VALUES ($1, $2, $3, $4, $5, $6)
         ON CONFLICT DO NOTHING`,
        [...args, requestHash, requestId],
      );
      if (inserted.rowCount === 1) return { state: "new", requestId };
      const { rows } = await pool.query<{
        request_hash: string;
        response_status: number | null;
        response_body: unknown;
      }>(
        `SELECT request_hash, response_status, response_body FROM idempotency_keys
         WHERE organization_id = $1 AND application_id = $2 AND subject_id = $3 AND key = $4`,
        args,
      );
      const row = rows[0];
      // Expired between the insert and the select: treat as still in progress.
      if (!row) return { state: "in_progress" };
      if (row.request_hash !== requestHash) return { state: "mismatch" };
      if (row.response_status === null) return { state: "in_progress" };
      return {
        state: "replay",
        status: row.response_status,
        body: row.response_body,
      };
    },

    async complete(caller: Caller, key: string, status: number, body: unknown) {
      await pool.query(
        `UPDATE idempotency_keys SET response_status = $5, response_body = $6
         WHERE organization_id = $1 AND application_id = $2 AND subject_id = $3 AND key = $4`,
        [
          caller.organizationId,
          caller.applicationId,
          caller.subjectId,
          key,
          status,
          JSON.stringify(body),
        ],
      );
    },

    /** Frees a key whose request never reached a provider, so a retry can try again. */
    async release(caller: Caller, key: string) {
      await pool.query(
        `DELETE FROM idempotency_keys
         WHERE organization_id = $1 AND application_id = $2 AND subject_id = $3 AND key = $4
           AND response_status IS NULL`,
        [caller.organizationId, caller.applicationId, caller.subjectId, key],
      );
    },

    async budget(organizationId: string) {
      const period = periodOf();
      const { rows } = await pool.query<{
        limit: string;
        spent: string;
        reserved: string;
      }>(
        `SELECT limit_amount::text AS limit, spent::text AS spent, reserved::text AS reserved
         FROM budgets WHERE organization_id = $1 AND period = $2`,
        [organizationId, period],
      );
      return { period, ...rows[0] };
    },

    async usage(caller: Omit<Caller, "subjectId">, from: Date, to: Date) {
      const { rows } = await pool.query<{
        model: string;
        outcome: string;
        requests: string;
        input_tokens: string;
        output_tokens: string;
        cost: string;
      }>(
        `SELECT model_id AS model, outcome, count(*)::text AS requests,
           coalesce(sum(input_tokens), 0)::text AS input_tokens,
           coalesce(sum(output_tokens), 0)::text AS output_tokens,
           sum(cost)::text AS cost
         FROM usage_events
         WHERE organization_id = $1 AND application_id = $2 AND time >= $3 AND time < $4
         GROUP BY 1, 2 ORDER BY 1, 2`,
        [caller.organizationId, caller.applicationId, from, to],
      );
      return rows.map((r) => ({
        model: r.model,
        outcome: r.outcome,
        requests: Number(r.requests),
        inputTokens: Number(r.input_tokens),
        outputTokens: Number(r.output_tokens),
        cost: r.cost,
      }));
    },

    /** Records the configured price book; refuses one that changed under the same version. */
    async publishPrices(
      prices: Record<string, { inputPerMTok: string; outputPerMTok: string }>,
    ) {
      const client = await pool.connect();
      try {
        await client.query("BEGIN");
        await client.query(
          "INSERT INTO price_books (version) VALUES ($1) ON CONFLICT DO NOTHING",
          [priceBookVersion],
        );
        for (const [model, p] of Object.entries(prices)) {
          await client.query(
            `INSERT INTO prices (price_book_version, model_id, input_per_mtok, output_per_mtok)
             VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING`,
            [priceBookVersion, model, p.inputPerMTok, p.outputPerMTok],
          );
          const same = await client.query(
            `SELECT 1 FROM prices WHERE price_book_version = $1 AND model_id = $2
               AND input_per_mtok = $3::numeric AND output_per_mtok = $4::numeric`,
            [priceBookVersion, model, p.inputPerMTok, p.outputPerMTok],
          );
          if (same.rowCount !== 1) {
            throw new Error(
              `price book ${priceBookVersion} already has a different price for ${model}; publish a new version`,
            );
          }
        }
        await client.query("COMMIT");
      } catch (err) {
        await client.query("ROLLBACK");
        throw err;
      } finally {
        client.release();
      }
    },
  };
}

export type Ledger = ReturnType<typeof createLedger>;
