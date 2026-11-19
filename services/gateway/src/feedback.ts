import type pg from "pg";
import { HttpError } from "./http.js";

export const LABELS = [
  "inaccurate",
  "unsafe",
  "refused",
  "too_slow",
  "other",
] as const;

export interface FeedbackInput {
  organizationId: string;
  submitterId: string;
  requestId: string;
  verdict: "positive" | "negative";
  label: (typeof LABELS)[number] | null;
  comment: string | null;
}

const PRUNE_BATCH = 1000;
const MAX_COMMENT_BYTES = 1000;
const COMPLETION_ID = /^chatcmpl-(led_[0-9a-f-]{36})$/;

const invalid = (message: string) =>
  new HttpError(400, "invalid_request", message);

/** Accepts the `id` of a chat completion and returns the ledger request ID behind it. */
export function parseFeedback(
  body: unknown,
): Omit<FeedbackInput, "organizationId" | "submitterId"> {
  if (typeof body !== "object" || body === null || Array.isArray(body)) {
    throw invalid("Body must be an object.");
  }
  const b = body as Record<string, unknown>;
  for (const k of Object.keys(b)) {
    if (!["requestId", "verdict", "label", "comment"].includes(k)) {
      throw new HttpError(
        400,
        "unsupported_parameter",
        `${k} is not supported.`,
      );
    }
  }
  const id = typeof b.requestId === "string" && COMPLETION_ID.exec(b.requestId);
  if (!id) throw invalid("requestId must be the id of a chat completion.");
  if (b.verdict !== "positive" && b.verdict !== "negative") {
    throw invalid("verdict must be positive or negative.");
  }
  const label = b.label ?? null;
  if (label !== null && !LABELS.includes(label as (typeof LABELS)[number])) {
    throw invalid(`label must be one of ${LABELS.join(", ")}.`);
  }
  const comment = b.comment ?? null;
  if (
    comment !== null &&
    (typeof comment !== "string" ||
      comment.includes("\u0000") ||
      Buffer.byteLength(comment) > MAX_COMMENT_BYTES)
  ) {
    throw invalid(
      `comment must be a string of at most ${MAX_COMMENT_BYTES} bytes.`,
    );
  }
  return {
    requestId: id[1] as string,
    verdict: b.verdict,
    label: label as FeedbackInput["label"],
    comment: comment as string | null,
  };
}

/** Model feedback of ADR-0009, kept in the gateway's own database. */
export function createFeedback(pool: pg.Pool) {
  return {
    /** Records the verdict for a request this organization made; false when it has no such request. */
    async submit(f: FeedbackInput): Promise<boolean> {
      const result = await pool.query(
        `INSERT INTO feedback (organization_id, request_id, submitter_id, model_id, verdict, label, comment)
         SELECT organization_id, request_id, $3, model_id, $4, $5, $6
         FROM usage_events WHERE request_id = $1 AND organization_id = $2
         ON CONFLICT (organization_id, request_id, submitter_id) DO UPDATE
           SET verdict = EXCLUDED.verdict, label = EXCLUDED.label,
               comment = EXCLUDED.comment, updated_at = now()`,
        [
          f.requestId,
          f.organizationId,
          f.submitterId,
          f.verdict,
          f.label,
          f.comment,
        ],
      );
      return result.rowCount === 1;
    },

    async list(organizationId: string, limit: number, before: number | null) {
      const { rows } = await pool.query<{
        id: string;
        request_id: string;
        submitter_id: string;
        model_id: string;
        verdict: string;
        label: string | null;
        comment: string | null;
        updated_at: Date;
      }>(
        `SELECT id::text, request_id, submitter_id, model_id, verdict, label, comment, updated_at
         FROM feedback
         WHERE organization_id = $1 AND ($2::bigint IS NULL OR id < $2)
         ORDER BY id DESC LIMIT $3`,
        [organizationId, before, limit + 1],
      );
      const page = rows.slice(0, limit);
      return {
        data: page.map((r) => ({
          requestId: r.request_id,
          submitterId: r.submitter_id,
          model: r.model_id,
          verdict: r.verdict,
          label: r.label,
          comment: r.comment,
          updatedAt: r.updated_at.toISOString(),
        })),
        nextCursor: rows.length > limit ? (page.at(-1)?.id ?? null) : null,
      };
    },

    async prune(retentionDays: number) {
      await pool.query(
        `DELETE FROM feedback WHERE ctid IN (
           SELECT ctid FROM feedback
           WHERE created_at < now() - $1 * interval '1 day' LIMIT ${PRUNE_BATCH})`,
        [retentionDays],
      );
    },
  };
}

export type Feedback = ReturnType<typeof createFeedback>;
