import type pg from "pg";
import { HttpError } from "./http.js";
import { type Identity, SLUG } from "./identity.js";

export type Kind = "task_type" | "domain";
export type GroupBy = keyof typeof GROUPS;

export const TAG_ID = /^[a-z0-9][a-z0-9-]{0,47}$/;
export const RUN_ID = /^[A-Za-z0-9._-]{1,64}$/;
export const KIND_OF_PATH: Record<string, Kind> = {
  "task-types": "task_type",
  domains: "domain",
};

const MAX_TAGS = 50;
const MAX_GROUPS = 100;
const MAX_RUNS = 50;
const MAX_WINDOW_MS = 93 * 24 * 3600_000;
const ISO = /^\d{4}-\d{2}-\d{2}T[\d:.]+(Z|[+-]\d{2}:\d{2})$/;
// Fixed strings, never user input: they are spliced into SQL.
const GROUPS = {
  organization: "organization_id",
  taskType: "coalesce(task_type, 'untagged')",
  domain: "coalesce(domain, 'untagged')",
  model: "model_id",
  run: "run_id",
} as const;

export interface UsageQuery {
  from: Date;
  to: Date;
  groupBy: GroupBy | null;
  group: string | null;
  /** Null only for platform callers asking across organizations. */
  organization: string | null;
  runIds: string[];
}

const invalid = (message: string) =>
  new HttpError(400, "invalid_request", message);

/** The window, grouping and organization a caller may ask for; throws 400 or 403 otherwise. */
export function parseUsageQuery(
  url: URL,
  who: Identity,
  series: boolean,
): UsageQuery {
  const q = url.searchParams;
  const platform = who.scopes.includes("ai:usage:platform");
  const now = new Date();
  const time = (name: string, fallback: string) => {
    const v = q.get(name) ?? fallback;
    const d = new Date(v);
    if (!ISO.test(v) || Number.isNaN(d.getTime())) {
      throw invalid(`${name} must be an ISO 8601 time.`);
    }
    return d;
  };
  const from = time("from", `${now.toISOString().slice(0, 7)}-01T00:00:00Z`);
  const to = time("to", now.toISOString());
  const span = to.getTime() - from.getTime();
  if (span < 0 || span > MAX_WINDOW_MS) {
    throw invalid("to must not be before from, and at most 93 days after it.");
  }

  const org = q.get("organization");
  if (org !== null && !SLUG.test(org))
    throw invalid("organization is invalid.");
  if (org !== null && org !== who.organizationId && !platform) {
    throw new HttpError(403, "forbidden", "Missing scope ai:usage:platform.");
  }
  const organization = platform ? org : who.organizationId;

  const g = q.get("groupBy");
  if (g !== null && !Object.hasOwn(GROUPS, g)) {
    throw invalid(`groupBy must be one of ${Object.keys(GROUPS).join(", ")}.`);
  }
  const groupBy = g as GroupBy | null;
  if (groupBy === "organization" && !platform) {
    throw new HttpError(403, "forbidden", "Missing scope ai:usage:platform.");
  }

  const group = q.get("group");
  const runIds = q.getAll("runId");
  if (series) {
    if ((groupBy === null) !== (group === null)) {
      throw invalid("groupBy and group are given together.");
    }
    if (group !== null && (group === "" || group.length > 128)) {
      throw invalid("group must be 1 to 128 characters.");
    }
    if (groupBy === "run" && !RUN_ID.test(group ?? "")) {
      throw invalid("group must be a run id.");
    }
    if (runIds.length > 0) throw invalid("runId is only for the summary.");
  } else {
    if (groupBy === null) throw invalid("groupBy is required.");
    if (group !== null) throw invalid("group is only for the series.");
    if (groupBy === "run") {
      if (
        runIds.length < 1 ||
        runIds.length > MAX_RUNS ||
        !runIds.every((r) => RUN_ID.test(r))
      ) {
        throw invalid(`groupBy=run takes 1 to ${MAX_RUNS} valid runId values.`);
      }
    } else if (runIds.length > 0) {
      throw invalid("runId is only for groupBy=run.");
    }
  }
  return { from, to, groupBy, group, organization, runIds };
}

interface MeasureRow {
  requests: number;
  input_tokens: number;
  output_tokens: number;
  cost: string;
  l50: number | null;
  l95: number | null;
  f50: number | null;
  f95: number | null;
  au_day: number;
  au_week: number;
  au_month: number;
}

const measures = (m: MeasureRow) => ({
  requests: m.requests,
  inputTokens: m.input_tokens,
  outputTokens: m.output_tokens,
  cost: m.cost,
  latencyMs: { p50: m.l50, p95: m.l95 },
  firstTokenMs: { p50: m.f50, p95: m.f95 },
  activeUsers: { day: m.au_day, week: m.au_week, month: m.au_month },
});

const HOURS = { day: 24, week: 168, month: 720 };

/** The aggregates of one group of events; `f` picks the events inside the window. */
const aggregates = (f: string) => `
  count(*) FILTER (WHERE ${f})::int AS requests,
  coalesce(sum(input_tokens) FILTER (WHERE ${f}), 0)::float8 AS input_tokens,
  coalesce(sum(output_tokens) FILTER (WHERE ${f}), 0)::float8 AS output_tokens,
  coalesce(sum(cost) FILTER (WHERE ${f}), 0)::text AS cost,
  percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE ${f}) AS l50,
  percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE ${f}) AS l95,
  percentile_cont(0.5) WITHIN GROUP (ORDER BY first_token_ms) FILTER (WHERE ${f}) AS f50,
  percentile_cont(0.95) WITHIN GROUP (ORDER BY first_token_ms) FILTER (WHERE ${f}) AS f95`;

const PERSON = "(organization_id, application_id, subject_id)";

/** Usage by tag, run and day, and the tags themselves (ADR-0011); the same database as the ledger. */
export function createAnalytics(pool: pg.Pool, requestPool: pg.Pool = pool) {
  return {
    async tags(organizationId: string, kind: Kind) {
      const { rows } = await pool.query<{
        id: string;
        name: string;
        updated_at: Date;
      }>(
        `SELECT id, name, updated_at FROM tags
         WHERE organization_id = $1 AND kind = $2 ORDER BY id`,
        [organizationId, kind],
      );
      return rows.map((r) => ({
        id: r.id,
        name: r.name,
        updatedAt: r.updated_at.toISOString(),
      }));
    },

    async defined(organizationId: string, kind: Kind, id: string) {
      const { rowCount } = await requestPool.query(
        "SELECT 1 FROM tags WHERE organization_id = $1 AND kind = $2 AND id = $3",
        [organizationId, kind, id],
      );
      return rowCount === 1;
    },

    /** Creates or renames; throws 409 when a new tag would be over the limit. */
    async put(organizationId: string, kind: Kind, id: string, name: string) {
      const client = await pool.connect();
      try {
        await client.query("BEGIN");
        // Serializes creates, so two at the limit cannot both pass the count.
        await client.query(
          "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
          [`${organizationId}/${kind}`],
        );
        const { rows } = await client.query<{ n: number; has: boolean }>(
          `SELECT count(*)::int AS n, coalesce(bool_or(id = $3), false) AS has
           FROM tags WHERE organization_id = $1 AND kind = $2`,
          [organizationId, kind, id],
        );
        if (!rows[0]?.has && (rows[0]?.n ?? 0) >= MAX_TAGS) {
          throw new HttpError(
            409,
            "tag_limit",
            `An organization has at most ${MAX_TAGS} of these.`,
          );
        }
        const saved = await client.query<{ updated_at: Date }>(
          `INSERT INTO tags (organization_id, kind, id, name) VALUES ($1, $2, $3, $4)
           ON CONFLICT (organization_id, kind, id)
           DO UPDATE SET name = EXCLUDED.name, updated_at = now()
           RETURNING updated_at`,
          [organizationId, kind, id, name],
        );
        await client.query("COMMIT");
        return {
          id,
          name,
          updatedAt: (
            saved.rows[0] as { updated_at: Date }
          ).updated_at.toISOString(),
        };
      } catch (err) {
        await client.query("ROLLBACK");
        throw err;
      } finally {
        client.release();
      }
    },

    async remove(organizationId: string, kind: Kind, id: string) {
      const { rowCount } = await pool.query(
        "DELETE FROM tags WHERE organization_id = $1 AND kind = $2 AND id = $3",
        [organizationId, kind, id],
      );
      return rowCount === 1;
    },

    async summary(q: UsageQuery & { groupBy: GroupBy }) {
      const params: unknown[] = [q.from, q.to];
      const where = [
        // Reaches back 30 days before `to` for the active-user windows.
        "time >= least($1::timestamptz, $2::timestamptz - interval '720 hours')",
        "time < $2::timestamptz",
      ];
      if (q.organization !== null) {
        params.push(q.organization);
        where.push(`organization_id = $${params.length}`);
      }
      if (q.groupBy === "run") {
        params.push(q.runIds);
        where.push(`run_id = ANY($${params.length})`);
      }
      const inWindow = "time >= $1::timestamptz";
      const active = (hours: number) =>
        `count(DISTINCT ${PERSON}) FILTER (WHERE end_user AND time >= $2::timestamptz - interval '${hours} hours')::int`;
      const { rows } = await pool.query<MeasureRow & { key: string }>(
        `SELECT ${GROUPS[q.groupBy]} AS key, ${aggregates(inWindow)},
           ${active(HOURS.day)} AS au_day, ${active(HOURS.week)} AS au_week,
           ${active(HOURS.month)} AS au_month
         FROM usage_events WHERE ${where.join(" AND ")}
         GROUP BY 1 HAVING count(*) FILTER (WHERE ${inWindow}) > 0
         ORDER BY sum(cost) FILTER (WHERE ${inWindow}) DESC, 1
         LIMIT ${MAX_GROUPS + 1}`,
        params,
      );
      return {
        truncated: rows.length > MAX_GROUPS,
        data: rows
          .slice(0, MAX_GROUPS)
          .map((r) => ({ key: r.key, ...measures(r) })),
      };
    },

    async series(q: UsageQuery) {
      const params: unknown[] = [q.from, q.to];
      const extra: string[] = [];
      if (q.organization !== null) {
        params.push(q.organization);
        extra.push(`organization_id = $${params.length}`);
      }
      if (q.groupBy !== null) {
        params.push(q.group);
        extra.push(`${GROUPS[q.groupBy]} = $${params.length}`);
      }
      const aligned = "days.day_end = days.day + interval '1 day'";
      // A user is active on day d for a W-day window when they were active on any of d-W+1..d,
      // so each user's active days cover W end-days each; the covered ranges are merged per
      // user, and a running sum of range starts and ends gives the count for every day at once.
      const cover = (w: number) => `
        pieces_${w} AS (
          SELECT CASE WHEN prev IS NULL OR p >= prev + interval '${w} days'
                      THEN p ELSE prev + interval '${w} days' END AS s,
                 p + interval '${w - 1} days' AS e
          FROM (SELECT p, lag(p) OVER (PARTITION BY person ORDER BY p) AS prev FROM pairs) x
        ), edges_${w} AS (
          SELECT d, sum(v) AS v FROM (
            SELECT s AS d, 1 AS v FROM pieces_${w}
            UNION ALL SELECT e + interval '1 day', -1 FROM pieces_${w}) x GROUP BY d
        ), au_${w} AS (
          SELECT g.day, sum(coalesce(edges_${w}.v, 0)) OVER (ORDER BY g.day) AS n
          FROM calendar g(day) LEFT JOIN edges_${w} ON edges_${w}.d = g.day
        )`;
      const raw = (hours: number) =>
        `count(DISTINCT person) FILTER (WHERE t >= days.day_end - interval '${hours} hours')::int`;
      // Windows end at midnight except on the last day, which is counted over raw events.
      const { rows } = await pool.query<MeasureRow & { day: string }>(
        `WITH e AS MATERIALIZED (
           SELECT time AT TIME ZONE 'UTC' AS t, ${PERSON} AS person, end_user, input_tokens,
             output_tokens, cost, latency_ms, first_token_ms
           FROM usage_events
           WHERE time >= $1::timestamptz - interval '720 hours' AND time < $2::timestamptz
             ${extra.map((c) => `AND ${c}`).join(" ")}
         ), m AS (
           SELECT date_trunc('day', t) AS d, ${aggregates("true")}
           FROM e WHERE t >= $1::timestamptz AT TIME ZONE 'UTC' GROUP BY 1
         ), pairs AS (
           SELECT DISTINCT person, date_trunc('day', t) AS p FROM e WHERE end_user
         ), calendar AS (
           SELECT generate_series(
             date_trunc('day', ($1::timestamptz - interval '720 hours') AT TIME ZONE 'UTC'),
             date_trunc('day', ($2::timestamptz AT TIME ZONE 'UTC') - interval '1 microsecond'),
             interval '1 day')
         ), days AS (
           SELECT m.d AS day, least(m.d + interval '1 day', $2::timestamptz AT TIME ZONE 'UTC') AS day_end
           FROM m
         ), ${cover(1)}, ${cover(7)}, ${cover(30)}
         SELECT to_char(days.day, 'YYYY-MM-DD') AS day, m.*,
           (CASE WHEN ${aligned} THEN au_1.n ELSE r.au_day END)::int AS au_day,
           (CASE WHEN ${aligned} THEN au_7.n ELSE r.au_week END)::int AS au_week,
           (CASE WHEN ${aligned} THEN au_30.n ELSE r.au_month END)::int AS au_month
         FROM days
         JOIN m ON m.d = days.day
         JOIN au_1 ON au_1.day = days.day
         JOIN au_7 ON au_7.day = days.day
         JOIN au_30 ON au_30.day = days.day
         LEFT JOIN LATERAL (
           SELECT ${raw(HOURS.day)} AS au_day, ${raw(HOURS.week)} AS au_week,
             ${raw(HOURS.month)} AS au_month
           FROM e WHERE NOT (${aligned}) AND end_user
             AND t < days.day_end AND t >= days.day_end - interval '${HOURS.month} hours') r ON true
         ORDER BY days.day`,
        params,
      );
      return rows.map((r) => ({ day: r.day, ...measures(r) }));
    },
  };
}

export type Analytics = ReturnType<typeof createAnalytics>;
