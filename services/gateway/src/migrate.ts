import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { ClientBase } from "pg";

const MIGRATIONS_DIR = join(import.meta.dirname, "../migrations");
// Arbitrary constant: serializes concurrent runners (several replicas starting).
const LOCK_ID = 7_202_001;
const LOCK_POLL_MS = 200;
const LOCK_WAIT_MS = 10 * 60_000;
const NO_TRANSACTION = "-- no-transaction";

/** Applies every unapplied `migrations/*.sql` in order, each in its own transaction. */
export async function migrate(client: ClientBase): Promise<string[]> {
  // Polling, not pg_advisory_lock: a statement blocked on the lock holds a snapshot that
  // CREATE INDEX CONCURRENTLY waits for, which would deadlock.
  for (let waited = 0; ; waited += LOCK_POLL_MS) {
    const { rows } = await client.query<{ locked: boolean }>(
      "SELECT pg_try_advisory_lock($1) AS locked",
      [LOCK_ID],
    );
    if (rows[0]?.locked) break;
    if (waited >= LOCK_WAIT_MS)
      throw new Error("another migration is still running");
    await new Promise((r) => setTimeout(r, LOCK_POLL_MS));
  }
  try {
    await client.query(
      `CREATE TABLE IF NOT EXISTS schema_migrations (
         version text PRIMARY KEY,
         applied_at timestamptz NOT NULL DEFAULT now()
       )`,
    );
    const done = new Set(
      (
        await client.query<{ version: string }>(
          "SELECT version FROM schema_migrations",
        )
      ).rows.map((r) => r.version),
    );
    const applied: string[] = [];
    const files = readdirSync(MIGRATIONS_DIR)
      .filter((f) => f.endsWith(".sql"))
      .sort();
    for (const file of files) {
      if (done.has(file)) continue;
      const sql = readFileSync(join(MIGRATIONS_DIR, file), "utf8");
      const record = "INSERT INTO schema_migrations (version) VALUES ($1)";
      if (sql.startsWith(NO_TRANSACTION)) {
        // CREATE INDEX CONCURRENTLY cannot run in a transaction or a multi-statement query.
        try {
          for (const statement of sql.split(";\n").filter((s) => s.trim())) {
            await client.query(statement);
          }
          await client.query(record, [file]);
        } catch (err) {
          throw new Error(`migration ${file} failed`, { cause: err });
        }
        applied.push(file);
        continue;
      }
      await client.query("BEGIN");
      try {
        await client.query("SET LOCAL lock_timeout = '5s'");
        await client.query(sql);
        await client.query(record, [file]);
        await client.query("COMMIT");
      } catch (err) {
        await client.query("ROLLBACK");
        throw new Error(`migration ${file} failed`, { cause: err });
      }
      applied.push(file);
    }
    return applied;
  } finally {
    await client.query("SELECT pg_advisory_unlock($1)", [LOCK_ID]);
  }
}
