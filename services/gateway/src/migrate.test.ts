import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { test } from "node:test";
import pg from "pg";
import { migrate } from "./migrate.js";

const url = process.env.BAE6_DATABASE_URL;
const skip = url ? false : "BAE6_DATABASE_URL not set";

test(
  "a concurrent-index migration leaves the session's statement timeout as it was",
  { skip },
  async () => {
    const admin = new pg.Client({ connectionString: url });
    await admin.connect();
    const name = `migrate_${randomUUID().replaceAll("-", "").slice(0, 12)}`;
    await admin.query(`CREATE DATABASE ${name}`);
    const target = new URL(url as string);
    target.pathname = `/${name}`;
    const client = new pg.Client({
      connectionString: target.toString(),
      statement_timeout: 10_000,
    });
    try {
      await client.connect();
      const applied = await migrate(client);
      assert.ok(applied.includes("004_usage_events_org_time.sql"));
      const { rows } = await client.query("SHOW statement_timeout");
      assert.equal(rows[0].statement_timeout, "10s");
    } finally {
      await client.end();
      await admin.query(`DROP DATABASE ${name}`);
      await admin.end();
    }
  },
);
