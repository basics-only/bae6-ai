-- Usage analytics and task tags (bae6-contracts ADR-0011). Additive only: old replicas keep working.

ALTER TABLE reservations
  ADD COLUMN task_type text,
  ADD COLUMN domain text,
  ADD COLUMN run_id text,
  ADD COLUMN end_user boolean NOT NULL DEFAULT false;

ALTER TABLE usage_events
  ADD COLUMN task_type text,
  ADD COLUMN domain text,
  ADD COLUMN run_id text,
  -- Events from before this migration have no end-user flag and count as no active user.
  ADD COLUMN end_user boolean NOT NULL DEFAULT false,
  ADD COLUMN latency_ms integer CHECK (latency_ms >= 0),
  ADD COLUMN first_token_ms integer CHECK (first_token_ms >= 0);

CREATE INDEX usage_events_org_time ON usage_events (organization_id, time);

CREATE TABLE tags (
  organization_id text NOT NULL CHECK (organization_id ~ '^[a-z0-9][a-z0-9-]{0,62}$'),
  kind text NOT NULL CHECK (kind IN ('task_type', 'domain')),
  id text NOT NULL CHECK (id ~ '^[a-z0-9][a-z0-9-]{0,47}$' AND id <> 'untagged'),
  name text NOT NULL CHECK (char_length(name) BETWEEN 1 AND 64),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (organization_id, kind, id)
);
