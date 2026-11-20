CREATE TABLE agent_runs (
  seq bigserial NOT NULL UNIQUE,
  id uuid PRIMARY KEY,
  organization_id text NOT NULL,
  application_id text NOT NULL,
  submitter_id text NOT NULL,
  model text NOT NULL,
  task text NOT NULL,
  tools text[] NOT NULL DEFAULT '{}',
  limits jsonb NOT NULL,
  status text NOT NULL CHECK (status IN ('queued', 'running', 'finished', 'cancelled')),
  attempt integer NOT NULL DEFAULT 0,
  lease_until timestamptz,
  cancel_requested boolean NOT NULL DEFAULT false,
  idempotency_key text,
  request_sha256 text,
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz,
  stop text,
  result_text text,
  tokens bigint NOT NULL DEFAULT 0,
  receipts jsonb NOT NULL DEFAULT '[]'
);

CREATE UNIQUE INDEX agent_runs_idempotency
  ON agent_runs (organization_id, submitter_id, idempotency_key)
  WHERE idempotency_key IS NOT NULL;
CREATE INDEX agent_runs_org_seq ON agent_runs (organization_id, seq DESC);
CREATE INDEX agent_runs_queued ON agent_runs (seq) WHERE status = 'queued';
CREATE INDEX agent_runs_leases ON agent_runs (lease_until) WHERE status = 'running';
