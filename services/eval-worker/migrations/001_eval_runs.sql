CREATE TABLE eval_runs (
  seq bigserial UNIQUE,
  id uuid PRIMARY KEY,
  organization_id text NOT NULL,
  application_id text NOT NULL,
  submitter_id text NOT NULL,
  model text NOT NULL,
  name text NOT NULL,
  dataset_version text NOT NULL,
  cases jsonb NOT NULL,
  scorers text[] NOT NULL,
  status text NOT NULL CHECK (status IN ('queued', 'running', 'finished', 'cancelled')),
  attempt integer NOT NULL DEFAULT 0,
  lease_until timestamptz,
  cancel_requested boolean NOT NULL DEFAULT false,
  idempotency_key text,
  request_sha256 text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz,
  stop text,
  summary jsonb,
  report jsonb
);

CREATE UNIQUE INDEX eval_runs_idempotency
  ON eval_runs (organization_id, submitter_id, idempotency_key)
  WHERE idempotency_key IS NOT NULL;
CREATE INDEX eval_runs_org_seq ON eval_runs (organization_id, seq DESC);
CREATE INDEX eval_runs_queued ON eval_runs (seq) WHERE status = 'queued';
CREATE INDEX eval_runs_leases ON eval_runs (lease_until) WHERE status = 'running';
