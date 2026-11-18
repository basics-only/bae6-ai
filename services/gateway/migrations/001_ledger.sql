-- Metering ledger (bae6-contracts ADR-0004). Money is numeric; nothing here is a float.

CREATE FUNCTION reject_change() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION '% is append-only', TG_TABLE_NAME;
END $$;

CREATE TABLE price_books (
  version text PRIMARY KEY CHECK (version ~ '^[A-Za-z0-9._-]{1,64}$'),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE prices (
  price_book_version text NOT NULL REFERENCES price_books,
  model_id text NOT NULL,
  input_per_mtok numeric NOT NULL CHECK (input_per_mtok >= 0),
  output_per_mtok numeric NOT NULL CHECK (output_per_mtok >= 0),
  PRIMARY KEY (price_book_version, model_id)
);
CREATE TRIGGER prices_immutable BEFORE UPDATE OR DELETE ON prices
  FOR EACH ROW EXECUTE FUNCTION reject_change();

CREATE TABLE budgets (
  organization_id text NOT NULL CHECK (organization_id ~ '^[a-z0-9][a-z0-9-]{0,62}$'),
  period text NOT NULL CHECK (period ~ '^[0-9]{4}-[0-9]{2}$'),
  limit_amount numeric NOT NULL CHECK (limit_amount >= 0),
  spent numeric NOT NULL DEFAULT 0 CHECK (spent >= 0),
  reserved numeric NOT NULL DEFAULT 0 CHECK (reserved >= 0),
  PRIMARY KEY (organization_id, period)
);

CREATE TABLE reservations (
  request_id text PRIMARY KEY,
  organization_id text NOT NULL,
  period text NOT NULL,
  application_id text NOT NULL,
  subject_id text NOT NULL,
  provider text NOT NULL,
  model_id text NOT NULL,
  price_book_version text NOT NULL,
  estimate numeric NOT NULL CHECK (estimate >= 0),
  lease_expires timestamptz NOT NULL,
  FOREIGN KEY (organization_id, period) REFERENCES budgets,
  FOREIGN KEY (price_book_version, model_id) REFERENCES prices
);
CREATE INDEX reservations_lease ON reservations (lease_expires);

CREATE TABLE usage_events (
  request_id text PRIMARY KEY,
  organization_id text NOT NULL,
  application_id text NOT NULL,
  subject_id text NOT NULL,
  provider text NOT NULL,
  model_id text NOT NULL,
  price_book_version text NOT NULL,
  -- Null when the provider did not report usage (outcome uncertain).
  input_tokens integer CHECK (input_tokens >= 0),
  output_tokens integer CHECK (output_tokens >= 0),
  cost numeric NOT NULL CHECK (cost >= 0),
  outcome text NOT NULL CHECK (outcome IN ('measured', 'uncertain', 'failed')),
  time timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX usage_events_tenant ON usage_events (organization_id, application_id, time);
CREATE TRIGGER usage_events_append_only BEFORE UPDATE OR DELETE ON usage_events
  FOR EACH ROW EXECUTE FUNCTION reject_change();

CREATE TABLE idempotency_keys (
  organization_id text NOT NULL,
  application_id text NOT NULL,
  subject_id text NOT NULL,
  key text NOT NULL,
  request_hash text NOT NULL,
  request_id text NOT NULL UNIQUE,
  response_status integer,
  response_body jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (organization_id, application_id, subject_id, key)
);
CREATE INDEX idempotency_keys_created ON idempotency_keys (created_at);
