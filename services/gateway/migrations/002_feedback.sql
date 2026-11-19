-- Model feedback (bae6-contracts ADR-0009). One verdict per organization, request and submitter.

CREATE TABLE feedback (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  organization_id text NOT NULL,
  request_id text NOT NULL,
  submitter_id text NOT NULL,
  model_id text NOT NULL,
  verdict text NOT NULL CHECK (verdict IN ('positive', 'negative')),
  label text CHECK (label IN ('inaccurate', 'unsafe', 'refused', 'too_slow', 'other')),
  comment text CHECK (octet_length(comment) <= 1000),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (organization_id, request_id, submitter_id)
);
CREATE INDEX feedback_org_listing ON feedback (organization_id, id DESC);
CREATE INDEX feedback_created ON feedback (created_at);
