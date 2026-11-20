-- Model routing (bae6-contracts ADR-0014). Additive and nullable: requests that name a model
-- directly leave every column null, and replicas on the previous version keep working.

ALTER TABLE idempotency_keys ADD COLUMN model_id text;

ALTER TABLE reservations
  ADD COLUMN route_id text,
  ADD COLUMN decision text;

ALTER TABLE usage_events
  ADD COLUMN route_id text,
  ADD COLUMN decision text;

-- NOT VALID: existing rows are all NULL, so skip the validating scan under the table lock.
ALTER TABLE idempotency_keys
  ADD CONSTRAINT idempotency_keys_model_id_check
    CHECK (model_id ~ '^[a-z0-9][a-z0-9._-]{0,63}$') NOT VALID;

ALTER TABLE reservations
  ADD CONSTRAINT reservations_route_id_check
    CHECK (route_id ~ '^[a-z0-9][a-z0-9._-]{0,63}$') NOT VALID,
  ADD CONSTRAINT reservations_decision_check
    CHECK (decision ~ '^(default|default:skipped|rule:[0-9]{1,2})$') NOT VALID;

ALTER TABLE usage_events
  ADD CONSTRAINT usage_events_route_id_check
    CHECK (route_id ~ '^[a-z0-9][a-z0-9._-]{0,63}$') NOT VALID,
  ADD CONSTRAINT usage_events_decision_check
    CHECK (decision ~ '^(default|default:skipped|rule:[0-9]{1,2})$') NOT VALID;
