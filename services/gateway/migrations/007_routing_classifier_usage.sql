-- Same widening as 006, for usage_events.

ALTER TABLE usage_events
  DROP CONSTRAINT usage_events_decision_check,
  ADD CONSTRAINT usage_events_decision_check
    CHECK (decision ~ '^(default(:skipped|:classifier)?|rule:[0-9]{1,2}|classifier(:[a-z][a-z0-9_-]{0,15})?)$') NOT VALID;
