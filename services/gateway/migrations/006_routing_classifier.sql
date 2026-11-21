-- Classifier routing (bae6-contracts ADR-0014): widens the decision values. NOT VALID keeps the
-- table lock short; existing rows all match the old, narrower pattern. One table per file so
-- each lock is released at its own commit.

ALTER TABLE reservations
  DROP CONSTRAINT reservations_decision_check,
  ADD CONSTRAINT reservations_decision_check
    CHECK (decision ~ '^(default(:skipped|:classifier)?|rule:[0-9]{1,2}|classifier(:[a-z][a-z0-9_-]{0,15})?)$') NOT VALID;
