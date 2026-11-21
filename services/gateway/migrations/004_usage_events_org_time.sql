-- no-transaction
DROP INDEX CONCURRENTLY IF EXISTS usage_events_org_time;
CREATE INDEX CONCURRENTLY usage_events_org_time ON usage_events (organization_id, time);
