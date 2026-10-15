-- Review M7: rollups are keyed by node-chosen names, so they need an owner (host_id, NULL on rows from before this
-- migration: those are only dropped by age) and an index on the bucket time so the retention delete does not scan.
ALTER TABLE metric_rollups ADD COLUMN host_id TEXT;
CREATE INDEX idx_metric_rollups_age ON metric_rollups(res, bucket_ts);
