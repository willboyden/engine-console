-- Contract v2 (the fleet): every managed instance belongs to a host and records what model it loads.
-- host_id: which machine's node runs the container ('local' = the console's own machine; every pre-fleet row).
-- model: JSON ModelRef (format + repo id / root + path / ollama name); NULL on pre-fleet rows = a Hugging Face repo_id.
-- image_req: the image the operator asked for at launch, if any (else the settings pin or the adapter's default is used).
ALTER TABLE instances ADD COLUMN host_id TEXT NOT NULL DEFAULT 'local';
ALTER TABLE instances ADD COLUMN model TEXT;
ALTER TABLE instances ADD COLUMN image_req TEXT;
CREATE INDEX instances_host ON instances(host_id, state);
