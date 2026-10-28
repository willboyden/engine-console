-- Engine Console schema v1. Applied in filename order by services/store.py; never edit a shipped file,
-- add 000N_*.sql instead.
CREATE TABLE profiles (
  id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, engine TEXT NOT NULL, repo_id TEXT,
  params TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE downloads (
  id TEXT PRIMARY KEY, repo_id TEXT NOT NULL, revision TEXT, commit_sha TEXT, allow_patterns TEXT,
  state TEXT NOT NULL, total_bytes INTEGER NOT NULL DEFAULT 0, done_bytes INTEGER NOT NULL DEFAULT 0,
  files_total INTEGER NOT NULL DEFAULT 0, files_done INTEGER NOT NULL DEFAULT 0, current_file TEXT,
  error TEXT, error_code TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE instances (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, engine TEXT NOT NULL, repo_id TEXT NOT NULL, params TEXT NOT NULL,
  gpu_ids TEXT NOT NULL, gpu_uuids TEXT NOT NULL, port INTEGER, container_name TEXT NOT NULL, image TEXT NOT NULL,
  state TEXT NOT NULL, phase TEXT, progress_pct REAL, pinned INTEGER NOT NULL DEFAULT 0, ttl_idle_s INTEGER,
  profile_id TEXT, error TEXT, last_logs TEXT, fit TEXT, created_at REAL NOT NULL, started_at REAL,
  last_request_at REAL
);
CREATE TABLE metric_rollups (
  instance_id TEXT NOT NULL, res TEXT NOT NULL, bucket_ts INTEGER NOT NULL, key TEXT NOT NULL,
  sum REAL NOT NULL, count INTEGER NOT NULL, max REAL NOT NULL,
  PRIMARY KEY (instance_id, res, bucket_ts, key)
);
CREATE TABLE bench_runs (
  id TEXT PRIMARY KEY, instance_id TEXT NOT NULL, engine TEXT, repo_id TEXT, profile_id TEXT, params TEXT NOT NULL,
  suite TEXT NOT NULL, state TEXT NOT NULL, results TEXT NOT NULL, error TEXT, created_at REAL NOT NULL, finished_at REAL
);
CREATE TABLE conversations (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, system_prompt TEXT, instance_id TEXT,
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL, content TEXT NOT NULL, reasoning TEXT, usage TEXT, ts REAL NOT NULL
);
CREATE INDEX messages_conv ON messages(conversation_id, id);
CREATE TABLE prompts (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL, tags TEXT NOT NULL,
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE arena_matches (
  id TEXT PRIMARY KEY, prompt TEXT NOT NULL, system_prompt TEXT, blind INTEGER NOT NULL,
  model_a TEXT NOT NULL, model_b TEXT NOT NULL, response_a TEXT, response_b TEXT,
  winner TEXT, created_at REAL NOT NULL, voted_at REAL
);
CREATE TABLE arena_ratings (model TEXT PRIMARY KEY, rating REAL NOT NULL, games INTEGER NOT NULL, wins INTEGER NOT NULL,
  losses INTEGER NOT NULL, ties INTEGER NOT NULL);
CREATE TABLE usage_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, model TEXT NOT NULL, instance_id TEXT,
  prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL, latency_ms REAL NOT NULL, source TEXT NOT NULL
);
CREATE INDEX usage_ts ON usage_events(ts);
CREATE TABLE audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, actor TEXT NOT NULL, role TEXT NOT NULL,
  method TEXT NOT NULL, path TEXT NOT NULL, status INTEGER NOT NULL, params TEXT NOT NULL
);
-- Append-only: forbid UPDATE/DELETE at the database level, not just by convention.
CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit BEGIN SELECT RAISE(ABORT, 'audit is append-only'); END;
CREATE TABLE api_keys (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL, prefix TEXT NOT NULL, hash TEXT NOT NULL UNIQUE,
  created_at REAL NOT NULL, last_used_at REAL, revoked_at REAL
);
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
