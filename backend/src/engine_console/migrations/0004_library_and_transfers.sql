-- Contract v2 (docs/ARCHITECTURE.md 10.5 and 10.6). Applied in filename order by services/store.py; never edit a shipped
-- file, add 000N_*.sql instead.
--
-- library_snapshots: the last good `models.list` of every host, so a machine that is down still shows its last-known
-- models. roots_json/facts_json hold what is needed to rebuild a host's copies without its node (model roots, delete
-- policy, addresses, mounts); scans_json holds the per-root scan status the node reported.
CREATE TABLE library_snapshots (
  host_id TEXT PRIMARY KEY, generation INTEGER, roots_json TEXT NOT NULL, entries_json TEXT NOT NULL,
  scans_json TEXT NOT NULL DEFAULT '[]', facts_json TEXT NOT NULL DEFAULT '{}', updated_at REAL NOT NULL
);
-- Downloads and copies target one machine and one of its model roots. kind: 'hf' (Hugging Face -> hub -> node) or
-- 'copy' (node -> hub -> node). done_json lists the destination paths already finished by this row (files that are
-- written under their final name, e.g. in a gguf root, cannot be told from an older file of the same size);
-- current_dest is the destination of the file in flight, whose `.incomplete` partial a cancel may remove.
ALTER TABLE downloads ADD COLUMN host_id TEXT NOT NULL DEFAULT 'local';
ALTER TABLE downloads ADD COLUMN root_id TEXT;
ALTER TABLE downloads ADD COLUMN kind TEXT NOT NULL DEFAULT 'hf';
ALTER TABLE downloads ADD COLUMN source_copy_id TEXT;
ALTER TABLE downloads ADD COLUMN done_json TEXT;
ALTER TABLE downloads ADD COLUMN current_dest TEXT;
