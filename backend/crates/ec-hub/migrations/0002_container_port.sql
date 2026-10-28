-- Engine container port, needed to describe the internal endpoint (http://<alias>:<port>).
ALTER TABLE instances ADD COLUMN container_port INTEGER;
