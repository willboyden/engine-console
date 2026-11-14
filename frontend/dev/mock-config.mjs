// The Config files page's two routes for the mock server: GET /config/{file} and POST /config/validate (shapes of
// backend/crates/ec-hub/src/services/config_check.rs). The templates and the env settings list are read from the
// repository itself (node/examples, the hub's SETTINGS table), so the mock cannot drift from them. Validation here is
// a rough stand-in (JSON syntax, known keys and names); the real hub runs its own loaders.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = (rel, fallback = '') => {
  try {
    return fs.readFileSync(path.join(REPO, rel), 'utf8');
  } catch {
    return fallback;
  }
};

export function envSettings() {
  const src = read('backend/crates/ec-engines/src/envfile.rs');
  const block = /pub const SETTINGS: &\[\(&str, &str\)\] = &\[([\s\S]*?)\];/.exec(src)?.[1] || '';
  const prefixed = ['host', 'port', 'allowed_hosts', 'trust_loopback', 'data_dir'];
  const noEffect = [
    'hf_cache_dir',
    'docker_bin',
    'gateway_image',
    'gateway_image_allowlist',
    'discovery_enabled',
    'discovery_ports',
    'discovery_interval_s',
    'enable_api_docs',
  ];
  return [...block.matchAll(/\("(\w+)", "(\w+)"\)/g)].map(([, name, kind]) => ({
    name,
    var: prefixed.includes(name) ? `ENGINE_CONSOLE_${name.toUpperCase()}` : name.toUpperCase(),
    kind,
    no_effect: noEffect.includes(name),
  }));
}

const HOSTS_TEMPLATE =
  '{"hosts": [\n  {"id": "local", "label": "This machine"},\n  {"id": "gpu-box-2", "label": "GPU box 2", "ssh": "ops@192.0.2.5"},\n  {"id": "edge-1", "label": "Edge 1", "ssh": "ops@192.0.2.6", "port": 2222, "enabled": false}\n]}\n';
const ENV_TEMPLATE =
  '# Engine Console hub settings (dotenv). The hub reads it once at start-up.\n# EGRESS_PROXY=http://127.0.0.1:8082\n# (an absolute path: the hub never expands ~)\n# HF_TOKEN_FILE=/path/to/engine-console/hf_token\n';

export function configFile(file) {
  if (file === 'hosts') {
    return {
      file,
      format: 'json',
      path: '/path/to/.config/engine-console/hosts.json',
      exists: true,
      served: true,
      text: '{"hosts": [\n  {"id": "local", "label": "This machine"},\n  {"id": "gpu-a", "label": "GPU box A", "ssh": "ops@192.0.2.21"}\n]}\n',
      reason: null,
      problem: null,
      max_bytes: 65536,
      templates: { hosts: HOSTS_TEMPLATE },
    };
  }
  if (file === 'env') {
    return {
      file,
      format: 'dotenv',
      path: '/path/to/.config/engine-console/env',
      exists: true,
      served: false,
      text: null,
      reason: 'secret',
      problem: null,
      max_bytes: 1048576,
      max_lines: 4000,
      templates: { env: ENV_TEMPLATE },
      fields: envSettings(),
    };
  }
  if (file === 'policy') {
    return {
      file,
      format: 'json',
      path: null,
      locations: ['/etc/ecnode/policy.json', '~/.config/ecnode/policy.json'],
      node_policy: null,
      exists: null,
      served: false,
      text: null,
      reason: 'remote',
      problem: null,
      max_bytes: 262144,
      templates: {
        monitor: read('node/examples/policy.monitor.json', '{"version": 1, "mode": "monitor"}\n'),
        manage: read('node/examples/policy.manage.json', '{"version": 1, "mode": "manage"}\n'),
      },
    };
  }
  return null;
}

const HOST_KEYS = ['id', 'label', 'ssh', 'port', 'ssh_config', 'enabled'];
const POLICY_KEYS = [
  'version',
  'mode',
  'docker',
  'engine_images',
  'gateway_image',
  'engine_network',
  'port_range',
  'discovery_ports',
  'model_roots',
  'scripts',
  'allow_pull',
  'allow_delete',
  'allow_ipc_host',
  'external_control',
  'max_shm_gib',
];

function json(text) {
  try {
    return { doc: JSON.parse(text) };
  } catch (e) {
    const pos = /position (\d+)/.exec(e.message);
    return {
      err: {
        message: `not valid JSON: ${e.message}`,
        ...(pos ? { line: text.slice(0, Number(pos[1])).split('\n').length } : {}),
      },
    };
  }
}

export function validate(file, text) {
  const errors = [],
    warnings = [];
  let outline = null;
  if (file === 'hosts') {
    const { doc, err } = json(text);
    if (err) errors.push(err);
    else if (!doc || !Array.isArray(doc.hosts) || Object.keys(doc).length !== 1)
      errors.push({ message: 'the file must be {"hosts": [ ... ]} and nothing else' });
    else {
      doc.hosts.forEach((x, i) => {
        const extra = Object.keys(x || {}).filter((k) => !HOST_KEYS.includes(k));
        if (extra.length) errors.push({ message: `hosts[${i}]: unknown key(s) ${extra.sort().join(', ')}` });
      });
      if (!doc.hosts.some((x) => x?.id === 'local'))
        warnings.push({ message: 'no host "local": the console then manages only other machines' });
      outline = {
        hosts: doc.hosts.map((x) => ({
          id: x?.id,
          label: x?.label || x?.id,
          remote: Boolean(x && x['ssh']),
          enabled: x?.enabled !== false,
        })),
      };
    }
  } else if (file === 'env') {
    const known = new Map(envSettings().map((f) => [f.var, f]));
    const set = [],
      unknown = [];
    text.split('\n').forEach((raw, i) => {
      const m = /^\s*(?:export\s+)?([^#=\s][^=]*?)\s*=/.exec(raw);
      if (!m || raw.trimStart().startsWith('#')) return;
      const f = known.get(m[1]);
      if (!f) {
        unknown.push(m[1]);
        warnings.push({ message: `${m[1]} is not a setting of the hub; it is ignored`, line: i + 1 });
      } else set.push(f.var);
    });
    outline = { set, unknown };
  } else {
    const { doc, err } = json(text);
    if (err) errors.push(err);
    else if (!doc || typeof doc !== 'object' || Array.isArray(doc))
      errors.push({ message: 'the policy must be one JSON object' });
    else {
      const bad = Object.keys(doc).find((k) => !k.startsWith('_') && !POLICY_KEYS.includes(k));
      if (bad)
        errors.push({ message: `policy: unknown key '${bad}' (allowed: ${[...POLICY_KEYS].sort().join(', ')})` });
      outline = {
        mode: doc.mode || 'monitor',
        engine_images: (doc.engine_images || []).length,
        model_roots: (doc.model_roots || []).length,
        writable_roots: (doc.model_roots || []).filter((r) => r?.writable).length,
        scripts: (doc.scripts || []).length,
        allow_pull: !!doc.allow_pull,
        allow_delete: !!doc.allow_delete,
        allow_ipc_host: !!doc.allow_ipc_host,
        external_control: doc.external_control || 'none',
      };
      warnings.push({
        message:
          "checked here without the node's own paths; run `ecnode check --policy <file>` on the machine for the final word",
      });
    }
  }
  return { file, ok: errors.length === 0, errors, warnings, outline: errors.length ? null : outline };
}
