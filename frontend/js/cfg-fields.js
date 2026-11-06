// What the config editor's Form view shows for each file, and where each field's (i) explanation lives in the learn
// bundle (concept keys `config.<file>.<path>`, backend learn/concepts.json; the coverage test there requires one per
// field the hub and the node know). Also the install commands of the Export card: text for the human to run, never
// run by the console. Pure data and functions, no DOM.

/** The files, in the order of the file tabs. `format` decides the parser; `name` is the download file name. */
export const CFG_FILES = [
  { id: 'hosts', format: 'json', name: 'hosts.json' },
  { id: 'env', format: 'dotenv', name: 'env' },
  { id: 'policy', format: 'json', name: 'policy.json' },
];

/** Learn concept of a file as a whole, and of one field (`path` without list indices: model_roots.path). */
export const fileConcept = (file) => `config.file.${file}`;
export const fieldConcept = (file, path) => `config.${file}.${path}`;

// Field kinds: text, int, float, bool (a checkbox), choice (options), lines (a list of strings, one per line),
// ints (a list of whole numbers, comma separated), pair (two whole numbers), nullable text (empty -> null).
export const HOST_FIELDS = [
  { key: 'id', kind: 'text', required: true },
  { key: 'label', kind: 'text' },
  { key: 'ssh', kind: 'text' },
  { key: 'port', kind: 'int' },
  { key: 'ssh_config', kind: 'bool', dflt: false },
  { key: 'enabled', kind: 'bool', dflt: true },
];

export const POLICY_FIELDS = [
  { key: 'version', kind: 'int' },
  { key: 'mode', kind: 'choice', options: ['monitor', 'manage'] },
  { key: 'docker.binary', kind: 'text' },
  { key: 'docker.context', kind: 'text', nullable: true },
  { key: 'engine_images', kind: 'lines' },
  { key: 'gateway_image', kind: 'text', nullable: true },
  { key: 'engine_network', kind: 'text' },
  { key: 'port_range', kind: 'pair' },
  { key: 'discovery_ports', kind: 'ints' },
  { key: 'allow_pull', kind: 'bool', dflt: false },
  { key: 'allow_delete', kind: 'bool', dflt: false },
  { key: 'allow_ipc_host', kind: 'bool', dflt: false },
  { key: 'external_control', kind: 'choice', options: ['none', 'stop_start'] },
  { key: 'max_shm_gib', kind: 'float' },
];
export const ROOT_FIELDS = [
  { key: 'id', kind: 'text', required: true },
  { key: 'path', kind: 'text', required: true },
  { key: 'kind', kind: 'choice', options: ['auto', 'hf-home', 'hf-hub', 'gguf', 'plain', 'ollama', 'other'] },
  { key: 'writable', kind: 'bool', dflt: false },
  { key: 'label', kind: 'text' },
];

/** The env file's settings come from the hub (GET /config/env `fields`: name, var, kind as the hub parses it). The
 * form's control for each parser kind. */
export const ENV_KIND = {
  str: 'text', path: 'text', opt_str: 'text', opt_path: 'text', port: 'int', int: 'int', u64: 'int', float: 'float', bool: 'bool',
  str_list: 'json', int_list: 'json', opt_str_list: 'json',
};
/** The settings a new env file most often needs: offered first in "Add a setting". */
export const ENV_COMMON = ['egress_proxy', 'egress_ca_bundle', 'require_egress_proxy', 'hf_token_file', 'hosts_file', 'node_policy', 'allowed_hosts', 'host', 'port'];

/** Every concept key the Form view can ask for, given the env field names (tests check each exists in the bundle). */
export function allFieldConcepts(envNames = []) {
  return [
    ...CFG_FILES.map((f) => fileConcept(f.id)),
    fieldConcept('hosts', 'hosts'), ...HOST_FIELDS.map((f) => fieldConcept('hosts', f.key)),
    ...POLICY_FIELDS.map((f) => fieldConcept('policy', f.key)), fieldConcept('policy', 'docker'), fieldConcept('policy', 'model_roots'),
    ...ROOT_FIELDS.map((f) => fieldConcept('policy', `model_roots.${f.key}`)), fieldConcept('policy', 'scripts'),
    ...envNames.map((n) => fieldConcept('env', n)),
  ];
}

/** Parse a form control's text into the JSON value for its kind: {ok, value} (value undefined = remove the key) or
 * {ok: false, why} with an i18n key. */
export function fieldValue(kind, raw, { nullable = false } = {}) {
  const s = String(raw ?? '').trim();
  if (kind === 'bool') return { ok: true, value: !!raw };
  if (s === '') return { ok: true, value: nullable ? null : undefined };
  if (kind === 'int') return /^-?\d+$/.test(s) ? { ok: true, value: Number(s) } : { ok: false, why: 'cfg.bad_int' };
  if (kind === 'float') return Number.isFinite(Number(s)) ? { ok: true, value: Number(s) } : { ok: false, why: 'cfg.bad_number' };
  if (kind === 'lines') return { ok: true, value: s.split('\n').map((x) => x.trim()).filter(Boolean) };
  if (kind === 'ints') {
    const parts = s.split(/[\s,]+/).filter(Boolean);
    return parts.every((p) => /^\d+$/.test(p)) ? { ok: true, value: parts.map(Number) } : { ok: false, why: 'cfg.bad_ints' };
  }
  if (kind === 'pair') {
    const parts = s.split(/[\s,]+/).filter(Boolean);
    return parts.length === 2 && parts.every((p) => /^\d+$/.test(p)) ? { ok: true, value: parts.map(Number) } : { ok: false, why: 'cfg.bad_pair' };
  }
  return { ok: true, value: s };
}

/** The value of a field shown in its control (the inverse of fieldValue). */
export function fieldText(kind, v) {
  if (v === undefined || v === null) return '';
  if (kind === 'lines') return Array.isArray(v) ? v.join('\n') : String(v);
  if (kind === 'ints' || kind === 'pair') return Array.isArray(v) ? v.join(', ') : String(v);
  return String(v);
}

// What a draft must not leave lying around in a download: the hub's secret-looking values (services/common.rs
// SECRET_VALUE), a URL with a password in it (a proxy), and a dotenv setting named like a secret (HF_TOKEN; not a
// *_FILE path) with a value.
const SECRET_VALUE = /\b(hf_[A-Za-z0-9]{8,}|ec_[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9._-]{8,})/;
// Anchored on '://' (not on a scheme, which a long run of letters would rescan from every position: quadratic).
const URL_PASSWORD = /:\/\/[^\s/@:]+:[^\s/@]+@/;
const SECRET_NAME = /(token|secret|password|passwd|api[_-]?key)/i;

/** True when the draft holds something that looks like a secret (the Export card then warns). `pairs`: the dotenv
 * assignments [{key, value}] when the file is the env file. Never returns or logs the value. */
export function looksSecret(text, pairs = []) {
  const s = String(text ?? '');
  if (SECRET_VALUE.test(s) || URL_PASSWORD.test(s)) return true;
  return pairs.some((p) => SECRET_NAME.test(p.key) && !/_FILE$/i.test(p.key) && String(p.value ?? '').trim() !== '');
}

/** Shell-quote one argument (single quotes; a path the user typed may hold spaces). */
export const shq = (s) => (/^[A-Za-z0-9_./~=:@%+-]+$/.test(s) ? s : `'${String(s).replace(/'/g, `'\\''`)}'`);

/** The commands that install a downloaded draft, as plain text (the page shows them with a Copy button). `path` is
 * the target the hub reported (hosts, env) or the node's policy location. Lines starting with # are comments. */
export function installCommands(file, { path, download }) {
  const src = shq(`./${download}`);
  if (file === 'hosts') {
    const dst = shq(path || '~/.config/engine-console/hosts.json');
    return [
      '# on the machine the console runs on, as the user that runs it',
      `[ -e ${dst} ] && cp -p ${dst} ${dst}.bak`,
      `install -D -m 600 ${src} ${dst}`,
      'engine-console hosts check',
      '# the console reads hosts.json only at start-up: restart it, e.g.',
      'systemctl --user restart engine-console',
    ].join('\n');
  }
  if (file === 'env') {
    const dst = shq(path || '~/.config/engine-console/env');
    return [
      '# on the machine the console runs on, as the user that runs it',
      `[ -e ${dst} ] && cp -p ${dst} ${dst}.bak`,
      `install -D -m 600 ${src} ${dst}`,
      '# the downloaded copy is plaintext and may hold a token or a password: remove it',
      `rm -- ${src}`,
      '# the console reads the env file only at start-up: restart it, e.g.',
      'systemctl --user restart engine-console',
    ].join('\n');
  }
  const dst = shq(path || '/etc/ecnode/policy.json');
  return [
    '# on the machine whose node reads this policy',
    `[ -e ${dst} ] && sudo cp -p ${dst} ${dst}.bak`,
    `sudo install -D -m 644 -o root -g root ${src} ${dst}`,
    `/usr/local/lib/ecnode/ecnode check --policy ${dst}`,
    '# the node applies it within seconds; the console sees it after the host reconnects:',
    '# press Reconnect on that host\'s page',
  ].join('\n');
}
