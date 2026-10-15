// Human-readable, actionable text for API failures: the hardening statuses (v1.1) and the node errors of contract v2
// (ARCHITECTURE section 10.11). Server detail is appended where it adds information (it already names the machine).
import { t } from './i18n.js';

// Advice first, then what the server said. The server's detail often starts in lower case ("a valid key is required"): once it
// follows a full stop it is a new sentence, so a plain lower-case word gets its capital (ids such as "gpu-box-2" are left alone).
const join = (a, b) => {
  if (!b || String(a).includes(b)) return a;
  const sentence = /[.!?:]$/.test(String(a).trim()) && /^[a-z]+(\s|$)/.test(String(b));
  return `${a} ${sentence ? b.charAt(0).toUpperCase() + b.slice(1) : b}`;
};

// code -> i18n key. Checked before the status so a specific problem gets its specific advice.
const BY_CODE = {
  instance_not_managed: 'err.not_managed',
  csrf_header_required: 'err.csrf',
  cross_origin: 'err.cross_origin',
  node_unavailable: 'err.node_unavailable',
  busy: 'err.busy',
  monitor_only: 'err.monitor_only',
  policy_denied: 'err.policy_denied',
  not_owned: 'err.not_owned',
  in_use: 'err.in_use',
  port_in_use: 'err.port_in_use',
  conflict: 'err.conflict',
  cancelled: 'err.cancelled',
  no_space: 'err.no_space',
  timeout: 'err.timeout',
  unsupported: 'err.unsupported',
  hash_mismatch: 'err.integrity',
  size_mismatch: 'err.integrity',
  docker_error: 'err.node_failed',
  http_error: 'err.node_failed',
  io_error: 'err.node_failed',
  unknown_host: 'err.unknown_host',
  egress_proxy_unavailable: 'err.egress',
  // codes of the hub's own services (library, scripts, launch)
  model_in_use: 'err.in_use',
  copy_not_found: 'err.not_found',
  root_not_deletable: 'err.policy_denied',
  script_busy: 'err.busy',
  unknown_script: 'err.not_found',
  unknown_action: 'err.not_found',
  unknown_job: 'err.not_found',
  invalid_arg: 'err.bad_request',
  arg_required: 'err.bad_request',
  model_missing: 'err.model_missing',
  format_unsupported: 'err.format_unsupported',
  image_not_allowed: 'err.policy_denied',
  no_free_port: 'err.port_in_use',
  preflight_failed: 'err.preflight_failed',
};
const BY_STATUS = { 400: 'err.bad_request', 401: 'err.unauthorized', 403: 'err.forbidden', 404: 'err.not_found', 409: 'err.conflict_state', 413: 'err.too_large', 415: 'err.unsupported_media',
  421: 'err.bad_host', 501: 'err.unsupported', 502: 'err.node_failed', 503: 'err.unavailable', 504: 'err.timeout', 507: 'err.no_space' };

export function describeError(e) {
  if (!e) return t('common.error');
  const status = e.status, code = e.code, detail = e.message;
  if (code === 'secret_in_argv') return detail; // the server explains which secret and what to do instead
  if (status === 0 || code === 'network') return join(t('err.network'), detail && detail !== t('err.network') ? `(${detail})` : '');
  if (status === 429) return code === 'too_many_streams' ? t('err.too_many_streams') : t('err.rate_limited');
  if (status === 422) {
    const errs = (e.problem?.errors || []).map((x) => `${(x.loc || []).filter((p) => p !== 'body').join('.')}: ${x.msg}`).filter(Boolean);
    return errs.length ? `${t('err.invalid')} ${errs.join('; ')}` : (detail || t('err.invalid'));
  }
  const key = BY_CODE[code] || BY_STATUS[status];
  if (!key) return detail || t('common.error');
  const base = t(key);
  // These are fully explained by the advice; the server's wording would only repeat it.
  if (code === 'instance_not_managed' || status === 413 || status === 415 || status === 421) return base;
  return join(base, detail);
}

/** Whether a failure is the kind where the action may or may not have happened (a node timeout): the UI must say so. */
export const isUncertain = (e) => e?.status === 504 || e?.code === 'timeout';

/** Whether the key's role is what refused the request (a viewer on an admin-only route), not a policy, a gated model or
 * the CSRF check, which share the 403 status. Screens that a viewer may open but not use show a plain note instead. */
export const isRoleRefusal = (e) => e?.status === 403 && e?.code === 'forbidden';
