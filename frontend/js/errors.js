// Human-readable text for API failures, including the hardening statuses added in contract v1.1.
import { t } from './i18n.js';

export function describeError(e) {
  if (!e) return t('common.error');
  const status = e.status, code = e.code, detail = e.message;
  if (code === 'instance_not_managed') return t('err.not_managed');
  if (code === 'secret_in_argv') return detail; // the server explains which secret and what to do instead
  if (status === 413) return t('err.too_large');
  if (status === 415) return t('err.unsupported_media');
  if (status === 421) return t('err.bad_host');
  if (status === 429) return code === 'too_many_streams' ? t('err.too_many_streams') : t('err.rate_limited');
  if (status === 422) {
    const errs = (e.problem?.errors || []).map((x) => `${(x.loc || []).filter((p) => p !== 'body').join('.')}: ${x.msg}`).filter(Boolean);
    return errs.length ? `${t('err.invalid')} ${errs.join('; ')}` : (detail || t('err.invalid'));
  }
  return detail || t('common.error');
}
