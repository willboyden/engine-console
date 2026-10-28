// The Python app's route table (backend/parity/routes.python.json, generated from the live app by
// gen/routes_python.py) plus the OpenAPI route FastAPI adds as a plain Starlette route. Matching follows Starlette:
// the first route (in table order) whose path regex and method both match; a path match with no method match is 405.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { PARITY } from './backend.mjs';

// Not an APIRoute, so not in the generated table: FastAPI's own `GET /api/v1/openapi.json` (Starlette also answers
// HEAD on a plain GET route).
export const EXTRA_ROUTES = [
  { method: 'GET', path: '/api/v1/openapi.json', module: 'fastapi.openapi', name: 'openapi' },
  { method: 'HEAD', path: '/api/v1/openapi.json', module: 'fastapi.openapi', name: 'openapi' },
];

function compile(template) {
  let re = '';
  let last = 0;
  for (const m of template.matchAll(/\{([A-Za-z_][A-Za-z0-9_]*)(:path)?\}/g)) {
    re += template.slice(last, m.index).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    re += m[2] ? '(.*)' : '([^/]+)';
    last = m.index + m[0].length;
  }
  re += template.slice(last).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`^${re}$`);
}

export class RouteTable {
  constructor(routes) {
    this.routes = routes.map((r) => ({ ...r, key: `${r.method} ${r.path}`, re: compile(r.path) }));
  }

  static load(file = join(PARITY, 'routes.python.json')) {
    const doc = JSON.parse(readFileSync(file, 'utf8'));
    // the openapi route sits before the API routers in Starlette's list (it is added at app construction)
    return new RouteTable([...EXTRA_ROUTES, ...doc.routes]);
  }

  /** {key, route} for a concrete request; key "405 <path>" when only the method is wrong, "unmatched <path>". */
  routeFor(method, rawPath) {
    let path = rawPath.split('?')[0];
    try { path = decodeURIComponent(path); } catch { /* keep as sent */ }
    let partial = null;
    for (const r of this.routes) {
      if (!r.re.test(path)) continue;
      if (r.method === method) return { key: r.key, route: r };
      partial ||= r;
    }
    return partial ? { key: `405 ${path}`, route: null } : { key: `unmatched ${path}`, route: null };
  }

  /** Every route a complete corpus must reach. */
  keys() {
    return this.routes.map((r) => r.key);
  }
}
