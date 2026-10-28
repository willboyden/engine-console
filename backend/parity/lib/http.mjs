// A raw HTTP/1.1 client on node:http: full control of Host, Origin, Content-Length vs chunked bodies, HEAD and
// unusual methods (fetch forbids some of these headers). Never follows redirects.
import { request } from 'node:http';

/**
 * @param {object} o
 * @param {number} o.port
 * @param {string} o.method
 * @param {string} o.path   path + query, already encoded
 * @param {Record<string,string>} o.headers
 * @param {Buffer|null} o.body
 * @param {Buffer[]|null} o.chunks  send these as a chunked body (no content-length)
 * @param {number} o.timeoutMs
 */
export function send({ port, method, path, headers = {}, body = null, chunks = null, timeoutMs = 15000 }) {
  return new Promise((resolve) => {
    const h = { ...headers };
    if (body !== null && !Object.keys(h).some((k) => k.toLowerCase() === 'content-length')) {
      h['Content-Length'] = String(body.length);
    }
    if (!Object.keys(h).some((k) => k.toLowerCase() === 'host')) h.Host = `127.0.0.1:${port}`;
    const req = request({ host: '127.0.0.1', port, method, path, headers: h, setHost: false });
    let done = false;
    const finish = (r) => { if (!done) { done = true; clearTimeout(timer); resolve(r); } };
    const timer = setTimeout(() => {
      req.destroy();
      finish({ status: 0, headers: {}, body: Buffer.alloc(0), error: `timeout after ${timeoutMs} ms` });
    }, timeoutMs);
    req.on('response', (res) => {
      const parts = [];
      res.on('data', (d) => parts.push(d));
      res.on('end', () => finish({ status: res.statusCode, headers: lowerHeaders(res.headers), body: Buffer.concat(parts) }));
      res.on('error', (e) => finish({ status: res.statusCode, headers: lowerHeaders(res.headers), body: Buffer.concat(parts), error: e.message }));
    });
    req.on('error', (e) => finish({ status: 0, headers: {}, body: Buffer.alloc(0), error: e.message }));
    // A server may answer (413 by Content-Length, 401...) and close before a large body is fully written: the
    // late EPIPE/ECONNRESET on the socket is transport noise, not a result (the response is what is compared).
    req.on('socket', (s) => { if (!s.ecQuiet) { s.ecQuiet = true; s.on('error', () => {}); } });
    if (chunks) {
      for (const c of chunks) req.write(c);
      req.end();
    } else {
      req.end(body ?? undefined);
    }
  });
}

function lowerHeaders(h) {
  const out = {};
  for (const [k, v] of Object.entries(h)) out[k.toLowerCase()] = Array.isArray(v) ? v.join(', ') : String(v);
  return out;
}
