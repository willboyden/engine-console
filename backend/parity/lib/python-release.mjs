// The Python implementation for the parity harness after the cutover: the Python hub (backend/) and node agent (node/)
// of a git ref. Default: the tag `python-final` when this checkout has it, else commit 15ac7a6, the last commit with the
// Python code, which the Rust port was verified against (backend/PORTING.md). The ref's backend/ and node/ trees are
// exported with `git archive` (no worktree is created) into backend/target/python-release/<commit>/ and get their
// virtualenvs with `uv sync --frozen`: the hub with Python 3.13, the node with Python 3.12 (the oldest interpreter a
// managed machine runs). uv's cache is used, or the network the first time. Nothing outside backend/target is written.
//
//   node backend/parity/lib/python-release.mjs [ref]      # prepare (once) and print the release root
//   EC_PYTHON_REF=<ref> node backend/parity/run.mjs        # every harness script takes the ref from EC_PYTHON_REF
//   node backend/parity/run.mjs --a python@<ref>           # or, for run.mjs, from the backend name
//   PY=$(node backend/parity/lib/python-release.mjs)/backend/.venv/bin/python   # e.g. for the golden generators
//
// A ref may also be an absolute path to a Python tree already on disk (backend/pyproject.toml and node/pyproject.toml,
// each with its .venv): it is used as it is, never synced or written to.
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const PARITY = dirname(dirname(fileURLToPath(import.meta.url)));
const WORKSPACE = dirname(PARITY);
const REPO = dirname(WORKSPACE);

/** The last commit with the Python hub and node: the code the port was verified against. */
export const LAST_PYTHON_COMMIT = '15ac7a6702d13c485bd621d0fd80f27f67ad3713';
/** The tag the release puts on that commit. */
export const PYTHON_FINAL_TAG = 'python-final';

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { encoding: 'buffer', maxBuffer: 512 * 1024 * 1024, ...opts });
  if (r.error || r.status !== 0) {
    const out = `${r.stdout ?? ''}${r.stderr ?? ''}`.slice(-2000);
    throw new Error(`${cmd} ${args.join(' ')} failed (${r.error?.message ?? r.status}): ${out}`);
  }
  return r.stdout;
}

/** The commit a ref names, or null. */
function commitOf(ref) {
  const r = spawnSync('git', ['-C', REPO, 'rev-parse', '--verify', '--quiet', `${ref}^{commit}`], { encoding: 'utf8' });
  return r.status === 0 ? r.stdout.trim() : null;
}

/** The default reference: `python-final` if this checkout has the tag, else the last Python commit. */
export function defaultRef() {
  if (process.env.EC_PYTHON_REF) return process.env.EC_PYTHON_REF;
  return commitOf(PYTHON_FINAL_TAG) ? PYTHON_FINAL_TAG : LAST_PYTHON_COMMIT;
}

/** `python` or `python@<ref>` -> the ref (or an absolute tree path); anything else -> null. */
export function refOf(impl) {
  if (impl === 'python') return defaultRef();
  if (typeof impl === 'string' && impl.startsWith('python@') && impl.length > 7) return impl.slice(7);
  return null;
}

const prepared = new Map();

/** Prepare (once) the Python implementation of `ref`. Returns absolute paths: { root, backend, node, hubPy, nodePy }. */
export function pythonRelease(ref = defaultRef()) {
  if (!prepared.has(ref)) prepared.set(ref, prepare(ref));
  return prepared.get(ref);
}

function prepare(ref) {
  const paths = (root) => ({
    root,
    backend: join(root, 'backend'),
    node: join(root, 'node'),
    hubPy: join(root, 'backend', '.venv', 'bin', 'python'),
    nodePy: join(root, 'node', '.venv', 'bin', 'python'),
  });
  if (isAbsolute(ref) && existsSync(join(ref, 'backend', 'pyproject.toml'))) {
    const out = paths(ref);
    for (const py of [out.hubPy, out.nodePy]) {
      if (!existsSync(py)) throw new Error(`${ref}: ${py} is missing (run \`uv sync --frozen\` in its backend/ and node/)`);
    }
    return out;
  }
  const sha = commitOf(ref);
  if (!sha) throw new Error(`${ref} is not a commit in this checkout (fetch the tag ${PYTHON_FINAL_TAG}, or name a commit)`);
  const out = paths(join(WORKSPACE, 'target', 'python-release', sha));
  if (existsSync(join(out.root, '.ready'))) return out;
  rmSync(out.root, { recursive: true, force: true });
  mkdirSync(out.root, { recursive: true });
  const tar = run('git', ['-C', REPO, 'archive', '--format=tar', sha, '--', 'backend', 'node']);
  run('tar', ['-x', '-C', out.root], { input: tar });
  for (const [dir, python] of [[out.node, '3.12'], [out.backend, '3.13']]) {
    if (!existsSync(join(dir, 'pyproject.toml'))) throw new Error(`${ref} has no Python package at ${dir}`);
    run('uv', ['sync', '--frozen', '--python', python], { cwd: dir, env: { UV_PYTHON_DOWNLOADS: 'never', ...process.env } });
  }
  writeFileSync(join(out.root, '.ready'), `${ref}\n`);
  return out;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    console.log(pythonRelease(process.argv[2] || defaultRef()).root);
  } catch (e) {
    console.error(`python-release: ${e.message}`);
    process.exit(1);
  }
}
