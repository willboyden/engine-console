// A fake `nvidia-smi` (node/tests/fakes/fake_smi.py in Node): GPUs from $FAKE_STATE_DIR/smi.json, `--query-*` CSV.
'use strict';
const fs = require('node:fs');
const path = require('node:path');

const state = JSON.parse(fs.readFileSync(path.join(process.env.FAKE_STATE_DIR, 'smi.json'), 'utf8'));
function main(args) {
  if (!state.gpus.length) { process.stdout.write('No devices were found\n'); return 6; }
  if (!args.length) {
    process.stdout.write(`| NVIDIA-SMI ${state.driver}   Driver Version: ${state.driver}   CUDA Version: ${state.cuda}  |\n`);
    return 0;
  }
  const query = args.find((a) => a.startsWith('--query-'));
  if (!query) return 2;
  const [kind, fields] = query.split('=');
  const rows = kind === '--query-gpu' ? state.gpus : (state.apps || []);
  for (const row of rows) {
    const vals = [];
    for (const f of fields.split(',')) {
      if (!(f in row)) { process.stderr.write(`Field "${f}" is not a valid field to query.\n`); return 2; }
      vals.push(String(row[f]));
    }
    process.stdout.write(vals.join(', ') + '\n');
  }
  return 0;
}
process.exitCode = main(process.argv.slice(2));
