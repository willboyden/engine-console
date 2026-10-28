// A fake `nvidia-smi` (port of node/tests/fakes/fake_smi.py): GPUs and compute apps from $FAKE_SMI_STATE (JSON);
// `--query-*` CSV like the real one. Each call is appended to $FAKE_SMI_STATE.log as {"args", "env_names"}.
'use strict';
const fs = require('node:fs');

function main(child) {
  const state = JSON.parse(fs.readFileSync(process.env.FAKE_SMI_STATE, 'utf8'));
  const args = process.argv.slice(2);
  fs.appendFileSync(process.env.FAKE_SMI_STATE + '.log', JSON.stringify({ args, env_names: Object.keys(child).sort() }) + '\n');
  if (!state.gpus.length) { process.stdout.write('No devices were found\n'); return 6; }
  if (!args.length) {
    process.stdout.write(`| NVIDIA-SMI ${state.driver}   Driver Version: ${state.driver}   CUDA Version: ${state.cuda}  |\n`);
    return 0;
  }
  const query = args.find((a) => a.startsWith('--query-'));
  if (!query) return 2;
  const [kind, fields] = [query.split('=')[0], query.slice(query.indexOf('=') + 1)];
  const rows = kind === '--query-gpu' ? state.gpus : (state.apps || []);
  for (const row of rows) {
    const vals = [];
    for (const field of fields.split(',')) {
      if (!(field in row)) { process.stderr.write(`Field "${field}" is not a valid field to query.\n`); return 2; }
      vals.push(String(row[field]));
    }
    process.stdout.write(vals.join(', ') + '\n');
  }
  return 0;
}

module.exports = { main };
