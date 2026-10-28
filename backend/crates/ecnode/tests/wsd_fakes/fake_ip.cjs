// A fake `ip -j addr` (port of node/tests/fakes/fake_ip.py): interfaces from $FAKE_IP_STATE (JSON) or a default.
'use strict';
const fs = require('node:fs');

const DEFAULT = [
  { ifname: 'lo', addr_info: [{ family: 'inet', local: '127.0.0.1', prefixlen: 8 }] },
  { ifname: 'eth0', addr_info: [{ family: 'inet', local: '10.0.0.5', prefixlen: 24 },
    { family: 'inet6', local: 'fe80::1', prefixlen: 64 }, { family: 'inet6', local: 'fd00::5', prefixlen: 64 }] },
  { ifname: 'docker0', addr_info: [{ family: 'inet', local: '172.17.0.1', prefixlen: 16 }] },
];

function main() {
  const args = process.argv.slice(2);
  if (args.length !== 2 || args[0] !== '-j' || args[1] !== 'addr') return 2;
  const p = process.env.FAKE_IP_STATE;
  const data = p && fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : DEFAULT;
  process.stdout.write(JSON.stringify(data) + '\n');
  return 0;
}

module.exports = { main };
