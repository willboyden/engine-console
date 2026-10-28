// A fake `ip -j addr` (node/tests/fakes/fake_ip.py in Node): documentation-range addresses only.
'use strict';
const DATA = [
  { ifname: 'lo', addr_info: [{ family: 'inet', local: '127.0.0.1', prefixlen: 8 }] },
  { ifname: 'eth0', addr_info: [{ family: 'inet', local: '192.0.2.5', prefixlen: 24 },
    { family: 'inet6', local: 'fe80::1', prefixlen: 64 }, { family: 'inet6', local: '2001:db8::5', prefixlen: 64 }] },
];
const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== '-j' || args[1] !== 'addr') process.exitCode = 2;
else process.stdout.write(JSON.stringify(DATA) + '\n');
