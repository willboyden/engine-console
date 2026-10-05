# ADR-0015: Remote nodes over ssh, with the hub's own key restricted to the node command

Status: accepted (contract v2, `docs/ARCHITECTURE.md` section 10.1, decision 15); implemented in
`backend/src/engine_console/services/hosts.py`, `cli.py` and `node/install.sh`.
**Fakes only:** no remote login has been made yet. The ssh command line the hub builds was checked against the real
`ssh` client with `ssh -G` (an offline evaluation that proves each option is known and takes effect), and the whole
link logic runs against an in-process fake node. The restricted `authorized_keys` line has not been installed on a real
machine.

## Context

Remote nodes need a transport that opens no new listening service. The machines are on a LAN, and the operator already
reaches them with ssh. The hub must start the transport unattended. Two things can go wrong: the hub's key is stolen,
and ssh quietly uses credentials or configuration the operator did not intend.

## Decision

- **A key of its own.** `engine-console hosts init-key --from <hub addresses>` creates an ed25519 key
  (`~/.config/engine-console/ssh/id_ed25519`, mode 0600, no passphrase because the hub uses it unattended) and prints the
  one line to add to the login user's authorized keys on each machine:
  `restrict,from="<hub addresses>",command="/usr/local/lib/ecnode/ecnode --stdio" ssh-ed25519 AAAA... engine-console-hub`.
  `restrict` turns off pty allocation, port, agent and X11 forwarding and user rc files; `command` makes sshd run the
  node whatever the client asked for; `from` pins the key to the hub's addresses. The public key's own comment is
  dropped so nothing in it can add options. An existing key is never replaced.
- **A command line that is the whole truth.** `ssh -T -e none -F none -o BatchMode=yes -o StrictHostKeyChecking=yes
  -o UpdateHostKeys=no -o VerifyHostKeyDNS=no -o IdentityAgent=none -o PKCS11Provider=none -o PasswordAuthentication=no
  -o KbdInteractiveAuthentication=no -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no -o ForwardX11=no
  -o ClearAllForwardings=yes -o PermitLocalCommand=no -o ConnectTimeout=5 -o ServerAliveInterval=5
  -o ServerAliveCountMax=3 -o IdentityFile=<hub key> -o IdentitiesOnly=yes [-p port] -- <ssh> <remote command>`, with
  an environment of only `HOME`, `PATH`, `LANG` and `USER`. `-F none` is left out only for a host that sets
  `ssh_config: true`. known_hosts is read and never written: an unknown or changed host key is an error that tells the
  operator to connect once by hand.
- **Never fall back to other identities.** The hub does not start ssh while its key file is missing (the link says to
  run `hosts init-key`), and passes the key as `-o IdentityFile=` rather than `-i`. With `-i` and a missing file ssh only
  warns and then offers the user's default identity files, even with `IdentitiesOnly=yes` (observed with `ssh -G`),
  which would log in with an unrestricted key. `-o IdentityFile=` fails closed even if the file disappears after the
  check. Key paths with whitespace, quotes or `%` are refused.
- **Say when the restriction is missing.** A node reports `forced_cmd` (sshd set `SSH_ORIGINAL_COMMAND`). When a remote
  node says it is false, the link carries the note "reached with an UNRESTRICTED ssh key".
- **Failures in words.** Login refused, host key unknown or changed, name not resolved, no route, node not installed
  (exit status 127) each become a sentence with the fix; failures only a person can fix are retried every 300 s.

## Consequences

- Nothing new listens on either side. A stolen hub key can start a node, from the hub's address, and nothing else; the
  node then obeys its own policy (ADR-0016).
- The key has no passphrase and lives on the hub's disk. Anyone who can read it as the hub's user can start nodes on
  every machine that authorized it, from the hub's address, or from anywhere if it was installed with `--from '*'` (the
  CLI prints a warning for that).
- A person must add the line on every machine. The hub cannot do it and does not try.
- `-F none` ignores the operator's ssh configuration (ProxyJump, Port, Hostname aliases). `ssh_config: true` brings it
  back for one host, and the configuration may then add identity files, which is exactly the case the unrestricted-key
  note exists for.
- The destination pattern has no `:`, so a bare IPv6 address is refused; use a DNS name or an ssh_config alias.
- No connection multiplexing: a reconnect pays a full handshake. The link is long-lived, so this is rare.
- The remote command is fixed by the `REMOTE_NODE_COMMAND` setting (default `/usr/local/lib/ecnode/ecnode --stdio`) and
  must equal the `command=` of the installed line.
- The node runs as the login user. If that user is in the `docker` group of a rootful daemon, the user is root-equivalent
  on that machine: the policy limits what the hub asks for, not what a bug in the node could do.

## Alternatives considered

- **The operator's default keys or an ssh agent.** Unrestricted keys that give a shell: a hub compromise would be a login
  on every machine.
- **A passphrase-protected key with an agent.** The hub could not restart unattended.
- **ssh certificates from a CA.** Better revocation, but more infrastructure than a few machines justify.
- **A TLS agent port with client certificates.** A new listening service on each machine plus a PKI.
- **Password authentication.** Not usable unattended and weaker.
- **Trust the operator's ssh configuration by default.** ProxyCommand, LocalCommand and extra identities become part of
  what the console runs; `-F none` plus an explicit option list makes the command line the complete truth.
