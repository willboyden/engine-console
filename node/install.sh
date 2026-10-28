#!/usr/bin/env bash
# install.sh - install (or remove) the Engine Console node agent `ecnode` on THIS machine.
#
#   sudo node/install.sh --binary FILE   install the prebuilt ecnode binary FILE (a release asset
#                                        ecnode-<version>-<arch>, or `make node-dist` output) as
#                                        /usr/local/lib/ecnode/ecnode (root-owned). FILE.sha256 must sit next to it.
#                                        Writes /etc/ecnode/policy.json (MONITOR mode) only if it does not exist
#                                        yet, then runs `ecnode check` as the user who called sudo.
#   sudo node/install.sh --uninstall     remove /usr/local/lib/ecnode. /etc/ecnode (your policy) is kept.
#
# Safe to re-run: the whole /usr/local/lib/ecnode is replaced with one rename, an existing policy is never touched.
# Nothing here opens a port, creates a user, edits ssh or sudo configuration, or starts a service. It never changes
# the owner or mode of /usr/local/lib itself. No Python, compiler or network is needed on the machine.
# Needs: bash, coreutils (sha256sum, od, install, mktemp, stat).
# Going back to the Python node: run node/install.sh of the tag python-final (see node/README.md, "Rollback").
set -euo pipefail

# A fixed PATH and umask: nothing in the caller's environment picks the programs this root script runs, and what it
# creates is root-owned and world-readable (directories 0755, the binary 0755, the policy 0644) whatever umask sudo
# passed on.
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH
unset CDPATH
umask 022

PREFIX=/usr/local/lib/ecnode
BIN="$PREFIX/ecnode"
POLICY_DIR=/etc/ecnode
POLICY="$POLICY_DIR/policy.json"
MAX_BYTES=$((256 * 1024 * 1024))      # a sanity cap: the release binary is a few MiB

# Resolve our own location before anything changes directory.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { printf 'install.sh: ERROR: %s\n' "$*" >&2; exit 1; }
say() { printf 'install.sh: %s\n' "$*"; }

usage() {
    sed -n '2,15p' "$SELF_DIR/install.sh"
    return 0
}

MODE=""
SRC_BIN=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --binary)
            [ "$#" -ge 2 ] || die "--binary needs a file"
            [ -z "$MODE" ] || die "use --binary or --uninstall, not both"
            MODE=install
            SRC_BIN="$2"
            shift 2
            ;;
        --binary=*)
            [ -z "$MODE" ] || die "use --binary or --uninstall, not both"
            MODE=install
            SRC_BIN="${1#--binary=}"
            shift
            ;;
        --uninstall)
            [ -z "$MODE" ] || die "use --binary or --uninstall, not both"
            MODE=uninstall
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option '$1' (use --binary FILE, --uninstall or --help)" ;;
    esac
done
if [ -z "$MODE" ]; then
    die "nothing to do: install a release binary with --binary ecnode-<version>-<arch> (node/README.md). The Python node is no longer part of this checkout; to go back to it, run node/install.sh of the tag python-final."
fi

if [ "$(id -u)" -ne 0 ]; then
    die "run me with sudo: I write to $PREFIX and $POLICY_DIR"
fi

# The user who called sudo: the node runs as that user (locally or through the hub's ssh key).
INVOKER="${SUDO_USER:-}"
if [ "$INVOKER" = "root" ]; then
    INVOKER=""
fi

# Run a command as the invoking user when there is one, else as root (with a warning printed once).
as_user() {
    if [ -n "$INVOKER" ]; then
        sudo -u "$INVOKER" -H -- "$@"
    else
        "$@"
    fi
}

# ── uninstall ────────────────────────────────────────────────────────────────────────────────────────────
if [ "$MODE" = "uninstall" ]; then
    if [ -e "$PREFIX" ] || [ -L "$PREFIX" ]; then
        rm -rf -- "$PREFIX" || die "could not remove $PREFIX"
        say "removed $PREFIX"
    else
        say "$PREFIX is not installed; nothing to remove"
    fi
    if [ -d "$POLICY_DIR" ]; then
        say "kept $POLICY_DIR (your policy). Remove it yourself if you no longer want it: sudo rm -r $POLICY_DIR"
    fi
    exit 0
fi

# ── checks on the file we were given ─────────────────────────────────────────────────────────────────────
# This machine's ELF e_machine number: the binary must be built for it.
case "$(uname -m)" in
    x86_64 | amd64) WANT_MACHINE=62; ARCH=x86_64 ;;
    aarch64 | arm64) WANT_MACHINE=183; ARCH=aarch64 ;;
    *) die "this machine is $(uname -m); ecnode is built for x86_64 and aarch64 only" ;;
esac

[ -n "$SRC_BIN" ] || die "--binary needs a file"
if [ -L "$SRC_BIN" ]; then
    die "$SRC_BIN is a symlink; name the file itself"
fi
[ -f "$SRC_BIN" ] || die "$SRC_BIN is not a regular file"
SUM_FILE="$SRC_BIN.sha256"
if [ -L "$SUM_FILE" ]; then
    die "$SUM_FILE is a symlink; name the file itself"
fi
[ -f "$SUM_FILE" ] || die "$SUM_FILE not found: put the release's .sha256 file next to the binary"
size="$(stat -c '%s' -- "$SRC_BIN")" || die "could not check $SRC_BIN"
if [ "$size" -le 0 ] || [ "$size" -gt "$MAX_BYTES" ]; then
    die "$SRC_BIN is $size bytes; that is not an ecnode binary"
fi
# The first field of the .sha256 file (sha256sum's own format: "<64 hex>  <name>"). The name in it is not used:
# the hash is compared with the file named on the command line, so the .sha256 file can never point elsewhere.
WANT_SUM="$(head -c 4096 -- "$SUM_FILE" | head -n 1 | cut -d ' ' -f 1 | tr 'A-F' 'a-f')"
case "$WANT_SUM" in
    *[!0-9a-f]* | "") die "$SUM_FILE does not start with a sha256 hex digest" ;;
esac
[ "${#WANT_SUM}" -eq 64 ] || die "$SUM_FILE does not start with a sha256 hex digest"

# ── stage the new prefix next to the old one (same filesystem, so the swap is one rename) ───────────────────
# The parent (/usr/local/lib) belongs to the system: never change its owner or mode (install -d would, even on an
# existing directory). Create it only if it is missing, and refuse one that anyone but root can write to, because
# the staging directory and the swap below happen in it.
PARENT="$(dirname "$PREFIX")"
if [ ! -d "$PARENT" ]; then
    mkdir -p -- "$PARENT" || die "could not create $PARENT"       # 0755 at every level: umask 022
fi
parent_owner="$(stat -L -c '%u' "$PARENT")" || die "could not check $PARENT"
parent_mode="$(stat -L -c '%a' "$PARENT")" || die "could not check $PARENT"
if [ "$parent_owner" != "0" ] || (( 8#$parent_mode & 8#022 )); then
    die "$PARENT must be owned and writable by root only (uid $parent_owner, mode $parent_mode); fix that first"
fi
STAGE="$(mktemp -d "$PARENT/.ecnode-new.XXXXXX")" || die "could not create a staging directory in $PARENT"
OLD=""
# shellcheck disable=SC2329  # called by the EXIT trap
cleanup() {
    if [ -n "${STAGE:-}" ] && [ -d "$STAGE" ]; then
        rm -rf -- "$STAGE"
    fi
    return 0
}
trap cleanup EXIT

# Copy first, then check the copy: the checks and the install see the same bytes, whoever can write the original.
install -o root -g root -m 0755 -- "$SRC_BIN" "$STAGE/ecnode" || die "could not copy $SRC_BIN"
GOT_SUM="$(sha256sum -- "$STAGE/ecnode" | cut -d ' ' -f 1)" || die "could not hash $SRC_BIN"
if [ "$GOT_SUM" != "$WANT_SUM" ]; then
    die "$SRC_BIN does not match $SUM_FILE (sha256 $GOT_SUM, expected $WANT_SUM); nothing was installed"
fi
# ELF header: magic, 64-bit (EI_CLASS 2), little-endian (EI_DATA 1), an executable or PIE (e_type 2 or 3) for this
# machine (e_machine, bytes 18-19).
# od prints 16 bytes per line: join the lines (-v: never collapse repeated lines into '*').
hdr_text="$(od -A n -v -t u1 -N 20 -- "$STAGE/ecnode" | tr '\n' ' ')" || die "could not read $SRC_BIN"
read -r -a hdr <<<"$hdr_text"
if [ "${#hdr[@]}" -ne 20 ] || [ "${hdr[0]}" -ne 127 ] || [ "${hdr[1]}" -ne 69 ] || [ "${hdr[2]}" -ne 76 ] \
    || [ "${hdr[3]}" -ne 70 ]; then
    die "$SRC_BIN is not an ELF executable"
fi
if [ "${hdr[4]}" -ne 2 ] || [ "${hdr[5]}" -ne 1 ]; then
    die "$SRC_BIN is not a 64-bit little-endian ELF file"
fi
e_type=$(( hdr[16] + 256 * hdr[17] ))
e_machine=$(( hdr[18] + 256 * hdr[19] ))
if [ "$e_type" -ne 2 ] && [ "$e_type" -ne 3 ]; then
    die "$SRC_BIN is an ELF file but not an executable"
fi
if [ "$e_machine" -ne "$WANT_MACHINE" ]; then
    die "$SRC_BIN is built for ELF machine $e_machine, but this machine is $ARCH ($WANT_MACHINE): use the $ARCH binary"
fi
chown root:root "$STAGE" || die "could not chown $STAGE"
chmod 0755 "$STAGE" || die "could not chmod $STAGE"

if [ -z "$INVOKER" ]; then
    say "warning: not called through sudo by a normal user; the policy template and the check run as root"
fi
NEW_VERSION="$("$STAGE/ecnode" version)" \
    || die "$SRC_BIN does not run on this machine (ecnode version failed); nothing was installed"
case "$NEW_VERSION" in
    "ecnode "*) ;;
    *) die "$SRC_BIN is not ecnode (its version command printed something else); nothing was installed" ;;
esac
say "installing $NEW_VERSION ($ARCH, sha256 $GOT_SUM)"

# ── swap it in ───────────────────────────────────────────────────────────────────────────────────────────
# The whole prefix moves, so whatever an older release put there (the Python node's lib/ and launcher) goes with
# the old prefix.
if [ -e "$PREFIX" ] || [ -L "$PREFIX" ]; then
    OLD="$PARENT/.ecnode-old.$$"
    rm -rf -- "$OLD"
    mv -- "$PREFIX" "$OLD" || die "could not move the old $PREFIX aside"
fi
if ! mv -- "$STAGE" "$PREFIX"; then
    if [ -n "$OLD" ]; then
        mv -- "$OLD" "$PREFIX" || true
    fi
    die "could not move the new copy into $PREFIX (the old one was put back)"
fi
STAGE=""
if [ -n "$OLD" ]; then
    rm -rf -- "$OLD" || say "warning: could not remove the old copy $OLD"
fi

VERSION_LINE="$("$BIN" version)" || die "the installed binary does not run: $BIN version failed"
say "installed $VERSION_LINE in $PREFIX ($BIN, sha256 $GOT_SUM)"

# ── policy: written once, never overwritten ─────────────────────────────────────────────────────────────
if [ -e "$POLICY" ] || [ -L "$POLICY" ]; then
    say "kept the existing policy $POLICY (not changed)"
else
    install -d -o root -g root -m 0755 "$POLICY_DIR" || die "could not create $POLICY_DIR"
    TMP_POLICY="$(mktemp "$POLICY_DIR/.policy.XXXXXX")" || die "could not create a temporary file in $POLICY_DIR"
    # generated as the calling user, so the detected model directories and docker context are theirs; the node
    # validates the template against its own schema before printing it (exit 1, nothing printed, if it does not load)
    if ! as_user "$BIN" policy-template > "$TMP_POLICY"; then
        rm -f -- "$TMP_POLICY"
        die "ecnode policy-template failed; no policy was written"
    fi
    chown root:root "$TMP_POLICY" || { rm -f -- "$TMP_POLICY"; die "could not chown the policy"; }
    chmod 0644 "$TMP_POLICY" || { rm -f -- "$TMP_POLICY"; die "could not chmod the policy"; }
    mv -- "$TMP_POLICY" "$POLICY" || { rm -f -- "$TMP_POLICY"; die "could not move the policy into place"; }
    say "wrote $POLICY (MONITOR mode: the console can watch this machine but change nothing)."
    say "Read it, check every model root, then set \"mode\": \"manage\" yourself (docs/FLEET-PROTOCOL.md section 6)."
fi

# ── what this machine would report ──────────────────────────────────────────────────────────────────────
say "running: $BIN check --policy $POLICY"
check_rc=0
as_user "$BIN" check --policy "$POLICY" || check_rc=$?
if [ "$check_rc" -ne 0 ]; then
    say "ecnode check exited with $check_rc: see the lines marked '!' above (the install itself is complete)"
fi
say "done. To let a hub reach this machine, add the key line it prints (engine-console hosts init-key) to the"
say "login user's authorized keys; it is restricted to: command=\"$BIN --stdio\"."
exit 0
