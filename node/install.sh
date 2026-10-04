#!/usr/bin/env bash
# install.sh - install (or remove) the Engine Console node agent `ecnode` on THIS machine.
#
#   sudo node/install.sh              copy the package to /usr/local/lib/ecnode/lib (root-owned), write the
#                                     launcher /usr/local/lib/ecnode/ecnode, write /etc/ecnode/policy.json
#                                     (MONITOR mode) only if it does not exist yet, then run `ecnode check`
#                                     as the user who called sudo.
#   sudo node/install.sh --uninstall  remove /usr/local/lib/ecnode. /etc/ecnode (your policy) is kept.
#
# Safe to re-run: the code is replaced atomically, an existing policy is never touched.
# Nothing here opens a port, creates a user, edits ssh or sudo configuration, or starts a service.
# Needs: bash, coreutils, /usr/bin/python3 >= 3.10 (the machine's system Python; no pip packages).
set -euo pipefail

PREFIX=/usr/local/lib/ecnode
LIB="$PREFIX/lib"
LAUNCHER="$PREFIX/ecnode"
POLICY_DIR=/etc/ecnode
POLICY="$POLICY_DIR/policy.json"
PYTHON=/usr/bin/python3

# Resolve our own location before anything changes directory.
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SELF_DIR/src/ecnode"

die() { printf 'install.sh: ERROR: %s\n' "$*" >&2; exit 1; }
say() { printf 'install.sh: %s\n' "$*"; }

usage() {
    sed -n '2,11p' "$SELF_DIR/install.sh"
    return 0
}

MODE=install
case "${1:-}" in
    "") ;;
    --uninstall) MODE=uninstall ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (use --uninstall, --help or nothing)" ;;
esac
[ "$#" -le 1 ] || die "too many arguments"

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

# ── checks ───────────────────────────────────────────────────────────────────────────────────────────────
if [ ! -f "$SRC/__init__.py" ] || [ ! -f "$SRC/proto.py" ] || [ ! -f "$SRC/__main__.py" ]; then
    die "cannot find the ecnode package next to this script (looked in $SRC)"
fi
if [ ! -x "$PYTHON" ]; then
    die "$PYTHON not found: install the system python3 (3.10 or newer) first"
fi
if ! "$PYTHON" -I -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    die "$PYTHON is older than 3.10; ecnode needs 3.10 or newer"
fi
if [ -z "$INVOKER" ]; then
    say "warning: not called through sudo by a normal user; the policy template and the check run as root"
fi

# ── stage the new copy next to the old one (same filesystem, so the swap is one rename) ─────────────────────
PARENT="$(dirname "$PREFIX")"
install -d -o root -g root -m 0755 "$PARENT" || die "could not create $PARENT"
STAGE="$(mktemp -d "$PARENT/.ecnode-new.XXXXXX")" || die "could not create a staging directory in $PARENT"
OLD=""
cleanup() {
    if [ -n "${STAGE:-}" ] && [ -d "$STAGE" ]; then
        rm -rf -- "$STAGE"
    fi
    return 0
}
trap cleanup EXIT

install -d -o root -g root -m 0755 "$STAGE/lib/ecnode" || die "could not create $STAGE/lib/ecnode"
copied=0
for f in "$SRC"/*.py "$SRC"/py.typed; do
    if [ -f "$f" ]; then
        install -o root -g root -m 0644 "$f" "$STAGE/lib/ecnode/" || die "could not copy $f"
        copied=$((copied + 1))
    fi
done
if [ "$copied" -lt 5 ]; then
    die "only $copied files copied from $SRC; the source tree looks incomplete"
fi

# The launcher: isolated Python (no PYTHON* variables, no user site-packages, no current directory on
# sys.path), with only the root-owned copy above added to the path.
cat > "$STAGE/ecnode" <<LAUNCHER || die "could not write the launcher"
#!$PYTHON -I
"""Engine Console node agent (installed by install.sh). Runs the root-owned copy in $LIB."""
import sys
sys.path.insert(0, "$LIB")
from ecnode.__main__ import main
sys.exit(main())
LAUNCHER
chown root:root "$STAGE/ecnode" || die "could not chown the launcher"
chmod 0755 "$STAGE/ecnode" || die "could not chmod the launcher"

if ! "$PYTHON" -I -m compileall -q "$STAGE/lib" >/dev/null; then
    die "the ecnode sources do not compile with $PYTHON"
fi
chown -R root:root "$STAGE" || die "could not chown $STAGE"
chmod 0755 "$STAGE" || die "could not chmod $STAGE"

# ── swap it in ───────────────────────────────────────────────────────────────────────────────────────────
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

VERSION_LINE="$("$LAUNCHER" version)" || die "the installed launcher does not run: $LAUNCHER version failed"
say "installed $VERSION_LINE in $PREFIX (launcher: $LAUNCHER)"

# ── policy: written once, never overwritten ─────────────────────────────────────────────────────────────
if [ -e "$POLICY" ] || [ -L "$POLICY" ]; then
    say "kept the existing policy $POLICY (not changed)"
else
    install -d -o root -g root -m 0755 "$POLICY_DIR" || die "could not create $POLICY_DIR"
    TMP_POLICY="$(mktemp "$POLICY_DIR/.policy.XXXXXX")" || die "could not create a temporary file in $POLICY_DIR"
    # generated as the calling user, so the detected model directories and docker context are theirs
    if ! as_user "$LAUNCHER" policy-template > "$TMP_POLICY"; then
        rm -f -- "$TMP_POLICY"
        die "ecnode policy-template failed; no policy was written"
    fi
    if ! "$PYTHON" -I -c 'import json, sys; json.load(open(sys.argv[1]))' "$TMP_POLICY"; then
        rm -f -- "$TMP_POLICY"
        die "the generated policy is not valid JSON; no policy was written"
    fi
    chown root:root "$TMP_POLICY" || { rm -f -- "$TMP_POLICY"; die "could not chown the policy"; }
    chmod 0644 "$TMP_POLICY" || { rm -f -- "$TMP_POLICY"; die "could not chmod the policy"; }
    mv -- "$TMP_POLICY" "$POLICY" || { rm -f -- "$TMP_POLICY"; die "could not move the policy into place"; }
    say "wrote $POLICY (MONITOR mode: the console can watch this machine but change nothing)."
    say "Read it, check every model root, then set \"mode\": \"manage\" yourself (docs/FLEET-PROTOCOL.md section 6)."
fi

# ── what this machine would report ──────────────────────────────────────────────────────────────────────
say "running: $LAUNCHER check"
check_rc=0
as_user "$LAUNCHER" check || check_rc=$?
if [ "$check_rc" -ne 0 ]; then
    say "ecnode check exited with $check_rc: see the lines marked '!' above (the install itself is complete)"
fi
say "done. To let a hub reach this machine, add the key line it prints (engine-console hosts init-key) to the"
say "login user's authorized keys; it is restricted to: command=\"$LAUNCHER --stdio\"."
exit 0
