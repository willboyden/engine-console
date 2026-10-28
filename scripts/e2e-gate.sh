#!/usr/bin/env bash
# Ship gate for the browser end-to-end suite (frontend/e2e/run.mjs).
#
#   scripts/e2e-gate.sh run           build the hub and the node, run the suite; on a pass with a clean backend/,
#                                     frontend/ and node/, record it
#   scripts/e2e-gate.sh check <rev>   exit 0 only if the suite passed on exactly <rev>'s backend/, frontend/ and node/
#
# The record is the triple of git tree ids of backend/ (the Rust workspace: the hub, the node agent ecnode, the wire
# protocol, the parity harness), frontend/ and node/ (the node's installer and policy examples), kept inside .git
# (never committed). The suite runs the real hub and its real local node. A docs-only commit after a pass needs no rerun; any change to the code does.
# The pre-push hook (.githooks/pre-push, enabled by `make hooks`) calls `check` for every commit or tag being pushed. CI
# runs the suite again on every push. No `set -e`: every failure below is handled and reported on purpose.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
STAMP="$(git rev-parse --git-path engine-console-e2e.pass)" || { echo "e2e-gate: not a git checkout" >&2; exit 2; }
CODE=(backend frontend node)

# "<backend tree> <frontend tree> <node tree>" of a commit (an annotated tag is peeled to its commit).
trees() {
  local d out=() id
  for d in "${CODE[@]}"; do
    id=$(git rev-parse --verify --quiet "$1^{commit}:$d") || return 1
    out+=("$id")
  done
  echo "${out[*]}"
}

case "${1:-}" in
  run)
    # A Chrome that is not on PATH can be named once per checkout: git config engine-console.chrome /path/to/chrome
    if [ -z "${CHROME_BIN:-}" ] && chrome=$(git config --get engine-console.chrome); then export CHROME_BIN="$chrome"; fi
    # the suite runs backend/target/debug/{engine-console,ecnode}: build them from the working tree it is about to test
    (cd backend && cargo build --locked -p ec-hub -p ecnode --bins) || { echo "e2e-gate: cargo build failed" >&2; exit 1; }
    node frontend/e2e/run.mjs || exit 1
    # The suite tested the working tree, so a pass only counts for a commit when the code matches it.
    if [ -n "$(git status --porcelain -- "${CODE[@]}")" ]; then
      echo "e2e-gate: passed, but backend/, frontend/ or node/ has uncommitted changes, so no pass was recorded." >&2
      echo "e2e-gate: commit, then run 'make e2e' again before pushing." >&2
      exit 0
    fi
    want=$(trees HEAD) || { echo "e2e-gate: cannot read HEAD's trees" >&2; exit 1; }
    echo "$want" > "$STAMP" || exit 1
    echo "e2e-gate: pass recorded for $(git rev-parse --short HEAD)"
    ;;
  check)
    rev="${2:-}"
    [ -n "$rev" ] || { echo "usage: scripts/e2e-gate.sh check <rev>" >&2; exit 2; }
    want=$(trees "$rev") || { echo "e2e-gate: $rev has no backend/, frontend/ and node/ trees" >&2; exit 1; }
    [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$want" ]
    ;;
  *)
    echo "usage: scripts/e2e-gate.sh run | check <rev>" >&2
    exit 2
    ;;
esac
