# Engine Console developer tasks. Nothing here (except the opt-in *-desktop targets, which write launcher files) touches
# Docker, systemd or another machine, or uses the network beyond what cargo needs (crates.io, and the RustSec advisory
# database for `audit`); `crate-age` queries the crates.io API, and `parity` / `proto` need uv's cache (or the network,
# the first time) for the Python reference. `make run` serves the console against your real state: read
# docs/OPERATIONS.md first. The hub, the node agent and their wire protocol are one Rust workspace in backend/; its
# toolchain is pinned by backend/rust-toolchain.toml. Every cargo call uses the committed Cargo.lock (--locked): a gate
# never resolves a different dependency set than the one that was reviewed.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
BACKEND  := backend
FRONTEND := frontend
CARGO    := cargo
RELEASE  := $(BACKEND)/target/release
DIST     := $(BACKEND)/target/dist

.PHONY: help setup hooks build release run test fmt lint clippy typecheck deny audit crate-age test-map parity proto \
        frontend-test e2e all node-test node-dist install-desktop install-desktop-icon uninstall-desktop

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup:  ## build the hub and the node (debug, locked dependencies) and enable the git hooks
	cd $(BACKEND) && $(CARGO) build --locked --workspace --bins
	if git rev-parse --git-dir >/dev/null 2>&1; then $(MAKE) --no-print-directory hooks; fi

hooks:  ## use the versioned git hooks in .githooks (pre-push refuses code without a recorded e2e pass)
	git config core.hooksPath .githooks

build:  ## debug binaries: backend/target/debug/{engine-console,ecnode}
	cd $(BACKEND) && $(CARGO) build --locked --workspace --bins

# Release binaries carry no path of the build machine: source paths in panic messages and debug data are rewritten
# (the rest of $HOME to ~, the cargo home to /cargo, the workspace to /build), and the build fails if one remains.
# Cargo's trim-paths would do this but is not stable on the pinned toolchain. CI's release job uses the same flags.
CARGO_HOME ?= $(HOME)/.cargo
REMAP := --remap-path-prefix=$(HOME)=~ --remap-path-prefix=$(CARGO_HOME)=/cargo --remap-path-prefix=$(CURDIR)/$(BACKEND)=/build
RELEASE_BINS := engine-console ecnode

release:  ## optimised binaries for real use: backend/target/release/{engine-console,ecnode}
	cd $(BACKEND) && RUSTFLAGS="$(strip $(RUSTFLAGS) $(REMAP))" $(CARGO) build --locked --release --workspace --bins
	for b in $(RELEASE_BINS); do \
	  for p in "$(CURDIR)/" "$(HOME)/"; do \
	    if [ "$$p" != "//" ] && grep -aqF "$$p" "$(RELEASE)/$$b"; then echo "release: $$b still contains the path $$p" >&2; exit 1; fi; \
	  done; \
	  echo "release: $(RELEASE)/$$b (no build-machine path inside)"; \
	done

run: release  ## serve API + UI on 127.0.0.1:8791 (foreground; ENGINE_CONSOLE_* env and ~/.config/engine-console/env)
	$(RELEASE)/engine-console

test:  ## every Rust test: hub, node, protocol (hermetic: fakes, temp dirs, closed ports)
	cd $(BACKEND) && $(CARGO) test --locked --workspace --all-features

fmt:  ## rustfmt the workspace in place
	cd $(BACKEND) && $(CARGO) fmt --all

lint:  ## rustfmt check + clippy with warnings as errors
	cd $(BACKEND) && $(CARGO) fmt --all --check
	$(MAKE) --no-print-directory clippy

clippy:  ## clippy on every crate and target (warnings are errors)
	cd $(BACKEND) && $(CARGO) clippy --locked --workspace --all-targets --all-features -- -D warnings

typecheck: clippy  ## the compiler and clippy are the type check

deny:  ## cargo-deny: advisories, licenses, bans, sources (backend/deny.toml)
	cd $(BACKEND) && $(CARGO) deny --locked check

audit:  ## cargo-audit against the RustSec database
	cd $(BACKEND) && $(CARGO) audit

crate-age:  ## the 14-day dependency cooldown (queries the crates.io API; exceptions: backend/tools/crate-age-exceptions.toml)
	python3 -I $(BACKEND)/tools/crate-age.py

test-map:  ## every test of the last Python release names the Rust test that covers it, and that test exists
	node $(BACKEND)/parity/check-test-map.mjs --cargo

parity: build  ## HTTP differential run: the last Python release (needs uv) vs the Rust hub, then Python vs Python
	node $(BACKEND)/parity/run.mjs
	node $(BACKEND)/parity/run.mjs --b python

proto: build  ## the node protocol: Python x Rust driver and node matrix, Python hub + Rust node, Rust hub + Python node
	node $(BACKEND)/parity/proto/matrix.mjs --no-build
	node $(BACKEND)/parity/proto/hub-node.mjs --no-build
	node $(BACKEND)/parity/proto/rust-hub.mjs --no-build

frontend-test:  ## node --test on the pure-logic modules (no npm install needed)
	cd $(FRONTEND) && node --test tests/*.test.mjs

e2e:  ## browser end-to-end: real hub + real local node (monitor mode) + headless Chrome, every page; records a pass for pre-push
	scripts/e2e-gate.sh run

all: lint test test-map deny audit frontend-test e2e  ## everything the quality bar requires (plus crate-age, parity and proto)

node-test:  ## the node agent's and the wire protocol's tests only
	cd $(BACKEND) && $(CARGO) test --locked -p ecnode -p ec-proto --all-features

# The node for THIS machine's architecture as a release asset: ecnode-<version>-<arch> and its .sha256, the files
# `node/install.sh --binary` takes. CI's release job builds x86_64 and aarch64 (glibc) and attaches both.
node-dist: release  ## backend/target/dist/ecnode-<version>-<arch> + .sha256, for node/install.sh --binary
	ver=$$(sed -n 's/^version = "\(.*\)"/\1/p' $(BACKEND)/Cargo.toml | head -n 1); \
	arch=$$(uname -m); name="ecnode-$$ver-$$arch"; \
	mkdir -p $(DIST); \
	install -m 0755 $(RELEASE)/ecnode "$(DIST)/$$name"; \
	(cd $(DIST) && sha256sum "$$name" > "$$name.sha256"); \
	echo "node-dist: $(DIST)/$$name"; cat "$(DIST)/$$name.sha256"

# The desktop launcher runs `engine-console open` (start in the background if needed, then open the UI) and offers a
# "Stop" action. A fixed applications dir, not $XDG_DATA_HOME: snap-packaged terminals and IDEs point that elsewhere.
# Paths go into a .desktop Exec line and a sed replacement, so the checkout path is limited to a safe character set.
# For the same reason gio runs without GIO_MODULE_DIR, which a snap terminal points at modules that cannot set metadata.
APPS_DIR := $(HOME)/.local/share/applications
LAUNCHER := engine-console.desktop

install-desktop:  ## add "Engine Console" to your app menu (starts the console if needed, then opens the UI); needs `make release`
	test -x $(RELEASE)/engine-console || { echo "run 'make release' first" >&2; exit 1; }
	dir=$$(pwd -P); \
	case "$$dir" in *[!A-Za-z0-9._/+-]*) echo "install-desktop: move the checkout to a path of [A-Za-z0-9._/+-] only" >&2; exit 1;; esac; \
	mkdir -p "$(APPS_DIR)"; \
	sed -e "s|@BIN@|$$dir/$(RELEASE)/engine-console|g" -e "s|@ICON@|$$dir/deploy/engine-console.svg|g" \
	    deploy/$(LAUNCHER) > "$(APPS_DIR)/$(LAUNCHER)"; \
	chmod 644 "$(APPS_DIR)/$(LAUNCHER)"; \
	echo "installed $(APPS_DIR)/$(LAUNCHER)"

install-desktop-icon: install-desktop  ## also put the launcher on your desktop (GNOME: marked trusted so it runs on double-click)
	desk=$$(xdg-user-dir DESKTOP 2>/dev/null || true); [ -n "$$desk" ] && [ "$$desk" != "$(HOME)" ] || desk="$(HOME)/Desktop"; \
	[ -d "$$desk" ] || { echo "install-desktop-icon: no desktop folder at $$desk" >&2; exit 1; }; \
	install -m 755 "$(APPS_DIR)/$(LAUNCHER)" "$$desk/$(LAUNCHER)"; \
	if command -v gio >/dev/null; then env -u GIO_MODULE_DIR gio set "$$desk/$(LAUNCHER)" metadata::trusted true || echo "note: could not mark it trusted; right-click it > Allow Launching" >&2; fi; \
	echo "installed $$desk/$(LAUNCHER)"

uninstall-desktop:  ## remove the menu entry and the desktop icon
	rm -f "$(APPS_DIR)/$(LAUNCHER)"
	desk=$$(xdg-user-dir DESKTOP 2>/dev/null || true); [ -n "$$desk" ] && [ "$$desk" != "$(HOME)" ] || desk="$(HOME)/Desktop"; \
	rm -f "$$desk/$(LAUNCHER)"
