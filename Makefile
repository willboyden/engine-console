# Engine Console developer tasks. Nothing here touches Docker, the network or systemd.
# Backend uses uv + Python 3.13 (newer interpreters often lack wheels for some dependencies).
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
BACKEND  := backend
FRONTEND := frontend
NODE     := node

.PHONY: help setup run test lint typecheck frontend-test node-test node-lint node-typecheck node-matrix all \
	install-desktop install-desktop-icon uninstall-desktop

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup:  ## create the backend and node venvs and install runtime + dev deps
	cd $(BACKEND) && uv venv --python 3.13 && uv sync --frozen
	cd $(NODE) && uv venv --python 3.12 && uv sync

run:  ## serve API + UI on 127.0.0.1:8791 (foreground)
	cd $(BACKEND) && uv run engine-console

test:  ## backend pytest (offline: fakes for HF and docker)
	cd $(BACKEND) && uv run pytest -q

lint:  ## ruff
	cd $(BACKEND) && uv run ruff check src tests

typecheck:  ## mypy --strict
	cd $(BACKEND) && uv run mypy src

frontend-test:  ## node --test on the pure-logic modules (no npm install needed)
	cd $(FRONTEND) && node --test tests/*.test.mjs

node-test:  ## node agent pytest on its oldest supported target (3.12: Ubuntu 24.04 / DGX OS system Python)
	cd $(NODE) && uv run pytest -q

node-lint:  ## ruff on the node agent
	cd $(NODE) && uv run ruff check src tests

node-typecheck:  ## mypy --strict on the node agent
	cd $(NODE) && uv run mypy src

node-matrix:  ## node tests on every interpreter a machine might have (3.12, 3.13, 3.14)
	cd $(NODE) && for v in 3.12 3.13 3.14; do echo "== python $$v"; uv run --isolated --python $$v --with pytest -- python -m pytest -q -p no:cacheprovider; done

all: lint typecheck test frontend-test node-lint node-typecheck node-test  ## everything the quality bar requires

# The desktop launcher runs `engine-console open` (start in the background if needed, then open the UI) and offers a
# "Stop" action. A fixed applications dir, not $XDG_DATA_HOME: snap-packaged terminals and IDEs point that elsewhere.
# Paths go into a .desktop Exec line and a sed replacement, so the checkout path is limited to a safe character set.
APPS_DIR := $(HOME)/.local/share/applications
LAUNCHER := engine-console.desktop

install-desktop:  ## add "Engine Console" to your app menu (starts the console if needed, then opens the UI)
	test -x $(BACKEND)/.venv/bin/engine-console || { echo "run 'make setup' first" >&2; exit 1; }
	dir=$$(pwd -P); \
	case "$$dir" in *[!A-Za-z0-9._/+-]*) echo "install-desktop: move the checkout to a path of [A-Za-z0-9._/+-] only" >&2; exit 1;; esac; \
	mkdir -p "$(APPS_DIR)"; \
	sed -e "s|@BIN@|$$dir/$(BACKEND)/.venv/bin/engine-console|g" -e "s|@ICON@|$$dir/deploy/engine-console.svg|g" \
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
