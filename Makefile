# Engine Console developer tasks. Nothing here touches Docker, the network or systemd.
# Backend uses uv + Python 3.13 (newer interpreters often lack wheels for some dependencies).
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
BACKEND  := backend
FRONTEND := frontend
NODE     := node

.PHONY: help setup run test lint typecheck frontend-test node-test node-lint node-typecheck node-matrix all

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
