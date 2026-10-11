.PHONY: help test lint check install

SHELL := /bin/bash
BENCH_DIR ?= ../europriv-bench

VENV := .venv/bin/activate
RUN := source $(VENV) &&

help:
	@echo "  make install  - editable install + dev deps into .venv"
	@echo "  make test     - run tests with coverage"
	@echo "  make lint     - run ruff"
	@echo "  make check    - test + lint"
	@echo "  make install-moe - isolated Transformers 5 privacy-filter environment"

install:
	# europriv-bench is the shared source of truth (taxonomy + spans) AND the eval harness.
	# Install the sibling editable first so the dependency resolves locally without an index.
	$(RUN) pip install -e $(BENCH_DIR)
	$(RUN) pip install -e '.[dev]'

test:
	$(RUN) coverage run -m pytest
	$(RUN) coverage report

lint:
	$(RUN) ruff check .

check: test lint

.PHONY: install-moe
install-moe:
	python3.13 -m venv .venv-moe
	.venv-moe/bin/pip install -e $(BENCH_DIR)
	.venv-moe/bin/pip install -e '.[moe,dev]'
