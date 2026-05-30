.PHONY: help test lint check install

VENV := .venv/bin/activate
RUN := source $(VENV) &&

help:
	@echo "  make install  - editable install + dev deps into .venv"
	@echo "  make test     - run tests with coverage"
	@echo "  make lint     - run ruff"
	@echo "  make check    - test + lint"

install:
	# europriv-bench is the shared source of truth (taxonomy + spans) AND the eval harness.
	# Install the sibling editable first so the dependency resolves locally without an index.
	$(RUN) pip install -e ../europriv-bench
	$(RUN) pip install -e '.[dev]'

test:
	$(RUN) coverage run -m pytest
	$(RUN) coverage report

lint:
	$(RUN) ruff check .

check: test lint
