.PHONY: install lint format format-check test check clean

install:
	uv sync

lint:
	uv run ruff check scripts tests

format:
	uv run ruff format scripts tests

format-check:
	uv run ruff format --check scripts tests

test:
	uv run pytest

compile:
	uv run python -m py_compile scripts/*.py

## Canonical local gate: what CI runs before a push.
check: lint format-check compile test

clean:
	rm -rf .ruff_cache .pytest_cache __pycache__ scripts/__pycache__ tests/__pycache__