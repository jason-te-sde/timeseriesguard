.PHONY: lint type test leakage protocol data check

PYTEST_ARGS ?=

lint:
	uv run ruff check .
	uv run ruff format --check .

type:
	uv run mypy

test:
	uv run pytest --cov --cov-report=term-missing $(PYTEST_ARGS)

leakage:
	uv run pytest tests/leakage -m "not data"

protocol:
	uv run pytest tests/protocol -m "data or not data"

data:
	uv run tsguard data fetch

check: lint type test
