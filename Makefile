.PHONY: lint type test leakage protocol data smoke reproduce check

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

# A run exports every run in its tracking store, so `smoke` starts from a fresh one.
smoke:
	rm -rf artifacts/smoke artifacts/mlruns.db
	uv run tsguard reproduce --profile smoke --out artifacts/smoke --tracking-uri sqlite:///artifacts/mlruns.db

# The full run takes hours. It refuses to start while reports/mlruns.db exists: delete it first.
reproduce:
	uv run tsguard reproduce --profile full --out reports

check: lint type test
