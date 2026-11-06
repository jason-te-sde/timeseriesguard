.PHONY: lint type test leakage protocol data smoke reproduce check bundle container-up latency

PYTEST_ARGS ?=
ENTITY ?= machine-1-1
BUNDLE_ARGS ?= --metrics reports/metrics.json

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

# Fits one machine and writes artifacts/bundle/bundle.joblib and its .sha256, which the container
# mounts. The default is the headline model of reports/metrics.json with its tuned parameters;
# `make bundle BUNDLE_ARGS="--model iforest --default-params"` needs no metrics file.
bundle:
	uv run tsguard bundle --entity $(ENTITY) $(BUNDLE_ARGS) --out artifacts/bundle/bundle.joblib

# Needs Docker and `make bundle` first. Stop it with `docker compose down`.
container-up:
	docker compose up --build --wait

# Needs the service on 127.0.0.1:8000 (`make container-up`) and the SMD files of the machine it
# serves: the script loads that machine's test rows to build the requests.
latency:
	uv run python scripts/latency.py --n 2000 --warmup 100 --budget-ms 50 --out artifacts/latency.json
