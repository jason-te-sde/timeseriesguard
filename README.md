# TimeSeriesGuard

Anomaly detection on server metrics, built so that its numbers can be trusted. It runs four
detectors (a robust z-score, Isolation Forest, and 1D-CNN and LSTM autoencoders) over 63 named
features on the Server Machine Dataset (SMD: 28 machines, 38 channels, 708,405 train and 708,420
test rows, one row per minute). The models are not the point. The evaluation protocol was fixed
before any model ran; every threshold, hyper-parameter and the headline model are chosen without a
single test label; the common ways to leak future data or labels are refused by the code itself;
and the model results quoted here are generated from one file and checked against it in CI. A FastAPI service
scores one row at a time and returns the same score the batch evaluation computed for that row.

## What is different

### 1. A protocol that random scores cannot beat

Most published SMD results use *point adjust*, which counts a whole anomaly segment as detected
when any one of its rows is flagged. SMD's segments run up to 3,161 rows, so a detector that knows
nothing does very well. Uniform random scores, five seeds, measured on the real labels:

| | point adjust F1\* | point-wise F1\* | event-level AUC-PR |
| --- | --- | --- | --- |
| machine-1-1 | 0.963 to 0.989 | 0.173 to 0.175 (= flagging every row) | |
| mean of 28 machines | 0.767 to 0.776 | about 0.08 | 0.039 to 0.049 |

The headline here is the **event-level AUC-PR**: each anomaly segment is one positive sample, and
the normal time is cut into one-hour chunks, each one negative sample, with a one-hour cooldown
after every incident. It counts incidents caught and quiet hours paged, which is how alerts are
experienced on call. Point-wise, point adjust and PA%K are reported next to it for every model.
A test (`tests/protocol`) keeps the random-score result true on every run. Why each choice was made:
[docs/evaluation.md](docs/evaluation.md) and [ADR 0001](docs/adr/0001-event-level-protocol.md).

### 2. Leakage guards that the code enforces

Leaks make results better, so they never look like bugs. Instead of relying on review, each
protection is a refusal by the object that holds the data: the feature pipeline refuses to evaluate
the rows its scaling was fitted on; test labels refuse to be read before the final evaluation; a
threshold cannot be frozen after a label was read; fold and split builders are re-checked from their
numbers; and a perturbation test replaces future rows with noise and requires every earlier output to
stay bit-for-bit the same. Ten known leaks (a centered window, a backward fill, a whole-series
scaler, a threshold tuned on test labels, an unpurged fold, a shuffled split, ...) are injected on
purpose, and the tests assert *which* guard catches each one:
[docs/testing.md](docs/testing.md#the-injected-defects) and
[ADR 0003](docs/adr/0003-guards-live-in-the-classes.md).

## Results

The block below is written by `tsguard claims --write` from `reports/metrics.json`, and CI's
`claims` job fails if it disagrees with that file; no number in it is typed by hand. If it is
empty, the full run has not been committed yet.

<!-- claims:begin -->
<!-- claims:end -->

## Quickstart

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --all-extras                # runtime, the experiment extra (MLflow, Optuna) and dev tools
uv run tsguard data fetch           # SMD at a pinned commit, every file sha256-verified
make check                          # ruff, mypy, and the test suite with coverage
make leakage                        # the injected-defect suite
uv run tsguard reproduce --profile smoke --out artifacts/smoke --tracking-uri sqlite:///artifacts/mlruns.db
make bundle                         # fit one model for machine-1-1 into artifacts/bundle/
docker compose up --build --wait    # serve it on 127.0.0.1:8000
curl -s http://127.0.0.1:8000/v1/model
```

The smoke profile runs every phase of the experiment on two machines in minutes. The service, its
error responses and the latency budget (p95 below 50 ms) are described in
[docs/operations.md](docs/operations.md).

## Where every number comes from

- `reports/metrics.json` is the only source. It is written by `tsguard reproduce --profile full`
  (`make reproduce`), run locally on an Apple M2 with 16 GB of memory, and it records the commit
  the code was at in `environment.git_sha` (with `+dirty` if the working tree had changes), next to
  the platform, the library versions and the thread count.
- `reports/mlflow_runs.csv` is the export of every tracked run: 8 tuning trials per model plus one
  final run per model and machine. `reports/claims.json` and the block above are derived from these
  files by `tsguard claims --write`; `tsguard claims --check` recomputes them.
- The headline model is chosen by its validation objective on injected anomalies, before any test
  label is read, and `metrics.json` records that zero labels were read during selection.

## What the suite does not cover

- **Bit-identity across platforms.** One seed reproduces every digest bit for bit on one machine.
  Another BLAS or libm can change the last bit of a feature, and nothing claims a Linux run matches
  a macOS one.
- **The full 28-machine run in CI.** It takes hours and runs locally; CI runs a two-machine smoke
  profile. A manual workflow (`.github/workflows/full-run.yml`) can re-derive the full run on a
  Linux runner.
- **CI itself, so far.** The GitHub repository has not been created yet, so the workflows are
  written and linted but have never executed, and the container image has not been built.
- **Alerting.** The service returns a score and a decision per row. Deduplication, routing and
  paging (Slack or otherwise) are out of scope.
- **The realism of the injected anomalies.** Model selection runs on synthetic anomalies of four
  kinds. Their distribution is a modelling choice, not a measurement of real incidents.
- **Hostile clients.** The service has no authentication, and a model bundle is a pickle: load only
  bundles you built. See [SECURITY.md](SECURITY.md).

## Repository

| | |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | the packages, the selection data flow, where each guard sits |
| [docs/evaluation.md](docs/evaluation.md) | the four protocols, label-free selection, variance |
| [docs/features.md](docs/features.md) | the 63 feature specs |
| [docs/testing.md](docs/testing.md) | test layers, CI jobs, the injected defects |
| [docs/operations.md](docs/operations.md) | running the service and the experiment, and what to do when they fail |
| [docs/adr/](docs/adr/) | four recorded decisions |
| [docs/lessons.md](docs/lessons.md) | what building it taught |
| [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [CHANGELOG.md](CHANGELOG.md) | |

## License

MIT, copyright Jason Te. See [LICENSE](LICENSE). SMD itself comes from the
[OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) repository under its own MIT license and is
downloaded, never committed.
