# TimeSeriesGuard

Anomaly detection on server metrics, built so that its numbers can be trusted. It compares four
detectors on the Server Machine Dataset (SMD: 28 machines, 38 channels, 708,405 train and 708,420
test rows, one row per minute): a robust z-score and Isolation Forest over 63 named features, and
1D-CNN and LSTM autoencoders over the 38 standardized raw channels. The models are not the point.
The evaluation protocol was fixed before any model ran; every threshold, hyper-parameter and the
headline model are chosen without a single test label; the common ways to leak future data or
labels are refused by the code itself; and the numbers in the results block are generated from the
files in `reports/` and checked against them. A FastAPI service scores one row at a time, and its
score matches the batch evaluation's score for that row within a relative 1e-6.

## What is different

### 1. A protocol that random scores cannot beat

Most published SMD results use *point adjust*, which counts a whole anomaly segment as detected
when any one of its rows is flagged. SMD's segments run up to 3,161 rows, so a detector that knows
nothing does very well. Uniform random scores, five seeds, measured on the real labels:

| | point adjust F1\* | point-wise F1\* | event-level AUC-PR |
| --- | --- | --- | --- |
| machine-1-1 | 0.963 to 0.989 | 0.173 to 0.175 (within 0.002 of flagging every row) | not measured |
| mean of 28 machines | 0.767 to 0.776 | about 0.08 | 0.039 to 0.049 |

The headline here is the **event-level AUC-PR**: each anomaly segment is one positive sample, and
the normal time is cut into one-hour chunks, each one negative sample, with a one-hour cooldown
after every incident. It counts incidents caught and quiet hours paged, which is how alerts are
experienced on call. Point-wise, point adjust and PA%K are reported next to it for every model.
`tests/protocol` pins the result as bounds: the default test run covers a synthetic-label version,
and the real-label versions run in the `protocol` CI job. Why each choice was made:
[docs/evaluation.md](docs/evaluation.md) and [ADR 0001](docs/adr/0001-event-level-protocol.md).

### 2. Leakage guards that the code enforces

Leaks make results better, so they never look like bugs. Instead of relying on review, each
protection is a refusal by the object that holds the data: the feature pipeline refuses to evaluate
the rows its scaling was fitted on; test labels refuse to be read before the final evaluation; a
threshold cannot be frozen during the final evaluation, and each frozen threshold records how many
label reads preceded it, so the experiment stops if any threshold was frozen after a read; fold and
split builders are re-checked from their numbers; and a perturbation test replaces future rows with
noise and requires every earlier output to stay bit-for-bit the same. Ten known leaks (a centered
window, a backward fill, a whole-series scaler, a threshold tuned on test labels, an unpurged fold, a
shuffled split, ...) are injected on purpose, and the tests assert *which* guard catches each one:
[docs/testing.md](docs/testing.md#the-injected-defects) and
[ADR 0003](docs/adr/0003-guards-live-in-the-classes.md).

## Results

The block below is written by `tsguard claims --write` from `reports/`: the model numbers from
`metrics.json`, the run count from `mlflow_runs.csv` and the latency from `latency.json`. No number
in it is typed by hand. Once `reports/` holds any `.json` or `.csv` file, CI's `claims` job fails
when the block or `reports/claims.json` disagrees with those files. If the block is empty, the full
run has not been committed yet.

<!-- claims:begin -->
<!-- claims:end -->

## Quickstart

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --all-extras                # runtime, the experiment extra (MLflow, Optuna) and dev tools
uv run tsguard data fetch           # SMD at a pinned commit, every file sha256-verified
make check                          # ruff, mypy, and the test suite with coverage
make leakage                        # the injected-defect suite
make smoke                          # the whole experiment on two machines, from a fresh MLflow store
make bundle                         # fit one model for machine-1-1 into artifacts/bundle/
docker compose up --build --wait    # serve it on 127.0.0.1:8000
curl -s http://127.0.0.1:8000/v1/model
```

`make smoke` runs `tsguard reproduce --profile smoke` into `artifacts/smoke`, every phase of the
experiment on two machines, in minutes. It deletes `artifacts/mlruns.db` first, because `reproduce`
refuses a tracking store that already exists. The service, its error responses and the latency
budget (p95 below 50 ms) are described in [docs/operations.md](docs/operations.md).

## Where every number comes from

- `reports/metrics.json` is the source of every model number. It is written by
  `tsguard reproduce --profile full` (`make reproduce`), run locally on an Apple M2 with 16 GB of
  memory, and it records the commit the code was at in `environment.git_sha` (with `+dirty` if the
  working tree had changes), next to the platform, the library versions and the thread count.
- `reports/mlflow_runs.csv` is the export of every tracked run (8 tuning trials per model plus one
  final run per model and machine) and the source of the run count. `reports/latency.json`, written
  by `scripts/latency.py` against a running service, is the source of the latency figure.
- `reports/claims.json` and the block above are derived from these three files by
  `tsguard claims --write`; `tsguard claims --check` recomputes them.
- The headline model is chosen by its validation objective on injected anomalies, before any test
  label is read, and `metrics.json` records that zero labels were read during selection.

## What the suite does not cover

- **Bit-identity across platforms.** Two runs with one seed, in one process on one machine,
  reproduce every digest bit for bit. Another BLAS or libm can change the last bit of a feature,
  and nothing claims a Linux run matches a macOS one.
- **The full 28-machine run in CI.** It takes hours and runs locally; CI runs a two-machine smoke
  profile. A manual workflow (`.github/workflows/full-run.yml`) can re-derive the full run on a
  Linux runner.
- **CI itself, so far.** The GitHub repository has not been created yet, so the workflows are
  written and linted but have never executed, and the container image has not been built.
- **Latency, so far.** No latency has been measured against a running service yet; the 50 ms budget
  is asserted by the `container` CI job, which has not run.
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
