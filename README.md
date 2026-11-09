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
| machine-1-1 | 0.963 to 0.989 | 0.173 to 0.175 (within 0.0022 of flagging every row, 0.1728) | not measured |
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
when the block or `reports/claims.json` disagrees with those files.

<!-- claims:begin -->
| Claim | Value |
| --- | --- |
| Training steps | 708,405 |
| Test steps | 708,420 |
| Channels | 38 |
| Feature specs | 63 |
| Models compared | 4 |
| Headline model | zscore |
| Headline event-level AUC-PR | 0.70 (95% CI 0.62 to 0.78) |
| MLflow runs | 144 |
| p95 HTTP round trip, client-side (ms) | 6.7 (budget 50) |

Macro-mean validation objective (on injected anomalies, what the headline was chosen by) and AUC-PR under each evaluation protocol, over entities (headline model in bold):

| Model | Validation objective | point_wise | point_adjust | pa_percent_k | event_level |
| --- | ---: | ---: | ---: | ---: | ---: |
| **zscore** | **0.671** | **0.50** | **0.85** | **0.69** | **0.70** |
| iforest | 0.240 | 0.47 | 0.84 | 0.68 | 0.54 |
| cnn_ae | 0.542 | 0.44 | 0.92 | 0.64 | 0.68 |
| lstm_ae | 0.484 | 0.43 | 0.95 | 0.66 | 0.73 |

Effect of the FFT features on event-level F1 and AUC-PR (mean ± sample std over seeds):

| Model | ΔF1 (points) | ΔF1 (relative %) | ΔAUC-PR |
| --- | ---: | ---: | ---: |
| zscore | +2.5 ± 1.8 | +7.4 ± 5.5 | -0.006 ± 0.000 |
| iforest | +2.0 ± 1.6 | +11.4 ± 8.3 | +0.013 ± 0.014 |
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
  memory, and it records the commit the code was at in `environment.git_sha` (with `+dirty` if a
  tracked file had uncommitted changes) and the git tree hash of `src/` in `environment.source_tree`,
  which names the code across merges, next to the platform, the library versions and the thread
  count.
- The committed run was made at commit `b85d382`. Its `git_sha` carries `+dirty` and it has no
  `source_tree`, because at that commit untracked files also counted as changes, the run's own
  output directory was untracked when it started, and `source_tree` did not exist yet; `git status`
  in that checkout lists no tracked change after the run. The commits since then changed the CLI,
  validation and the report schema: selecting machine-1-1 again at the current commit, with the
  run's tuned parameters, reproduces its pipeline, detector and test-score digests, thresholds and
  validation objectives bit for bit for all four models.
- `reports/mlflow_runs.csv` is the export of every tracked run (8 tuning trials per model plus one
  final run per model and machine) and the source of the run count. `reports/latency.json`, written
  by `scripts/latency.py` against a running service, is the source of the latency figure: a
  client-side HTTP round trip, taken on the headline model (the file records the model and bundle
  it measured, and `tsguard claims` refuses any other).
- `reports/claims.json` and the block above are derived from these three files by
  `tsguard claims --write`; `tsguard claims --check` recomputes them.
- The headline model is chosen by its validation objective on injected anomalies, before any test
  label is read, and `metrics.json` records that zero labels were read during selection.

## What the suite does not cover

- **Bit-identity across platforms.** Two runs with one seed, in one process on one machine,
  reproduce every digest bit for bit. Another BLAS or libm can change the last bit of a feature,
  and nothing claims a Linux run matches a macOS one.
- **The full 28-machine run in CI.** It takes hours and runs locally; CI runs a two-machine smoke
  profile. A manual workflow (`.github/workflows/full-run.yml`) is written for it, but it has never
  run, and it may not fit a hosted runner: its reproduce step is capped at 340 minutes, and the
  hosted runners of a private repository have 2 vCPUs, while the run is configured for four threads
  and was made on an Apple M2. Nothing here claims that workflow can re-derive the run.
- **CI on other platforms.** Every CI job runs on GitHub's hosted ubuntu runners (x86_64) and has
  passed there, `container` included: it built the image on linux/amd64 and served it healthy. No
  job runs on macOS, where the full run was made, or on arm64; the image was built for linux/arm64
  only on the M2.
- **Latency beyond one host.** Both measurements are sequential requests from a client on the same
  machine as the container, over loopback: the p95 in the results block, on the M2, and the
  `container` job's p95 of 14.7 ms (p99 16.1 ms) on a hosted runner (x86_64, 2 vCPUs) in the first
  pull request's run, where the job asserts the same 50 ms budget. A network hop and concurrent
  clients are not measured.
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
