# Architecture

One package, `tsguard`, in a `src` layout. The split follows one rule: **features do not know about
models, models do not know about protocols, and protocols do not know which model produced a
score.** `tsguard.experiment.runner.EntityRunner` is the one place that connects them, so it is also
the one place that has to call every leakage guard, and `tests/leakage/test_runner_wiring.py` checks
that it does.

## Unit map

| Package | What it does | How to use it | Depends on |
| --- | --- | --- | --- |
| `tsguard.errors` | The exception hierarchy. Every deliberate error is a `TsguardError`; a `LeakageError` also carries `guard`, the name of the guard that fired, and a `detail` dict. | `except LeakageError as e: e.guard` | nothing |
| `tsguard.data` | SMD from one pinned OmniAnomaly commit: `smd.py` (manifest, verified fetch, `load_entity`), `contract.py` (six shape, range and label rules), `labels.py` (`LabelAudit`, `Phase`, `AuditedLabels`, `FrozenThreshold`), `splits.py` (walk-forward folds and the two structural checks). | `load_entity("machine-1-1", audit=LabelAudit())` returns read-only train and test arrays and labels that refuse to be read outside the final evaluation. | `errors` |
| `tsguard.features` | `specs.py`: the 63 named feature specs, each a pure function of a trailing window of at most 64 rows (table in [features.md](features.md)). `pipeline.py`: `FeaturePipeline`, which computes them in blocks, standardizes them with statistics fitted once, and refuses to evaluate its own fit rows. `causality.py`: the future-perturbation tests. | `FeaturePipeline().fit_transform(train, end_index=n)`, then `transform(series, start_index=n)` | `errors` |
| `tsguard.models` | The `Detector` protocol (`fit`, `score(rows, context)`, `state_digest`), four detectors (z-score and Isolation Forest on features, 1D-CNN and LSTM autoencoders on raw channels), their defaults and Optuna search spaces. | `build_detector("iforest", params, seed=s)` | `repro` (digests) |
| `tsguard.eval` | `protocols.py`: point-wise, point-adjust, PA%K and event-level views, average precision, F1 at a frozen threshold, oracle F1. `inject.py`: seeded synthetic anomalies for selection. | `evaluate(scores, labels, frozen_threshold, ProtocolParams())` | `data` (`FrozenThreshold`) |
| `tsguard.experiment` | `runner.py`: `EntityRunner`, which selects a model on one entity without labels and evaluates it once. `tuning.py`: Optuna search over the validation objective. `tracking.py`: MLflow or in-memory run records and the CSV export. `ablation.py`: the FFT ablation. `reproduce.py`: the whole experiment in a fixed phase order. | `reproduce(PROFILES["smoke"], loader=..., tracker=..., out_dir=...)`, or `tsguard reproduce` | `data`, `errors`, `eval`, `features`, `models`, `repro` |
| `tsguard.repro` | `seeding.py` (`derive_seed`, `configure_runtime`), `hashing.py` (array and JSON digests), `report.py` (the `metrics.json` schema and the bootstrap interval), `claims.py` (`claims.json` and the README block, derived from `reports/`). | `read_metrics(path)`, `tsguard claims --check` | `eval` (protocol names) |
| `tsguard.serve` | `bundle.py`: a fitted pipeline, detector and frozen threshold for one entity in one checksummed file, and `score_rows`. `schemas.py`: strict Pydantic request and response bodies. `app.py`: the FastAPI service. `commands.py`: `bundle_command` and `serve_command`. | see [operations.md](operations.md) | `data`, `errors`, `experiment`, `features`, `models`, `repro` |
| `tsguard.cli` | The `tsguard` command: `data fetch`, `data verify`, `reproduce`, `claims`, `bundle`, `serve`. A `TsguardError` becomes one line on stderr and exit code 1. Nothing it imports at module level may need MLflow or Optuna, because the container installs neither. | `uv run tsguard --help` | `data`, `errors`, `experiment`, `models`, `repro`, `serve` |

Outside the package: `scripts/latency.py` measures a running service against the latency budget,
and `Dockerfile` with `docker-compose.yml` package the service (see [operations.md](operations.md)).

## Rules that hold everywhere

- **Every feature value is a pure function of its trailing window.** A spec's value at row `t` reads
  rows `t - lookback + 1 .. t` and nothing else: no running sums, no recursion from the series
  start. The EWMA is a 64-row windowed EWMA for this reason, and skewness and kurtosis are
  population moments of the window. A block computation therefore gives the same bits whatever the
  block size, and the service, which receives only the last rows, gives the same value as the batch
  run. [ADR 0004](adr/0004-windowed-features.md) records why.
- **Flat windows give zero, never NaN.** A window of identical values has exactly zero spread, so its
  spread, skew, kurtosis and FFT features are 0. Skewness and kurtosis are also 0 when the
  denominator (`m2^1.5`, `m2^2`) underflows to zero for a window of tiny but unequal values.
- **Standardization is bounded.** The pipeline subtracts a per-column mean and divides by a
  per-column scale fitted on the fit rows only. A column constant over the fit rows gets that
  constant as its mean, exactly, so it standardizes to 0 instead of rounding noise. Every scale is
  at least `MIN_SCALE = 1e-3` (SMD is min-max scaled to [0, 1], so a smaller change is noise), and
  every standardized value is clipped to `±50` (`STANDARD_CLIP`). The warmup rows at the series
  start become 0; a NaN anywhere else raises `RuntimeError` naming the column, so a broken spec
  cannot hide behind a zero fill.
- **All randomness comes from an explicit seed.** `derive_seed(root, *labels)` hashes a root seed
  and a purpose (entity, model, fold, `"inject"`, `"causal"`) into an independent 32-bit seed, so
  changing one purpose never shifts another's stream. `configure_runtime` fixes the torch thread
  count and turns on deterministic algorithms; entry points call it, library code never does.

## The data flow of `EntityRunner.select`

`select(model, params, seed=...)` chooses a configuration's threshold and scores the test rows
without reading a test label. In order:

1. **Enter the selection phase.** The whole call runs inside `audit.phase(Phase.SELECTION)`, even when
   the caller is inside a final-evaluation block, so any label read in it is refused.
2. **Split.** `split_fn(n_train, n_test)` (by default `chronological_split`) gives the train and test
   indices, and `check_split_monotonic` refuses them unless every train index precedes every test
   index. Guard `split_monotonic`.
3. **Plan.** A probe detector, built with the fold-0 seed, reports its input kind (`features` or
   `raw`) and lookback. The specs are the registry for feature models and the single identity spec
   for the autoencoders. `walk_forward_folds(n_train, k=3, initial_frac=0.5, purge=64)` cuts the
   train rows into three expanding folds: fold `j` fits on `[0, b_{j-1})` and validates on
   `[b_{j-1} + 64, b_j)`, with `b_0 = floor(n/2)` and the remaining half split in three.
   `check_fold_boundaries` refuses the folds unless every gap is at least the purge, the purge is at
   least the lookback, and the folds move forward in time. Guard `fold_boundary`.
4. **Per fold, fit and score synthetic anomalies.**
   1. Fit a pipeline on `train[:fit_end]` (cached per input kind and fold, since it does not depend on
      the model or the seed).
   2. Inject anomalies into `train[:val_end]`, inside the validation block only, with per-channel
      scales from `train[:fit_end]` and a generator seeded from `(seed, entity, "inject", fold)`
      (cached per seed and fold).
   3. Build the detector with the seed `(seed, entity, model, fold)`, standardize the validation
      rows with `transform(start_index=val_start, context=lookback - 1)` (guard `fit_end`: the
      evaluated rows must not start before the fit end), fit the detector on the in-sample
      features, and score the validation rows. A non-finite score raises `RuntimeError`.
   4. The fold's objective is the event-level AUC-PR of those scores against the injected labels.
5. **Freeze the threshold.** `select_threshold` picks the threshold that maximizes event-level F1 on
   the *last* fold's injected labels, and `audit.freeze_threshold` turns it into a `FrozenThreshold`.
   Freezing is refused during the final evaluation, and the frozen value records how many label reads
   preceded it. Guard `label_access_audit`.
6. **Score the test rows** with the last fold's pipeline and detector: `transform(full_series,
   start_index=train_end, context=lookback - 1)`, then `detector.score`. The scores must be finite and
   are stored read-only. The final model is the last fold's model, so the threshold and the test
   scores come from the same fitted model and share one score scale.
7. **Check causality** (when `RunConfig.causal_check` is on), each check once per runner:
   `assert_causal` on the pipeline's transform of the test rows (once per input kind),
   `assert_fit_ignores_future` on a fresh fit with the last fold's fit end (once per input kind), and
   `assert_causal` on the detector's scoring of the standardized test rows (once per input kind and
   model). Guard `assert_causal`.
8. **Return a `Selection`**: the mean of the three fold objectives, the frozen threshold, the test
   scores, and the digests of the pipeline, the detector and the scores. With `keep_models=True` it
   also holds the fitted pipeline and detector, which is what a service bundle is made from.

`EntityRunner.evaluate(selection)` is the only read of test labels in the normal flow. It enters
`Phase.FINAL_EVALUATION`, reads the labels once as `evaluate[<model>]`, and computes the four
protocols at the frozen threshold plus the event-level sensitivity grid.

A leakage error raised inside `select` or `objective` gets a note naming the entity and the model,
so a failure deep in a 28-entity run says where it happened.

### Around the runner: `reproduce`

`tsguard.experiment.reproduce.reproduce` runs a profile in a fixed order, and the leakage claims rest
on that order:

1. **Tune** each model on the tuning entities (machine-1-1, machine-2-1, machine-3-1 in the full
   profile): Optuna TPE, 8 trials per model, the first trial always the defaults, each trial scored by
   the validation objective and recorded as one MLflow run.
2. **Select** all four models on every entity, one entity at a time, each under a fresh `LabelAudit`.
   A runner holds 2 to 3 GB of cached features on a real entity, so it is released before the next
   entity is loaded.
3. **Blind check and headline.** Every audit must show zero reads and every frozen threshold zero
   reads before it, or the run stops with `RuntimeError`. The headline model is the one with the
   highest macro validation objective, a quantity computed from selections only, so the choice
   cannot depend on a test label.
4. **Evaluate** every selection once on its test labels.
5. **Checkpoint**: `metrics.partial.json`, a valid `Metrics` with an empty ablation, is written before
   the ablation so that a failure in its long tail does not discard the rest.
6. **Ablate** the FFT family on the two feature models over several seeds. On each entity every
   selection (every seed, model and arm) is made before the first test label is read. The ablation
   runs with `causal_check=False`: both arms use specs the selection phase already verified.
7. Write `metrics.json` and delete the checkpoint. The CLI then exports the MLflow store to
   `mlflow_runs.csv` and refuses the result if the row count is not `4 × (trials + entities)`.

## Where each guard sits

| Guard | Code | Called by | What it refuses | Defects |
| --- | --- | --- | --- | --- |
| `fit_end` | `FeaturePipeline.transform` | the runner for every validation block and the test rows; `score_rows` in the service | a start row before the fit end: evaluating rows the scaling was fitted on | 2 |
| `assert_causal` | `features/causality.py`: `assert_causal`, `assert_fit_ignores_future` | `EntityRunner._check_causal` | an output row before `t0` that changes when the inputs from `t0` on are replaced by noise; a fitted state that changes when the rows after `end_index` are replaced | 1, 2b, 3, 4, 8 |
| `fold_boundary` | `data/splits.py`: `check_fold_boundaries` | `EntityRunner._plan` | a purge shorter than the lookback, a gap shorter than the purge, folds that do not move forward | 6 |
| `split_monotonic` | `data/splits.py`: `check_split_monotonic` | `EntityRunner._select` | a train index at or after a test index, empty or non-increasing index arrays | 7 |
| `label_access_audit` | `data/labels.py`: `AuditedLabels.read`, `LabelAudit.freeze_threshold` | every label read; the runner's threshold freeze | a label read outside `Phase.FINAL_EVALUATION`; a threshold frozen during it | 5, 5b |

The defect ids are those of [testing.md](testing.md#the-injected-defects). Guards that are not leakage
guards, and so raise other errors:

- `eval.protocols.evaluate` accepts only a `FrozenThreshold` (`TypeError` otherwise), and only
  `LabelAudit.freeze_threshold` can create one.
- `LabelAudit` and `AuditedLabels` refuse to be copied or pickled: a copy would keep a stale phase or
  carry the values out of the guard.
- `reproduce.label_reads_at_selection` and the ablation stop the run with `RuntimeError` if any
  selection-time read happened, so the headline is label-free by check, not by convention.
- `FeaturePipeline` raises `RuntimeError` for a NaN past the warmup, and `transform` with a window
  that starts after the series start demands the full history its first row needs.
- `data.smd` verifies every file against the sha256 and size in `smd_manifest.json` before parsing
  it, and `data.contract` checks shape, finiteness, the [0, 1] range and binary labels.
- `serve.bundle.load_bundle` checks the sha256 sidecar before unpickling, then the pipeline state
  digest, the detector digest, and a fingerprint of the feature code (see
  [operations.md](operations.md#verify-a-bundle)).
