# Testing

The suite has to show two different things. That the code computes what it says it computes, which
is what tests usually do. And that the protections against leakage work, which a suite that only
runs honest code cannot show: a guard that never fires looks exactly like a guard that is broken.
So the leakage layer injects known defects and asserts *which* guard stopped each one.

## Layers

Markers decide what a plain `pytest` runs. `pyproject.toml` deselects `data` and `slow` by default,
so `uv run pytest` is the fast suite and needs neither the dataset nor more than a few minutes.

| Layer | Where | Command | CI job |
| --- | --- | --- | --- |
| Unit | `tests/unit` | `make test` (`uv run pytest` with coverage, floor 80 %) | `test` (Python 3.11 and 3.12) |
| Leakage | `tests/leakage` | `make leakage` | `leakage`, and `test` (the default run includes it) |
| Protocol | `tests/protocol` | `make protocol` (needs the dataset) | `protocol` |
| Determinism | `tests/determinism`: the whole `reproduce` run twice with one seed must give identical digests (two `slow` tests on synthetic machines, one `data` test on machine-1-1); plus a same-seed test per component in `tests/unit` | `uv run pytest tests/determinism -m "slow or data"`; the unit ones with `uv run pytest -k "deterministic or same_seed"` | `determinism` runs `tests/determinism`, `test` the unit ones |
| Data | anything marked `data` | `uv run tsguard data fetch`, then `uv run pytest -m data` | `protocol` runs `tests/protocol` and `tests/data` |
| Slow | anything marked `slow`: the tiny end-to-end `reproduce` in `tests/unit/test_reproduce.py` and `tests/determinism` | `uv run pytest -m slow` | `smoke` (all but `tests/determinism`, which has its own job) |

Around the tests:

- `make lint type` runs `ruff check`, `ruff format --check` and `mypy` (strict for `src` and
  `scripts`; test functions may be unannotated). CI job `lint`.
- `make smoke` runs `tsguard reproduce --profile smoke` on two real machines, every phase included.
  CI jobs `smoke` (warm caches, every push and pull request) and `clean-clone` (pushes to `main`
  only, with no cache of any kind, so a dependency or a data file that only one machine happens to
  have cannot hide).
- `make bundle container-up latency` builds a bundle for machine-1-1, starts the service in its
  container and asserts a p95 below 50 ms over 2,000 requests. CI job `container`.
- `uv run tsguard claims --check` compares the README numbers with `reports/`. CI job `claims`.
- The `hygiene` job rejects a commit message that carries a `Co-Authored-By:` trailer.

Gaps, stated so nobody assumes otherwise:

- **One `data` test is outside every CI job.** `tests/unit/test_smd_fetch.py::test_real_cache_matches_the_packaged_manifest`
  carries the `data` marker but lives in `tests/unit`, and the jobs that fetch data select
  `tests/protocol`, `tests/data` and `tests/determinism` only. It runs locally with
  `uv run pytest -m data`.
- **Determinism is checked on one platform.** Two runs on the same machine must agree bit for bit;
  nothing compares a macOS run with a Linux run, and nothing claims they agree.

None of the CI jobs has run yet: the GitHub repository does not exist, so the workflows are written
and linted (`actionlint`) but have never executed. Until they have, every row of the table above is
verified locally only.

### Tests that need the dataset fail, they never skip

A test marked `data` uses the `smd_dir` fixture, which verifies the cache against the packaged
manifest and fails with `SMD data missing: run uv run tsguard data fetch` when it cannot. A skipped
data test in CI would read as a pass.

### Randomness in tests

A test that needs random numbers takes the `rng` fixture, a `numpy.random.Generator` seeded from
`TSGUARD_TEST_SEED` (default `20261103`). A failing test that used it prints `seed=<n>` in its
report, so the failure can be replayed:

```sh
TSGUARD_TEST_SEED=<n> uv run pytest tests/leakage -k <test>
```

## The injected defects

Each defect is a known way to leak, written down on purpose in `tests/leakage/defects.py`, and each
must be caught by the one guard named for it. The table below is that file's `DEFECTS` mapping;
`tests/unit/test_docs.py` parses this table and fails if an id or a guard differs from the code.

| id | defect | guard |
| --- | --- | --- |
| `1` | rolling window centered on the row, reading rows after it | `assert_causal` |
| `2` | scaler fitted on every row and recorded as such | `fit_end` |
| `2b` | scaler fitted on every row but recorded with the requested fit end | `assert_causal` |
| `3` | gaps (values at or below 0.5) filled from the next value, backwards | `assert_causal` |
| `4` | spectrum computed over the whole block instead of a trailing window | `assert_causal` |
| `5` | threshold searched on test labels during model selection | `label_access_audit` |
| `5b` | threshold frozen after test labels were read during evaluation | `label_access_audit` |
| `6` | walk-forward folds without a purge between fit and validation | `fold_boundary` |
| `7` | rows shuffled into train and test, ignoring time order | `split_monotonic` |
| `8` | gaps (values at or below 0.5) interpolated from both sides | `assert_causal` |

Where each guard lives, and which code calls it, is in [architecture.md](architecture.md#where-each-guard-sits).

Three files exercise the table:

- `tests/leakage/test_injected_defects.py` injects each defect directly into the guard's input and
  asserts the guard's name from the raised `LeakageError`, never just "an exception". The
  `covers(...)` decorator records which defects a test injects, and
  `test_every_defect_has_a_test` fails if an id in `DEFECTS` has no test.
- `tests/leakage/test_runner_wiring.py` hands the same defects to an `EntityRunner` as constructor
  arguments (a leaky spec, pipeline class, fold builder, split or threshold search) and asserts which
  guard stops the runner. This is what shows the guards are *called*, not merely correct.
- `tests/leakage/test_must_not_flag.py` is the other direction: lag features, trailing and expanding
  windows, a forward fill behind the same mask defects 3 and 8 use, two independent entities, and the
  full 63-spec registry must all pass `assert_causal`. Without these, a guard that flags everything
  would pass the defect tests.

Each defect is caught for the leak it names and for no other. The two fill defects therefore open
their gaps with a mask that reads only the value itself (`block > 0.5`). An earlier version used a
whole-block median as the mask; `assert_causal` then fired on the median, a forward fill behind the
same mask was flagged as well, and the tests passed for the wrong reason.

## Proving a guard can go red

A green guard test is only evidence if the same test turns red when the guard is gone. To check one:

1. Disable the guard in the source, for example make `check_fold_boundaries` in
   `src/tsguard/data/splits.py` return at its first line, or delete the `start_index < fit_end`
   branch of `FeaturePipeline.transform`.
2. Run the defect test and the wiring test for it:

   ```sh
   uv run pytest tests/leakage -k purge       # fold_boundary: 2 tests
   uv run pytest tests/leakage -k full_fit    # fit_end: 3 tests
   ```

   Expected: the tests fail, and the failure names the defect that got through.
3. Restore the file (`git checkout -- <file>`) and run `make leakage` again.

The same method applies to ordering claims, with one addition: deleting a step proves the step
exists, but only moving it proves its position is pinned. For example, the test
`test_the_headline_line_comes_after_the_blind_check_and_before_evaluation` must fail when the
headline choice is moved after the evaluation loop in `reproduce`, not only when it is removed.
