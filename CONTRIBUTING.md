# Contributing

## Prerequisites

- Python 3.11 (CI also runs 3.12) and [uv](https://docs.astral.sh/uv/). `uv sync --all-extras` installs
  everything: the runtime, the `experiment` extra (MLflow, Optuna) and the dev tools. Run every
  command through `uv run`.
- The SMD files for anything marked `data`: `uv run tsguard data fetch` (about 1.4 million rows,
  verified against the checksums in `src/tsguard/data/smd_manifest.json`). Never commit them.

## Before you push

```sh
make check leakage
```

That is `ruff check`, `ruff format --check`, `mypy` (strict for `src` and `scripts`), the default
test suite with an 80 % coverage floor, and the injected-defect suite. All of it must pass with no
warnings. Depending on what you touched, also run:

| You changed | Also run |
| --- | --- |
| protocols, labels, data loading | `make protocol` and `uv run pytest -m data` |
| the runner, tuning, the ablation, `reproduce` | `uv run pytest -m slow`, then `make smoke` |
| the service or the bundle | `make bundle container-up latency` (needs Docker) |
| a feature spec | nothing extra, but `docs/features.md` must list it: a test compares the table with the registry |

## Workflow

1. **Open an issue first.** State the problem, the proposed approach and the acceptance criteria. If
   the change touches features, scaling, folds, thresholds or labels, name the leakage guard that is
   meant to keep it honest.
2. **Branch** from `main` as `<type>/<issue>-<slug>`, for example `fix/12-kurtosis-underflow`.
3. Commit in small steps that each pass `make check`.
4. Open a pull request that closes the issue (`Closes #12`). It merges by squash, once CI is green and
   the branch is up to date.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/), with the module as the scope:

```
<type>(<scope>): <imperative summary, no trailing period>

<body: why this change, what was considered and rejected, what it does not do,
and how far it was verified>

Refs #<issue>
```

Types in use: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `ci`, `build`, `chore`. Scopes are
package names (`data`, `features`, `models`, `eval`, `experiment`, `repro`, `serve`, `cli`) or
`deploy`, `leakage`, `determinism`.

**No co-author trailers.** A `Co-Authored-By:` line fails the `hygiene` job. Tools add one by
default, so delete it before committing.

## Tests

- A change in behaviour comes with a test that fails without it. Run it and watch it fail first.
- Randomness comes from an explicit seed: the `rng` fixture in tests, an explicit
  `numpy.random.Generator` or integer seed in code, never `np.random.<function>` or `random`.
- A test that needs the dataset carries `@pytest.mark.data` and uses the `smd_dir` fixture, which
  fails when the data is missing. It never skips.
- A leakage test asserts the guard's name (`LeakageError.guard`), not just that something raised.
  A new guard comes with an injected defect in `tests/leakage/defects.py`, a row in the table in
  `docs/testing.md` (a test compares the two), and a control that must not be flagged.
- A claim about order is pinned by a test that fails when the step is moved, not only when it is
  deleted.

## Code

- Every public module, class and function in `src` has a Google-style docstring; comments explain
  why, not what.
- No module-level mutable state, no wildcard imports, no `print` outside `src/tsguard/cli.py` and
  `scripts/`.
- Every deliberate exception subclasses `tsguard.errors.TsguardError` and names the entity, row,
  column or phase it concerns.
- Where the code follows a paper, cite its section instead of paraphrasing the argument.
- Everything in the repository is in English.
