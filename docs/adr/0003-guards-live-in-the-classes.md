# 3. Leakage guards live inside the classes that hold the data

Status: accepted, 2026-11-03.

## Context

A leak in a time-series pipeline makes results better, so nothing about it looks like a failure. The
code that leaks also looks innocent: a centered rolling window, a scaler fitted on the whole series, a
backward fill, a spectrum over the whole block, a threshold picked on the test labels, folds without a
purge, a shuffled split. Each is one argument or one line, and each is a plausible thing to write.

Two properties made the usual defences insufficient here. The pipeline has many places where a leak
can enter (features, scaling, folds, splits, thresholds, labels), and a correct-looking change in one
place can feed future rows into another. And the project's results claim to be leak-free, so that
claim needs evidence, not intent.

## Decision

Each protection is a refusal by the object that holds the resource, raised as a `LeakageError` that
names the guard:

- `FeaturePipeline.transform` refuses a start row before the fit end (`fit_end`).
- `AuditedLabels.read` refuses outside `Phase.FINAL_EVALUATION`, and `LabelAudit.freeze_threshold`
  refuses during it (`label_access_audit`). `evaluate` accepts only a `FrozenThreshold`, and labels
  and audits refuse to be copied or pickled.
- `check_fold_boundaries` and `check_split_monotonic` re-derive the fold and split invariants from the
  numbers instead of trusting the code that built them (`fold_boundary`, `split_monotonic`).
- `assert_causal` and `assert_fit_ignores_future` treat any transform as a black box: replace the
  inputs from row `t0` on with noise, and require every earlier output, or the fitted state, to stay
  bit-for-bit the same (`assert_causal`).

`EntityRunner` is the one place that wires data, features, models and protocols together, and it
calls every guard itself. The components through which a leak could enter (the feature specs, the
pipeline class, the fold and split builders, the threshold search) are constructor arguments, so a test
can hand it a leaky one.

Ten known defects are written down in `tests/leakage/defects.py`, and each must be caught by the
guard named for it, not merely by some error. Controls show the guards stay quiet on honest code. See
[testing.md](../testing.md#the-injected-defects).

## Consequences

- A leak is an exception at the moment it happens, naming the guard, the row or fold, and (through a
  note) the machine and the model, instead of a number that is quietly too good.
- New code inherits the guards: a new feature spec is checked by `assert_causal` in every selection
  without anyone writing a test for it.
- The defect table is executable, so the claim "each guard catches what it is for" is re-checked on
  every run of the suite.

Costs:

- **Runtime.** The perturbation tests recompute features and scores on noise. They run once per input
  kind and model in a runner, and the FFT ablation skips them because both of its arms use specs the
  selection already verified.
- **Friction.** Callers pass `context` and `offset` explicitly, thresholds need the `FrozenThreshold`
  ceremony, and labels cannot be copied, pickled or sent to another process.
- **The guards bound what they watch, nothing more.** The perturbation test sees dependence on the rows
  it replaces, in the input it is given; a leak through a path no guard watches (a decision a person
  makes after looking at test results, for example) is still a matter of discipline. The guards make
  the known leaks impossible; they do not prove the absence of unknown ones.

## Rejected alternatives

**Review-only discipline.** A reviewer catches the leaks the reviewer thinks of. The innocent-looking
ones are exactly the ones that pass review. Review also has to be repeated for every change, and it
leaves no evidence that a reader of the results can check. Review still happens; it is not what the
leak-free claim rests on.

**A lint rule.** It sees syntax, not data flow. It could flag `rolling(center=True)`, but not a fill
whose direction depends on an argument, a statistic computed over an array that happens to include the
future, or a label read through a helper in another module.
