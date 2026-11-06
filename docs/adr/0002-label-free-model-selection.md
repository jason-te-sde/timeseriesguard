# 2. Select thresholds, hyper-parameters and the headline without test labels

Status: accepted, 2026-11-03.

## Context

SMD's train half carries no labels. Only the test half does, and those labels are what the reported
numbers are computed from. Yet four choices need a target to be made against: each detector's
hyper-parameters, each detector's decision threshold, the model behind the headline, and which fitted
model scores the test rows.

Any of these chosen with the test labels in view turns the reported number into an upper bound. That
is the familiar F1\* of the literature, the best F1 over all thresholds, which this project reports
only as an oracle next to the real result.

## Decision

Every choice is made on **synthetic anomalies injected into the train half**, the surrogate that
Goswami et al. (ICLR 2023, arXiv:2210.01078) describe in §3, in the paragraph "Synthetic Anomaly
Injection" (their generator is in Appendix A.5):

- The train rows are cut into three expanding walk-forward folds with a 64-row purge between each fit
  window and its validation block.
- Each validation block receives seeded anomalies of four kinds (spike, level shift, variance burst,
  flatline) with lengths, channel counts and magnitudes drawn from fixed ranges.
- A configuration's objective is the event-level AUC-PR against the injected labels, averaged over the
  three folds. Optuna tunes against it; the headline is the model with the best mean objective over
  the machines.
- The threshold that maximizes event-level F1 on the last fold is frozen and applied unchanged to the
  test scores, and the last fold's fitted model is the model that produces them.

The rule is enforced, not trusted. Test labels live in `AuditedLabels`, which refuses every read
outside `Phase.FINAL_EVALUATION`. A threshold exists only as a `FrozenThreshold`, which
`LabelAudit.freeze_threshold` refuses to create during the final evaluation and which records how many
reads preceded it. `reproduce` stops the run unless every audit shows zero reads and every threshold
zero earlier reads, and it chooses the headline before the first test label is read.

## Consequences

- The F1 reported for each model comes from a threshold that never saw a test label, and
  `metrics.json` records `label_reads_at_selection: 0` for the headline.
- The same machinery serves the model bundle: building one is the experiment's selection, run once,
  and reads no label.

Costs:

- **Selection is only as good as the injected anomalies.** Their distribution is a modelling choice,
  not a measurement of SMD's incidents. A kind of real anomaly the injector never plants (a slow
  drift, say) is not rewarded during selection.
- **The final model sees about 83 % of the train rows**, the last fold's fit window. Refitting on all
  of them would put the frozen threshold on another model's score scale.
- **The frozen threshold rests on 3 or 4 synthetic segments**, those injected into the last fold's
  validation block (one per 1,000 rows, at least two). F1 at the frozen threshold is therefore a
  noisy number from machine to machine.
- **Selection is expensive.** Every configuration is fitted and scored three times, and the injected
  blocks and feature matrices are cached per machine (2 to 3 GB on a real machine).
- **F1 at the frozen threshold is lower than F1\*,** sometimes by a lot. That gap is the honest part
  of the number, and it makes the results look worse than published ones that report F1\*.

## Rejected alternatives

**Tuning and thresholding on the test labels.** The simplest option, and the one that produces the
optimistic numbers this project exists to avoid. F1\* is still computed, labelled as an oracle, and
used for nothing.

**Carving a labelled validation slice out of the test half.** It would select on real anomalies, but
SMD has too few per machine: machine-2-8 has a single anomaly segment, so either the validation slice
or the remaining test slice of that machine would have none. A slice also shortens the test period
every number is reported on, and every real segment it used for selection is one fewer the
evaluation can count. Injection does not run out: the synthetic segments come from the train half,
which no number is reported on.

**Refitting the chosen configuration on the whole train half.** It would fit on 20 % more rows (the
last fold leaves out the final sixth of the train half), but the threshold was frozen on the scores
of the fold model, and a refitted model scores on a different scale.
