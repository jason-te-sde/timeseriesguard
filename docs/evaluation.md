# Evaluation protocol

This document fixes how a score becomes a number, and why. Every choice in it was made before any
model was run on SMD, and none of it is tuned to a result. Results are not here: they are in the
README block that `tsguard claims --write` generates and in `reports/`: `metrics.json` is the
source of every model number, `mlflow_runs.csv` of the run count and `latency.json` of the latency.

## The data the protocol has to handle

SMD (Server Machine Dataset) as read from the pinned OmniAnomaly commit: 28 machines, 38 channels
each, sampled once a minute, 708,405 train rows and 708,420 test rows in all. Only the test half is
labelled. It holds 327 anomaly segments whose lengths are very uneven: median 11 rows, 10th
percentile 3, 90th percentile 239, longest 3,161. The share of anomalous test rows ranges from
0.42 % to 15.65 % per machine, and machine-2-8 has a single anomaly segment.

Those lengths matter more than anything else below. A protocol that counts rows lets the long
segments decide the score: one 3,161-row segment weighs as much as 287 segments of the median length.

## Four protocols, one headline

Every (entity, model) pair is scored under four protocols. Each protocol is a *view*, a pair
`(scores, labels)`, and the same three numbers are computed on every view: average precision
(`auc_pr`), F1 at a threshold frozen before any test label was read (`f1`), and the best F1 any
threshold reaches on the view (`f1_oracle`, written F1\* below). F1\* needs the labels to be found,
so it is an upper bound reported for comparison with the literature, never a result, and it takes
no part in any choice. Macro values are the mean over entities, so every machine weighs the same.

| Protocol | Samples | Sample score |
| --- | --- | --- |
| `point_wise` | every test row | the row's score |
| `point_adjust` | every test row | inside an anomaly segment, the segment's maximum |
| `pa_percent_k` | every test row | inside a segment, `max(own, k-th largest of the segment)`, see below |
| `event_level` | each anomaly segment (positive) and each normal chunk (negative) | the sample's maximum |

Why report all four instead of one:

- **Point-wise** is the strict baseline, but it asks a detector to flag every minute of an incident,
  and it is dominated by the longest segments. It is not what an operator cares about.
- **Point adjust (PA)** is what most published SMD results use, so it is reported for comparison. It
  is also broken, and the project carries a test that shows it (next section).
- **PA%K** is the published repair of PA. It is reported because it is the obvious alternative to the
  protocol this project chose, and a reviewer should be able to compare the two.
- **Event-level** is the headline. Its unit is the incident, which is how alerts are experienced:
  one incident detected or missed, one quiet hour paged or not.

## Random scores beat point adjust

`tests/protocol/test_point_adjust_is_gameable.py` scores a detector that knows nothing (uniform
random scores) under every protocol, with five seeds (0 to 4). Measured on the real SMD labels:

| | point adjust F1\* | point-wise F1\* | event-level AUC-PR |
| --- | --- | --- | --- |
| machine-1-1 | 0.963 to 0.989 | 0.173 to 0.175 | not measured |
| 28-machine macro mean | 0.767 to 0.776 | about 0.08 | 0.039 to 0.049 |

Under PA, random scores reach an F1\* that would pass for state of the art. The mechanism is the
segment lengths: a segment of a few hundred rows almost surely contains one high random score, and
PA then credits every row of it. Flagging every row of machine-1-1 scores a point-wise F1\* of
`2a / (1 + a)` for the machine's anomaly rate `a`, which is 0.1728 for `a` = 0.0946, and the best
point-wise F1\* of random scores is 0.175, within 0.0022 of it. The same random
scores get an event-level AUC-PR below 0.05 on average over the 28 machines: counting incidents and
quiet hours instead of rows leaves them nothing to exploit.

The test asserts these as bounds, for every one of the five seeds: on machine-1-1, PA F1\* > 0.9 and
point-wise F1\* at most `2a / (1 + a) + 0.02`; over the 28 machines, macro PA F1\* > 0.7, macro
point-wise F1\* < 0.1 and macro event-level AUC-PR < 0.1. A version on synthetic labels (four
1,000-row segments in 40,000 rows) needs no data and runs in the default suite; if it fails, PA is
implemented wrongly.

## The event-level protocol

Given one score per test row and the labels:

1. **Positives.** Every maximal run of anomalous rows is one positive sample. Its score is the
   highest score inside the run.
2. **Cooldown.** The `C` normal rows that follow the end of each anomaly segment are dropped.
3. **Negatives.** The normal rows that remain form maximal runs; each run is cut from its first row
   into consecutive chunks of `L` rows (the last chunk of a run may be shorter). Every chunk is one
   negative sample, scored by its maximum. A chunk never spans a segment or a dropped row.
4. **AUC-PR** is `sklearn.metrics.average_precision_score` over these samples, and F1 is computed on
   them at the frozen threshold.

Implementation: `tsguard.eval.protocols.event_samples` and `event_level_auc_pr`.

**`L = 60` (one hour).** SMD is sampled once a minute. Alerting deduplicates within a window, and an
hour is the usual one: a normal hour in which any minute crosses the threshold is one false page,
however many minutes crossed it. Scoring a chunk by its maximum is exactly that rule.

**`C = 60` (one hour).** An alert in the hour after an incident ends belongs, operationally, to the
same incident: on-call is still looking at it, so it is not a new false page. Features computed over
trailing windows also carry an anomaly forward for up to 64 rows by construction. Without a
cooldown, the protocol would charge a detector for the tail of an incident it already caught.

Both constants live in `ProtocolParams` (`chunk = 60`, `cooldown = 60`) and were fixed before the
first model ran.

### What it does not measure

- **Detection delay.** A segment counts as detected if its maximum crosses the threshold, whether that
  happens in its first minute or its last. For a 3,161-row segment that is a real difference the
  number does not show.
- **Length.** Every segment is one positive, so a 3-row blip and a two-day incident weigh the same.
  That is the intended correction of point-wise and PA, and also a choice a reader may disagree with.
- **Alarms in the cooldown.** A genuine false alarm inside the hour after an incident is forgiven.

### Sensitivity grid

To show how much the headline depends on the two constants, every evaluation also computes the
event-level AUC-PR for `(L, C)` in `{30, 60, 120} x {0, 60}`, for every model and entity
(`EntityRunner` and `SENSITIVITY_GRID` in `tsguard.experiment.runner`). `metrics.json` holds them
per entity under `sensitivity` and as macro means under `sensitivity_macro`, keyed `L<chunk>_C<cooldown>`.
The headline uses `L60_C60` only.

What the grid shows (macro-mean event-level AUC-PR, from `sensitivity_macro` in `reports/metrics.json`):

| `(L, C)` | zscore (headline) | lstm_ae | cnn_ae | iforest |
| --- | ---: | ---: | ---: | ---: |
| `L30_C0` | 0.5589 | 0.7192 | 0.6571 | 0.4547 |
| `L30_C60` | 0.6803 | 0.7296 | 0.6708 | 0.5237 |
| `L60_C0` | 0.5935 | 0.7247 | 0.6683 | 0.4802 |
| `L60_C60` | 0.6989 | 0.7347 | 0.6812 | 0.5427 |
| `L120_C0` | 0.6120 | 0.7328 | 0.6820 | 0.4994 |
| `L120_C60` | 0.7156 | 0.7413 | 0.6920 | 0.5666 |

- **The z-score headline does depend on the cooldown.** Its value runs from 0.5589 to 0.7156 over the
  six cells, and moving from `L60_C60` to `L60_C0` takes it from 0.6989 to 0.5935. With the cooldown
  on, it ranks second behind `lstm_ae` in all three chunk lengths; at `C = 0` it ranks third, behind
  both autoencoders, in all three. Dropping the cooldown costs it 0.10 to 0.12 at each `L`, against
  about 0.01 for `lstm_ae` and `cnn_ae` and 0.06 to 0.07 for `iforest`.
- **The chunk length matters less.** At `C = 60` the z-score moves from 0.6803 (`L = 30`) to 0.7156
  (`L = 120`).
- **The autoencoders barely move** (`lstm_ae` 0.7192 to 0.7413 over the grid), and `lstm_ae` is first
  in every cell. The headline is the z-score because the validation objective chose it, not because
  it leads the grid.

## PA%K

PA%K is defined by Kim et al., "Towards a Rigorous Evaluation of Time-series Anomaly Detection",
AAAI 2022, arXiv:2109.05257, §4.2 ("New evaluation protocol PA%K"): point adjust is applied to a
segment only if the share of its rows that are detected exceeds K percent. This project uses
`K = 20`.

PA%K is stated for one threshold; average precision needs every threshold at once. The
implementation (`adjust_pa_percent_k`) therefore rewrites the scores so that any threshold applied to
them gives the PA%K decision. Inside a segment of `len` rows, every row's score becomes
`max(own, kth)`, where `kth` is the `k`-th largest score of the segment and

```
k = min(len, floor(K * len / 100) + 1)
```

A threshold at or below `kth` flags at least `k` rows of the segment, so the whole segment is lifted
over it; a higher threshold leaves the rows as they were. `K = 0` gives point adjust and `K = 100`
gives point-wise.

**Why `floor(K * len / 100) + 1` and not `ceil(K * len / 100)`.** The paper's condition is "exceeds",
a strict inequality. The smallest count of rows that exceeds `K` percent of `len` is
`floor(K * len / 100) + 1`. `ceil` gives the same count except when `K * len` divides by 100, where
it accepts a count *equal* to `K` percent: for a 5-row segment and `K = 20`, `ceil` credits the
segment for one detected row (exactly 20 %), and `floor + 1` demands two. An early draft of the design
used `ceil`; the code and this document use `floor + 1`.

## Choosing thresholds and models without test labels

SMD's train half has no labels, and the test labels may be read only once, at the end. Thresholds,
hyper-parameters and the headline model are therefore all chosen on synthetic anomalies injected into
the train half. Goswami et al., "Unsupervised Model Selection for Time-series Anomaly Detection",
ICLR 2023, arXiv:2210.01078, list synthetic anomaly injection among their surrogate measures of model
quality in §3 (the paragraph "Synthetic Anomaly Injection"; their generator is in Appendix A.5). Their
premise is that a model that performs well on data with injected anomalies is a good model. This
project uses the idea, not their method: there is no rank aggregation, only one surrogate.

How it is done here (`tsguard.eval.inject`, `tsguard.experiment.runner`):

- **Walk-forward folds inside the train half.** Three expanding folds: the first fit window is the
  first 50 % of the train rows, and the remaining half is split into three validation blocks. Each
  validation block starts 64 rows (the purge, equal to the longest feature lookback) after its fit
  window ends.
- **Injection.** Each validation block is cut into one slot per 1,000 rows (at least two slots) and
  one anomaly is planted per slot, which gives 3 or 4 per block on SMD's machines. The kind is
  chosen uniformly: a spike (1 to 3 rows) or a level shift adds a constant, and a variance burst adds
  Gaussian noise, each with a magnitude of 3 to 8 times the channel's standard deviation over the fit
  rows (0.1 for a channel that is constant there); a flatline holds the channel at its value in the
  segment's first row and uses no magnitude. Non-spike lengths are log-uniform between 3 and 240
  rows, matching the 10th to 90th percentile of SMD's segment lengths, and each anomaly touches 1 to
  8 channels. Neighbouring anomalies are at least 120 rows apart, one cooldown plus one chunk, so at
  least one whole negative chunk survives the cooldown between them. Every draw comes from a
  generator seeded from `(seed, entity, "inject", fold)`.
- **Objective.** A configuration's objective on an entity is the mean over the three folds of the
  event-level AUC-PR against the injected labels. Tuning maximizes its mean over the tuning entities.
- **Threshold.** The threshold that maximizes event-level F1 on the last fold's injected anomalies is
  frozen through `LabelAudit.freeze_threshold` and applied unchanged to the test scores. It rests on
  the 3 or 4 synthetic segments of that one block, so F1 at the frozen threshold is a noisy number.
  AUC-PR needs no threshold and does not inherit that noise.
- **Final model.** The model that scores the test rows is the last fold's model, fitted on about the
  first 83 % (five sixths) of the train rows. Refitting on the whole train half would put the frozen
  threshold on a different model's score scale. The cost is the last sixth of the training rows.
- **Headline.** The headline model is the one with the highest validation objective, averaged over
  the evaluated entities. It is chosen and announced before the first test label is read, and
  `metrics.json` records `selected_by: "validation_objective"` and `label_reads_at_selection: 0`.
  Reading the file checks the choice again: a `metrics.json` whose headline is not the model with
  the highest validation objective (a tie goes to the earlier of `zscore`, `iforest`, `cnn_ae`,
  `lstm_ae`) is invalid, and the README's model table prints that objective next to the scores.

Why not the alternatives (see [ADR 0002](adr/0002-label-free-model-selection.md)): tuning on test
labels reports an oracle, not a result; and carving a labelled validation slice out of the test half
fails on SMD's per-machine segment counts, where machine-2-8 has one segment to split.

The injection distribution is a modelling choice, not a measurement of real SMD anomalies. If real
incidents look unlike all four kinds (a slow drift, for example), selection favours detectors of the
injected kinds. The distribution was fixed before any test evaluation and is not tuned afterwards.

## Per-entity variance and the bootstrap interval

A macro mean over 28 machines hides how differently the machines behave. Two consequences are
reported instead of hidden:

- **Per-entity values.** `metrics.json` holds every protocol's numbers for every machine under
  `models.<model>.per_entity`, next to each machine's validation objective and frozen threshold.
  Machines with few segments give coarse values. machine-2-8 has one segment, so its event-level
  AUC-PR has a single positive sample and can only be 1, 1/2, 1/3, ...: one over the number of
  samples that score at least as high as that one incident. In the macro mean it still weighs 1/28.
- **A bootstrap interval for the headline.** `headline.ci95` is the percentile bootstrap 95 %
  interval of the macro mean: the 28 per-entity values are resampled with replacement 1,000 times
  with the run's seed, and the interval is the 2.5th and 97.5th percentiles of the resampled means
  (`tsguard.repro.report.bootstrap_ci`). It measures how much the number depends on which machines
  are in the benchmark. It does not include the variation from the model's or the injection's seed;
  the FFT ablation reports that spread separately, over five seeds.

## The FFT ablation

The ten FFT specs are removed from the registry and both feature models (z-score, Isolation Forest)
are selected and evaluated again with everything else unchanged: the same entities, tuned
parameters, folds and seeds. The comparison is event-level F1 at the frozen threshold and
event-level AUC-PR, each a macro mean per seed, reported over seeds 0 to 4 as the mean and sample
standard deviation of the absolute difference (in F1 points) and of the relative difference (in
percent). The autoencoders read raw channels, never the FFT features, so they are not ablated.
