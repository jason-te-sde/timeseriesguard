# 4. Every feature is a pure function of a trailing window of at most 64 rows

Status: accepted, 2026-11-03.

## Context

The same features are computed in two places. The experiment computes them for whole machines, in
blocks of rows so that a 28,000-row machine with 2,246 feature columns fits in memory. The service
computes them for one row from the few rows a request carries. A score is only meaningful if both
places compute the same value for the same row: the threshold was frozen on batch scores.

The obvious implementations do not give that. pandas' rolling mean, variance, skew and kurtosis use
running sums that are updated as the window slides, so the rounding error at row `t` depends on every
row the computation passed through before it. A recursive EWMA, `y[t] = a x[t] + (1 - a) y[t-1]`, has
infinite memory: its value depends on where the series started. Under either, a value computed from a
block, or from a request's few rows, differs from the value computed from the whole series. The first
design accepted that for the EWMA, required 512 rows of history per request, and compared service and
batch scores with a tolerance.

## Decision

Every spec's value at row `t` is a pure function of rows `t - lookback + 1 .. t`, with
`lookback <= MAX_LOOKBACK = 64`, and of nothing else:

- Rolling statistics reduce each `sliding_window_view` window independently. No running sums.
- The EWMA is windowed: a weighted mean of the last 64 rows with weights `(1 - a)^i`, divided by the
  sum of the weights.
- Skewness and kurtosis are the population moments of the window.
- A window's mean is taken about its last value, so a window of identical values has exactly zero
  deviation, and every spread statistic, skewness, kurtosis and FFT feature of it is exactly 0. Skewness
  and kurtosis are also 0 when their denominator underflows to zero for tiny but unequal values.
- The FFT features use a 64-row window with its mean removed.

`MAX_LOOKBACK` then sets three other numbers: each feature block is fed 63 rows of history, the
walk-forward purge is 64 rows, and a request to a feature model carries 64 rows.

## Consequences

- Computing in blocks changes no bit of any value, whatever the block size
  (`test_block_size_does_not_change_a_single_bit`), and a value computed from its window alone equals
  the value computed inside a long block (`test_spec_value_depends_only_on_its_window`).
- A feature-model request carries 64 rows, not 512, and the service's score for a row matches the
  batch score of that row (tested to a relative tolerance of 1e-6 for a feature model and an
  autoencoder; a loaded bundle reproduces the batch test scores bit for bit).
- `assert_causal` can treat the pipeline as a black box: with no hidden state, any dependence on a
  later row is a leak and nothing else.

Costs:

- **Compute.** Each window is reduced on its own, `O(w)` per row and statistic instead of `O(1)` for a
  running update. Large temporary arrays are avoided by processing windows in chunks.
- **The EWMA is not the textbook EWMA.** It forgets everything older than 64 rows and renormalizes the
  weights. For `a = 0.1` the dropped tail is `0.9^64`, about 0.1 % of the weight, so the two differ
  slightly; for `a = 0.3` and `0.7` the difference is negligible.
- **No feature looks back more than 64 minutes.** Daily and weekly patterns enter only through the four
  calendar features.
- **Bit-identity holds on one platform.** Another BLAS or libm can change the last bit, and
  `fft_dom_freq` takes an argmax over spectral bins that can be nearly tied. The service bundle
  therefore compares its feature fingerprint with a tolerance, and nothing claims cross-platform
  bit-identity.

## Rejected alternatives

**pandas running moments.** The fastest and the most familiar, but origin-dependent: the value at a row
depends on where the computation started, so block computation and serving cannot match the batch run
exactly, and every comparison needs a tolerance chosen by hand.

**A recursive EWMA with a long history.** Its value depends on the series origin. Serving it needs
hundreds of rows per request (512 in the first design, to make the truncated weight negligible), and the
batch and service values still differ, only by little. Blocks would have to carry the recursion's state
from one block to the next, so they are no longer independent.
