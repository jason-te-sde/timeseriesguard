# Lessons

What building this project taught that was specific to it: each entry is a thing that went wrong or
nearly did, what it looked like, and the rule it left behind in the code.

- **A near-constant channel turns noise into a 1e5-sigma event.** machine-1-1 has a channel whose
  standard deviation over the train rows is about 3e-6. Its first real movement in the test rows
  standardized to values around 1e5 and would have decided every detector's score. SMD is min-max
  scaled to [0, 1], so a change below 0.1 % of a channel's range is noise: every scale is now at least
  `MIN_SCALE = 1e-3`.
- **Scale-free features ignore any scale floor.** Windowed kurtosis of a channel that is flat in
  training reaches 1e4 to 1e5 standard deviations in the test rows even with the floor, because
  kurtosis does not shrink with the values. Standardized features are clipped to `±50`, a bound
  fixed before any model ran.
- **The mean of identical numbers is not always that number.** The arithmetic mean of many copies of
  0.1 misses 0.1 by about 1e-15, so a column that is constant in training standardized to noise
  instead of to zero. The fitted mean of a constant column is now the constant itself, and window
  means are taken about the window's last value for the same reason.
- **A robust scale can be sharper than the data.** For a column that is almost always constant with
  rare spikes, the median absolute deviation is close to 0 while the standard deviation is not, and
  the z-score detector produced scores and frozen thresholds in the thousands. Its scale is now at
  least one training standard deviation.
- **Feature bagging is not free at 2,246 columns.** Isolation Forest with `max_features < 1.0` copies
  a column subset for every tree. Measured with 200 trees, fitting 24,000 rows and scoring 28,000,
  against `max_features=1.0`: `0.5` was 73 times slower to fit and 39 times slower to score, `0.1` 15
  and 8 times. The forest already picks a random feature at every split, so `max_features` is fixed
  at 1.0 and is not tuned.
- **A guard test can pass for the wrong reason.** The first versions of the backward-fill and
  interpolation defects opened their gaps with a whole-block median. `assert_causal` caught them, but
  for reading the median, not for the fill direction, and a forward fill behind the same mask was
  flagged too. The masks now read only the value itself, and a forward-fill control must stay
  unflagged.
- **Underflow reaches the service before it reaches the data.** A request whose values differ only
  by about 1e-100 made `m2^1.5` and `m2^2` underflow to zero, skewness and kurtosis became NaN, the
  pipeline refused the NaN, and the client got a 500. A window whose denominator underflows is now
  treated as flat. No SMD window is anywhere near this; a request can be.
- **An error body that echoes the input can fail to serialize.** FastAPI's default 422 body repeats
  the rejected input, and a `NaN` token in a request made that body impossible to write as JSON, so
  a validation error became a 500. Schema errors are now reported as type, location and message
  only.
- **An import can keep gigabytes alive.** Importing Optuna for the first time inside the tuning
  function left a traceback in module state that pinned the function's frame, and with it the
  tuning runners' 2 to 3 GB of cached features each, for the rest of a multi-hour run. `reproduce`
  now imports Optuna before any runner exists.
- **Ordering is part of a leakage claim.** The first `reproduce` chose the headline model after the
  evaluation. The choice read no label and gave the same model, yet it broke the claim that the
  choice precedes every test label, and nothing could show it did not depend on them. The tests now
  pin that the choice is made once, while every audit still shows zero reads, and they fail when the
  choice is moved after the evaluation, not only when it is removed.
- **Label-free selection has a price, and the table shows it.** In the run committed on 2026-11-04
  the validation objective chose the z-score (0.671, next best 0.542), while the LSTM autoencoder
  scored higher on the test labels (event-level AUC-PR 0.73 against 0.70). Choosing by the test
  score would have reported the higher number and made it meaningless as an estimate. Every model
  is reported under every protocol, so the gap stays visible instead of being chosen away.
