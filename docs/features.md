# Feature registry

TimeSeriesGuard turns each 38-channel SMD series into 63 named feature specs: 59 are computed per channel and 4 calendar specs once per entity, so a full feature row has 59 x 38 + 4 = 2,246 columns. The registry lives in `src/tsguard/features/specs.py` and the table below lists it in registry order. Notation: `x[t]` is the channel value at row `t`, and `lookback` counts the rows a value reads, the current one included. Every value at row `t` is a pure function of rows `t - lookback + 1 .. t` and nothing else (no running sums, no recursion from the start of the series), which is why a block-wise batch computation and a request that carries only the last 64 rows give bit-identical values; the first `lookback - 1` rows of a block are NaN, and calendar specs never are. A "flat" window is one whose values are all identical; its spread statistics (standard deviation, range, skew, kurtosis, deviation from the mean) are exactly 0, and so are its FFT features, never NaN. For the FFT family, `p` is the squared magnitude of the real FFT of the window after its mean is removed, and bin `k` is the frequency `k / 64` cycles per row.

| name | family | lookback | definition |
| --- | --- | --- | --- |
| lag_1 | lag | 2 | x[t-1] |
| lag_2 | lag | 3 | x[t-2] |
| lag_3 | lag | 4 | x[t-3] |
| lag_4 | lag | 5 | x[t-4] |
| lag_5 | lag | 6 | x[t-5] |
| lag_6 | lag | 7 | x[t-6] |
| lag_7 | lag | 8 | x[t-7] |
| lag_8 | lag | 9 | x[t-8] |
| lag_9 | lag | 10 | x[t-9] |
| lag_10 | lag | 11 | x[t-10] |
| roll_mean_5 | rolling | 5 | Mean of x[t-4 .. t] |
| roll_mean_15 | rolling | 15 | Mean of x[t-14 .. t] |
| roll_mean_60 | rolling | 60 | Mean of x[t-59 .. t] |
| roll_std_5 | rolling | 5 | Population standard deviation (ddof 0) of x[t-4 .. t] |
| roll_std_15 | rolling | 15 | Population standard deviation (ddof 0) of x[t-14 .. t] |
| roll_std_60 | rolling | 60 | Population standard deviation (ddof 0) of x[t-59 .. t] |
| roll_min_5 | rolling | 5 | Minimum of x[t-4 .. t] |
| roll_min_15 | rolling | 15 | Minimum of x[t-14 .. t] |
| roll_min_60 | rolling | 60 | Minimum of x[t-59 .. t] |
| roll_max_5 | rolling | 5 | Maximum of x[t-4 .. t] |
| roll_max_15 | rolling | 15 | Maximum of x[t-14 .. t] |
| roll_max_60 | rolling | 60 | Maximum of x[t-59 .. t] |
| roll_median_5 | rolling | 5 | Median of x[t-4 .. t] |
| roll_median_15 | rolling | 15 | Median of x[t-14 .. t] |
| roll_median_60 | rolling | 60 | Median of x[t-59 .. t] |
| roll_range_5 | rolling | 5 | roll_max_5 - roll_min_5 |
| roll_range_15 | rolling | 15 | roll_max_15 - roll_min_15 |
| roll_range_60 | rolling | 60 | roll_max_60 - roll_min_60 |
| dev_mean_15 | rolling | 15 | x[t] - roll_mean_15 |
| dev_mean_60 | rolling | 60 | x[t] - roll_mean_60 |
| roll_skew_5 | rolling_higher | 5 | m3 / m2^1.5 of x[t-4 .. t], m_k the k-th central moment of the window (population); 0 for a flat window |
| roll_skew_15 | rolling_higher | 15 | m3 / m2^1.5 of x[t-14 .. t], m_k the k-th central moment of the window (population); 0 for a flat window |
| roll_skew_60 | rolling_higher | 60 | m3 / m2^1.5 of x[t-59 .. t], m_k the k-th central moment of the window (population); 0 for a flat window |
| roll_kurt_5 | rolling_higher | 5 | m4 / m2^2 - 3 of x[t-4 .. t], population central moments; 0 for a flat window |
| roll_kurt_15 | rolling_higher | 15 | m4 / m2^2 - 3 of x[t-14 .. t], population central moments; 0 for a flat window |
| roll_kurt_60 | rolling_higher | 60 | m4 / m2^2 - 3 of x[t-59 .. t], population central moments; 0 for a flat window |
| diff1_1 | diff | 2 | x[t] - x[t-1] |
| diff1_5 | diff | 6 | x[t] - x[t-5] |
| diff1_15 | diff | 16 | x[t] - x[t-15] |
| diff2_1 | diff | 3 | x[t] - 2 x[t-1] + x[t-2] |
| diff2_5 | diff | 11 | x[t] - 2 x[t-5] + x[t-10] |
| diff2_15 | diff | 31 | x[t] - 2 x[t-15] + x[t-30] |
| fft_mag_1 | fft | 64 | 1st largest magnitude among rfft bins 1-32 of the mean-removed window x[t-63 .. t] |
| fft_mag_2 | fft | 64 | 2nd largest magnitude among rfft bins 1-32 of the mean-removed window x[t-63 .. t] |
| fft_mag_3 | fft | 64 | 3rd largest magnitude among rfft bins 1-32 of the mean-removed window x[t-63 .. t] |
| fft_mag_4 | fft | 64 | 4th largest magnitude among rfft bins 1-32 of the mean-removed window x[t-63 .. t] |
| fft_mag_5 | fft | 64 | 5th largest magnitude among rfft bins 1-32 of the mean-removed window x[t-63 .. t] |
| fft_entropy | fft | 64 | -sum(q ln q) with q = p / sum(p), p = squared magnitude of rfft bins 1-32; 0 when sum(p) = 0 |
| fft_dom_freq | fft | 64 | (bin index of the largest p among bins 1-32) / 64, in cycles per row; 0 when sum(p) = 0 |
| fft_band_low | fft | 64 | sum of p over bins 1-10, divided by sum(p); 0 when sum(p) = 0 |
| fft_band_mid | fft | 64 | sum of p over bins 11-21, divided by sum(p); 0 when sum(p) = 0 |
| fft_band_high | fft | 64 | sum of p over bins 22-32, divided by sum(p); 0 when sum(p) = 0 |
| ewma_0p1 | ewma | 64 | windowed EWMA, a = 0.1: sum over i = 0..63 of (1-a)^i x[t-i], divided by sum over i = 0..63 of (1-a)^i |
| ewma_0p3 | ewma | 64 | windowed EWMA, a = 0.3: sum over i = 0..63 of (1-a)^i x[t-i], divided by sum over i = 0..63 of (1-a)^i |
| ewma_0p7 | ewma | 64 | windowed EWMA, a = 0.7: sum over i = 0..63 of (1-a)^i x[t-i], divided by sum over i = 0..63 of (1-a)^i |
| q10_15 | quantile | 15 | 10th percentile of x[t-14 .. t] (linear interpolation) |
| q90_15 | quantile | 15 | 90th percentile of x[t-14 .. t] (linear interpolation) |
| q10_60 | quantile | 60 | 10th percentile of x[t-59 .. t] (linear interpolation) |
| q90_60 | quantile | 60 | 90th percentile of x[t-59 .. t] (linear interpolation) |
| cal_hod_sin | calendar | 1 | sin(2 pi (minute mod 1440) / 1440), minute the absolute minute index |
| cal_hod_cos | calendar | 1 | cos(2 pi (minute mod 1440) / 1440) |
| cal_dow_sin | calendar | 1 | sin(2 pi (minute mod 10080) / 10080) |
| cal_dow_cos | calendar | 1 | cos(2 pi (minute mod 10080) / 10080) |
