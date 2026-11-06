"""The registry of 63 named feature specs.

Every spec's value at row ``t`` is a pure function of the rows ``[t - lookback + 1, t]`` and
of nothing else: no running sums, no recursion from the start of the series. A value
therefore does not depend on where a computation began, so a pipeline can compute in
blocks, and a service that receives only the last 64 rows returns exactly the value a batch
run produced for the same row.

That rule decides how the statistics are computed. Rolling statistics reduce over
``numpy.lib.stride_tricks.sliding_window_view`` windows, one independent reduction per
window, never over pandas' running-moment algorithms, whose rounding error depends on the
whole history. The EWMA is windowed for the same reason. The window mean is taken about the
window's own last value, so a window of identical values has exactly zero deviation and
every spread statistic of it is exactly zero.

A channel spec maps a float64 block ``(m, c)`` to a float64 block ``(m, c)``; the first
``lookback - 1`` rows are NaN because the block holds too little history. An entity spec
maps the absolute minute index ``(m,)`` to ``(m,)`` and never emits NaN.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

BlockCache = dict[str, np.ndarray]
"""Scratch space shared by the specs of one block computation, keyed by family."""

MAX_LOOKBACK: Final = 64

_LAGS: Final = tuple(range(1, 11))
_ROLLING_WINDOWS: Final = (5, 15, 60)
_DEV_MEAN_WINDOWS: Final = (15, 60)
_DIFF_STEPS: Final = (1, 5, 15)
_EWMA_ALPHAS: Final = (0.1, 0.3, 0.7)
_QUANTILE_LEVELS: Final = ((10, 15), (90, 15), (10, 60), (90, 60))
_HOUR_PERIOD: Final = 1440
_WEEK_PERIOD: Final = 10080

_FFT_WINDOW: Final = MAX_LOOKBACK
_FFT_BINS: Final = 32  # bins 1..32 of the 33-bin rfft; bin 0 is the removed mean
_FFT_TOP: Final = 5
_FFT_BANDS: Final = ((0, 10), (10, 21), (21, 32))  # bins 1-10, 11-21, 22-32
_FFT_KEY: Final = "fft"
_FFT_NAMES: Final = (
    *(f"fft_mag_{i}" for i in range(1, _FFT_TOP + 1)),
    "fft_entropy",
    "fft_dom_freq",
    "fft_band_low",
    "fft_band_mid",
    "fft_band_high",
)

# Upper bound on the elements of one temporary window array; it keeps a 28,000-row block
# from materialising hundreds of megabytes per spec. Chunking changes no value, because
# every window is reduced on its own.
_CHUNK_ELEMENTS: Final = 1 << 22


@dataclass(frozen=True)
class FeatureSpec:
    """One named feature and the function that computes it.

    Attributes:
        name: Unique feature name, for example ``roll_mean_15``.
        family: The group the ablation drops together, for example ``rolling``.
        lookback: Number of rows, the current one included, the value at row ``t`` reads.
        scope: ``"channel"`` for a spec computed per channel from a float64 block
            ``(m, c)`` into ``(m, c)``; ``"entity"`` for a calendar spec computed from the
            absolute minute index ``(m,)`` into ``(m,)``.
        fn: The computation. Rows ``t < lookback - 1`` of a channel spec are NaN. The
            second argument is the scratch space shared by the specs of one block.
    """

    name: str
    family: str
    lookback: int
    scope: Literal["channel", "entity"]
    fn: Callable[[np.ndarray, BlockCache], np.ndarray]


# ---------------------------------------------------------------- window machinery


def _window_chunks(block: np.ndarray, w: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(first_row, windows)`` for the trailing windows of every channel.

    ``windows`` is a C-contiguous float64 array ``(c, n, w)``; window ``j`` of the chunk
    ends at block row ``w - 1 + first_row + j``. Contiguity makes each reduction over the
    last axis run the same inner loop whatever the number of windows, which is what keeps a
    value bit-identical between a long block and a single window.
    """
    n = block.shape[0] - w + 1
    if n <= 0:
        return
    channels = block.shape[1]
    windows = sliding_window_view(np.ascontiguousarray(block.T), w, axis=1)
    step = max(1, _CHUNK_ELEMENTS // (max(channels, 1) * w))
    for start in range(0, n, step):
        yield start, np.ascontiguousarray(windows[:, start : start + step])


def _reduce_windows(
    block: np.ndarray, w: int, reduce: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Apply ``reduce`` ``(c, n, w) -> (c, n)`` to every trailing window of ``block``."""
    out = np.full((block.shape[1], block.shape[0]), np.nan)
    for start, windows in _window_chunks(block, w):
        reduced = reduce(windows)
        out[:, w - 1 + start : w - 1 + start + reduced.shape[1]] = reduced
    return np.ascontiguousarray(out.T)


def _window_mean(windows: np.ndarray) -> np.ndarray:
    """Mean of each window, taken about the window's last value.

    ``last + mean(window - last)`` equals the plain mean up to rounding, and is exactly the
    shared value for a window of identical values, whose plain mean can miss it by one ulp.
    """
    last = windows[..., -1:]
    return np.asarray(last[..., 0] + (windows - last).mean(axis=-1))


def _deviations(windows: np.ndarray) -> np.ndarray:
    return np.asarray(windows - _window_mean(windows)[..., None])


def _std(windows: np.ndarray) -> np.ndarray:
    dev = _deviations(windows)
    return np.asarray(np.sqrt((dev * dev).mean(axis=-1)))


def _skew(windows: np.ndarray) -> np.ndarray:
    dev = _deviations(windows)
    d2 = dev * dev
    m2, m3 = d2.mean(axis=-1), (d2 * dev).mean(axis=-1)
    denominator = m2**1.5
    flat = denominator == 0.0  # m2 == 0, or so small that m2 ** 1.5 underflows: skew 0 either way
    return np.where(flat, 0.0, m3 / np.where(flat, 1.0, denominator))


def _kurt(windows: np.ndarray) -> np.ndarray:
    dev = _deviations(windows)
    d2 = dev * dev
    m2, m4 = d2.mean(axis=-1), (d2 * d2).mean(axis=-1)
    denominator = m2 * m2
    flat = denominator == 0.0  # m2 == 0, or so small that m2 * m2 underflows: kurtosis 0 either way
    return np.where(flat, 0.0, m4 / np.where(flat, 1.0, denominator) - 3.0)


def _range(windows: np.ndarray) -> np.ndarray:
    return np.asarray(windows.max(axis=-1) - windows.min(axis=-1))


def _dev_mean(windows: np.ndarray) -> np.ndarray:
    return np.asarray(windows[..., -1] - _window_mean(windows))


_ROLLING_STATS: Final[tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...]] = (
    ("mean", _window_mean),
    ("std", _std),
    ("min", lambda windows: np.asarray(windows.min(axis=-1))),
    ("max", lambda windows: np.asarray(windows.max(axis=-1))),
    ("median", lambda windows: np.asarray(np.median(windows, axis=-1))),
    ("range", _range),
)


def _rolling_spec(
    name: str, family: str, w: int, reduce: Callable[[np.ndarray], np.ndarray]
) -> FeatureSpec:
    def fn(block: np.ndarray, cache: BlockCache) -> np.ndarray:
        return _reduce_windows(block, w, reduce)

    return FeatureSpec(name, family, w, "channel", fn)


# ---------------------------------------------------------------- families


def _lag_spec(k: int) -> FeatureSpec:
    def fn(block: np.ndarray, cache: BlockCache) -> np.ndarray:
        out = np.full(block.shape, np.nan)
        if block.shape[0] > k:
            out[k:] = block[:-k]
        return out

    return FeatureSpec(f"lag_{k}", "lag", k + 1, "channel", fn)


def _diff1_spec(k: int) -> FeatureSpec:
    def fn(block: np.ndarray, cache: BlockCache) -> np.ndarray:
        out = np.full(block.shape, np.nan)
        if block.shape[0] > k:
            out[k:] = block[k:] - block[:-k]
        return out

    return FeatureSpec(f"diff1_{k}", "diff", k + 1, "channel", fn)


def _diff2_spec(k: int) -> FeatureSpec:
    def fn(block: np.ndarray, cache: BlockCache) -> np.ndarray:
        out = np.full(block.shape, np.nan)
        if block.shape[0] > 2 * k:
            out[2 * k :] = block[2 * k :] - 2.0 * block[k:-k] + block[: -2 * k]
        return out

    return FeatureSpec(f"diff2_{k}", "diff", 2 * k + 1, "channel", fn)


def _ewma_spec(alpha: float) -> FeatureSpec:
    # weights[j] multiplies the window row j (oldest first), i.e. (1 - alpha) ** (63 - j).
    weights = np.power(1.0 - alpha, np.arange(MAX_LOOKBACK - 1, -1, -1))
    weights.flags.writeable = False
    total = float(weights.sum())

    def reduce(windows: np.ndarray) -> np.ndarray:
        last = windows[..., -1:]  # shifting by the last value keeps a flat window exact
        return np.asarray(last[..., 0] + ((windows - last) * weights).sum(axis=-1) / total)

    name = f"ewma_{alpha:.1f}".replace(".", "p")
    return _rolling_spec(name, "ewma", MAX_LOOKBACK, reduce)


def _quantile_spec(percent: int, w: int) -> FeatureSpec:
    level = percent / 100.0

    def reduce(windows: np.ndarray) -> np.ndarray:
        return np.asarray(np.quantile(windows, level, axis=-1))

    return _rolling_spec(f"q{percent}_{w}", "quantile", w, reduce)


def _fft_features(block: np.ndarray, cache: BlockCache) -> np.ndarray:
    """Return the ten FFT features of every row as ``(10, c, m)``, computed once per block."""
    m, c = block.shape
    cached = cache.get(_FFT_KEY)
    if cached is not None:
        if cached.shape[1:] != (c, m):
            raise ValueError(
                f"the cache entry {_FFT_KEY!r} holds spectra for a block of "
                f"{cached.shape[2]} rows x {cached.shape[1]} channels, "
                f"but this block has {m} rows x {c} channels; use one cache per block"
            )
        return cached
    out = np.full((len(_FFT_NAMES), c, m), np.nan)
    for start, windows in _window_chunks(block, _FFT_WINDOW):
        frames = windows - _window_mean(windows)[..., None]
        spectrum = np.fft.rfft(frames, axis=-1)[..., 1 : _FFT_BINS + 1]
        power = spectrum.real**2 + spectrum.imag**2
        magnitude = np.abs(spectrum)
        top = np.sort(np.partition(magnitude, _FFT_BINS - _FFT_TOP, axis=-1)[..., -_FFT_TOP:])
        total = power.sum(axis=-1)
        live = total > 0.0  # a window with no variation has no spectrum: every feature is 0
        safe = np.where(live, total, 1.0)
        share = power / safe[..., None]
        log_share = np.log(np.where(share > 0.0, share, 1.0))
        features = [
            *(top[..., -i] for i in range(1, _FFT_TOP + 1)),
            0.0 - (share * log_share).sum(axis=-1),
            np.where(live, (power.argmax(axis=-1) + 1) / _FFT_WINDOW, 0.0),
            *(np.where(live, power[..., a:b].sum(axis=-1) / safe, 0.0) for a, b in _FFT_BANDS),
        ]
        stop = start + features[0].shape[1]
        out[:, :, _FFT_WINDOW - 1 + start : _FFT_WINDOW - 1 + stop] = np.stack(features)
    cache[_FFT_KEY] = out
    return out


def _fft_spec(index: int, name: str) -> FeatureSpec:
    def fn(block: np.ndarray, cache: BlockCache) -> np.ndarray:
        return np.asarray(_fft_features(block, cache)[index].T.copy())

    return FeatureSpec(name, "fft", _FFT_WINDOW, "channel", fn)


def _calendar_spec(name: str, period: int, trig: Callable[[np.ndarray], np.ndarray]) -> FeatureSpec:
    def fn(minutes: np.ndarray, cache: BlockCache) -> np.ndarray:
        return np.asarray(trig(2.0 * np.pi * np.mod(minutes, period) / period))

    return FeatureSpec(name, "calendar", 1, "entity", fn)


CHANNEL_SPECS: Final[tuple[FeatureSpec, ...]] = (
    *(_lag_spec(k) for k in _LAGS),
    *(
        _rolling_spec(f"roll_{stat}_{w}", "rolling", w, reduce)
        for stat, reduce in _ROLLING_STATS
        for w in _ROLLING_WINDOWS
    ),
    *(_rolling_spec(f"dev_mean_{w}", "rolling", w, _dev_mean) for w in _DEV_MEAN_WINDOWS),
    *(_rolling_spec(f"roll_skew_{w}", "rolling_higher", w, _skew) for w in _ROLLING_WINDOWS),
    *(_rolling_spec(f"roll_kurt_{w}", "rolling_higher", w, _kurt) for w in _ROLLING_WINDOWS),
    *(_diff1_spec(k) for k in _DIFF_STEPS),
    *(_diff2_spec(k) for k in _DIFF_STEPS),
    *(_fft_spec(index, name) for index, name in enumerate(_FFT_NAMES)),
    *(_ewma_spec(alpha) for alpha in _EWMA_ALPHAS),
    *(_quantile_spec(percent, w) for percent, w in _QUANTILE_LEVELS),
)
"""The 59 per-channel specs, in registry order."""

CALENDAR_SPECS: Final[tuple[FeatureSpec, ...]] = (
    _calendar_spec("cal_hod_sin", _HOUR_PERIOD, np.sin),
    _calendar_spec("cal_hod_cos", _HOUR_PERIOD, np.cos),
    _calendar_spec("cal_dow_sin", _WEEK_PERIOD, np.sin),
    _calendar_spec("cal_dow_cos", _WEEK_PERIOD, np.cos),
)
"""The 4 calendar specs, computed from the absolute minute index."""

FEATURE_SPECS: Final = CHANNEL_SPECS + CALENDAR_SPECS
"""All 63 specs: the channel specs followed by the calendar specs."""


def specs_without(family: str) -> tuple[FeatureSpec, ...]:
    """Return the registry with one family removed, for the ablation.

    Args:
        family: A family name present in the registry, for example ``"fft"``.

    Returns:
        The remaining specs, in registry order.

    Raises:
        ValueError: ``family`` names no spec; a misspelt family would otherwise return the
            full registry and pass off an unablated run as an ablation.
    """
    families = sorted({spec.family for spec in FEATURE_SPECS})
    if family not in families:
        raise ValueError(f"unknown feature family {family!r}; the families are {families}")
    return tuple(spec for spec in FEATURE_SPECS if spec.family != family)
