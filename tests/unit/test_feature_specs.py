from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from tsguard.features import specs as specs_module
from tsguard.features.specs import (
    CALENDAR_SPECS,
    CHANNEL_SPECS,
    FEATURE_SPECS,
    MAX_LOOKBACK,
    FeatureSpec,
    specs_without,
)

FEATURES_DOC = Path(__file__).resolve().parents[2] / "docs" / "features.md"
WINDOW = 64
FFT_BINS = 32


def spec_named(name: str) -> FeatureSpec:
    for spec in FEATURE_SPECS:
        if spec.name == name:
            return spec
    raise AssertionError(f"no spec named {name!r}")


def value(name: str, x: np.ndarray) -> np.ndarray:
    """Run one channel spec on ``x`` of shape (m, c)."""
    return spec_named(name).fn(x, {})


def fft_value(name: str, x: np.ndarray) -> np.ndarray:
    """Run one spec on the single series ``x`` and return its (m,) output."""
    return spec_named(name).fn(x[:, None], {})[:, 0]


def inputs_for(spec: FeatureSpec, rng: np.random.Generator, rows: int = 300) -> np.ndarray:
    if spec.scope == "entity":
        return np.arange(rows, dtype=np.float64) + 12_345.0
    return rng.random((rows, 3))


def tone(cycles: int, rows: int = 256, phase: float = 0.3) -> np.ndarray:
    """A cosine with ``cycles`` whole periods in every 64-row window."""
    return np.cos(2.0 * np.pi * cycles * np.arange(rows) / WINDOW + phase)


# ---------------------------------------------------------------- registry


def test_registry_has_63_specs_and_at_least_60():
    assert len(FEATURE_SPECS) == 63
    assert len(FEATURE_SPECS) >= 60


def test_channel_and_calendar_split():
    assert len(CHANNEL_SPECS) == 59
    assert len(CALENDAR_SPECS) == 4
    assert FEATURE_SPECS == CHANNEL_SPECS + CALENDAR_SPECS
    assert {s.scope for s in CHANNEL_SPECS} == {"channel"}
    assert {s.scope for s in CALENDAR_SPECS} == {"entity"}


def test_family_counts():
    assert Counter(s.family for s in FEATURE_SPECS) == {
        "lag": 10,
        "rolling": 20,
        "rolling_higher": 6,
        "diff": 6,
        "fft": 10,
        "ewma": 3,
        "quantile": 4,
        "calendar": 4,
    }


def test_names_are_unique():
    names = [s.name for s in FEATURE_SPECS]
    assert len(set(names)) == len(names)


def test_max_lookback_is_64():
    assert max(s.lookback for s in FEATURE_SPECS) == MAX_LOOKBACK == 64


def expected_lookback(name: str) -> int:
    parts = name.split("_")
    if name.startswith("lag_") or name.startswith("diff1_"):
        return int(parts[-1]) + 1
    if name.startswith("diff2_"):
        return 2 * int(parts[-1]) + 1
    if name.startswith(("fft_", "ewma_")):
        return 64
    if name.startswith("cal_"):
        return 1
    return int(parts[-1])  # roll_*, dev_mean_*, q10_*, q90_*: the window length


@pytest.mark.parametrize("spec", FEATURE_SPECS, ids=lambda s: s.name)
def test_declared_lookback_follows_the_naming_rule(spec):
    assert spec.lookback == expected_lookback(spec.name)


# ---------------------------------------------------------------- causality and shape


@pytest.mark.parametrize("spec", FEATURE_SPECS, ids=lambda s: s.name)
def test_spec_is_causal(spec, rng):
    x = inputs_for(spec, rng)
    perturbed = x.copy()
    perturbed[200:] = rng.random(perturbed[200:].shape) * 1000.0
    before = spec.fn(x, {})
    after = spec.fn(perturbed, {})
    assert np.array_equal(before[:200], after[:200], equal_nan=True)


@pytest.mark.parametrize("spec", CHANNEL_SPECS, ids=lambda s: s.name)
def test_spec_value_depends_only_on_its_window(spec, rng):
    x = rng.random((300, 3))
    x[:, 2] = 0.3  # a constant channel whose mean does not round-trip exactly
    whole = spec.fn(x, {})
    for t in (100, 250):
        window = x[t - spec.lookback + 1 : t + 1]
        alone = spec.fn(window, {})[-1]
        assert np.array_equal(whole[t], alone), f"row {t} of {spec.name} reads outside its window"


@pytest.mark.parametrize("spec", FEATURE_SPECS, ids=lambda s: s.name)
def test_spec_output_shape_dtype_and_warmup(spec, rng):
    x = inputs_for(spec, rng, rows=150)
    if spec.scope == "channel":
        x[:, 0] = 0.3
    x.flags.writeable = False  # a spec must neither need nor attempt to write into its input
    out = spec.fn(x, {})
    assert out.dtype == np.float64
    assert out.shape == x.shape
    warmup = spec.lookback - 1
    assert np.isnan(out[:warmup]).all()
    assert np.isfinite(out[warmup:]).all()


@pytest.mark.parametrize("spec", CHANNEL_SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("short", [0, 1])
def test_block_shorter_than_the_lookback_is_all_nan(spec, short, rng):
    rows = 0 if short == 0 else spec.lookback - 1
    out = spec.fn(rng.random((rows, 3)), {})
    assert out.shape == (rows, 3)
    assert np.isnan(out).all()


def test_chunking_does_not_change_a_value(rng, monkeypatch):
    block = rng.random((200, 3))
    block[:, 1] = 0.3
    expected = {spec.name: spec.fn(block, {}) for spec in CHANNEL_SPECS}
    monkeypatch.setattr(specs_module, "_CHUNK_ELEMENTS", 50)  # one window per chunk
    for spec in CHANNEL_SPECS:
        assert np.array_equal(spec.fn(block, {}), expected[spec.name], equal_nan=True), spec.name


# ---------------------------------------------------------------- lag and diff


def test_lag_values():
    x = np.arange(10.0)[:, None]
    out = value("lag_3", x)[:, 0]
    np.testing.assert_array_equal(out, [np.nan, np.nan, np.nan, 0, 1, 2, 3, 4, 5, 6])


def test_diff1_values():
    x = np.arange(40.0)[:, None] * 2.0
    out = value("diff1_5", x)[:, 0]
    assert np.isnan(out[:5]).all()
    assert (out[5:] == 10.0).all()


def test_diff2_values():
    x = (np.arange(40.0) ** 2)[:, None]
    assert (value("diff2_1", x)[2:, 0] == 2.0).all()
    assert (value("diff2_5", x)[10:, 0] == 50.0).all()
    assert (value("diff2_15", x)[30:, 0] == 450.0).all()


# ---------------------------------------------------------------- rolling


def test_roll_mean_matches_manual(rng):
    x = rng.random((50, 2))
    expected = x[6:11].mean(axis=0)
    np.testing.assert_allclose(value("roll_mean_5", x)[10], expected, rtol=1e-12)


@pytest.mark.parametrize("w", [5, 15, 60])
def test_rolling_statistics_match_manual(w, rng):
    x = rng.random((120, 2))
    t = 100
    win = x[t - w + 1 : t + 1]
    np.testing.assert_allclose(value(f"roll_std_{w}", x)[t], win.std(axis=0), rtol=1e-10)
    np.testing.assert_array_equal(value(f"roll_min_{w}", x)[t], win.min(axis=0))
    np.testing.assert_array_equal(value(f"roll_max_{w}", x)[t], win.max(axis=0))
    np.testing.assert_allclose(value(f"roll_median_{w}", x)[t], np.median(win, axis=0), rtol=1e-12)
    np.testing.assert_allclose(
        value(f"roll_range_{w}", x)[t], win.max(axis=0) - win.min(axis=0), rtol=1e-12
    )


@pytest.mark.parametrize("w", [5, 15, 60])
def test_skew_and_kurt_are_population_moments(w, rng):
    x = rng.random((120, 2)) ** 3
    t = 100
    win = x[t - w + 1 : t + 1]
    dev = win - win.mean(axis=0)
    m2, m3, m4 = (dev**2).mean(axis=0), (dev**3).mean(axis=0), (dev**4).mean(axis=0)
    np.testing.assert_allclose(value(f"roll_skew_{w}", x)[t], m3 / m2**1.5, rtol=1e-9)
    np.testing.assert_allclose(value(f"roll_kurt_{w}", x)[t], m4 / m2**2 - 3.0, rtol=1e-9)


@pytest.mark.parametrize("constant", [0.3, 0.1, 0.7, 1.0 / 3.0])
@pytest.mark.parametrize("name", [f"roll_{k}_{w}" for k in ("skew", "kurt") for w in (5, 15, 60)])
def test_flat_window_skew_and_kurt_are_zero(name, constant):
    x = np.full((100, 2), constant)
    out = value(name, x)
    warmup = spec_named(name).lookback - 1
    assert (out[warmup:] == 0.0).all()


# With deviations of about 1e-100 the second moment is 1e-200 and its square underflows to 0, so
# the ratio is 0/0. A window is flat when the denominator itself is zero. SMD never gets near
# this (its smallest nonzero step is 1e-6); the service accepts any magnitude up to 1e6 in
# either direction, so the registry has to be finite everywhere in between.
DENOMINATOR_UNDERFLOWS = {("roll_kurt", -100), ("roll_kurt", -160), ("roll_skew", -160)}
SKEW_AND_KURT = [f"roll_{k}_{w}" for k in ("skew", "kurt") for w in (5, 15, 60)]


@pytest.mark.parametrize("exponent", [-80, -100, -160])
@pytest.mark.parametrize("name", SKEW_AND_KURT)
def test_tiny_magnitude_window_skew_and_kurt_are_finite(name, exponent, rng):
    x = 10.0**exponent * rng.standard_normal((100, 2))
    out = value(name, x)[spec_named(name).lookback - 1 :]
    assert np.isfinite(out).all(), (
        f"{name} at magnitude 1e{exponent} has {(~np.isfinite(out)).sum()} non-finite values"
    )
    if (name.rsplit("_", 1)[0], exponent) in DENOMINATOR_UNDERFLOWS:
        assert (out == 0.0).all(), (
            f"{name} at 1e{exponent}: a window whose denominator is 0 is flat"
        )


@pytest.mark.parametrize("exponent", [-20, -60])
@pytest.mark.parametrize("name", SKEW_AND_KURT)
def test_small_but_representable_windows_keep_their_moments(name, exponent, rng):
    unit = rng.standard_normal((100, 2))
    start = spec_named(name).lookback - 1
    scaled = value(name, 10.0**exponent * unit)[start:]
    reference = value(name, unit)[start:]
    assert (reference != 0.0).any()
    np.testing.assert_allclose(scaled, reference, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("constant", [0.3, 0.1, 0.7, 1.0 / 3.0])
def test_flat_window_is_exact_for_the_other_statistics(constant):
    x = np.full((100, 2), constant)
    for w in (5, 15, 60):
        assert (value(f"roll_mean_{w}", x)[w - 1 :] == constant).all()
        assert (value(f"roll_std_{w}", x)[w - 1 :] == 0.0).all()
        assert (value(f"roll_range_{w}", x)[w - 1 :] == 0.0).all()
    for w in (15, 60):
        assert (value(f"dev_mean_{w}", x)[w - 1 :] == 0.0).all()
    for name in ("q10_15", "q90_60", "roll_median_15", "ewma_0p1", "ewma_0p3", "ewma_0p7"):
        spec = spec_named(name)
        assert (spec.fn(x, {})[spec.lookback - 1 :] == constant).all(), name


def test_dev_mean_is_x_minus_mean(rng):
    x = rng.random((100, 3))
    for w in (15, 60):
        expected = x - value(f"roll_mean_{w}", x)
        np.testing.assert_allclose(value(f"dev_mean_{w}", x), expected, atol=1e-15, equal_nan=True)


@pytest.mark.parametrize("name", ["q10_15", "q90_15", "q10_60", "q90_60"])
def test_quantiles_match_numpy_with_linear_interpolation(name, rng):
    x = rng.random((120, 2))
    w, q = int(name.split("_")[1]), int(name[1:3]) / 100.0
    t = 100
    expected = np.quantile(x[t - w + 1 : t + 1], q, axis=0)
    np.testing.assert_allclose(value(name, x)[t], expected, rtol=1e-12)


# ---------------------------------------------------------------- ewma


@pytest.mark.parametrize(
    ("name", "alpha"), [("ewma_0p1", 0.1), ("ewma_0p3", 0.3), ("ewma_0p7", 0.7)]
)
def test_windowed_ewma_matches_formula(name, alpha, rng):
    x = rng.random((200, 2))
    t = 100
    weights = np.array([(1.0 - alpha) ** i for i in range(64)])
    expected = sum(weights[i] * x[t - i] for i in range(64)) / weights.sum()
    np.testing.assert_allclose(value(name, x)[t], expected, rtol=1e-12)


# ---------------------------------------------------------------- fft


def test_fft_dominant_frequency_of_a_pure_tone():
    x = np.sin(2 * np.pi * 8 * np.arange(256) / 64)  # 8 cycles per 64-step window
    assert fft_value("fft_dom_freq", x)[-1] == pytest.approx(8 / 64)
    assert fft_value("fft_band_low", x)[-1] == pytest.approx(1.0, abs=1e-9)


def test_fft_magnitude_and_entropy_of_a_pure_tone():
    x = np.sin(2 * np.pi * 8 * np.arange(256) / 64)
    assert fft_value("fft_mag_1", x)[-1] == pytest.approx(32.0, rel=1e-9)  # N / 2
    assert fft_value("fft_mag_2", x)[-1] == pytest.approx(0.0, abs=1e-9)
    assert fft_value("fft_entropy", x)[-1] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("bin_", "band"),
    [
        (1, "low"),
        (10, "low"),
        (11, "mid"),
        (21, "mid"),
        (22, "high"),
        (32, "high"),  # the Nyquist bin belongs to the spectrum
    ],
)
def test_fft_bands_split_bins_1_10_11_21_22_32(bin_, band):
    x = tone(bin_)
    assert fft_value("fft_dom_freq", x)[-1] == pytest.approx(bin_ / 64)
    for name in ("low", "mid", "high"):
        expected = 1.0 if name == band else 0.0
        assert fft_value(f"fft_band_{name}", x)[-1] == pytest.approx(expected, abs=1e-9)


def test_fft_of_an_impulse_has_a_flat_spectrum():
    x = np.zeros(256)
    x[200] = 1.0  # inside the last window, rows 192..255
    assert fft_value("fft_entropy", x)[-1] == pytest.approx(np.log(FFT_BINS), rel=1e-9)
    assert fft_value("fft_band_low", x)[-1] == pytest.approx(10 / 32)
    assert fft_value("fft_band_mid", x)[-1] == pytest.approx(11 / 32)
    assert fft_value("fft_band_high", x)[-1] == pytest.approx(11 / 32)


@pytest.mark.parametrize("constant", [0.0, 0.3, 0.1, 1.0])
def test_fft_of_constant_series_is_zero_not_nan(constant):
    x = np.full(200, constant)
    for spec in CHANNEL_SPECS:
        if spec.family == "fft":
            out = fft_value(spec.name, x)
            assert (out[63:] == 0.0).all(), spec.name


def test_fft_magnitudes_are_sorted_descending(rng):
    x = rng.random(300)
    mags = np.stack([fft_value(f"fft_mag_{i}", x) for i in range(1, 6)], axis=1)[63:]
    assert (np.diff(mags, axis=1) <= 0.0).all()
    t = 100
    window = x[t - 63 : t + 1]
    spectrum = np.abs(np.fft.rfft(window - window.mean()))[1:33]
    np.testing.assert_allclose(mags[t - 63], np.sort(spectrum)[::-1][:5], rtol=1e-9)


def test_fft_entropy_and_dominant_bin_match_manual(rng):
    x = rng.random(300)
    t = 150
    window = x[t - 63 : t + 1]
    power = np.abs(np.fft.rfft(window - window.mean()))[1:33] ** 2
    q = power / power.sum()
    assert fft_value("fft_entropy", x)[t] == pytest.approx(-(q * np.log(q)).sum(), rel=1e-9)
    assert fft_value("fft_dom_freq", x)[t] == pytest.approx((power.argmax() + 1) / 64)
    low, mid, high = power[:10].sum(), power[10:21].sum(), power[21:].sum()
    total = power.sum()
    assert fft_value("fft_band_low", x)[t] == pytest.approx(low / total, rel=1e-9)
    assert fft_value("fft_band_mid", x)[t] == pytest.approx(mid / total, rel=1e-9)
    assert fft_value("fft_band_high", x)[t] == pytest.approx(high / total, rel=1e-9)


def test_fft_family_shares_one_computation_per_block(rng):
    block = rng.random((200, 3))
    fft_specs = [s for s in CHANNEL_SPECS if s.family == "fft"]
    cache: dict[str, np.ndarray] = {}
    shared = [spec.fn(block, cache) for spec in fft_specs]
    assert list(cache) == ["fft"]
    spectrum = cache["fft"]
    for spec in fft_specs:
        spec.fn(block, cache)
    assert cache["fft"] is spectrum  # later specs read the cached spectrum; nothing recomputed
    for spec, got in zip(fft_specs, shared, strict=True):
        assert np.array_equal(got, spec.fn(block, {}), equal_nan=True)


def test_fft_cache_from_another_block_is_refused(rng):
    spec = spec_named("fft_mag_1")
    cache: dict[str, np.ndarray] = {}
    spec.fn(rng.random((100, 3)), cache)
    with pytest.raises(ValueError, match="fft"):
        spec.fn(rng.random((120, 3)), cache)


# ---------------------------------------------------------------- calendar


def test_calendar_period():
    minutes = np.array([0.0, 1440.0, 10080.0, 720.0, 2520.0, 7_000_000.0])
    hod_sin = spec_named("cal_hod_sin").fn(minutes, {})
    hod_cos = spec_named("cal_hod_cos").fn(minutes, {})
    dow_sin = spec_named("cal_dow_sin").fn(minutes, {})
    dow_cos = spec_named("cal_dow_cos").fn(minutes, {})
    assert hod_sin[0] == hod_sin[1]
    assert dow_cos[0] == 1.0
    assert dow_sin[0] == dow_sin[2]
    assert dow_cos[0] == dow_cos[2]
    assert hod_cos[3] == pytest.approx(-1.0)  # noon
    assert dow_sin[4] == pytest.approx(1.0)  # a quarter of the week
    assert hod_sin[5] == pytest.approx(np.sin(2 * np.pi * (7_000_000 % 1440) / 1440))


# ---------------------------------------------------------------- specs_without


def test_specs_without_fft_drops_exactly_ten():
    kept = specs_without("fft")
    assert len(kept) == 53
    assert all(s.family != "fft" for s in kept)
    assert kept == tuple(s for s in FEATURE_SPECS if s.family != "fft")


def test_specs_without_an_unknown_family_is_refused():
    with pytest.raises(ValueError, match="no_such_family"):
        specs_without("no_such_family")


# ---------------------------------------------------------------- docs


def parse_doc_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for line in FEATURES_DOC.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells[0] in ("name", "") or set(cells[0]) <= {"-", ":"}:
            continue
        rows.append(cells)
    return rows


def test_docs_table_matches_registry():
    rows = parse_doc_rows()
    assert [row[0] for row in rows] == [s.name for s in FEATURE_SPECS]


def test_docs_table_states_family_lookback_and_a_definition():
    rows = parse_doc_rows()
    for row, spec in zip(rows, FEATURE_SPECS, strict=True):
        assert len(row) == 4, spec.name
        assert row[1] == spec.family, spec.name
        assert row[2] == str(spec.lookback), spec.name
        assert row[3], spec.name
