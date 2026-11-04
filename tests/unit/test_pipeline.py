from __future__ import annotations

import warnings

import numpy as np
import pytest

from tests.conftest import CONSTANT_CHANNEL, make_entity
from tsguard.errors import GUARD_FIT_END, LeakageError
from tsguard.features.pipeline import RAW_SPECS, FeaturePipeline, compute_raw_features
from tsguard.features.specs import FEATURE_SPECS, MAX_LOOKBACK, FeatureSpec

N_CHANNELS = 38
N_COLUMNS = 59 * N_CHANNELS + 4


@pytest.fixture(scope="module")
def entity():
    return make_entity()


@pytest.fixture(scope="module")
def full(entity):
    return entity.full


@pytest.fixture(scope="module")
def fitted(full, entity):
    return FeaturePipeline().fit(full, end_index=entity.train_end)


@pytest.fixture(scope="module")
def evaluated(fitted, full, entity):
    """The standardized test rows of the shared entity."""
    return fitted.transform(full, start_index=entity.train_end)


def test_columns_and_count():
    p = FeaturePipeline()
    assert len(p.columns) == 59 * 38 + 4 == 2246
    assert p.columns[0] == "lag_1[c00]"
    assert p.columns[37] == "lag_1[c37]"
    assert p.columns[-1] == "cal_dow_cos"
    assert p.columns[38] == "lag_2[c00]"  # spec-major: all channels of a spec come together
    assert p.columns[-4:] == ("cal_hod_sin", "cal_hod_cos", "cal_dow_sin", "cal_dow_cos")
    assert len(set(p.columns)) == len(p.columns)


def test_transform_before_fit_end_is_refused(full, entity):
    p = FeaturePipeline().fit(full, end_index=entity.train_end)
    with pytest.raises(LeakageError) as ei:
        p.transform(full, start_index=entity.train_end - 1)
    assert ei.value.guard == GUARD_FIT_END == "fit_end"
    assert ei.value.detail == {"start_index": entity.train_end - 1, "fit_end": entity.train_end}
    s, f = entity.train_end - 1, entity.train_end
    assert str(ei.value).endswith(
        f"evaluation starts at row {s} but the pipeline was fitted on rows up to {f}"
    )


def test_transform_from_fit_end_is_allowed_and_shaped(evaluated, entity):
    assert evaluated.shape == (entity.n_test, N_COLUMNS)
    assert evaluated.dtype == np.float32
    assert evaluated.flags.c_contiguous
    assert np.isfinite(evaluated).all()


def test_fit_never_reads_rows_after_end_index(full, entity, rng):
    end = entity.train_end
    noisy = full.copy()
    noisy[end:] = rng.random(noisy[end:].shape)
    a = FeaturePipeline().fit(full, end_index=end)
    b = FeaturePipeline().fit(noisy, end_index=end)
    assert a.state_digest() == b.state_digest()
    assert a.transform_in_sample(full).tobytes() == b.transform_in_sample(noisy).tobytes()


def test_test_rows_see_train_history(fitted, evaluated, full, entity):
    end = entity.train_end
    raw = compute_raw_features(full, FEATURE_SPECS, rows=(0, len(full)))
    head = raw[:end]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean, scale = np.nanmean(head, axis=0), np.nanstd(head, axis=0)
    scale = np.where((scale == 0.0) | np.isnan(scale), 1.0, scale)
    expected = np.nan_to_num((raw[end] - np.nan_to_num(mean)) / scale)
    np.testing.assert_allclose(evaluated[0], expected, rtol=1e-4, atol=1e-5)

    lag = fitted.columns.index("lag_1[c00]")
    last_train_value = (full[end - 1, 0] - mean[lag]) / scale[lag]
    assert evaluated[0, lag] != 0.0  # a cold start would have filled the warmup with 0.0
    assert evaluated[0, lag] == pytest.approx(last_train_value, rel=1e-5)


def test_constant_column_scales_to_zero_not_nan(fitted, evaluated, full):
    columns = fitted.columns
    std_columns = [
        i
        for i, name in enumerate(columns)
        if name.startswith("roll_std_") and name.endswith(f"[c{CONSTANT_CHANNEL}]")
    ]
    assert len(std_columns) == 3
    in_sample = fitted.transform_in_sample(full)
    for block in (in_sample, evaluated):
        assert np.isfinite(block).all()
        assert (block[:, std_columns] == 0.0).all()


def test_warmup_nans_become_zero(full, entity):
    p = FeaturePipeline()
    out = p.fit_transform(full, end_index=entity.train_end)
    assert out.shape == (entity.train_end, N_COLUMNS)
    assert np.isfinite(out).all()
    assert out[0, p.columns.index("lag_1[c00]")] == 0.0  # NaN warmup, not (NaN - mean) / scale
    assert out[MAX_LOOKBACK, p.columns.index("lag_10[c00]")] != 0.0


def test_unfitted_pipeline_raises(full):
    p = FeaturePipeline()
    assert p.fit_end is None
    with pytest.raises(RuntimeError, match="not fitted"):
        p.transform(full, start_index=0)
    with pytest.raises(RuntimeError, match="not fitted"):
        p.transform_in_sample(full)
    with pytest.raises(RuntimeError, match="not fitted"):
        p.state_digest()


def test_wrong_channel_count_is_value_error(full, entity, fitted):
    with pytest.raises(ValueError, match="37"):
        FeaturePipeline().fit(full[:, :37], end_index=entity.train_end)
    with pytest.raises(ValueError, match="37"):
        fitted.transform(full[:, :37], start_index=entity.train_end)
    with pytest.raises(ValueError, match="37"):
        fitted.transform_in_sample(full[:, :37])


@pytest.mark.parametrize("end_index", [0, -1, 4001])
def test_fit_end_index_outside_the_series_is_value_error(full, end_index):
    assert len(full) == 4000
    with pytest.raises(ValueError, match="end_index"):
        FeaturePipeline().fit(full, end_index=end_index)


def test_raw_specs_pipeline_is_standardized_identity(full, entity):
    p = FeaturePipeline(RAW_SPECS)
    assert p.columns == tuple(f"raw[c{j:02d}]" for j in range(N_CHANNELS))
    out = p.fit_transform(full, end_index=entity.train_end)
    train = full[: entity.train_end]
    live = [j for j in range(N_CHANNELS) if j != CONSTANT_CHANNEL]
    np.testing.assert_allclose(out[:, live].mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(out[:, live].std(axis=0), 1.0, atol=1e-5)
    expected = (train[:, 0] - train[:, 0].mean()) / train[:, 0].std()
    np.testing.assert_allclose(out[:, 0], expected, atol=1e-5)
    assert (out[:, CONSTANT_CHANNEL] == 0.0).all()


def test_context_rows_may_reach_into_the_fit_range(fitted, evaluated, full, entity):
    ctx = 31
    out = fitted.transform(full, start_index=entity.train_end, context=ctx)
    assert out.shape == (entity.n_test + ctx, N_COLUMNS)
    assert out[ctx:].tobytes() == evaluated.tobytes()
    assert np.isfinite(out).all()


def test_context_reaching_before_offset_is_value_error(fitted, full, entity):
    end = entity.train_end
    with pytest.raises(ValueError, match="context"):
        fitted.transform(full, start_index=end, context=end + 1)
    with pytest.raises(ValueError, match="context"):
        fitted.transform(full[100:], start_index=end, context=end - 100 + 1, offset=100)
    with pytest.raises(ValueError, match="context"):
        fitted.transform(full, start_index=end, context=-1)


def test_start_index_beyond_the_series_is_value_error(fitted, full, entity):
    with pytest.raises(ValueError, match="start_index"):
        fitted.transform(full, start_index=len(full) + 1)


def test_block_size_does_not_change_a_single_bit(rng):
    x = rng.random((3000, N_CHANNELS))
    x[:, CONSTANT_CHANNEL] = 0.5
    wide = compute_raw_features(x, FEATURE_SPECS, rows=(0, 3000), block=4096)
    narrow = compute_raw_features(x, FEATURE_SPECS, rows=(0, 3000), block=97)
    assert wide.shape == (3000, N_COLUMNS)
    assert wide.tobytes() == narrow.tobytes()


def test_a_row_range_and_an_offset_do_not_change_a_single_bit(rng):
    x = rng.random((1500, N_CHANNELS))
    whole = compute_raw_features(x, FEATURE_SPECS, rows=(0, 1500))
    part = compute_raw_features(x, FEATURE_SPECS, rows=(700, 1100), block=61)
    shifted = compute_raw_features(x[600:], FEATURE_SPECS, rows=(700, 1100), offset=600, block=61)
    assert part.tobytes() == whole[700:1100].tobytes()
    assert shifted.tobytes() == whole[700:1100].tobytes()
    # with fewer than lookback rows of history before the range, the missing window is NaN
    cold = compute_raw_features(x[650:], FEATURE_SPECS, rows=(700, 1100), offset=650)
    ewma = FeaturePipeline().columns.index("ewma_0p1[c00]")
    assert np.isnan(cold[: MAX_LOOKBACK - 1 - 50, ewma]).all()
    assert np.isfinite(cold[MAX_LOOKBACK - 1 - 50 :, ewma]).all()


def test_warmup_rows_of_the_raw_features_stay_nan(rng):
    x = rng.random((200, N_CHANNELS))
    out = compute_raw_features(x, FEATURE_SPECS, rows=(0, 200), block=50)
    columns = FeaturePipeline().columns
    lag10 = columns.index("lag_10[c00]")
    assert np.isnan(out[:10, lag10]).all()
    assert np.isfinite(out[10:, lag10]).all()
    ewma = columns.index("ewma_0p1[c00]")
    assert np.isnan(out[: MAX_LOOKBACK - 1, ewma]).all()
    assert np.isfinite(out[MAX_LOOKBACK - 1 :, ewma]).all()
    assert np.isfinite(out[:, columns.index("cal_hod_sin")]).all()


def test_offset_window_equals_full_series_rows(fitted, evaluated, full, entity):
    r = entity.train_end + 500
    window = full[r - 63 : r + 1]
    out = fitted.transform(window, start_index=r, offset=r - 63)
    assert out.shape == (1, N_COLUMNS)
    assert out[0].tobytes() == evaluated[500].tobytes()


def test_transform_ignores_rows_before_its_history(fitted, evaluated, full, entity):
    end = entity.train_end
    first_read = end - fitted.lookback + 1
    garbled = full.copy()
    garbled[:first_read] = np.nan
    assert fitted.transform(garbled, start_index=end).tobytes() == evaluated.tobytes()

    ctx = 20
    garbled = full.copy()
    garbled[: first_read - ctx] = np.nan
    out = fitted.transform(garbled, start_index=end, context=ctx)
    assert out[ctx:].tobytes() == evaluated.tobytes()
    assert np.isfinite(out).all()


def test_transform_feeds_the_specs_only_the_rows_it_needs(full, entity):
    seen = []

    def probe(block, cache):
        seen.append(len(block))
        return block.copy()

    spec = FeatureSpec("probe", "toy", 5, "channel", probe)
    p = FeaturePipeline((spec,)).fit(full, end_index=entity.train_end)
    seen.clear()
    p.transform(full, start_index=entity.train_end, context=3)
    # the evaluated rows, the context rows, and lookback - 1 rows of history: nothing earlier
    assert seen == [entity.n_test + 3 + 4]


def test_spec_with_oversized_lookback_is_rejected():
    long_spec = FeatureSpec("too_long", "toy", MAX_LOOKBACK + 1, "channel", lambda b, c: b.copy())
    with pytest.raises(ValueError, match="too_long"):
        FeaturePipeline((*RAW_SPECS, long_spec))
    with pytest.raises(ValueError, match="too_long"):
        compute_raw_features(np.zeros((10, N_CHANNELS)), (long_spec,), rows=(0, 10))


def test_no_specs_is_value_error():
    with pytest.raises(ValueError, match="spec"):
        FeaturePipeline(())


def test_required_history_is_the_lookback():
    assert FeaturePipeline().required_history == 64
    assert FeaturePipeline().lookback == 64
    assert FeaturePipeline(RAW_SPECS).required_history == 1
    assert FeaturePipeline(RAW_SPECS).lookback == 1


def test_fit_transform_is_fit_then_transform_in_sample(full, entity):
    end = entity.train_end
    one_pass = FeaturePipeline()
    out = one_pass.fit_transform(full, end_index=end)
    two_step = FeaturePipeline().fit(full, end_index=end)
    assert one_pass.fit_end == two_step.fit_end == end
    assert one_pass.state_digest() == two_step.state_digest()
    assert out.tobytes() == two_step.transform_in_sample(full).tobytes()


def test_state_digest_is_stable_and_sensitive(full, entity):
    end = entity.train_end
    a = FeaturePipeline().fit(full, end_index=end).state_digest()
    assert a == FeaturePipeline().fit(full, end_index=end).state_digest()
    assert a != FeaturePipeline().fit(full, end_index=end - 100).state_digest()
    assert len(a) == 64
    assert a != FeaturePipeline(RAW_SPECS).fit(full, end_index=end).state_digest()


def test_a_fit_too_short_for_any_window_neither_warns_nor_yields_nan(full):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = FeaturePipeline().fit(full, end_index=1)
        out = p.transform(full, start_index=1)
    assert np.isfinite(out).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rows": (0, 10), "block": 0}, "block"),
        ({"rows": (0, 31)}, "rows"),
        ({"rows": (-1, 10)}, "rows"),
        ({"rows": (12, 10)}, "rows"),
        ({"rows": (0, 10), "offset": 5}, "rows"),
    ],
)
def test_compute_raw_features_rejects_bad_ranges(kwargs, match):
    x = np.zeros((30, N_CHANNELS))
    with pytest.raises(ValueError, match=match):
        compute_raw_features(x, RAW_SPECS, **kwargs)


def test_a_spec_returning_the_wrong_shape_is_named():
    broken = FeatureSpec("broken", "toy", 1, "channel", lambda b, c: b[:, :1])
    with pytest.raises(ValueError, match="broken"):
        compute_raw_features(np.zeros((30, N_CHANNELS)), (broken,), rows=(0, 30))


def test_in_sample_transform_needs_the_fit_rows(full, fitted, entity):
    with pytest.raises(ValueError, match="fitted on"):
        fitted.transform_in_sample(full[: entity.train_end - 1])


def test_zero_channels_is_value_error():
    with pytest.raises(ValueError, match="n_channels"):
        FeaturePipeline(RAW_SPECS, n_channels=0)
