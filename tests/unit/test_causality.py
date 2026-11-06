from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_entity
from tsguard.errors import GUARD_CAUSAL, LeakageError
from tsguard.features.causality import _COMPARE_BLOCK, assert_causal, assert_fit_ignores_future
from tsguard.features.pipeline import FeaturePipeline


@pytest.fixture
def x100(rng):
    """A (100, 2) array of uniform values."""
    return rng.uniform(0.0, 1.0, size=(100, 2))


def centered_mean(X: np.ndarray, window: int = 5) -> np.ndarray:
    return pd.DataFrame(X).rolling(window, center=True).mean().to_numpy()


def test_trailing_pipeline_is_causal(rng):
    e = make_entity()
    p = FeaturePipeline().fit(e.full, end_index=e.train_end)
    assert_causal(
        lambda X: p.transform(X, start_index=e.train_end),
        e.full,
        t0=e.train_end + 1000,
        start=e.train_end,
        rng=rng,
    )


def test_centered_window_is_caught_at_the_exact_row(rng, x100):
    with pytest.raises(LeakageError) as ei:
        assert_causal(centered_mean, x100, t0=50, rng=rng)
    assert ei.value.guard == GUARD_CAUSAL == "assert_causal"
    assert ei.value.detail == {"t0": 50, "row": 48, "column": 0}
    assert str(ei.value) == (
        "[assert_causal] output row 48 changed when inputs at rows >= 50 were replaced (column 0)"
    )


def test_row_is_absolute_when_the_output_starts_later(rng, x100):
    def fn(X):
        return centered_mean(X)[20:]

    with pytest.raises(LeakageError) as ei:
        assert_causal(fn, x100, t0=50, start=20, rng=rng)
    assert ei.value.detail["row"] == 48


def test_column_name_is_reported(rng, x100):
    def leak_in_b(X):
        return np.column_stack([X[:, 0], centered_mean(X[:, [1]], 3)[:, 0]])

    with pytest.raises(LeakageError) as ei:
        assert_causal(leak_in_b, x100, t0=50, rng=rng, columns=("a", "b"))
    assert ei.value.detail == {"t0": 50, "row": 49, "column": "b"}
    assert "(column b)" in str(ei.value)


def test_column_index_is_reported_without_names(rng, x100):
    def leak_in_second_column(X):
        return np.column_stack([X[:, 0], centered_mean(X[:, [1]], 3)[:, 0]])

    with pytest.raises(LeakageError) as ei:
        assert_causal(leak_in_second_column, x100, t0=50, rng=rng)
    assert ei.value.detail["column"] == 1


def test_one_dimensional_output_reports_column_zero(rng, x100):
    def fn(X):
        return centered_mean(X[:, [0]], 3)[:, 0]

    with pytest.raises(LeakageError) as ei:
        assert_causal(fn, x100, t0=50, rng=rng, columns=("only",))
    assert ei.value.detail == {"t0": 50, "row": 49, "column": "only"}


def test_a_difference_far_below_any_tolerance_is_caught(rng, x100):
    def fn(X):
        return X[:, 0] + 1e-9 * X[-1, 0]

    with pytest.raises(LeakageError) as ei:
        assert_causal(fn, x100, t0=50, rng=rng)
    assert ei.value.detail["row"] == 0


def test_identical_nan_is_not_a_difference_but_nan_against_a_value_is(rng, x100):
    last = x100[-1, 0]

    def nan_head(X):
        out = X.copy()
        out[:10] = np.nan
        return out

    def nan_when_the_future_changes(X):
        out = X.copy()
        if X[-1, 0] != last:
            out[10, 0] = np.nan
        return out

    assert_causal(nan_head, x100, t0=50, rng=rng)
    with pytest.raises(LeakageError) as ei:
        assert_causal(nan_when_the_future_changes, x100, t0=50, rng=rng)
    assert ei.value.detail["row"] == 10


@pytest.mark.parametrize(("t0", "start"), [(0, 0), (100, 0), (101, 0), (-1, 0), (60, 60), (50, 60)])
def test_t0_outside_range_is_value_error(rng, x100, t0, start):
    with pytest.raises(ValueError, match="t0"):
        assert_causal(lambda X: X[start:], x100, t0=t0, start=start, rng=rng)


def test_negative_start_is_value_error(rng, x100):
    with pytest.raises(ValueError, match="start"):
        assert_causal(lambda X: X, x100, t0=50, start=-1, rng=rng)


def test_outputs_of_different_shape_are_value_error(rng, x100):
    calls = []

    def shrinks_on_the_second_call(X):
        calls.append(1)
        return X if len(calls) == 1 else X[:-1]

    with pytest.raises(ValueError, match="shape"):
        assert_causal(shrinks_on_the_second_call, x100, t0=50, rng=rng)


def test_output_that_does_not_cover_the_rows_is_value_error(rng, x100):
    with pytest.raises(ValueError, match="rows"):
        assert_causal(lambda X: X[:30], x100, t0=50, rng=rng)


def test_names_that_do_not_match_the_columns_are_value_error(rng, x100):
    with pytest.raises(ValueError, match="columns"):
        assert_causal(lambda X: X, x100, t0=50, rng=rng, columns=("a", "b", "c"))


def test_the_input_is_not_modified(rng, x100):
    x100.flags.writeable = False
    before = x100.copy()
    assert_causal(lambda X: X, x100, t0=50, rng=rng)
    np.testing.assert_array_equal(x100, before)


def test_whole_series_normalization_is_caught(rng, x100):
    with pytest.raises(LeakageError):
        assert_causal(lambda X: (X - X.mean(axis=0)) / X.std(axis=0), x100, t0=60, rng=rng)


def test_honest_pipeline_fit_ignores_future(rng):
    e = make_entity()
    assert_fit_ignores_future(
        lambda X: FeaturePipeline().fit(X, end_index=e.train_end).state_digest(),
        e.full,
        end_index=e.train_end,
        rng=rng,
    )


def test_fit_that_reads_past_end_index_is_caught(rng):
    e = make_entity()
    with pytest.raises(LeakageError) as ei:
        assert_fit_ignores_future(
            lambda X: FeaturePipeline().fit(X, end_index=len(X)).state_digest(),
            e.full,
            end_index=e.train_end,
            rng=rng,
        )
    assert ei.value.guard == "assert_causal"
    assert ei.value.detail == {"phase": "fit", "end_index": e.train_end}
    assert str(ei.value) == (
        f"[assert_causal] fitted state changed when rows >= {e.train_end} were replaced; "
        "fit read past its end_index"
    )


@pytest.mark.parametrize("end_index", [0, 100, 101])
def test_fit_end_index_outside_range_is_value_error(rng, x100, end_index):
    with pytest.raises(ValueError, match="end_index"):
        assert_fit_ignores_future(lambda X: "digest", x100, end_index=end_index, rng=rng)


def test_fit_check_does_not_modify_the_input(rng, x100):
    x100.flags.writeable = False
    before = x100.copy()
    assert_fit_ignores_future(lambda X: "digest", x100, end_index=50, rng=rng)
    np.testing.assert_array_equal(x100, before)


def test_a_function_that_reuses_its_output_buffer_is_still_checked(rng, x100):
    buffer = np.empty_like(x100)

    def fn(X):
        buffer[:] = centered_mean(X)
        return buffer

    with pytest.raises(LeakageError) as ei:
        assert_causal(fn, x100, t0=50, rng=rng)
    assert ei.value.detail["row"] == 48


@pytest.mark.parametrize("start", [0, 100])
def test_a_leak_first_visible_past_the_first_compare_block_is_located(rng, start):
    t0 = _COMPARE_BLOCK + 104
    X = rng.uniform(0.0, 1.0, size=(t0 + 1800, 3))

    def leak_in_third_column(X):
        out = X.copy()
        out[:, 2] = centered_mean(X[:, [2]], 3)[:, 0]
        return out[start:]

    with pytest.raises(LeakageError) as ei:
        assert_causal(leak_in_third_column, X, t0=t0, start=start, rng=rng)
    assert ei.value.detail == {"t0": t0, "row": t0 - 1, "column": 2}


def test_fit_that_reads_exactly_one_row_past_end_index_is_caught(rng):
    e = make_entity()
    with pytest.raises(LeakageError) as ei:
        assert_fit_ignores_future(
            lambda X: FeaturePipeline().fit(X, end_index=e.train_end + 1).state_digest(),
            e.full,
            end_index=e.train_end,
            rng=rng,
        )
    assert ei.value.detail == {"phase": "fit", "end_index": e.train_end}


def test_integer_input_is_value_error(rng):
    X = np.arange(200).reshape(100, 2)
    with pytest.raises(ValueError, match="floating"):
        assert_causal(lambda X: X, X, t0=50, rng=rng)
    with pytest.raises(ValueError, match="floating"):
        assert_fit_ignores_future(lambda X: "digest", X, end_index=50, rng=rng)


def test_object_output_is_value_error(rng, x100):
    with pytest.raises(ValueError, match="dtype"):
        assert_causal(lambda X: X.astype(object), x100, t0=50, rng=rng)


def test_output_without_columns_is_value_error(rng, x100):
    with pytest.raises(ValueError, match="column"):
        assert_causal(lambda X: X[:, :0], x100, t0=50, rng=rng)
