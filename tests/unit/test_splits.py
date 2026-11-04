import numpy as np
import pytest

from tsguard.data.splits import (
    Fold,
    check_fold_boundaries,
    check_split_monotonic,
    chronological_split,
    walk_forward_folds,
)
from tsguard.errors import FoldBoundaryError, SplitOrderError


def test_folds_for_1000_rows():
    assert walk_forward_folds(1000) == (
        Fold(1, 500, 564, 667),
        Fold(2, 667, 731, 833),
        Fold(3, 833, 897, 1000),
    )


def test_last_fold_ends_at_n_for_real_entity_size():
    folds = walk_forward_folds(28_479)
    assert folds[0].fit_end == 14_239
    assert folds[-1].val_end == 28_479


def test_folds_follow_k_initial_frac_and_purge():
    assert walk_forward_folds(1000, k=2, initial_frac=0.6, purge=10) == (
        Fold(1, 600, 610, 800),
        Fold(2, 800, 810, 1000),
    )


def test_folds_pass_their_own_check():
    check_fold_boundaries(walk_forward_folds(28_479), lookback=64)


def test_zero_purge_folds_are_rejected_with_fold_index():
    with pytest.raises(FoldBoundaryError) as ei:
        check_fold_boundaries(walk_forward_folds(1000, purge=0), lookback=64)
    assert ei.value.guard == "fold_boundary"
    assert ei.value.detail == {"fold": 1, "gap": 0, "purge": 64}
    assert "fold 1" in str(ei.value)
    assert "gap 0" in str(ei.value)


def test_a_later_fold_with_too_small_a_gap_is_named():
    folds = (Fold(1, 500, 564, 667), Fold(2, 667, 700, 833))
    with pytest.raises(FoldBoundaryError) as ei:
        check_fold_boundaries(folds, lookback=64)
    assert ei.value.detail == {"fold": 2, "gap": 33, "purge": 64}


def test_purge_shorter_than_lookback_is_rejected():
    with pytest.raises(FoldBoundaryError, match="lookback"):
        check_fold_boundaries(walk_forward_folds(1000, purge=32), lookback=64, purge=32)


def test_purge_equal_to_lookback_is_accepted():
    check_fold_boundaries(walk_forward_folds(1000, purge=32), lookback=32, purge=32)


@pytest.mark.parametrize(("fit_end", "gap"), [(600, -100), (699, -1)])
def test_overlapping_folds_are_rejected(fit_end, gap):
    folds = (Fold(1, 500, 564, 700), Fold(2, fit_end, fit_end + 64, 833))
    with pytest.raises(FoldBoundaryError) as ei:
        check_fold_boundaries(folds, lookback=64)
    assert ei.value.detail == {"fold": 2, "gap": gap, "purge": 64}
    assert "fold 2" in str(ei.value)


@pytest.mark.parametrize(
    ("folds", "field", "gap"),
    [
        pytest.param((Fold(1, 500, 564, 700), Fold(2, 500, 564, 800)), "fit_end", 0, id="fit_end"),
        pytest.param(
            (Fold(1, 500, 600, 700), Fold(2, 501, 565, 800)), "val_start", -35, id="val_start"
        ),
        pytest.param(
            (Fold(1, 500, 564, 700), Fold(2, 501, 565, 650)), "val_end", -50, id="val_end"
        ),
    ],
)
def test_boundaries_that_do_not_increase_are_rejected(folds, field, gap):
    with pytest.raises(FoldBoundaryError, match=field) as ei:
        check_fold_boundaries(folds, lookback=64)
    assert ei.value.detail == {"fold": 2, "gap": gap, "purge": 64}


def test_fold_without_validation_rows_is_rejected():
    with pytest.raises(FoldBoundaryError, match="no validation rows") as ei:
        check_fold_boundaries((Fold(1, 500, 564, 564),), lookback=64)
    assert ei.value.detail == {"fold": 1, "gap": 64, "purge": 64}


def test_no_folds_is_rejected():
    with pytest.raises(FoldBoundaryError, match="no folds"):
        check_fold_boundaries((), lookback=64)


def test_fold_detail_values_are_plain_ints():
    with pytest.raises(FoldBoundaryError) as ei:
        check_fold_boundaries(walk_forward_folds(1000, purge=0), lookback=64)
    assert all(type(value) is int for value in ei.value.detail.values())


@pytest.mark.parametrize("field", ["fit_end", "val_start", "val_end"])
@pytest.mark.parametrize("bad", [564.0, float("nan")])
def test_a_fold_field_that_is_not_an_integer_is_a_type_error_naming_the_fold(field, bad):
    good = {"index": 2, "fit_end": 667, "val_start": 731, "val_end": 833}
    folds = (Fold(1, 500, 564, 667), Fold(**{**good, field: bad}))
    with pytest.raises(TypeError, match=rf"fold 2: {field} must be an integer, got float"):
        check_fold_boundaries(folds, lookback=64)


def test_a_fold_index_that_is_not_an_integer_is_a_type_error():
    folds = (Fold(2.0, 500, 564, 667),)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"fold 2\.0: index must be an integer"):
        check_fold_boundaries(folds, lookback=64)


def test_a_nan_fit_end_no_longer_slips_through_the_gap_comparisons():
    folds = (Fold(1, float("nan"), 564, 667),)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"fold 1: fit_end must be an integer, got float nan"):
        check_fold_boundaries(folds, lookback=64)


def test_numpy_integers_are_accepted_and_reported_as_plain_ints():
    folds = (
        Fold(np.int64(1), np.int64(500), np.int64(564), np.int64(667)),  # type: ignore[arg-type]
        Fold(np.int64(2), np.int64(667), np.int64(700), np.int64(833)),  # type: ignore[arg-type]
    )
    with pytest.raises(FoldBoundaryError) as ei:
        check_fold_boundaries(folds, lookback=64)
    assert ei.value.detail == {"fold": 2, "gap": 33, "purge": 64}
    assert all(type(value) is int for value in ei.value.detail.values())


@pytest.mark.parametrize("fit_end", [0, -5])
def test_a_fold_that_fits_on_no_rows_is_rejected(fit_end):
    folds = (Fold(1, fit_end, fit_end + 64, fit_end + 164),)
    with pytest.raises(FoldBoundaryError, match=r"fold 1: the fit window") as ei:
        check_fold_boundaries(folds, lookback=64)
    assert ei.value.detail == {"fold": 1, "gap": 64, "purge": 64}


def test_a_fit_window_of_one_row_is_accepted():
    check_fold_boundaries((Fold(1, 1, 65, 100),), lookback=64)


def test_lookback_below_one_is_a_value_error():
    with pytest.raises(ValueError, match="lookback"):
        check_fold_boundaries(walk_forward_folds(1000, purge=0), lookback=0, purge=0)


def test_too_small_series_is_a_value_error():
    with pytest.raises(ValueError, match="fold 1"):
        walk_forward_folds(200)


def test_a_fold_with_no_validation_row_is_a_value_error():
    assert walk_forward_folds(1000, purge=165)[1] == Fold(2, 667, 832, 833)
    with pytest.raises(ValueError, match="fold 2"):
        walk_forward_folds(1000, purge=166)


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        pytest.param({"k": 0}, "k", id="k"),
        pytest.param({"initial_frac": 0.0}, "initial_frac", id="frac-zero"),
        pytest.param({"initial_frac": 1.0}, "initial_frac", id="frac-one"),
        pytest.param({"purge": -1}, "purge", id="purge"),
    ],
)
def test_invalid_fold_parameters_are_value_errors(kwargs, name):
    with pytest.raises(ValueError, match=rf"\b{name}\b"):
        walk_forward_folds(1000, **kwargs)


def test_chronological_split_is_monotonic():
    tr, te = chronological_split(5, 3)
    check_split_monotonic(tr, te)
    assert tr.tolist() == [0, 1, 2, 3, 4]
    assert te.tolist() == [5, 6, 7]


@pytest.mark.parametrize(("n_train", "n_test"), [(0, 3), (5, 0), (-1, 3)])
def test_chronological_split_needs_rows_on_both_sides(n_train, n_test):
    with pytest.raises(ValueError, match="at least 1"):
        chronological_split(n_train, n_test)


def test_shuffled_split_is_rejected():
    train = np.array([0, 2, 3, 5, 7])
    test = np.array([1, 4, 6])
    with pytest.raises(SplitOrderError) as ei:
        check_split_monotonic(train, test)
    assert ei.value.guard == "split_monotonic"
    assert ei.value.detail == {"train_max": 7, "test_min": 1}
    assert "maximum train index 7" in str(ei.value)
    assert "minimum test index 1" in str(ei.value)


def test_train_max_equal_to_test_min_is_rejected():
    with pytest.raises(SplitOrderError):
        check_split_monotonic(np.array([0, 1, 2]), np.array([2, 3]))


@pytest.mark.parametrize(
    ("train", "test", "name", "position"),
    [
        pytest.param([0, 2, 1, 3], [4, 5], "train", 2, id="train-unsorted"),
        pytest.param([0, 1, 1, 2], [4, 5], "train", 2, id="train-duplicate"),
        pytest.param([0, 1, 2], [5, 4, 6], "test", 1, id="test-unsorted"),
        pytest.param([0, 1, 2], [3, 3], "test", 1, id="test-duplicate"),
    ],
)
def test_indices_that_do_not_strictly_increase_are_rejected(train, test, name, position):
    with pytest.raises(SplitOrderError, match=name) as ei:
        check_split_monotonic(np.array(train), np.array(test))
    assert ei.value.detail["position"] == position


def test_empty_index_arrays_are_rejected():
    with pytest.raises(SplitOrderError, match="empty"):
        check_split_monotonic(np.array([], dtype=np.int64), np.array([1, 2]))


def test_split_detail_values_are_plain_ints():
    with pytest.raises(SplitOrderError) as ei:
        check_split_monotonic(np.array([0, 2, 3, 5, 7]), np.array([1, 4, 6]))
    assert all(type(value) is int for value in ei.value.detail.values())


def test_split_indices_must_be_a_one_dimensional_integer_array():
    with pytest.raises(TypeError, match="integer"):
        check_split_monotonic(np.array([0.0, 1.0]), np.array([2, 3]))
    with pytest.raises(ValueError, match="1-D"):
        check_split_monotonic(np.array([[0, 1]]), np.array([2, 3]))
