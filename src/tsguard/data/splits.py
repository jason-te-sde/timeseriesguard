"""Chronological splits and walk-forward folds, with the two structural leakage checks.

Walk-forward validation grows the fit window over time: fold ``j`` fits on rows
``[0, fit_end)`` and validates on rows ``[val_start, val_end)``. The ``purge`` rows between
``fit_end`` and ``val_start`` are dropped so that a feature whose window reaches back
``lookback`` rows from a validation row never touches a row the model was fitted on.

There is no embargo parameter. An embargo removes training rows that follow a validation
block, and in an expanding walk-forward no training row ever does, so it would remove
nothing.

`check_fold_boundaries` and `check_split_monotonic` do not trust the code that built the
windows: they re-derive the invariants from the numbers and raise a `LeakageError` naming
the fold or the index that breaks them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tsguard.errors import FoldBoundaryError, SplitOrderError


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold, in row positions of the series it was cut from.

    Attributes:
        index: 1-based fold number.
        fit_end: Exclusive end of the fit window ``[0, fit_end)``.
        val_start: Inclusive start of the validation window; ``val_start - fit_end`` rows
            in between are purged.
        val_end: Exclusive end of the validation window.
    """

    index: int
    fit_end: int
    val_start: int
    val_end: int


def walk_forward_folds(
    n: int, *, k: int = 3, initial_frac: float = 0.5, purge: int = 64
) -> tuple[Fold, ...]:
    """Cut a series of ``n`` rows into expanding walk-forward folds.

    The block boundaries are ``b_0 = floor(n * initial_frac)`` and
    ``b_j = b_0 + round(j * (n - b_0) / k)`` for ``j = 1..k``, so ``b_k == n``. Fold ``j``
    (1-based) fits on ``[0, b_{j-1})`` and validates on ``[b_{j-1} + purge, b_j)``.
    ``round`` is Python's round-half-to-even.

    Args:
        n: Number of rows of the series.
        k: Number of folds, at least 1.
        initial_frac: Fraction of the rows the first fit window covers, strictly between
            0 and 1.
        purge: Rows dropped between a fit window and its validation window, at least 0.

    Returns:
        The ``k`` folds, in time order.

    Raises:
        ValueError: A parameter is out of range, or the series is too short for some fold
            to keep any validation row after the purge; the message names the fold.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not 0.0 < initial_frac < 1.0:
        raise ValueError(f"initial_frac must be strictly between 0 and 1, got {initial_frac}")
    if purge < 0:
        raise ValueError(f"purge must be at least 0, got {purge}")
    first = math.floor(n * initial_frac)
    bounds = [first] + [first + round(j * (n - first) / k) for j in range(1, k + 1)]
    folds = tuple(Fold(j, bounds[j - 1], bounds[j - 1] + purge, bounds[j]) for j in range(1, k + 1))
    for fold in folds:
        if fold.val_start >= fold.val_end:
            raise ValueError(
                f"fold {fold.index}: {n} rows are too few for k={k}, "
                f"initial_frac={initial_frac} and purge={purge}: validation would start at "
                f"row {fold.val_start} but end at row {fold.val_end}"
            )
    return folds


def check_fold_boundaries(folds: Sequence[Fold], *, lookback: int, purge: int = 64) -> None:
    """Check that no fold fits on rows its features or its neighbours validate on.

    Per fold, in order: ``val_start - fit_end`` must be at least ``purge`` and the
    validation window must hold at least one row. Between a fold and the next, ``fit_end``,
    ``val_start`` and ``val_end`` must each strictly increase, and ``val_end`` must not
    exceed the next fold's ``fit_end`` (equality is fine: the next fold may fit on exactly
    the rows this one validated on, but not validate on rows it has already trained past).

    Args:
        folds: The folds to check, in time order.
        lookback: Rows a feature reaches back, at least 1 (``MAX_LOOKBACK`` of the
            feature registry).
        purge: The purge the folds are meant to have; must be at least ``lookback``.

    Raises:
        ValueError: ``lookback`` is below 1.
        FoldBoundaryError: A rule above is violated. The message names the fold and the gap;
            ``detail`` holds the plain integers ``fold``, ``gap`` and ``purge``. ``gap`` is
            the offending distance in rows: ``val_start - fit_end`` for the per-fold rules,
            and for the cross-fold rules the difference to the previous fold (the next
            ``fit_end`` minus this ``val_end`` for the overlap rule, negative when the
            windows overlap). ``fold`` is 0 when no single fold is at fault.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be at least 1, got {lookback}")
    if purge < lookback:
        raise FoldBoundaryError(
            f"purge {purge} is shorter than lookback {lookback}: the features of a validation "
            f"row could reach back into the fit window",
            detail={"fold": 0, "gap": purge, "purge": purge},
        )
    if not folds:
        raise FoldBoundaryError(
            "no folds to check: nothing would be validated",
            detail={"fold": 0, "gap": 0, "purge": purge},
        )
    previous: Fold | None = None
    for fold in folds:
        _check_one_fold(fold, purge)
        if previous is not None:
            _check_fold_pair(previous, fold, purge)
        previous = fold


def _check_one_fold(fold: Fold, purge: int) -> None:
    gap = fold.val_start - fold.fit_end
    if gap < purge:
        raise FoldBoundaryError(
            f"fold {fold.index}: validation starts at row {fold.val_start}, gap {gap} rows "
            f"after the fit window ends at row {fold.fit_end}; the purge is {purge}",
            detail={"fold": fold.index, "gap": gap, "purge": purge},
        )
    if fold.val_end <= fold.val_start:
        raise FoldBoundaryError(
            f"fold {fold.index}: no validation rows, the window [{fold.val_start}, "
            f"{fold.val_end}) is empty; gap {gap} rows after the fit window",
            detail={"fold": fold.index, "gap": gap, "purge": purge},
        )


def _check_fold_pair(previous: Fold, fold: Fold, purge: int) -> None:
    pairs = (
        ("fit_end", previous.fit_end, fold.fit_end),
        ("val_start", previous.val_start, fold.val_start),
        ("val_end", previous.val_end, fold.val_end),
    )
    for name, before, after in pairs:
        if after <= before:
            raise FoldBoundaryError(
                f"fold {fold.index}: {name} {after} does not exceed {name} {before} of "
                f"fold {previous.index}; gap {after - before} rows",
                detail={"fold": fold.index, "gap": after - before, "purge": purge},
            )
    gap = fold.fit_end - previous.val_end
    if gap < 0:
        raise FoldBoundaryError(
            f"fold {fold.index}: fit window ends at row {fold.fit_end}, gap {gap} rows from "
            f"the end of fold {previous.index}'s validation window at row {previous.val_end}; "
            f"the windows overlap",
            detail={"fold": fold.index, "gap": gap, "purge": purge},
        )


def chronological_split(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the index arrays of a train block followed by a test block.

    Args:
        n_train: Number of train rows, at least 1.
        n_test: Number of test rows, at least 1.

    Returns:
        ``(train_idx, test_idx)``: ``[0, n_train)`` and ``[n_train, n_train + n_test)``,
        both ``int64``.

    Raises:
        ValueError: ``n_train`` or ``n_test`` is below 1.
    """
    if n_train < 1:
        raise ValueError(f"n_train must be at least 1, got {n_train}")
    if n_test < 1:
        raise ValueError(f"n_test must be at least 1, got {n_test}")
    train_idx = np.arange(n_train, dtype=np.int64)
    test_idx = np.arange(n_train, n_train + n_test, dtype=np.int64)
    return train_idx, test_idx


def check_split_monotonic(train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Check that the train indices lie strictly before the test indices in time.

    Both arrays must be non-empty, strictly increasing, and ``train_idx.max()`` must be
    below ``test_idx.min()``.

    Args:
        train_idx: Row positions of the train block, 1-D integer array.
        test_idx: Row positions of the test block, 1-D integer array.

    Raises:
        TypeError: An array is not of integer dtype.
        ValueError: An array is not 1-D.
        SplitOrderError: An array is empty, an array does not strictly increase, or the
            blocks are not in time order. The message gives the offending values; ``detail``
            holds plain integers: ``size`` for an empty array, ``position``, ``previous``
            and ``value`` for an array that does not strictly increase, ``train_max`` and
            ``test_min`` for blocks out of order.
    """
    _check_index_array("train", train_idx)
    _check_index_array("test", test_idx)
    train_max, test_min = int(train_idx.max()), int(test_idx.min())
    if train_max >= test_min:
        raise SplitOrderError(
            f"the maximum train index {train_max} is not below the minimum test index "
            f"{test_min}; train rows must all precede test rows",
            detail={"train_max": train_max, "test_min": test_min},
        )


def _check_index_array(name: str, idx: np.ndarray) -> None:
    if not np.issubdtype(idx.dtype, np.integer):
        raise TypeError(f"{name}_idx must be an integer array, got dtype {idx.dtype}")
    if idx.ndim != 1:
        raise ValueError(f"{name}_idx must be 1-D, got shape {idx.shape}")
    if idx.size == 0:
        raise SplitOrderError(
            f"{name} indices are empty; the order of the split cannot be established",
            detail={"size": 0},
        )
    bad = np.flatnonzero(idx[1:] <= idx[:-1])
    if bad.size:
        position = int(bad[0]) + 1  # bad[i] compares positions i and i + 1
        previous, value = int(idx[position - 1]), int(idx[position])
        raise SplitOrderError(
            f"{name} indices are not strictly increasing: index {value} at position "
            f"{position} does not exceed its predecessor {previous}",
            detail={"position": position, "previous": previous, "value": value},
        )
