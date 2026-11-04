"""The Future Perturbation Test: a lookahead detector that holds for any pipeline.

A transform is causal when its output at row ``t`` does not depend on inputs at rows after
``t``. Reading the code cannot prove that, and a centered window, a whole-series
normalization or a stray ``shift(-1)`` each look innocent. The test below needs no knowledge
of the code. It replaces every input row from ``t0`` on with uniform noise and compares the
outputs before ``t0`` with the outputs on the original input. Any difference, however small,
means a row before ``t0`` read a row from ``t0`` on.

`assert_causal` covers evaluation, with the transform already fitted. `assert_fit_ignores_future`
covers fitting: the state fitted on the first ``end_index`` rows must not depend on the rows
after them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

import numpy as np

from tsguard.errors import GUARD_CAUSAL, LeakageError

_COMPARE_BLOCK: Final = 4096  # rows compared at once; bounds the temporaries of the comparison


def _with_noise_from(X: np.ndarray, first_row: int, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of ``X`` whose rows from ``first_row`` on are uniform noise in [0, 1)."""
    perturbed = X.copy()
    perturbed[first_row:] = rng.uniform(0.0, 1.0, size=X[first_row:].shape)
    return perturbed


def _first_difference(before: np.ndarray, after: np.ndarray) -> tuple[int, int] | None:
    """Return the first ``(row, column)`` where two equal-shaped outputs differ, or None.

    The comparison is exact, and NaN equals NaN. Rows are scanned in blocks so a large output
    needs no full-size temporaries. A 1-D output counts as one column.
    """
    a = before.reshape(len(before), -1)
    b = after.reshape(len(after), -1)
    for lo in range(0, len(a), _COMPARE_BLOCK):
        block_a, block_b = a[lo : lo + _COMPARE_BLOCK], b[lo : lo + _COMPARE_BLOCK]
        if np.array_equal(block_a, block_b, equal_nan=True):
            continue
        differs = ~((block_a == block_b) | (np.isnan(block_a) & np.isnan(block_b)))
        row = int(np.argmax(differs.any(axis=1)))
        return lo + row, int(np.argmax(differs[row]))
    return None


def assert_causal(
    fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    *,
    t0: int,
    start: int = 0,
    rng: np.random.Generator,
    columns: Sequence[str] | None = None,
) -> None:
    """Assert that ``fn`` output before row ``t0`` does not depend on inputs from row ``t0`` on.

    ``X`` is copied and its rows from ``t0`` on are replaced by uniform noise in [0, 1); the
    outputs of ``fn`` on both inputs are compared exactly (NaN equals NaN) over the output
    rows that lie before ``t0``. ``X`` itself is not modified.

    ``fn`` must not refit anything between the two calls, so pass a transform that is already
    fitted. Fitting is covered by `assert_fit_ignores_future`.

    Args:
        fn: Maps an input array to its output rows ``[start, len(X))``, 1-D or 2-D.
        X: The input, ``(n, ...)``.
        t0: First absolute row that is replaced; must satisfy ``start < t0 < len(X)``.
        start: Absolute row of the first output row.
        rng: Source of the noise.
        columns: Names of the output columns, used to name the offending column. A 1-D
            output has one column.

    Raises:
        LeakageError: An output row before ``t0`` changed (guard ``assert_causal``). Its
            ``detail`` holds ``t0``, the absolute ``row`` and the ``column``, as a name when
            ``columns`` is given and as an index otherwise.
        ValueError: ``t0`` or ``start`` is out of range, the output does not hold the
            rows ``[start, len(X))``, the two outputs differ in shape, or ``columns`` does
            not match the number of output columns.
    """
    n = len(X)
    if not 0 <= start < t0 < n:
        raise ValueError(
            f"t0 must satisfy 0 <= start < t0 < len(X), got t0={t0}, start={start}, len(X)={n}"
        )
    before = np.asarray(fn(X))
    after = np.asarray(fn(_with_noise_from(X, t0, rng)))
    if before.shape != after.shape:
        raise ValueError(
            f"fn returned shape {before.shape} on the original input and shape {after.shape} "
            f"once rows >= {t0} were replaced; the output shape must not depend on the values"
        )
    if before.ndim not in (1, 2) or len(before) != n - start:
        raise ValueError(
            f"fn must return a 1-D or 2-D array with the {n - start} rows [{start}, {n}), "
            f"got shape {before.shape}"
        )
    n_columns = 1 if before.ndim == 1 else before.shape[1]
    if columns is not None and len(columns) != n_columns:
        raise ValueError(f"got {len(columns)} column names for {n_columns} output columns")
    cut = t0 - start
    found = _first_difference(before[:cut], after[:cut])
    if found is None:
        return
    row, col = found
    column: str | int = col if columns is None else str(columns[col])
    absolute_row = start + row
    raise LeakageError(
        f"output row {absolute_row} changed when inputs at rows >= {t0} were replaced "
        f"(column {column})",
        guard=GUARD_CAUSAL,
        detail={"t0": int(t0), "row": absolute_row, "column": column},
    )


def assert_fit_ignores_future(
    fit: Callable[[np.ndarray], str],
    X: np.ndarray,
    *,
    end_index: int,
    rng: np.random.Generator,
) -> None:
    """Assert that the state fitted with ``end_index`` does not depend on rows from it on.

    ``fit`` is called on ``X`` and on a copy of ``X`` whose rows from ``end_index`` on are
    uniform noise in [0, 1). The two digests must be equal. ``X`` itself is not modified.

    Args:
        fit: Fits a fresh object on the array it is given, with the same ``end_index``, and
            returns a digest of the fitted state.
        X: The input, ``(n, ...)``.
        end_index: Number of rows ``fit`` is allowed to read; must satisfy
            ``0 < end_index < len(X)``.
        rng: Source of the noise.

    Raises:
        LeakageError: The digests differ (guard ``assert_causal``); ``detail`` holds
            ``phase="fit"`` and ``end_index``.
        ValueError: ``end_index`` is out of range, which would leave no rows to replace.
    """
    n = len(X)
    if not 0 < end_index < n:
        raise ValueError(
            f"end_index must satisfy 0 < end_index < len(X), got {end_index} for len(X)={n}"
        )
    if fit(X) != fit(_with_noise_from(X, end_index, rng)):
        raise LeakageError(
            f"fitted state changed when rows >= {end_index} were replaced; "
            "fit read past its end_index",
            guard=GUARD_CAUSAL,
            detail={"phase": "fit", "end_index": int(end_index)},
        )
