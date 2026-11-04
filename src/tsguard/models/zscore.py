"""A robust z-score detector: how far a row sits from the training median, in MAD units.

It is the baseline every other model has to beat. It reads one row at a time, so it needs no
context, and it has no random state.
"""

from __future__ import annotations

from typing import Final, Literal, get_args

import numpy as np

from tsguard.models.base import InputKind, check_context
from tsguard.repro.hashing import array_digest, json_digest

Aggregation = Literal["mean", "max", "topk"]

_MAD_TO_SIGMA: Final = 1.4826  # makes the MAD a consistent estimator of sigma for normal data


class ZScoreDetector:
    """Score a row by its robust z-scores against the training median.

    ``fit`` stores, per column, the median and ``1.4826 * MAD`` (a column whose MAD is zero gets
    scale 1, so a constant column scores its raw distance and never divides by zero). ``score``
    takes ``|x - median| / scale`` per column and reduces each row to one number.

    Attributes:
        name: ``"zscore"``.
        input_kind: ``"features"``.
        lookback: ``1``; a score reads only its own row.
        aggregation: How the per-column z-scores of a row are reduced: ``"mean"``, ``"max"``, or
            ``"topk"`` (the mean of the ``k`` largest).
        k: The number of largest z-scores ``"topk"`` averages; at most the column count is used.
    """

    def __init__(self, *, aggregation: Aggregation = "topk", k: int = 32) -> None:
        """Create an unfitted detector.

        Args:
            aggregation: ``"mean"``, ``"max"`` or ``"topk"``.
            k: Number of largest z-scores to average for ``"topk"``. Must be at least 1.

        Raises:
            ValueError: ``aggregation`` is not one of the three names, or ``k < 1``.
        """
        if aggregation not in get_args(Aggregation):
            raise ValueError(
                f"zscore aggregation must be one of {get_args(Aggregation)}, got {aggregation!r}"
            )
        if k < 1:
            raise ValueError(f"zscore k must be at least 1, got {k}")
        self.name: str = "zscore"
        self.input_kind: InputKind = "features"
        self.lookback: int = 1
        self.aggregation: Aggregation = aggregation
        self.k: int = k
        self._median: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        """Store the per-column median and scaled MAD of ``X``.

        Args:
            X: Array of shape ``(n_rows, n_columns)`` with at least one row.

        Raises:
            ValueError: ``X`` is not a non-empty 2-D array of finite values.
        """
        data = _as_matrix(X)
        if len(data) == 0:
            raise ValueError("zscore cannot fit on zero rows")
        median = np.median(data, axis=0)
        scale = _MAD_TO_SIGMA * np.median(np.abs(data - median), axis=0)
        scale[scale == 0.0] = 1.0
        median.flags.writeable = False
        scale.flags.writeable = False
        self._median, self._scale = median, scale

    def score(self, X: np.ndarray, *, context: int) -> np.ndarray:
        """Score every row of ``X`` after its first ``context`` rows.

        Args:
            X: Array of shape ``(n_rows, n_columns)``, float32 or float64.
            context: Number of leading rows to skip. Any value from 0 up to ``len(X) - 1``.

        Returns:
            A float64 array of ``len(X) - context`` scores; higher is more anomalous.

        Raises:
            ValueError: ``context`` is negative or leaves no rows, the column count differs
                from the fitted one, or the scored rows hold NaN or infinity.
            RuntimeError: The detector is not fitted.
        """
        check_context(self, X, context)
        if self._median is None or self._scale is None:
            raise RuntimeError("zscore is not fitted: call fit before score")
        rows = _as_matrix(X[context:])
        if rows.shape[1] != len(self._median):
            raise ValueError(
                f"zscore was fitted on {len(self._median)} columns, got {rows.shape[1]} columns"
            )
        z = np.abs(rows - self._median) / self._scale
        if self.aggregation == "mean":
            return np.asarray(z.mean(axis=1))
        if self.aggregation == "max":
            return np.asarray(z.max(axis=1))
        n_columns = z.shape[1]
        top = min(self.k, n_columns)
        largest = np.partition(z, n_columns - top, axis=1)[:, n_columns - top :]
        return np.asarray(largest.mean(axis=1))

    def state_digest(self) -> str:
        """Return a digest of the aggregation, ``k``, the median and the scale.

        Returns:
            A 64-character hex string, equal for two detectors fitted on the same rows with
            the same configuration.

        Raises:
            RuntimeError: The detector is not fitted.
        """
        if self._median is None or self._scale is None:
            raise RuntimeError("zscore is not fitted: call fit before state_digest")
        return json_digest(
            {
                "aggregation": self.aggregation,
                "k": self.k,
                "median": array_digest(self._median),
                "scale": array_digest(self._scale),
            }
        )


def _as_matrix(X: np.ndarray) -> np.ndarray:
    """Return ``X`` as a float64 2-D array, refusing other shapes and non-finite values."""
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"zscore needs a 2-D array of rows and columns, got shape {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("zscore needs finite values: X holds NaN or infinity")
    return data
