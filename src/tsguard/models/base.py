"""The Detector protocol: what the evaluation protocols require of an anomaly model.

A detector is fitted on rows that carry no labels and scores rows it may read ``lookback - 1``
earlier rows for. The evaluation code supplies those earlier rows as ``context``, so a score is
never computed from fewer rows than the model needs, and never from a row after the one scored.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np

InputKind = Literal["features", "raw"]


@runtime_checkable
class Detector(Protocol):
    """An anomaly detector that is fitted on unlabeled rows and scores later rows.

    Attributes:
        name: Registry name, used in reports and error messages.
        input_kind: ``"features"`` if the detector reads the standardized feature matrix,
            ``"raw"`` if it reads the 38 raw channels.
        lookback: Number of rows, including the current one, that a score reads. A detector
            with lookback ``w`` needs ``w - 1`` rows of context.
    """

    name: str
    input_kind: InputKind
    lookback: int

    def fit(self, X: np.ndarray) -> None:
        """Fit on unlabeled rows.

        Args:
            X: Array of shape ``(n_rows, n_columns)``.
        """
        ...

    def score(self, X: np.ndarray, *, context: int) -> np.ndarray:
        """Score every row of ``X`` after its first ``context`` rows.

        Args:
            X: Array of shape ``(n_rows, n_columns)``; its first ``context`` rows are history
                that the detector reads but does not score.
            context: Number of leading rows of ``X`` that are history.

        Returns:
            A float64 array of ``len(X) - context`` scores, where a higher score means more
            anomalous.

        Raises:
            ValueError: ``context < lookback - 1``, or ``context >= len(X)``.
        """
        ...

    def state_digest(self) -> str:
        """Return a digest of the fitted state, equal for two identically fitted detectors."""
        ...


def check_context(detector: Detector, X: np.ndarray, context: int) -> None:
    """Check that ``context`` is enough history for ``detector`` and leaves rows to score.

    Detectors call this first in `Detector.score`, so every detector raises the same error.

    Args:
        detector: The detector whose ``name`` and ``lookback`` define the requirement.
        X: The array passed to ``score``.
        context: The number of leading history rows.

    Raises:
        ValueError: ``context < detector.lookback - 1``, or ``context >= len(X)``.
    """
    needed = detector.lookback - 1
    if context < needed:
        raise ValueError(f"{detector.name} needs context >= {needed} rows, got {context}")
    if context >= len(X):
        raise ValueError(
            f"{detector.name} has no rows to score: context {context} >= {len(X)} rows in X"
        )
