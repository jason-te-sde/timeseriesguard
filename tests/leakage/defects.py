"""Known ways to leak, injected on purpose so the guards can be shown to catch them.

Test support, not shipped. Each feature defect is a channel spec like the real ones, a
function of the whole block ``(m, c)``, with a dishonest ``lookback`` of 1: it claims to read
only the current row while it reads the rest of the block, which is how a pipeline that feeds
history and future in one array leaks. The pipeline cuts its blocks at fixed rows, so the
leak shows on every row of a block that has rows after it. None of them returns NaN: the
pipeline refuses a NaN past a spec's warmup (and these specs claim a lookback of 1, so they
have none), and a refusal would hide the leak from the guard that is meant to catch it.

Each defect must be caught for the leak it names and for no other. The two fill defects
therefore open their gaps with a mask that reads only the value itself (``block > 0.5``),
never a statistic of the block: with a block median as the mask, `assert_causal` would fire
on the median, and a forward fill behind the same mask would be flagged just the same.

The pipeline defects subclass `FeaturePipeline` and break the one rule it exists to keep:
the standardization is fitted on the rows before ``end_index`` and on nothing else.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd

from tsguard.errors import (
    GUARD_CAUSAL,
    GUARD_FIT_END,
    GUARD_FOLD,
    GUARD_LABELS,
    GUARD_SPLIT,
    LeakageError,
)
from tsguard.features.pipeline import FeaturePipeline
from tsguard.features.specs import BlockCache, FeatureSpec

DEFECT_FAMILY: Final = "defect"


def caught_by(action: Callable[[], object]) -> str | None:
    """Run ``action`` and name the guard that stopped it.

    Args:
        action: The code under test.

    Returns:
        The ``guard`` of the `LeakageError` it raised, or ``None`` if it raised nothing.
        Any other exception propagates, so a bug in the test is never read as a catch.
    """
    try:
        action()
    except LeakageError as error:
        return error.guard
    return None


def _defect_spec(name: str, fn: Callable[[np.ndarray, BlockCache], np.ndarray]) -> FeatureSpec:
    return FeatureSpec(f"defect_{name}", DEFECT_FAMILY, 1, "channel", fn)


GAP_LEVEL: Final = 0.5
"""The fill defects treat every value at or below this level as a gap."""


def gaps(block: np.ndarray) -> pd.DataFrame:
    """Return ``block`` with every value at or below `GAP_LEVEL` replaced by NaN.

    The mask reads each value on its own, so it is causal: the only thing that can make a
    fill built on it leak is the direction it fills in.

    Args:
        block: Channel values ``(m, c)``.

    Returns:
        A frame of the same shape; roughly half the cells of an SMD-like block are gaps.
    """
    return pd.DataFrame(block).where(block > GAP_LEVEL)


def _centered_rolling(block: np.ndarray, cache: BlockCache) -> np.ndarray:
    mean = pd.DataFrame(block).rolling(5, center=True).mean().fillna(0.0)
    return np.asarray(mean.to_numpy(dtype=np.float64))


def _bfill(block: np.ndarray, cache: BlockCache) -> np.ndarray:
    filled = gaps(block).bfill().fillna(0.0)
    return np.asarray(filled.to_numpy(dtype=np.float64))


def _whole_series_fft(block: np.ndarray, cache: BlockCache) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(block - block.mean(axis=0), axis=0))[1:]
    dominant = spectrum.max(axis=0, initial=0.0)
    return np.array(np.broadcast_to(dominant, block.shape), dtype=np.float64)


def _interpolate_both(block: np.ndarray, cache: BlockCache) -> np.ndarray:
    filled = gaps(block).interpolate(limit_direction="both").fillna(0.0)
    return np.asarray(filled.to_numpy(dtype=np.float64))


CENTERED_ROLLING: Final = _defect_spec("centered_rolling", _centered_rolling)
"""Defect 1: a 5-row rolling mean centered on the row, which reads two rows ahead."""

BFILL: Final = _defect_spec("bfill", _bfill)
"""Defect 3: gaps (the values at or below `GAP_LEVEL`) filled from the next value, backwards."""

WHOLE_SERIES_FFT: Final = _defect_spec("whole_series_fft", _whole_series_fft)
"""Defect 4: the dominant spectral magnitude of the entire block, the same on every row."""

INTERPOLATE_BOTH: Final = _defect_spec("interpolate_both", _interpolate_both)
"""Defect 8: gaps filled by interpolating between the values on both sides of them."""


class FullFitPipeline(FeaturePipeline):
    """Defect 2: standardizes with statistics fitted on every row it is given.

    It ignores ``end_index`` and records the fit end as ``len(X)``, so the guard on the
    evaluation side (``fit_end``) refuses every row it could be asked to evaluate.
    """

    def fit(self, X: np.ndarray, *, end_index: int) -> FeaturePipeline:
        """Fit on all of ``X``, whatever ``end_index`` says.

        Args:
            X: Channel values ``(n, n_channels)``.
            end_index: Ignored.

        Returns:
            This pipeline.
        """
        return super().fit(X, end_index=len(X))

    def fit_transform(self, X: np.ndarray, *, end_index: int) -> np.ndarray:
        """Fit on all of ``X`` and return all of its rows standardized.

        Args:
            X: Channel values ``(n, n_channels)``.
            end_index: Ignored.

        Returns:
            The in-sample features of every row of ``X``.
        """
        return super().fit_transform(X, end_index=len(X))


class LyingFullFitPipeline(FeaturePipeline):
    """Defect 2b: fits on every row but records the fit end the caller asked for.

    The evaluation-side guard compares against the recorded fit end, so it passes. Only
    the fit-side perturbation test, which changes the rows after ``end_index`` and compares
    the fitted state, can see that the statistics depend on them.
    """

    def fit(self, X: np.ndarray, *, end_index: int) -> FeaturePipeline:
        """Fit on all of ``X`` and record ``end_index`` as the fit end.

        Args:
            X: Channel values ``(n, n_channels)``.
            end_index: The fit end to report.

        Returns:
            This pipeline.

        Raises:
            ValueError: ``end_index`` is outside ``[1, len(X)]`` or ``X`` has the wrong
                number of channels.
        """
        self._check_fit_end(X, end_index)
        super().fit(X, end_index=len(X))
        self._fit_end = end_index
        return self

    def fit_transform(self, X: np.ndarray, *, end_index: int) -> np.ndarray:
        """Fit as `fit` does and return the first ``end_index`` rows standardized.

        Args:
            X: Channel values ``(n, n_channels)``.
            end_index: The fit end to report, and the number of rows returned.

        Returns:
            The in-sample features of the rows ``X[:end_index]``, scaled with statistics
            that were computed on every row.
        """
        self.fit(X, end_index=end_index)
        return self.transform_in_sample(X)


def train_test_split_shuffled(
    idx: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Defect 7: split row positions at random, as ``train_test_split(shuffle=True)`` does.

    Args:
        idx: Row positions to split.
        rng: Source of the shuffle.

    Returns:
        ``(train, test)``, each sorted and of dtype int64 so that only the time order of the
        two blocks, not the shape of either array, can fail a check; the first half of a
        random permutation is the train block.
    """
    shuffled = rng.permutation(np.asarray(idx, dtype=np.int64))
    half = len(shuffled) // 2
    return np.sort(shuffled[:half]), np.sort(shuffled[half:])


DEFECTS: Final[dict[str, tuple[str, str]]] = {
    "1": ("rolling window centered on the row, reading rows after it", GUARD_CAUSAL),
    "2": ("scaler fitted on every row and recorded as such", GUARD_FIT_END),
    "2b": ("scaler fitted on every row but recorded with the requested fit end", GUARD_CAUSAL),
    "3": ("gaps (values at or below 0.5) filled from the next value, backwards", GUARD_CAUSAL),
    "4": ("spectrum computed over the whole block instead of a trailing window", GUARD_CAUSAL),
    "5": ("threshold searched on test labels during model selection", GUARD_LABELS),
    "5b": ("threshold frozen after test labels were read during evaluation", GUARD_LABELS),
    "6": ("walk-forward folds without a purge between fit and validation", GUARD_FOLD),
    "7": ("rows shuffled into train and test, ignoring time order", GUARD_SPLIT),
    "8": ("gaps (values at or below 0.5) interpolated from both sides", GUARD_CAUSAL),
}
"""Every injected defect: its id, a description, and the guard that must catch it."""
