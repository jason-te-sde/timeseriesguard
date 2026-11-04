"""The feature pipeline, which refuses to evaluate the rows its scaling was fitted on.

The pipeline computes the registry's features and standardizes them. Fitting stores one
mean and one scale per feature column, computed from the fit rows only. Evaluating the same
rows afterwards would grade a detector on data its inputs were scaled with, so
`FeaturePipeline.transform` refuses a start row before the fit end.

Features are computed in blocks of output rows. Every spec's value at a row depends only on
the trailing ``lookback`` rows, so each block is fed ``MAX_LOOKBACK - 1`` rows of history
before it and the result does not depend on the block size.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Sequence
from typing import Final

import numpy as np

from tsguard.errors import GUARD_FIT_END, LeakageError
from tsguard.features.specs import FEATURE_SPECS, MAX_LOOKBACK, BlockCache, FeatureSpec

DEFAULT_BLOCK: Final = 4096
_STAT_COLUMNS: Final = 256  # columns reduced at once; bounds the temporaries of nanmean/nanstd

RAW_SPECS: Final[tuple[FeatureSpec, ...]] = (
    FeatureSpec("raw", "identity", 1, "channel", lambda x, c: x.copy()),
)
"""One spec that passes the channels through unchanged, for the no-feature baseline."""


def _check_specs(specs: Sequence[FeatureSpec]) -> None:
    """Reject a registry the block overlap cannot serve.

    Raises:
        ValueError: There are no specs, or a spec reads more than ``MAX_LOOKBACK`` rows,
            which the rows fed before each block would silently truncate.
    """
    if not specs:
        raise ValueError("a feature pipeline needs at least one spec")
    for spec in specs:
        if spec.lookback > MAX_LOOKBACK:
            raise ValueError(
                f"spec {spec.name!r} has lookback {spec.lookback}, above MAX_LOOKBACK "
                f"{MAX_LOOKBACK}; the block overlap would silently truncate its window"
            )


def _column_names(specs: Sequence[FeatureSpec], n_channels: int) -> tuple[str, ...]:
    """Return the column names: channel specs spec-major, then each entity spec once."""
    channel = [
        f"{spec.name}[c{j:02d}]"
        for spec in specs
        if spec.scope == "channel"
        for j in range(n_channels)
    ]
    entity = [spec.name for spec in specs if spec.scope == "entity"]
    return (*channel, *entity)


def _check_channels(x: np.ndarray, n_channels: int) -> None:
    if x.ndim != 2 or x.shape[1] != n_channels:
        raise ValueError(f"expected a 2-D array with {n_channels} channels, got shape {x.shape}")


def compute_raw_features(
    X: np.ndarray,  # noqa: N803
    specs: Sequence[FeatureSpec],
    *,
    rows: tuple[int, int],
    offset: int = 0,
    n_channels: int = 38,
    block: int = DEFAULT_BLOCK,
) -> np.ndarray:
    """Compute the unscaled features of a range of rows, in blocks.

    Each block of ``block`` output rows is computed from the input rows that begin
    ``MAX_LOOKBACK - 1`` before it (fewer at the series start) and keeps only its own output
    rows, so the result is bytewise independent of ``block`` and of where the range starts.
    A row whose window would reach before ``X[0]`` holds NaN for that spec.

    Args:
        X: Channel values ``(n, n_channels)``; ``X[0]`` is absolute row ``offset``.
        specs: The feature specs. Channel specs come first, spec-major, then entity specs.
        rows: The absolute output rows ``[start, stop)``; both must lie inside the rows
            ``X`` holds.
        offset: Absolute row index of ``X[0]``.
        n_channels: Number of channels of ``X``.
        block: Number of output rows computed at once.

    Returns:
        A C-contiguous float64 array ``(stop - start, n_columns)``, columns named as
        `FeaturePipeline.columns`.

    Raises:
        ValueError: ``X`` does not have ``n_channels`` columns, ``block`` is below 1,
            ``rows`` reaches outside ``X``, a spec reads more than ``MAX_LOOKBACK`` rows, or a
            channel spec returns the wrong shape.
    """
    _check_specs(specs)
    _check_channels(X, n_channels)
    start, stop = rows
    if block < 1:
        raise ValueError(f"block must be at least 1, got {block}")
    if not offset <= start <= stop <= offset + len(X):
        raise ValueError(
            f"rows {rows} must lie inside the {len(X)} rows of X, which are the absolute rows "
            f"[{offset}, {offset + len(X)})"
        )
    channel_specs = [spec for spec in specs if spec.scope == "channel"]
    entity_specs = [spec for spec in specs if spec.scope == "entity"]
    entity_base = len(channel_specs) * n_channels
    out = np.empty((stop - start, entity_base + len(entity_specs)))
    for first in range(start, stop, block):
        last = min(first + block, stop)
        read_from = max(offset, first - (MAX_LOOKBACK - 1))
        data = np.ascontiguousarray(X[read_from - offset : last - offset], dtype=np.float64)
        keep = slice(first - read_from, last - read_from)
        cache: BlockCache = {}
        for k, spec in enumerate(channel_specs):
            values = spec.fn(data, cache)
            if values.shape != data.shape:
                raise ValueError(
                    f"spec {spec.name!r} returned shape {values.shape} for a block of shape "
                    f"{data.shape}"
                )
            out[first - start : last - start, k * n_channels : (k + 1) * n_channels] = values[keep]
        minutes = np.arange(first, last, dtype=np.float64)
        for k, spec in enumerate(entity_specs):
            out[first - start : last - start, entity_base + k] = spec.fn(minutes, cache)
    return out


def _column_statistics(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-column ``(mean, scale)`` of the non-NaN values of ``raw``.

    A scale of 0 or NaN becomes 1.0 and a mean of NaN becomes 0.0, so a constant column
    standardizes to exactly zero and a column with no value at all passes through.
    """
    n_columns = raw.shape[1]
    mean = np.empty(n_columns)
    scale = np.empty(n_columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN column warns, then is NaN
        for lo in range(0, n_columns, _STAT_COLUMNS):
            hi = min(lo + _STAT_COLUMNS, n_columns)
            mean[lo:hi] = np.nanmean(raw[:, lo:hi], axis=0)
            scale[lo:hi] = np.nanstd(raw[:, lo:hi], axis=0)
    mean[np.isnan(mean)] = 0.0
    scale[(scale == 0.0) | np.isnan(scale)] = 1.0
    mean.flags.writeable = False
    scale.flags.writeable = False
    return mean, scale


def _standardize(raw: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Standardize ``raw`` in place, zero its NaN warmup, and return it as float32."""
    raw -= mean
    raw /= scale
    np.copyto(raw, 0.0, where=np.isnan(raw))
    return raw.astype(np.float32, order="C")


class FeaturePipeline:
    """Computes the registry's features and standardizes them with statistics fitted once.

    Attributes:
        columns: Column names; channel specs spec-major as ``name[cNN]``, then entity specs.
        lookback: The largest number of rows any spec reads.
    """

    def __init__(
        self, specs: Sequence[FeatureSpec] = FEATURE_SPECS, *, n_channels: int = 38
    ) -> None:
        """Create an unfitted pipeline.

        Args:
            specs: The feature specs to compute.
            n_channels: Number of input channels.

        Raises:
            ValueError: ``specs`` is empty, ``n_channels`` is below 1, or a spec's lookback
                exceeds ``MAX_LOOKBACK``.
        """
        _check_specs(specs)
        if n_channels < 1:
            raise ValueError(f"n_channels must be at least 1, got {n_channels}")
        self._specs = tuple(specs)
        self._n_channels = n_channels
        self.columns: tuple[str, ...] = _column_names(self._specs, n_channels)
        self.lookback: int = max(spec.lookback for spec in self._specs)
        self._fit_end: int | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    @property
    def fit_end(self) -> int | None:
        """Number of rows the pipeline was fitted on, or ``None`` before the first fit."""
        return self._fit_end

    @property
    def required_history(self) -> int:
        """Rows of history a caller must supply before the first row it wants evaluated."""
        return self.lookback

    def _require_fitted(self) -> tuple[int, np.ndarray, np.ndarray]:
        if self._fit_end is None or self._mean is None or self._scale is None:
            raise RuntimeError("the feature pipeline is not fitted; call fit() first")
        return self._fit_end, self._mean, self._scale

    def _check_fit_end(self, x: np.ndarray, end_index: int) -> None:
        _check_channels(x, self._n_channels)
        if not 1 <= end_index <= len(x):
            raise ValueError(f"end_index must be in [1, {len(x)}], got {end_index}")

    def _fit_rows(self, x: np.ndarray, end_index: int) -> np.ndarray:
        """Compute the raw features of ``x[:end_index]`` and store their statistics."""
        self._check_fit_end(x, end_index)
        raw = compute_raw_features(
            x[:end_index], self._specs, rows=(0, end_index), n_channels=self._n_channels
        )
        mean, scale = _column_statistics(raw)
        self._fit_end, self._mean, self._scale = end_index, mean, scale
        return raw

    def fit(self, X: np.ndarray, *, end_index: int) -> FeaturePipeline:  # noqa: N803
        """Fit the standardization on the rows ``X[:end_index]`` and nothing else.

        Args:
            X: Channel values ``(n, n_channels)``; only the first ``end_index`` rows are read.
            end_index: Number of rows to fit on; also the first row `transform` may evaluate.

        Returns:
            This pipeline.

        Raises:
            ValueError: ``end_index`` is outside ``[1, len(X)]`` or ``X`` has the wrong
                number of channels.
        """
        self._fit_rows(X, end_index)
        return self

    def fit_transform(self, X: np.ndarray, *, end_index: int) -> np.ndarray:  # noqa: N803
        """Fit on ``X[:end_index]`` and return those rows standardized, computing features once.

        Args:
            X: Channel values ``(n, n_channels)``; only the first ``end_index`` rows are read.
            end_index: Number of rows to fit on.

        Returns:
            The in-sample features, float32 ``(end_index, len(columns))``, C-contiguous.

        Raises:
            ValueError: As `fit`.
        """
        raw = self._fit_rows(X, end_index)
        _, mean, scale = self._require_fitted()
        return _standardize(raw, mean, scale)

    def transform_in_sample(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Return the standardized features of the rows the pipeline was fitted on.

        Args:
            X: Channel values ``(n, n_channels)`` with at least ``fit_end`` rows; only the
                first ``fit_end`` are read.

        Returns:
            Float32 ``(fit_end, len(columns))``, C-contiguous.

        Raises:
            RuntimeError: The pipeline is not fitted.
            ValueError: ``X`` has the wrong number of channels or fewer than ``fit_end`` rows.
        """
        fit_end, mean, scale = self._require_fitted()
        _check_channels(X, self._n_channels)
        if len(X) < fit_end:
            raise ValueError(
                f"X has {len(X)} rows, fewer than the {fit_end} the pipeline was fitted on"
            )
        raw = compute_raw_features(
            X[:fit_end], self._specs, rows=(0, fit_end), n_channels=self._n_channels
        )
        return _standardize(raw, mean, scale)

    def transform(
        self,
        X: np.ndarray,  # noqa: N803
        *,
        start_index: int,
        context: int = 0,
        offset: int = 0,
    ) -> np.ndarray:
        """Standardize the rows from ``start_index`` on, which must lie after the fit rows.

        ``context`` extra rows before ``start_index`` are returned too, for detectors that
        need a window; they may lie inside the fit range, because reading history is
        legal. Only the input rows from ``start_index - context - lookback + 1`` on are read.

        Args:
            X: Channel values ``(n, n_channels)``; ``X[0]`` is absolute row ``offset``.
            start_index: Absolute row of the first row being evaluated.
            context: Number of history rows to return before ``start_index``.
            offset: Absolute row index of ``X[0]``.

        Returns:
            Float32 features for the absolute rows ``[start_index - context, offset + len(X))``,
            C-contiguous.

        Raises:
            RuntimeError: The pipeline is not fitted.
            LeakageError: ``start_index`` is before the end of the fit rows (guard
                ``fit_end``).
            ValueError: ``X`` has the wrong number of channels, ``context`` is negative,
                ``start_index - context`` is before ``offset``, or ``start_index`` is
                after the last row of ``X``.
        """
        fit_end, mean, scale = self._require_fitted()
        _check_channels(X, self._n_channels)
        if start_index < fit_end:
            raise LeakageError(
                f"evaluation starts at row {start_index} but the pipeline was fitted on rows "
                f"up to {fit_end}",
                guard=GUARD_FIT_END,
                detail={"start_index": int(start_index), "fit_end": int(fit_end)},
            )
        if context < 0:
            raise ValueError(f"context must not be negative, got {context}")
        first = start_index - context
        if first < offset:
            raise ValueError(
                f"start_index {start_index} with context {context} reaches row {first}, "
                f"before offset {offset}, the first row of X"
            )
        if start_index > offset + len(X):
            raise ValueError(
                f"start_index {start_index} is after the last row of X, "
                f"which holds the absolute rows [{offset}, {offset + len(X)})"
            )
        read_from = max(offset, first - self.lookback + 1)
        raw = compute_raw_features(
            X[read_from - offset :],
            self._specs,
            rows=(first, offset + len(X)),
            offset=read_from,
            n_channels=self._n_channels,
        )
        return _standardize(raw, mean, scale)

    def state_digest(self) -> str:
        """Return a sha256 over the fit end, the column names and the fitted statistics.

        Returns:
            A 64-character hexadecimal digest; equal for two fits of the same rows.

        Raises:
            RuntimeError: The pipeline is not fitted.
        """
        fit_end, mean, scale = self._require_fitted()
        digest = hashlib.sha256()
        digest.update(f"{fit_end}\n".encode())
        digest.update("\n".join(self.columns).encode())
        digest.update(b"\n")
        digest.update(np.ascontiguousarray(mean, dtype="<f8").tobytes())
        digest.update(np.ascontiguousarray(scale, dtype="<f8").tobytes())
        return digest.hexdigest()
