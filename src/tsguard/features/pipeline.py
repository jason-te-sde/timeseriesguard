"""The feature pipeline, which refuses to evaluate the rows its scaling was fitted on.

The pipeline computes the registry's features and standardizes them. Fitting stores one
mean and one scale per feature column, computed from the fit rows only. Evaluating the same
rows afterwards would grade a detector on data its inputs were scaled with, so
`FeaturePipeline.transform` refuses a start row before the fit end.

A column that is constant over the fit rows standardizes to exactly zero, no scale is below
`MIN_SCALE`, and no standardized value is beyond `STANDARD_CLIP`. NaN is legal only in the
warmup at the series start, where it becomes 0.0; anywhere else it is an error, so a broken
spec cannot hide behind a zero fill.

Features are computed in blocks of output rows. Every spec's value at a row depends only on
the trailing ``lookback`` rows, so each block is fed ``MAX_LOOKBACK - 1`` rows of history
before it and the result does not depend on the block size.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from tsguard.errors import GUARD_FIT_END, LeakageError
from tsguard.features.specs import FEATURE_SPECS, MAX_LOOKBACK, BlockCache, FeatureSpec

DEFAULT_BLOCK: Final = 4096
MIN_SCALE: Final = 1e-3
"""Smallest scale a column is divided by.

SMD is min-max scaled to [0, 1] per channel, so a change below 0.1 % of a channel's range
is noise. Without a floor, a near-constant channel (machine-1-1 channel 33 has a train scale of
about 3e-6) turns the first real movement of the test rows into a value of 1e5 standard
deviations, which swamps every detector.
"""
STANDARD_CLIP: Final = 50.0
"""Bound on the absolute value of a standardized feature.

A value beyond 50 sigma says nothing more about being anomalous than 50 sigma does. Scale-free
features such as windowed kurtosis, on channels that are flat in training, otherwise reach
1e4 to 1e5 sigma and swamp every distance-based detector. The bound was fixed before any model
ran. Clipping acts on one value at a time, so it keeps every feature causal.
"""
_STAT_COLUMNS: Final = 256  # columns reduced at once; bounds the temporaries of nanmean/nanstd

_STATE_FIELDS: Final = (
    "fit_end",
    "n_channels",
    "columns",
    "spec_names",
    "mean",
    "scale",
    "digest",
)
"""The keys of `FeaturePipeline.get_state`, which `FeaturePipeline.from_state` requires."""

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


def _warmup_groups(
    specs: Sequence[FeatureSpec], n_channels: int
) -> tuple[tuple[int, np.ndarray], ...]:
    """Group the columns by the number of warmup rows their spec may leave NaN."""
    warmup = [
        spec.lookback - 1 for spec in specs if spec.scope == "channel" for _ in range(n_channels)
    ]
    warmup += [spec.lookback - 1 for spec in specs if spec.scope == "entity"]
    column_warmup = np.array(warmup)
    return tuple(
        (int(rows), np.flatnonzero(column_warmup == rows)) for rows in np.unique(column_warmup)
    )


def _state_statistic(state: Mapping[str, Any], key: str, n_columns: int) -> np.ndarray:
    """Return a private read-only copy of the statistic ``key`` of a pipeline state.

    Raises:
        ValueError: The statistic is not a vector with one finite value per column.
    """
    values = np.array(state[key], dtype=np.float64)
    if values.shape != (n_columns,):
        raise ValueError(
            f"state field {key!r} has shape {values.shape}; expected ({n_columns},), one value "
            f"per feature column"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"state field {key!r} holds NaN or infinity")
    values.flags.writeable = False
    return values


def _spec_mismatch(recorded: Sequence[str], given: Sequence[str]) -> str:
    """Describe the first place two lists of spec names differ."""
    for position, (a, b) in enumerate(zip(recorded, given, strict=False)):
        if a != b:
            return (
                f"the state was fitted with a different list of specs: position {position} is "
                f"{a!r} in the state and {b!r} in the specs given"
            )
    return (
        f"the state was fitted with {len(recorded)} specs and {len(given)} were given; the first "
        f"{min(len(recorded), len(given))} agree"
    )


def _check_channels(x: np.ndarray, n_channels: int) -> None:
    if x.ndim != 2 or x.shape[1] != n_channels:
        raise ValueError(f"expected a 2-D array with {n_channels} channels, got shape {x.shape}")


def compute_raw_features(
    X: np.ndarray,
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

    The mean of a column whose values are all equal is that value, exactly: the arithmetic
    mean of ``n`` copies of 0.1 misses it by about 1e-15, which would leave a constant
    column standardized to noise instead of zero. The scale is the standard deviation, at
    least `MIN_SCALE`. A column with no value at all gets mean 0.0 and scale 1.0, so it
    passes through.
    """
    n_columns = raw.shape[1]
    mean = np.empty(n_columns)
    scale = np.empty(n_columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN column warns, then is NaN
        for lo in range(0, n_columns, _STAT_COLUMNS):
            hi = min(lo + _STAT_COLUMNS, n_columns)
            chunk = raw[:, lo:hi]
            low, high = np.nanmin(chunk, axis=0), np.nanmax(chunk, axis=0)
            mean[lo:hi] = np.where(low == high, low, np.nanmean(chunk, axis=0))
            scale[lo:hi] = np.nanstd(chunk, axis=0)
    mean[np.isnan(mean)] = 0.0
    scale = np.where(np.isnan(scale), 1.0, np.maximum(scale, MIN_SCALE))
    mean.flags.writeable = False
    scale.flags.writeable = False
    return mean, scale


def _standardize(raw: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Standardize ``raw`` in place, zero its NaN warmup, clip it, and return it as float32."""
    raw -= mean
    raw /= scale
    np.copyto(raw, 0.0, where=np.isnan(raw))
    np.clip(raw, -STANDARD_CLIP, STANDARD_CLIP, out=raw)
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
        self._warmup_groups = _warmup_groups(self._specs, n_channels)
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

    def _reject_unexpected_nan(self, raw: np.ndarray, first_row: int) -> None:
        """Raise if ``raw`` holds NaN anywhere but in a spec's warmup at the series start.

        A spec may be NaN for the absolute rows below its ``lookback - 1``, because the
        series has not yet supplied a full window. Anywhere else a NaN means a spec has
        regressed, and zero-filling it would hide the regression inside a plausible number.

        Args:
            raw: Unscaled features of the absolute rows ``[first_row, first_row + len(raw))``.
            first_row: Absolute index of the first row of ``raw``.

        Raises:
            RuntimeError: A NaN lies outside the warmup; the message names the column and
                the absolute row of the first one.
        """
        nan = np.isnan(raw)
        for warmup, columns in self._warmup_groups:
            rows = min(max(warmup - first_row, 0), len(raw))
            if rows:
                nan[:rows, columns] = False
        if nan.any():
            row, column = divmod(int(np.argmax(nan)), nan.shape[1])
            raise RuntimeError(
                f"feature column {self.columns[column]!r} is NaN at row {first_row + row}, "
                f"past its warmup ({int(nan.sum())} such values in all); "
                f"a spec must not return NaN for a row with a full window"
            )

    def _fit_rows(self, x: np.ndarray, end_index: int) -> np.ndarray:
        """Compute the raw features of ``x[:end_index]`` and store their statistics."""
        self._check_fit_end(x, end_index)
        raw = compute_raw_features(
            x[:end_index], self._specs, rows=(0, end_index), n_channels=self._n_channels
        )
        self._reject_unexpected_nan(raw, 0)
        mean, scale = _column_statistics(raw)
        self._fit_end, self._mean, self._scale = end_index, mean, scale
        return raw

    def fit(self, X: np.ndarray, *, end_index: int) -> FeaturePipeline:
        """Fit the standardization on the rows ``X[:end_index]`` and nothing else.

        Args:
            X: Channel values ``(n, n_channels)``; only the first ``end_index`` rows are read.
            end_index: Number of rows to fit on; also the first row `transform` may evaluate.

        Returns:
            This pipeline.

        Raises:
            ValueError: ``end_index`` is outside ``[1, len(X)]`` or ``X`` has the wrong
                number of channels.
            RuntimeError: A spec returned NaN past its warmup; the pipeline is left as it was.
        """
        self._fit_rows(X, end_index)
        return self

    def fit_transform(self, X: np.ndarray, *, end_index: int) -> np.ndarray:
        """Fit on ``X[:end_index]`` and return those rows standardized, computing features once.

        Args:
            X: Channel values ``(n, n_channels)``; only the first ``end_index`` rows are read.
            end_index: Number of rows to fit on.

        Returns:
            The in-sample features, clipped to ``STANDARD_CLIP``: float32
            ``(end_index, len(columns))``, C-contiguous.

        Raises:
            ValueError: As `fit`.
            RuntimeError: As `fit`.
        """
        raw = self._fit_rows(X, end_index)
        _, mean, scale = self._require_fitted()
        return _standardize(raw, mean, scale)

    def transform_in_sample(self, X: np.ndarray) -> np.ndarray:
        """Return the standardized features of the rows the pipeline was fitted on.

        Args:
            X: Channel values ``(n, n_channels)`` with at least ``fit_end`` rows; only the
                first ``fit_end`` are read.

        Returns:
            Float32 ``(fit_end, len(columns))``, C-contiguous.

        Raises:
            RuntimeError: The pipeline is not fitted, or a spec returned NaN past its warmup.
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
        self._reject_unexpected_nan(raw, 0)
        return _standardize(raw, mean, scale)

    def transform(
        self,
        X: np.ndarray,
        *,
        start_index: int,
        context: int = 0,
        offset: int = 0,
    ) -> np.ndarray:
        """Standardize the rows from ``start_index`` on, which must lie after the fit rows.

        ``context`` extra rows before ``start_index`` are returned too, for detectors that
        need a window; they may lie inside the fit range, because reading history is
        legal. Only the input rows from ``start_index - context - lookback + 1`` on are read.
        With ``offset == 0``, ``X`` is the series itself and the first ``lookback - 1`` rows
        are the warmup, which standardizes to 0.0; with ``offset > 0`` the window must
        carry that history in full.

        Args:
            X: Channel values ``(n, n_channels)``; ``X[0]`` is absolute row ``offset``.
            start_index: Absolute row of the first row being evaluated.
            context: Number of history rows to return before ``start_index``.
            offset: Absolute row index of ``X[0]``.

        Returns:
            Float32 features for the absolute rows ``[start_index - context, offset + len(X))``,
            C-contiguous.

        Raises:
            RuntimeError: The pipeline is not fitted, or a spec returned NaN past its warmup.
            LeakageError: ``start_index`` is before the end of the fit rows (guard
                ``fit_end``).
            ValueError: ``X`` has the wrong number of channels, ``context`` is negative,
                ``start_index - context`` is before ``offset``, ``start_index`` is after the
                last row of ``X``, or ``offset`` is positive and fewer than
                ``lookback - 1`` rows precede the first returned row: a window that starts
                after the series start must carry the full history its first row needs.
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
        if offset > 0 and first - offset < self.lookback - 1:
            raise ValueError(
                f"needs {self.lookback + context} rows of history ending at row {start_index} "
                f"(lookback {self.lookback}, context {context}), got {start_index - offset + 1}: "
                f"a window that starts after the series start would hold NaN warmup rows that "
                f"standardize to a plausible mean value"
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
        self._reject_unexpected_nan(raw, first)
        return _standardize(raw, mean, scale)

    def get_state(self) -> dict[str, object]:
        """Return the fitted state as plain data, so a fit can be stored and restored.

        The feature specs are functions and are not part of the state; `from_state` takes them
        again and checks their names against the ones recorded here.

        Returns:
            A dict with ``fit_end``, ``n_channels``, ``columns`` and ``spec_names`` (lists),
            ``mean`` and ``scale`` (float64 arrays, one value per column; copies, so changing
            them changes nothing here) and ``digest``, the `state_digest` of this fit.

        Raises:
            RuntimeError: The pipeline is not fitted.
        """
        fit_end, mean, scale = self._require_fitted()
        return {
            "fit_end": fit_end,
            "n_channels": self._n_channels,
            "columns": list(self.columns),
            "spec_names": [spec.name for spec in self._specs],
            "mean": mean.copy(),
            "scale": scale.copy(),
            "digest": self.state_digest(),
        }

    @classmethod
    def from_state(cls, specs: Sequence[FeatureSpec], state: Mapping[str, Any]) -> FeaturePipeline:
        """Rebuild a fitted pipeline from `get_state`, without the rows it was fitted on.

        The result transforms exactly as the pipeline the state came from, and refuses rows
        before the same fit end. The statistics are copied, so the state can be discarded or
        changed afterwards.

        Args:
            specs: The feature specs the state was fitted with, in the same order.
            state: A state from `get_state`, possibly stored and read back.

        Returns:
            The fitted pipeline.

        Raises:
            ValueError: The state lacks a field; its spec names, columns or channel count do not
                match ``specs``; ``fit_end`` is below 1; a statistic is not one finite value per
                column, or a scale is below ``MIN_SCALE``; or the state's own digest does not
                match the digest of the pipeline built from it, which means the statistics
                changed after the state was taken.
        """
        missing = [key for key in _STATE_FIELDS if key not in state]
        if missing:
            raise ValueError(f"pipeline state has no field {', '.join(map(repr, missing))}")
        recorded, given = list(state["spec_names"]), [spec.name for spec in specs]
        if recorded != given:
            raise ValueError(_spec_mismatch(recorded, given))
        pipeline = cls(specs, n_channels=int(state["n_channels"]))
        if list(state["columns"]) != list(pipeline.columns):
            raise ValueError(
                f"the state has {len(state['columns'])} feature columns that do not match the "
                f"{len(pipeline.columns)} these specs produce"
            )
        fit_end = int(state["fit_end"])
        if fit_end < 1:
            raise ValueError(f"state field 'fit_end' must be at least 1, got {fit_end}")
        mean = _state_statistic(state, "mean", len(pipeline.columns))
        scale = _state_statistic(state, "scale", len(pipeline.columns))
        if (scale < MIN_SCALE).any():
            raise ValueError(
                f"state field 'scale' holds {int((scale < MIN_SCALE).sum())} values below "
                f"MIN_SCALE {MIN_SCALE}, which no fit produces"
            )
        pipeline._fit_end, pipeline._mean, pipeline._scale = fit_end, mean, scale
        if pipeline.state_digest() != state["digest"]:
            raise ValueError(
                "the state's digest does not match the pipeline built from it: its fit_end, "
                "columns or statistics changed after the state was taken"
            )
        return pipeline

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
