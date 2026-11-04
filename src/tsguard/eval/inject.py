"""Seeded synthetic anomalies for choosing a model without test labels.

The test labels of SMD are off limits until the final evaluation, so a detector's threshold and
hyper-parameters are selected on anomalies that are injected into a block of validation rows
instead (Goswami et al., ICLR 2023, arXiv:2210.01078). `inject` plants four kinds of anomaly
(spike, level shift, variance burst, flatline) into a block, labels the planted rows, and
leaves every row outside the planted segments byte-identical to the input.

The injection distribution is a modelling choice, not a measurement of real anomalies. The
draw order is fixed so that a seed reproduces the same segments; it is spelled out in `inject`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

InjectionKind = Literal["spike", "level_shift", "variance_burst", "flatline"]

KINDS: Final[tuple[InjectionKind, ...]] = ("spike", "level_shift", "variance_burst", "flatline")

_KIND_COUNT: Final[int] = len(KINDS)
_MAX_SPIKE_LEN: Final[int] = 3


@dataclass(frozen=True)
class InjectionConfig:
    """How many anomalies to plant, how long they are and how large.

    Attributes:
        min_len: Shortest non-spike segment, in rows.
        max_len: Longest non-spike segment, in rows.
        max_channels: Most channels one segment may touch; a segment touches between 1 and
            this many.
        min_sigma: Smallest magnitude, in units of the reference standard deviation.
        max_sigma: Largest magnitude, in units of the reference standard deviation.
        rows_per_segment: The block is cut into one slot per this many rows (at least 2 slots).
        gap: Rows kept free between the segments of two neighbouring slots, and half of it
            between a segment and the block edge.
        fallback_sigma: The reference standard deviation of a constant channel.
    """

    min_len: int = 3
    max_len: int = 240
    max_channels: int = 8
    min_sigma: float = 3.0
    max_sigma: float = 8.0
    rows_per_segment: int = 1000
    gap: int = 120
    fallback_sigma: float = 0.1


@dataclass(frozen=True)
class InjectedSegment:
    """One planted anomaly, in row positions of the series it was planted into.

    Attributes:
        start: First anomalous row.
        length: Number of anomalous rows.
        kind: Which of the four perturbations was applied.
        channels: The perturbed channels, sorted and distinct.
        magnitude: Size of the perturbation in reference standard deviations; drawn for every
            kind, but a flatline does not use it.
    """

    start: int
    length: int
    kind: InjectionKind
    channels: tuple[int, ...]
    magnitude: float


def reference_sigma(train_rows: np.ndarray, *, fallback: float = 0.1) -> np.ndarray:
    """Estimate the scale of each channel from rows the detector was fitted on.

    Args:
        train_rows: Array of shape ``(rows, channels)``.
        fallback: Value that replaces the standard deviation of a constant channel, so that an
            anomaly on it still has a size.

    Returns:
        The per-channel standard deviation, shape ``(channels,)``, with every zero replaced by
        ``fallback``.
    """
    sigma = np.std(train_rows, axis=0)
    return np.where(sigma == 0.0, fallback, sigma)


def inject(
    X: np.ndarray,
    *,
    start: int,
    end: int,
    sigma: np.ndarray,
    rng: np.random.Generator,
    config: InjectionConfig = InjectionConfig(),  # noqa: B008
) -> tuple[np.ndarray, np.ndarray, tuple[InjectedSegment, ...]]:
    """Plant synthetic anomalies in the rows ``[start, end)`` of ``X``.

    The block is cut into ``n = max(2, block // rows_per_segment)`` slots of
    ``slot = block // n`` rows, and each slot gets one segment, placed so that at least
    ``gap // 2`` rows separate it from the slot edges. For each slot in order the generator is
    drawn from as follows, and the segment is applied before the next slot is drawn:

    1. ``kind = KINDS[rng.integers(4)]``.
    2. ``length``: ``rng.integers(1, 4)`` for a spike, otherwise the rounded exponential of
       ``rng.uniform(log(min_len), log(max_len))``; then capped at ``slot - gap``.
    3. ``offset = rng.integers(0, slot - gap - length + 1)``; the segment starts at
       ``slot_start + gap // 2 + offset``.
    4. ``m = rng.integers(1, max_channels + 1)``, then ``rng.choice(channels, m, replace=False)``.
    5. ``magnitude = rng.uniform(min_sigma, max_sigma)``, then the sign from ``rng.random()``.
    6. Spike and level shift add ``sign * magnitude * sigma`` to the chosen channels, a variance
       burst adds ``rng.standard_normal((length, m)) * magnitude * sigma``, and a flatline
       holds the value of its first row.

    Args:
        X: Array of shape ``(rows, channels)``. It is never modified.
        start: First row of the block.
        end: Exclusive end of the block.
        sigma: Per-channel scale, shape ``(channels,)``; see `reference_sigma`.
        rng: The only source of randomness.
        config: Sizes and counts of the segments.

    Returns:
        A float64 copy of ``X`` with the anomalies planted, int8 labels for the rows
        ``[start, end)`` (1 on a planted row), and the planted segments in row order.

    Raises:
        ValueError: The block does not lie inside ``X``, ``sigma`` does not have one entry per
            channel, or a slot is too small to hold ``gap + min_len`` rows.
    """
    rows, n_channels = X.shape
    if not 0 <= start < end <= rows:
        raise ValueError(f"block [{start}, {end}) does not lie inside the {rows} rows of X")
    if sigma.shape != (n_channels,):
        raise ValueError(f"sigma must have shape ({n_channels},), got {sigma.shape}")
    block = end - start
    n = max(2, block // config.rows_per_segment)
    slot = block // n
    if slot < config.gap + config.min_len:
        raise ValueError(
            f"cannot plant {n} segments in the {block} rows [{start}, {end}): a slot of {slot} "
            f"rows is shorter than gap + min_len = {config.gap + config.min_len}"
        )

    out = np.array(X, dtype=np.float64, copy=True)
    labels = np.zeros(block, dtype=np.int8)
    segments: list[InjectedSegment] = []
    for i in range(n):
        kind = KINDS[int(rng.integers(_KIND_COUNT))]
        if kind == "spike":
            length = int(rng.integers(1, _MAX_SPIKE_LEN + 1))
        else:
            length = round(
                math.exp(rng.uniform(math.log(config.min_len), math.log(config.max_len)))
            )
        length = min(length, slot - config.gap)
        offset = int(rng.integers(0, slot - config.gap - length + 1))
        s = start + i * slot + config.gap // 2 + offset
        m = int(rng.integers(1, config.max_channels + 1))
        channels = tuple(sorted(int(c) for c in rng.choice(n_channels, size=m, replace=False)))
        magnitude = float(rng.uniform(config.min_sigma, config.max_sigma))
        sign = 1.0 if rng.random() < 0.5 else -1.0

        picked = list(channels)
        planted = slice(s, s + length)
        if kind == "variance_burst":
            noise = rng.standard_normal((length, m))
            out[planted, picked] += noise * magnitude * sigma[picked]
        elif kind == "flatline":
            out[planted, picked] = out[s, picked]
        else:
            out[planted, picked] += sign * magnitude * sigma[picked]
        labels[s - start : s - start + length] = 1
        segments.append(InjectedSegment(s, length, kind, channels, magnitude))
    return out, labels, tuple(segments)
