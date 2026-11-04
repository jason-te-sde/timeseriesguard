"""The four evaluation protocols: point-wise, point-adjust, PA%K and event-level.

A detector emits one score per row. How those scores are turned into a number decides what
the number means:

* ``point_wise`` scores every row on its own.
* ``point_adjust`` (PA) credits a whole anomaly segment when any of its rows is flagged. A
  detector that knows nothing reaches state-of-the-art F1 under it, because a segment of
  hundreds of rows almost surely contains one high random score.
* ``pa_percent_k`` (Kim et al., AAAI 2022, "Towards a Rigorous Evaluation of Time-series
  Anomaly Detection") credits a segment only when the detected fraction of it exceeds K
  percent.
* ``event_level`` is the protocol this project reports. Each anomaly segment is one positive
  sample (its highest score), and the normal rows between segments are cut into fixed-length
  chunks, one negative sample each (its highest score). A cooldown after every segment keeps
  the aftermath of an anomaly from counting as false alarms. Random scores cannot game it.

Every protocol is a *view*: a ``(scores, labels)`` pair on which the same three numbers are
computed (average precision, F1 at a threshold fixed beforehand, and the best F1 any
threshold could reach). `evaluate` runs all four.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from sklearn.metrics import average_precision_score

from tsguard.data.labels import FrozenThreshold

ProtocolName = Literal["point_wise", "point_adjust", "pa_percent_k", "event_level"]

PROTOCOLS: Final[tuple[ProtocolName, ...]] = (
    "point_wise",
    "point_adjust",
    "pa_percent_k",
    "event_level",
)
"""The protocols in report order."""

_NO_ANOMALY: Final = "no anomaly segment in labels"


@dataclass(frozen=True)
class ProtocolParams:
    """The constants of the protocols.

    Attributes:
        chunk: Length of a normal chunk in the event-level protocol, in rows.
        cooldown: Rows after the end of each anomaly segment that the event-level protocol
            leaves out.
        k_percent: K of PA%K, in percent.
    """

    chunk: int = 60
    cooldown: int = 60
    k_percent: int = 20


@dataclass(frozen=True)
class ProtocolScore:
    """What one protocol says about one set of scores.

    Attributes:
        auc_pr: Average precision on the protocol's view.
        f1: F1 on the view at the threshold that was frozen before the labels were read.
        f1_oracle: The best F1 any threshold reaches on the view. It needs the labels to be
            found, so it is an upper bound and never a result.
    """

    auc_pr: float
    f1: float
    f1_oracle: float


def _check_labels(labels: np.ndarray) -> np.ndarray:
    """Return ``labels`` as a 1-D ``int8`` array, or raise if they are not 0 or 1."""
    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {y.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must each be 0 or 1")
    return y.astype(np.int8)


def _check(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``scores`` as ``float64`` and ``labels`` as ``int8``, or raise if they do not fit."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {s.shape}")
    y = _check_labels(labels)
    if len(s) != len(y):
        raise ValueError(f"scores and labels differ in length: {len(s)} vs {len(y)}")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    return s, y


def segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """Find the maximal runs of anomalous rows.

    Args:
        labels: One label per row, each 0 or 1.

    Returns:
        The runs as half-open ``(start, end)`` row ranges, in order. Empty if no row is
        anomalous.

    Raises:
        ValueError: ``labels`` is not 1-D or holds anything but 0 and 1.
    """
    y = _check_labels(labels)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], y, [0]))))
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2], strict=True)]


def adjust_point_adjust(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Apply point adjust: every row of an anomaly segment takes the segment's top score.

    Args:
        scores: One finite anomaly score per row.
        labels: One label per row, each 0 or 1.

    Returns:
        A new array; ``scores`` is not modified. Normal rows keep their score.

    Raises:
        ValueError: The inputs differ in length, ``scores`` is not finite or not 1-D, or
            ``labels`` is not 1-D or holds anything but 0 and 1.
    """
    s, y = _check(scores, labels)
    out = s.copy()
    for start, end in segments(y):
        out[start:end] = s[start:end].max()
    return out


def adjust_pa_percent_k(scores: np.ndarray, labels: np.ndarray, *, k_percent: int) -> np.ndarray:
    """Apply PA%K: a segment is lifted only if more than K percent of it is detected.

    Each anomaly row's score becomes ``max(own, kth)``, where ``kth`` is the ``k``-th
    largest score of its segment and ``k = min(len, floor(k_percent * len / 100) + 1)``.
    A threshold at or below ``kth`` then flags at least ``k`` rows of the segment, and the
    segment counts as detected exactly when it flags more than ``k_percent`` percent of its
    rows. (The smallest count that exceeds K percent of ``len`` is ``floor(K * len / 100) +
    1``; ``ceil(K * len / 100)`` would accept a count equal to K percent when ``K * len``
    divides by 100.) ``k_percent=0`` is point adjust and ``k_percent=100`` is point-wise.

    Args:
        scores: One finite anomaly score per row.
        labels: One label per row, each 0 or 1.
        k_percent: K, from 0 to 100.

    Returns:
        A new array; ``scores`` is not modified. Normal rows keep their score.

    Raises:
        ValueError: ``k_percent`` is outside 0 to 100, the inputs differ in length,
            ``scores`` is not finite or not 1-D, or ``labels`` is not 1-D or holds anything
            but 0 and 1.
    """
    if not 0 <= k_percent <= 100:
        raise ValueError(f"k_percent must be between 0 and 100, got {k_percent}")
    s, y = _check(scores, labels)
    out = s.copy()
    for start, end in segments(y):
        length = end - start
        k = min(length, k_percent * length // 100 + 1)
        kth = np.partition(s[start:end], length - k)[length - k]
        out[start:end] = np.maximum(s[start:end], kth)
    return out


def event_samples(
    scores: np.ndarray, labels: np.ndarray, *, chunk: int, cooldown: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a series into the samples of the event-level protocol.

    Every anomaly segment is one positive sample whose score is the segment's maximum. The
    normal rows that remain after dropping the ``cooldown`` rows that follow each segment's
    end form maximal runs; each run is cut from its start into consecutive chunks of
    ``chunk`` rows (the last one may be shorter), and every chunk is one negative sample
    whose score is its maximum. A chunk never spans a segment or a dropped row.

    Args:
        scores: One finite anomaly score per row.
        labels: One label per row, each 0 or 1, with at least one anomaly.
        chunk: Rows per negative sample, at least 1.
        cooldown: Rows dropped after the end of every segment, at least 0.

    Returns:
        ``(sample_scores, sample_labels)``, ordered by the first row of each sample.

    Raises:
        ValueError: ``chunk`` is below 1, ``cooldown`` is negative, ``labels`` has no
            anomaly ("no anomaly segment in labels"), the inputs differ in length,
            ``scores`` is not finite or not 1-D, or ``labels`` is not 1-D or holds
            anything but 0 and 1.
    """
    if chunk < 1:
        raise ValueError(f"chunk must be at least 1, got {chunk}")
    if cooldown < 0:
        raise ValueError(f"cooldown must be at least 0, got {cooldown}")
    s, y = _check(scores, labels)
    anomalies = segments(y)
    if not anomalies:
        raise ValueError(_NO_ANOMALY)
    keep = y == 0
    for _, end in anomalies:
        keep[end : end + cooldown] = False
    firsts = [np.array([start for start, _ in anomalies])]
    values = [np.array([s[start:end].max() for start, end in anomalies])]
    kinds = [np.ones(len(anomalies), dtype=np.int8)]
    for run_start, run_end in segments(keep):
        starts = np.arange(run_start, run_end, chunk)
        firsts.append(starts)
        values.append(np.maximum.reduceat(s[run_start:run_end], starts - run_start))
        kinds.append(np.zeros(len(starts), dtype=np.int8))
    order = np.argsort(np.concatenate(firsts), kind="stable")
    return np.concatenate(values)[order], np.concatenate(kinds)[order]


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute average precision, the area under the precision-recall curve.

    Args:
        scores: One finite score per sample.
        labels: One label per sample, each 0 or 1, with at least one positive.

    Returns:
        ``sklearn.metrics.average_precision_score(labels, scores)``.

    Raises:
        ValueError: ``labels`` has no positive ("no anomaly segment in labels"), the inputs
            differ in length, ``scores`` is not finite or not 1-D, or ``labels`` is not 1-D
            or holds anything but 0 and 1.
    """
    s, y = _check(scores, labels)
    if not y.any():
        raise ValueError(_NO_ANOMALY)
    return float(average_precision_score(y, s))


def f1_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Compute F1 when every sample with ``score >= threshold`` is predicted anomalous.

    Args:
        scores: One finite score per sample.
        labels: One label per sample, each 0 or 1.
        threshold: The decision threshold; must be finite.

    Returns:
        The F1 score, or 0.0 when nothing is predicted or nothing is hit.

    Raises:
        ValueError: ``threshold`` is not finite, the inputs differ in length, ``scores`` is
            not finite or not 1-D, or ``labels`` is not 1-D or holds anything but 0 and 1.
    """
    s, y = _check(scores, labels)
    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold!r}")
    predicted = s >= threshold
    n_predicted = int(predicted.sum())
    if n_predicted == 0:
        return 0.0
    hits = int((predicted & (y == 1)).sum())
    return 2 * hits / (n_predicted + int(y.sum()))


def best_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Find the threshold that maximizes F1, among the distinct score values.

    Samples that share a score are predicted together. When two thresholds reach the same
    F1, the higher one wins.

    Args:
        scores: One finite score per sample, at least one sample.
        labels: One label per sample, each 0 or 1.

    Returns:
        ``(f1, threshold)``. With no positive label every F1 is 0 and the threshold is the
        highest score.

    Raises:
        ValueError: ``scores`` is empty, the inputs differ in length, ``scores`` is not
            finite or not 1-D, or ``labels`` is not 1-D or holds anything but 0 and 1.
    """
    s, y = _check(scores, labels)
    if len(s) == 0:
        raise ValueError("scores must not be empty")
    order = np.argsort(-s, kind="stable")
    ranked, hits = s[order], np.cumsum(y[order], dtype=np.int64)
    last_of_each_score = np.append(np.flatnonzero(ranked[1:] != ranked[:-1]), len(s) - 1)
    predicted = last_of_each_score + 1
    f1 = 2 * hits[last_of_each_score] / (predicted + int(y.sum()))
    best = int(np.argmax(f1))
    return float(f1[best]), float(ranked[last_of_each_score[best]])


def protocol_view(
    name: ProtocolName, scores: np.ndarray, labels: np.ndarray, params: ProtocolParams
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(scores, labels)`` a protocol scores on.

    Args:
        name: One of `PROTOCOLS`.
        scores: One finite anomaly score per row.
        labels: One label per row, each 0 or 1.
        params: The protocol constants.

    Returns:
        The point-wise view is the inputs; the two adjusted views keep the labels and
        replace the scores; the event-level view is `event_samples`.

    Raises:
        ValueError: ``name`` is not a protocol, or the inputs are invalid for it (see
            `adjust_point_adjust` and `event_samples`).
    """
    if name == "point_wise":
        return _check(scores, labels)
    if name == "point_adjust":
        return adjust_point_adjust(scores, labels), _check(scores, labels)[1]
    if name == "pa_percent_k":
        adjusted = adjust_pa_percent_k(scores, labels, k_percent=params.k_percent)
        return adjusted, _check(scores, labels)[1]
    if name == "event_level":
        return event_samples(scores, labels, chunk=params.chunk, cooldown=params.cooldown)
    raise ValueError(f"unknown protocol {name!r}; valid names: {', '.join(PROTOCOLS)}")


def event_level_auc_pr(scores: np.ndarray, labels: np.ndarray, params: ProtocolParams) -> float:
    """Compute average precision on the event-level samples.

    Args:
        scores: One finite anomaly score per row.
        labels: One label per row, each 0 or 1, with at least one anomaly.
        params: Supplies ``chunk`` and ``cooldown``.

    Returns:
        The event-level AUC-PR.

    Raises:
        ValueError: See `event_samples`.
    """
    return average_precision(*protocol_view("event_level", scores, labels, params))


def select_threshold(scores: np.ndarray, labels: np.ndarray, params: ProtocolParams) -> float:
    """Pick the threshold that maximizes F1 over the event-level samples.

    Used only on validation data whose labels are synthetic: the threshold must be fixed
    before any test label is read.

    Args:
        scores: One finite anomaly score per row.
        labels: One label per row, each 0 or 1, with at least one anomaly.
        params: Supplies ``chunk`` and ``cooldown``.

    Returns:
        The threshold of `best_f1` over `event_samples`; it is one of the sample scores.

    Raises:
        ValueError: See `event_samples`.
    """
    sample_scores, sample_labels = protocol_view("event_level", scores, labels, params)
    return best_f1(sample_scores, sample_labels)[1]


def evaluate(
    scores: np.ndarray, labels: np.ndarray, threshold: FrozenThreshold, params: ProtocolParams
) -> dict[ProtocolName, ProtocolScore]:
    """Score a detector's output under all four protocols.

    Args:
        scores: One finite anomaly score per row.
        labels: One label per row, each 0 or 1, with at least one anomaly.
        threshold: The decision threshold, frozen before any test label was read.
        params: The protocol constants.

    Returns:
        For each protocol in `PROTOCOLS`: average precision on its view, F1 on the view at
        ``threshold.value``, and the best F1 any threshold reaches on the view.

    Raises:
        TypeError: ``threshold`` is not a `FrozenThreshold`.
        ValueError: The inputs are invalid (see `event_samples`).
    """
    if not isinstance(threshold, FrozenThreshold):
        raise TypeError(
            "threshold must be a FrozenThreshold from LabelAudit.freeze_threshold, "
            f"got {type(threshold).__name__}"
        )
    result: dict[ProtocolName, ProtocolScore] = {}
    for name in PROTOCOLS:
        view_scores, view_labels = protocol_view(name, scores, labels, params)
        result[name] = ProtocolScore(
            auc_pr=average_precision(view_scores, view_labels),
            f1=f1_at(view_scores, view_labels, threshold.value),
            f1_oracle=best_f1(view_scores, view_labels)[0],
        )
    return result
