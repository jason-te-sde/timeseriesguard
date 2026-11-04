"""The data contract every entity must satisfy before anything else touches it.

Six rules, checked in this order; the first violation raises a `ContractError` whose message
starts with the entity name and names the rule:

1. ``train`` and ``test`` are 2-D float arrays with 38 columns.
2. Both have at least 1,000 rows.
3. All feature values are finite.
4. All feature values lie within ``[0.0, 1.0]``.
5. ``labels`` is 1-D with values in ``{0, 1}``.
6. ``len(labels) == len(test)``.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from tsguard.errors import ContractError

N_CHANNELS: Final = 38
"""Number of metric channels of every SMD machine."""

MIN_ROWS: Final = 1_000
"""Fewest rows a train or test split may have."""


def check_entity(name: str, train: np.ndarray, test: np.ndarray, labels: np.ndarray) -> None:
    """Check one entity against the six contract rules.

    Args:
        name: Entity name, used to start every error message.
        train: Training features, shape ``(n_train, 38)``.
        test: Test features, shape ``(n_test, 38)``.
        labels: Test labels, shape ``(n_test,)``, values 0 or 1.

    Raises:
        ContractError: On the first violated rule; the message starts with ``name`` and
            says which rule failed and where (split, row, column).
    """
    splits = (("train", train), ("test", test))
    for split, array in splits:
        _check_shape(name, split, array)
    for split, array in splits:
        _check_rows(name, split, array)
    for split, array in splits:
        _check_finite(name, split, array)
    for split, array in splits:
        _check_range(name, split, array)
    _check_labels(name, labels)
    if len(labels) != len(test):
        raise ContractError(
            f"{name}: rule 6 (label length): labels has {_count(len(labels), 'row')} but test has "
            f"{len(test):,}; there must be one label per test row"
        )


def _count(count: int, noun: str) -> str:
    """Return ``count`` with a thousands separator and ``noun`` pluralised by an ``s``."""
    return f"{count:,} {noun}" if count == 1 else f"{count:,} {noun}s"


def _check_shape(name: str, split: str, array: np.ndarray) -> None:
    valid = (
        isinstance(array, np.ndarray)
        and array.ndim == 2
        and np.issubdtype(array.dtype, np.floating)
        and array.shape[1] == N_CHANNELS
    )
    if not valid:
        described = (
            f"dtype {array.dtype} and shape {array.shape}"
            if isinstance(array, np.ndarray)
            else f"type {type(array).__name__}"
        )
        raise ContractError(
            f"{name}: rule 1 (shape): {split} must be a 2-D float array with {N_CHANNELS} "
            f"columns, got {described}"
        )


def _check_rows(name: str, split: str, array: np.ndarray) -> None:
    if len(array) < MIN_ROWS:
        raise ContractError(
            f"{name}: rule 2 (rows): {split} has {_count(len(array), 'row')}, need at least "
            f"{_count(MIN_ROWS, 'row')}"
        )


def _check_finite(name: str, split: str, array: np.ndarray) -> None:
    bad = ~np.isfinite(array)
    if bad.any():
        row, column = (int(i) for i in np.argwhere(bad)[0])
        raise ContractError(
            f"{name}: rule 3 (finite): {split} has {_count(int(bad.sum()), 'non-finite value')}, "
            f"first at row {row}, column {column}"
        )


def _check_range(name: str, split: str, array: np.ndarray) -> None:
    bad = (array < 0.0) | (array > 1.0)
    if bad.any():
        row, column = (int(i) for i in np.argwhere(bad)[0])
        first = float(array[row, column])
        raise ContractError(
            f"{name}: rule 4 (range): {split} has {_count(int(bad.sum()), 'value')} "
            f"outside [0.0, 1.0], first at row {row}, column {column} ({first!r})"
        )


def _check_labels(name: str, labels: np.ndarray) -> None:
    if not isinstance(labels, np.ndarray) or labels.ndim != 1:
        shape = labels.shape if isinstance(labels, np.ndarray) else type(labels).__name__
        raise ContractError(f"{name}: rule 5 (labels): labels must be 1-D, got {shape}")
    bad = ~np.isin(labels, (0, 1))
    if bad.any():
        row = int(np.argmax(bad))
        count = int(bad.sum())
        raise ContractError(
            f"{name}: rule 5 (labels): labels must have values in {{0, 1}}; "
            f"{count:,} {'does' if count == 1 else 'do'} not, first at row {row} "
            f"({labels[row].item()!r})"
        )
