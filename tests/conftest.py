"""Shared pytest fixtures and hooks."""

from __future__ import annotations

import os
from collections.abc import Generator, Sequence
from pathlib import Path

import numpy as np
import pytest

from tsguard.data.labels import AuditedLabels, LabelAudit
from tsguard.data.smd import EntityData, default_data_dir, verify
from tsguard.errors import TsguardError
from tsguard.experiment.reproduce import Profile
from tsguard.experiment.runner import RunConfig

pytest_plugins = ["pytester"]

SEED_ENV = "TSGUARD_TEST_SEED"
DEFAULT_SEED = 20261103
SEED_PROPERTY = "seed"

AR_COEFFICIENT = 0.8
DAILY_PERIOD = 1440
CONSTANT_CHANNEL = 37
SHIFTED_CHANNELS = 5
ANOMALY_SHIFT = 0.3
BAND_LOW, BAND_HIGH = 0.05, 0.95


@pytest.fixture
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """Return a Generator seeded from ``TSGUARD_TEST_SEED`` (default 20261103).

    The seed is recorded on the test node so a failure report can print it.
    """
    raw = os.environ.get(SEED_ENV, str(DEFAULT_SEED))
    try:
        seed = int(raw)
    except ValueError:
        raise ValueError(f"{SEED_ENV} must be an integer, got {raw!r}") from None
    request.node.user_properties.append((SEED_PROPERTY, seed))
    return np.random.default_rng(seed)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Append ``seed=<n>`` to the report of a failed test that used the ``rng`` fixture."""
    report = yield
    if report.failed:
        for name, value in item.user_properties:
            if name == SEED_PROPERTY:
                report.sections.append((SEED_PROPERTY, f"seed={value}"))
    return report


@pytest.fixture(scope="session")
def smd_dir() -> Path:
    """Return the verified SMD cache directory, or fail (never skip) if it is not usable."""
    root = default_data_dir()
    problem: TsguardError | None = None
    try:
        verify(dest=root)
    except TsguardError as error:
        problem = error
    if problem is not None:
        pytest.fail(f"SMD data missing: run `uv run tsguard data fetch` ({problem})", pytrace=False)
    return root


def make_entity(
    *,
    n_train: int = 2000,
    n_test: int = 2000,
    channels: int = 38,
    seed: int = 0,
    anomalies: Sequence[tuple[int, int]] = ((600, 20), (1400, 5)),
    audit: LabelAudit | None = None,
    name: str = "synthetic-0",
) -> EntityData:
    """Build a small synthetic entity that satisfies the data contract.

    Each channel is an AR(1) process (coefficient 0.8) plus a sinusoid of period 1440 with a
    random phase and amplitude, min-max scaled over train and test together into
    ``[0.05, 0.95]``. Channel 37 is constant at 0.5. Every ``(start, length)`` in
    ``anomalies`` adds 0.3 to channels 0-4 of the test rows ``[start, start + length)``
    (clipped to 1.0) and labels those rows 1. The result depends only on ``seed``.

    Args:
        n_train: Number of training rows.
        n_test: Number of test rows.
        channels: Number of channels; the contract needs 38.
        seed: Seed of the generator that draws the signal.
        anomalies: ``(start, length)`` segments, in test-row coordinates.
        audit: The audit the labels report to; a fresh `LabelAudit` if omitted. Reach it
            through ``entity.labels.audit``.
        name: Entity name.

    Returns:
        The entity, with read-only ``train`` and ``test``.

    Raises:
        ValueError: A segment does not fit inside the test rows.
    """
    for start, length in anomalies:
        if start < 0 or length < 1 or start + length > n_test:
            raise ValueError(
                f"anomaly (start={start}, length={length}) does not fit inside the "
                f"{n_test} test rows"
            )
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    noise = rng.standard_normal((n, channels))
    ar = np.empty((n, channels))
    ar[0] = noise[0] / np.sqrt(1.0 - AR_COEFFICIENT**2)
    for t in range(1, n):
        ar[t] = AR_COEFFICIENT * ar[t - 1] + noise[t]
    phase = rng.uniform(0.0, 2.0 * np.pi, channels)
    amplitude = rng.uniform(1.0, 3.0, channels)
    angle = 2.0 * np.pi * np.arange(n)[:, None] / DAILY_PERIOD + phase
    signal = ar + amplitude * np.sin(angle)
    low, high = signal.min(axis=0), signal.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    signal = BAND_LOW + (BAND_HIGH - BAND_LOW) * (signal - low) / span
    if channels > CONSTANT_CHANNEL:
        signal[:, CONSTANT_CHANNEL] = 0.5
    train, test = signal[:n_train].copy(), signal[n_train:].copy()
    labels = np.zeros(n_test, dtype=np.int8)
    for start, length in anomalies:
        rows = slice(start, start + length)
        test[rows, :SHIFTED_CHANNELS] = np.minimum(
            test[rows, :SHIFTED_CHANNELS] + ANOMALY_SHIFT, 1.0
        )
        labels[rows] = 1
    train.flags.writeable = False
    test.flags.writeable = False
    return EntityData(
        name=name,
        train=train,
        test=test,
        labels=AuditedLabels(labels, audit=LabelAudit() if audit is None else audit, entity=name),
    )


TINY_N_TRAIN = 3000
TINY_N_TEST = 2000
TINY_SEGMENTS_PER_ENTITY = 2
TINY = Profile(
    "tiny",
    ("synthetic-0", "synthetic-1"),
    ("synthetic-0",),
    1,
    (0,),
    RunConfig(ae_epochs=1, threads=1),
)
"""Two synthetic machines, one tuning trial, one ablation seed and one thread: seconds per run."""


def load_synthetic(name: str, audit: LabelAudit) -> EntityData:
    """Load ``synthetic-<i>`` for `TINY`: the entity `make_entity` builds from seed ``i``.

    Matches the loader signature of ``reproduce``, so a test passes it as ``loader=``.
    """
    index = int(name.rsplit("-", 1)[1])
    return make_entity(n_train=TINY_N_TRAIN, n_test=TINY_N_TEST, seed=index, name=name, audit=audit)
