"""Shared pytest fixtures and hooks."""

from __future__ import annotations

import os
from collections.abc import Generator

import numpy as np
import pytest

pytest_plugins = ["pytester"]

SEED_ENV = "TSGUARD_TEST_SEED"
DEFAULT_SEED = 20261103
SEED_PROPERTY = "seed"


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
