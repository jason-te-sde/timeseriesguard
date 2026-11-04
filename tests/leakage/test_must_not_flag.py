"""Honest pipelines that the leakage guards must leave alone.

A checker that flags everything is worthless, so each case below is legitimate and must come
back clean: the guard that catches the injected defects must not cry wolf on lags, trailing
and expanding windows, independent entities, or the full registry.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from tests.conftest import make_entity
from tests.leakage.defects import GAP_LEVEL, caught_by
from tsguard.data.smd import EntityData
from tsguard.features.causality import assert_causal, assert_fit_ignores_future
from tsguard.features.pipeline import FeaturePipeline
from tsguard.features.specs import FEATURE_SPECS, MAX_LOOKBACK, BlockCache, FeatureSpec

T0_OFFSET = 700  # the perturbed rows start this far into the test rows


def expanding_mean_spec() -> FeatureSpec:
    """A spec whose value at row ``t`` is the mean of every row of the block up to ``t``."""

    def fn(block: np.ndarray, cache: BlockCache) -> np.ndarray:
        counts = np.arange(1, len(block) + 1, dtype=np.float64)[:, None]
        return np.asarray(np.cumsum(block, axis=0) / counts)

    return FeatureSpec("expanding_mean", "expanding", MAX_LOOKBACK, "channel", fn)


def forward_fill_spec() -> FeatureSpec:
    """The twin of the backward-fill defect: the same gaps, filled from the value before them.

    The mask is `GAP_LEVEL`, which reads each value on its own, so the fill direction is
    the only difference from `BFILL`. A fill that reads only earlier rows is causal, so this
    must come back clean; if it were flagged, the defect would be caught for its mask.
    """

    def fn(block: np.ndarray, cache: BlockCache) -> np.ndarray:
        filled = pd.DataFrame(block).where(block > GAP_LEVEL).ffill().fillna(0.0)
        return np.asarray(filled.to_numpy(dtype=np.float64))

    return FeatureSpec("ffill_gaps", "fill", MAX_LOOKBACK, "channel", fn)


def flagged_by(
    specs: Sequence[FeatureSpec], entity: EntityData, rng: np.random.Generator
) -> str | None:
    """Return the guard that flags a pipeline of ``specs`` on ``entity``, or None.

    Runs both leakage checks: the evaluation side (`assert_causal` on the transform) and
    the fit side (`assert_fit_ignores_future` on the fitted state).
    """
    p = FeaturePipeline(specs).fit(entity.full, end_index=entity.train_end)
    evaluation = caught_by(
        lambda: assert_causal(
            lambda X: p.transform(X, start_index=entity.train_end),
            entity.full,
            t0=entity.train_end + T0_OFFSET,
            start=entity.train_end,
            rng=rng,
        )
    )
    fit = caught_by(
        lambda: assert_fit_ignores_future(
            lambda X: FeaturePipeline(specs).fit(X, end_index=entity.train_end).state_digest(),
            entity.full,
            end_index=entity.train_end,
            rng=rng,
        )
    )
    return evaluation or fit


def test_lag_features_are_not_flagged(rng):
    lags = tuple(spec for spec in FEATURE_SPECS if spec.family == "lag")
    assert len(lags) == 10
    assert flagged_by(lags, make_entity(), rng) is None


def test_trailing_and_expanding_windows_are_not_flagged(rng):
    e = make_entity()
    rolling = tuple(spec for spec in FEATURE_SPECS if spec.family.startswith("rolling"))
    assert len(rolling) > 10
    assert flagged_by((*rolling, expanding_mean_spec()), e, rng) is None

    def expanding_mean_of_series(X):
        return np.cumsum(X, axis=0) / np.arange(1, len(X) + 1)[:, None]

    assert (
        caught_by(lambda: assert_causal(expanding_mean_of_series, e.full, t0=1000, rng=rng)) is None
    )


def test_gaps_filled_forward_are_not_flagged(rng):
    specs = (*FEATURE_SPECS[:3], forward_fill_spec())
    assert flagged_by(specs, make_entity(), rng) is None


def test_entities_are_independent(rng):
    a, b = make_entity(seed=0, name="synthetic-a"), make_entity(seed=1, name="synthetic-b")
    pa = FeaturePipeline().fit(a.full, end_index=a.train_end)
    pb = FeaturePipeline().fit(b.full, end_index=b.train_end)
    assert pa.state_digest() != pb.state_digest()  # the two entities really differ
    digest_before = pa.state_digest()
    bytes_before = pa.transform(a.full, start_index=a.train_end).tobytes()

    # Replace entity B entirely and process it again, in the same process, after A was fitted.
    noise = rng.uniform(0.0, 1.0, size=b.full.shape)
    other = FeaturePipeline().fit(noise, end_index=b.train_end)
    other.transform(noise, start_index=b.train_end)

    assert pa.state_digest() == digest_before
    assert pa.transform(a.full, start_index=a.train_end).tobytes() == bytes_before
    caught = caught_by(
        lambda: assert_causal(
            lambda X: pa.transform(X, start_index=a.train_end),
            a.full,
            t0=a.train_end + T0_OFFSET,
            start=a.train_end,
            rng=rng,
        )
    )
    assert caught is None


def test_full_registry_is_not_flagged(rng):
    assert flagged_by(FEATURE_SPECS, make_entity(), rng) is None
