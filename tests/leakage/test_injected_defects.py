"""Each injected leakage defect, and the one guard that must catch it.

A suite that only shows the guards stay quiet on honest code proves nothing. Here every
known way to leak is injected on purpose and the test asserts which guard fired, not that
"some test failed". The defect ids follow the table of the design document.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import pytest

from tests.conftest import make_entity
from tests.leakage.defects import (
    BFILL,
    CENTERED_ROLLING,
    DEFECTS,
    INTERPOLATE_BOTH,
    WHOLE_SERIES_FFT,
    FullFitPipeline,
    LyingFullFitPipeline,
    caught_by,
    train_test_split_shuffled,
)
from tsguard.data.labels import Phase
from tsguard.data.splits import check_fold_boundaries, check_split_monotonic, walk_forward_folds
from tsguard.errors import (
    GUARD_CAUSAL,
    GUARD_FIT_END,
    GUARD_FOLD,
    GUARD_LABELS,
    GUARD_SPLIT,
)
from tsguard.features.causality import assert_causal, assert_fit_ignores_future
from tsguard.features.pipeline import FeaturePipeline
from tsguard.features.specs import FEATURE_SPECS, FeatureSpec

_Test = TypeVar("_Test", bound=Callable[..., object])

FEATURE_DEFECTS: dict[str, FeatureSpec] = {
    "1": CENTERED_ROLLING,
    "3": BFILL,
    "4": WHOLE_SERIES_FFT,
    "8": INTERPOLATE_BOTH,
}
"""The defects that live in a feature spec, by id."""

EXPECTED_GUARDS = {
    "1": GUARD_CAUSAL,
    "2": GUARD_FIT_END,
    "2b": GUARD_CAUSAL,
    "3": GUARD_CAUSAL,
    "4": GUARD_CAUSAL,
    "5": GUARD_LABELS,
    "5b": GUARD_LABELS,
    "6": GUARD_FOLD,
    "7": GUARD_SPLIT,
    "8": GUARD_CAUSAL,
}
"""The spec's table: which guard must catch which defect."""


def covers(*defect_ids: str) -> Callable[[_Test], _Test]:
    """Record on a test which defects it injects, for `test_every_defect_has_a_test`."""

    def mark(test: _Test) -> _Test:
        vars(test)["defect_ids"] = defect_ids
        return test

    return mark


@covers(*FEATURE_DEFECTS)
@pytest.mark.parametrize(
    "spec",
    [pytest.param(spec, id=f"{ident}-{spec.name}") for ident, spec in FEATURE_DEFECTS.items()],
)
def test_feature_defect_is_caught_by_assert_causal(spec, rng):
    e = make_entity()
    p = FeaturePipeline((*FEATURE_SPECS[:3], spec)).fit(e.full, end_index=e.train_end)
    caught = caught_by(
        lambda: assert_causal(
            lambda X: p.transform(X, start_index=e.train_end),
            e.full,
            t0=e.train_end + 700,
            start=e.train_end,
            rng=rng,
        )
    )
    assert caught == GUARD_CAUSAL


@covers("2")
def test_full_fit_scaler_is_caught_by_fit_end_guard():
    e = make_entity()
    p = FullFitPipeline().fit(e.full, end_index=e.train_end)
    assert p.fit_end == len(e.full)
    assert caught_by(lambda: p.transform(e.full, start_index=e.train_end)) == GUARD_FIT_END
    # The last row the scaler saw is refused too, so an off-by-one in the guard cannot pass.
    assert caught_by(lambda: p.transform(e.full, start_index=len(e.full) - 1)) == GUARD_FIT_END


@covers("2")
def test_full_fit_through_fit_transform_is_caught_by_fit_end_guard():
    e = make_entity()
    p = FullFitPipeline()
    in_sample = p.fit_transform(e.full, end_index=e.train_end)
    assert len(in_sample) == len(e.full)
    assert caught_by(lambda: p.transform(e.full, start_index=e.train_end)) == GUARD_FIT_END


@covers("2b")
def test_lying_scaler_is_caught_by_fit_side_perturbation(rng):
    e = make_entity()
    p = LyingFullFitPipeline().fit(e.full, end_index=e.train_end)
    # The lie works: the guard on the evaluation side sees nothing wrong.
    assert p.fit_end == e.train_end
    assert caught_by(lambda: p.transform(e.full, start_index=e.train_end)) is None
    caught = caught_by(
        lambda: assert_fit_ignores_future(
            lambda X: LyingFullFitPipeline().fit(X, end_index=e.train_end).state_digest(),
            e.full,
            end_index=e.train_end,
            rng=rng,
        )
    )
    assert caught == GUARD_CAUSAL


@covers("2b")
def test_lying_scaler_through_fit_transform_is_caught_by_fit_side_perturbation(rng):
    e = make_entity()

    def digest(X):
        p = LyingFullFitPipeline()
        in_sample = p.fit_transform(X, end_index=e.train_end)
        assert len(in_sample) == e.train_end
        return p.state_digest()

    assert (
        caught_by(lambda: assert_fit_ignores_future(digest, e.full, end_index=e.train_end, rng=rng))
        == GUARD_CAUSAL
    )


@covers("5")
def test_threshold_chosen_on_test_labels_is_caught_by_label_audit():
    e = make_entity()  # a fresh audit starts in the selection phase
    assert caught_by(lambda: e.labels.read("threshold_search")) == GUARD_LABELS


@covers("5b")
def test_threshold_chosen_during_final_evaluation_is_caught_by_label_audit():
    e = make_entity()
    audit = e.labels.audit

    def pick_on_test():
        with audit.phase(Phase.FINAL_EVALUATION):
            best = float(np.quantile(e.labels.read("pick"), 0.5))
            return audit.freeze_threshold(best, caller="pick_on_test")

    assert caught_by(pick_on_test) == GUARD_LABELS
    assert audit.current is Phase.SELECTION
    assert audit.reads == 1  # the read itself was legal; freezing a threshold afterwards was not


@covers("6")
def test_folds_without_purge_are_caught_by_fold_boundary_check():
    folds = walk_forward_folds(2000, purge=0)
    assert caught_by(lambda: check_fold_boundaries(folds, lookback=64)) == GUARD_FOLD


@covers("7")
def test_shuffled_split_is_caught_by_split_monotonic_check(rng):
    tr, te = train_test_split_shuffled(np.arange(100), rng)
    # Each half is a valid index array on its own, so only the time order can be at fault.
    assert tr.dtype == te.dtype == np.int64
    assert np.all(np.diff(tr) > 0)
    assert np.all(np.diff(te) > 0)
    assert caught_by(lambda: check_split_monotonic(tr, te)) == GUARD_SPLIT


def test_every_defect_has_a_test():
    assert set(DEFECTS) == set(EXPECTED_GUARDS)
    assert {ident: guard for ident, (_, guard) in DEFECTS.items()} == EXPECTED_GUARDS
    module = sys.modules[__name__]
    claimed = {
        ident
        for _, test in inspect.getmembers(module, inspect.isfunction)
        if test.__name__.startswith("test_")
        for ident in vars(test).get("defect_ids", ())
    }
    assert claimed == set(DEFECTS)
