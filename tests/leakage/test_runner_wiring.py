"""The runner must call every guard itself.

The guards are tested in isolation elsewhere. Here a leaky component is handed to the real
`EntityRunner`, and the test asserts which guard stopped `select`: if the runner stopped
calling a guard, the leak would pass silently and the matching test below would fail.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.conftest import make_entity
from tests.leakage.defects import (
    CENTERED_ROLLING,
    FullFitPipeline,
    LyingFullFitPipeline,
    caught_by,
    train_test_split_shuffled,
)
from tsguard.data.labels import LabelAudit, Phase
from tsguard.data.smd import EntityData
from tsguard.data.splits import walk_forward_folds
from tsguard.errors import (
    GUARD_CAUSAL,
    GUARD_FIT_END,
    GUARD_FOLD,
    GUARD_LABELS,
    GUARD_SPLIT,
    LeakageError,
)
from tsguard.experiment.runner import EntityRunner, RunConfig
from tsguard.features.specs import FEATURE_SPECS
from tsguard.models import Params, build_detector
from tsguard.models.base import Detector, InputKind
from tsguard.models.space import DEFAULT_PARAMS

CONFIG = RunConfig(ae_epochs=1, threads=1)
ZSCORE = DEFAULT_PARAMS["zscore"]


def entity() -> EntityData:
    return make_entity(n_train=3000, n_test=2000)


def select_with(data: EntityData, **kwargs: object) -> str | None:
    runner = EntityRunner(data, data.labels.audit, CONFIG, **kwargs)  # type: ignore[arg-type]
    return caught_by(lambda: runner.select("zscore", ZSCORE))


def test_runner_catches_a_centered_feature():
    specs = (*FEATURE_SPECS[:3], CENTERED_ROLLING)
    assert select_with(entity(), specs=specs) == GUARD_CAUSAL


def test_runner_catches_a_full_fit_scaler():
    assert select_with(entity(), pipeline_factory=FullFitPipeline) == GUARD_FIT_END


def test_runner_catches_a_lying_scaler():
    assert select_with(entity(), pipeline_factory=LyingFullFitPipeline) == GUARD_CAUSAL


def test_runner_catches_unpurged_folds():
    assert select_with(entity(), fold_fn=lambda n: walk_forward_folds(n, purge=0)) == GUARD_FOLD


def test_runner_catches_a_shuffled_split(rng):
    def shuffled(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
        return train_test_split_shuffled(np.arange(n_train + n_test), rng)

    assert select_with(entity(), split_fn=shuffled) == GUARD_SPLIT


def test_runner_catches_threshold_search_on_test_labels():
    data = entity()

    def peek(scores: np.ndarray, labels: np.ndarray, params: object) -> float:
        return float(np.quantile(data.labels.read("peek"), 0.5))

    assert select_with(data, threshold_fn=peek) == GUARD_LABELS


def test_honest_runner_raises_nothing():
    data = entity()
    audit: LabelAudit = data.labels.audit
    runner = EntityRunner(data, audit, CONFIG)
    selection = runner.select("zscore", ZSCORE)
    evaluation = runner.evaluate(selection)
    assert audit.reads == 1
    assert selection.threshold.reads_before == 0
    assert evaluation.protocols["event_level"].auc_pr > 0.0


class LookaheadDetector:
    """A detector whose score at row t is the mean of the features of row t + 1."""

    name = "lookahead"
    input_kind: InputKind = "features"
    lookback = 1

    def fit(self, X: np.ndarray) -> None:
        pass

    def score(self, X: np.ndarray, *, context: int) -> np.ndarray:
        rows = X[context:].astype(np.float64)
        return np.asarray(np.abs(np.roll(rows, -1, axis=0)).mean(axis=1), dtype=np.float64)

    def state_digest(self) -> str:
        return "0" * 64


def with_lookahead_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register a detector named ``lookahead`` next to the real ones."""

    def build(name: str, params: Params, **kwargs: Any) -> Detector:
        if name == "lookahead":
            return LookaheadDetector()
        return build_detector(name, params, **kwargs)

    monkeypatch.setattr("tsguard.experiment.runner.build_detector", build)


def test_runner_catches_a_detector_that_reads_the_future_after_an_honest_one(monkeypatch):
    # The honest model is selected first, so the per-kind checks are already done: only a check
    # that runs for every model can see the second one.
    with_lookahead_model(monkeypatch)
    data = entity()
    runner = EntityRunner(data, data.labels.audit, CONFIG)
    runner.select("zscore", ZSCORE)
    assert caught_by(lambda: runner.select("lookahead", {})) == GUARD_CAUSAL


def test_the_causal_checks_are_what_catches_the_future_readers(monkeypatch):
    # With the checks switched off neither leak is seen, so no other guard stops them.
    with_lookahead_model(monkeypatch)
    config = RunConfig(ae_epochs=1, threads=1, causal_check=False)
    data = entity()
    runner = EntityRunner(data, data.labels.audit, config)
    assert caught_by(lambda: runner.select("lookahead", {})) is None
    specs = (*FEATURE_SPECS[:3], CENTERED_ROLLING)
    centered = EntityRunner(data, data.labels.audit, config, specs=specs)
    assert caught_by(lambda: centered.select("zscore", ZSCORE)) is None


def test_peeking_inside_an_outer_final_evaluation_block_reads_nothing():
    data = entity()
    audit = data.labels.audit

    def peek(scores: np.ndarray, labels: np.ndarray, params: object) -> float:
        return float(np.quantile(data.labels.read("peek"), 0.5))

    runner = EntityRunner(data, audit, CONFIG, threshold_fn=peek)
    with audit.phase(Phase.FINAL_EVALUATION):
        guard = caught_by(lambda: runner.select("zscore", ZSCORE))
    assert guard == GUARD_LABELS
    assert audit.reads == 0


NOTE = "entity synthetic-0, model zscore"


def test_guard_errors_of_select_name_the_entity_and_model(rng):
    data = entity()

    def shuffled(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
        return train_test_split_shuffled(np.arange(n_train + n_test), rng)

    def peek(scores: np.ndarray, labels: np.ndarray, params: object) -> float:
        return float(np.quantile(data.labels.read("peek"), 0.5))

    cases = [
        ({"fold_fn": lambda n: walk_forward_folds(n, purge=0)}, GUARD_FOLD),
        ({"split_fn": shuffled}, GUARD_SPLIT),
        ({"threshold_fn": peek}, GUARD_LABELS),
        ({"pipeline_factory": FullFitPipeline}, GUARD_FIT_END),
        ({"specs": (*FEATURE_SPECS[:3], CENTERED_ROLLING)}, GUARD_CAUSAL),
    ]
    for kwargs, guard in cases:
        runner = EntityRunner(data, data.labels.audit, CONFIG, **kwargs)  # type: ignore[arg-type]
        with pytest.raises(LeakageError) as caught:
            runner.select("zscore", ZSCORE)
        assert caught.value.guard == guard
        assert caught.value.__notes__ == [NOTE]


def test_guard_errors_of_objective_name_the_entity_and_model():
    data = entity()
    runner = EntityRunner(
        data, data.labels.audit, CONFIG, fold_fn=lambda n: walk_forward_folds(n, purge=0)
    )
    with pytest.raises(LeakageError) as caught:
        runner.objective("zscore", ZSCORE)
    assert caught.value.guard == GUARD_FOLD
    assert caught.value.__notes__ == [NOTE]
