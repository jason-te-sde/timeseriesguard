"""The runner must call every guard itself.

The guards are tested in isolation elsewhere. Here a leaky component is handed to the real
`EntityRunner`, and the test asserts which guard stopped `select`: if the runner stopped
calling a guard, the leak would pass silently and the matching test below would fail.
"""

from __future__ import annotations

import numpy as np

from tests.conftest import make_entity
from tests.leakage.defects import (
    CENTERED_ROLLING,
    FullFitPipeline,
    LyingFullFitPipeline,
    caught_by,
    train_test_split_shuffled,
)
from tsguard.data.labels import LabelAudit
from tsguard.data.smd import EntityData
from tsguard.data.splits import walk_forward_folds
from tsguard.errors import GUARD_CAUSAL, GUARD_FIT_END, GUARD_FOLD, GUARD_LABELS, GUARD_SPLIT
from tsguard.experiment.runner import EntityRunner, RunConfig
from tsguard.features.specs import FEATURE_SPECS
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
