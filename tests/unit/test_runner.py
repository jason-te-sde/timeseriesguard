from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch

from tests.conftest import make_entity
from tsguard.data.labels import FrozenThreshold, LabelAudit, Phase
from tsguard.data.smd import EntityData
from tsguard.data.splits import Fold, walk_forward_folds
from tsguard.eval.inject import InjectionConfig, inject, reference_sigma
from tsguard.eval.protocols import PROTOCOLS, ProtocolParams, event_level_auc_pr, select_threshold
from tsguard.experiment.runner import (
    SENSITIVITY_GRID,
    EntityRunner,
    FoldOutcome,
    RunConfig,
    Selection,
)
from tsguard.features.pipeline import FeaturePipeline
from tsguard.features.specs import FeatureSpec
from tsguard.models import build_detector
from tsguard.models.base import Detector
from tsguard.models.space import DEFAULT_PARAMS
from tsguard.repro.seeding import configure_runtime, derive_seed

N_TRAIN = 3000
N_TEST = 2000
LAST_FIT_END = 2500
CONFIG = RunConfig(ae_epochs=1, threads=1)
ZSCORE = DEFAULT_PARAMS["zscore"]
IFOREST = DEFAULT_PARAMS["iforest"]
CNN_AE = {"window": 16, "width": 16, "latent": 4, "lr": 1e-3}
DIGEST_KEYS = {"pipeline", "detector", "test_scores"}


@pytest.fixture(scope="module", autouse=True)
def single_thread_runtime() -> Iterator[None]:
    """Fix torch to one thread and deterministic ops, then restore what the process had."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    configure_runtime(threads=1)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


@dataclass
class World:
    data: EntityData
    audit: LabelAudit
    runner: EntityRunner


def build_world(config: RunConfig = CONFIG, **kwargs: object) -> World:
    audit = LabelAudit()
    data = make_entity(n_train=N_TRAIN, n_test=N_TEST, audit=audit)
    return World(data, audit, EntityRunner(data, audit, config, **kwargs))  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def world() -> World:
    """One runner that only selects; its audit must stay at zero reads."""
    return build_world()


@dataclass
class Selected:
    selection: Selection
    reads_after_select: int


@pytest.fixture(scope="module")
def zscore(world: World) -> Selected:
    selection = world.runner.select("zscore", ZSCORE)
    return Selected(selection, world.audit.reads)


def test_select_never_reads_labels(world, zscore):
    assert zscore.reads_after_select == 0
    assert world.audit.reads == 0


def test_selection_fields(zscore):
    sel = zscore.selection
    assert (sel.entity, sel.model, sel.seed) == ("synthetic-0", "zscore", 0)
    assert dict(sel.params) == dict(ZSCORE)
    assert [fold.fold for fold in sel.folds] == [1, 2, 3]
    assert all(isinstance(fold, FoldOutcome) for fold in sel.folds)
    assert all(0.0 <= fold.objective <= 1.0 for fold in sel.folds)
    assert all(fold.n_segments == 2 for fold in sel.folds)
    assert sel.objective == pytest.approx(np.mean([fold.objective for fold in sel.folds]))
    assert 0.0 <= sel.objective <= 1.0
    assert len(sel.test_scores) == N_TEST
    assert np.isfinite(sel.test_scores).all()
    assert not sel.test_scores.flags.writeable
    assert set(sel.digests) == DIGEST_KEYS
    assert all(len(digest) == 64 for digest in sel.digests.values())
    assert sel.pipeline is None
    assert sel.detector is None


def test_threshold_is_frozen(zscore):
    threshold = zscore.selection.threshold
    assert isinstance(threshold, FrozenThreshold)
    assert np.isfinite(threshold.value)
    assert threshold.reads_before == 0


def test_evaluate_reads_labels_once_with_its_name(zscore):
    # A fresh audit, so the absolute read count is meaningful. The selection came from another
    # runner on the same entity: evaluate is a function of the selection and the labels only.
    other = build_world()
    ev = other.runner.evaluate(zscore.selection)
    assert other.audit.accesses[-1][1] == "evaluate[zscore]"
    assert other.audit.reads == 1
    assert ev.selection is zscore.selection
    assert set(ev.protocols) == set(PROTOCOLS)
    assert set(ev.sensitivity) == {"L30_C0", "L30_C60", "L60_C0", "L60_C60", "L120_C0", "L120_C60"}
    assert all(0.0 <= value <= 1.0 for value in ev.sensitivity.values())
    assert [f"L{chunk}_C{cooldown}" for chunk, cooldown in SENSITIVITY_GRID] == list(ev.sensitivity)
    assert ev.sensitivity["L60_C60"] == pytest.approx(ev.protocols["event_level"].auc_pr)


def test_injected_anomalies_are_found_better_than_chance(zscore):
    assert zscore.selection.objective > 0.3


def test_select_is_deterministic():
    first = build_world(RunConfig(ae_epochs=1, threads=1, causal_check=False))
    second = build_world(RunConfig(ae_epochs=1, threads=1, causal_check=False))
    a = first.runner.select("iforest", IFOREST)
    b = second.runner.select("iforest", IFOREST)
    assert a.digests == b.digests
    assert a.objective == b.objective
    assert a.folds == b.folds
    assert a.threshold.value == b.threshold.value
    assert np.array_equal(a.test_scores, b.test_scores)


def test_seed_override_changes_injection_not_features(world, zscore):
    other = world.runner.select("zscore", ZSCORE, seed=1)
    assert other.seed == 1
    assert other.digests["pipeline"] == zscore.selection.digests["pipeline"]
    assert other.objective != zscore.selection.objective


def test_pipeline_fits_are_shared_across_models():
    counts = {"fit": 0, "fit_transform": 0}

    class Counting(FeaturePipeline):
        def fit(self, X: np.ndarray, *, end_index: int) -> FeaturePipeline:
            counts["fit"] += 1
            return super().fit(X, end_index=end_index)

        def fit_transform(self, X: np.ndarray, *, end_index: int) -> np.ndarray:
            counts["fit_transform"] += 1
            return super().fit_transform(X, end_index=end_index)

    def factory(specs: Sequence[FeatureSpec]) -> FeaturePipeline:
        return Counting(specs)

    runner = build_world(pipeline_factory=factory).runner
    runner.select("zscore", ZSCORE)
    assert counts == {"fit": 2, "fit_transform": 3}  # one fit check = two fits, one per fold
    runner.select("iforest", IFOREST)
    assert counts == {"fit": 2, "fit_transform": 3}  # nothing is fitted or checked again


def test_autoencoder_path_uses_context(world):
    sel = world.runner.select("cnn_ae", CNN_AE)
    assert len(sel.test_scores) == N_TEST
    assert np.isfinite(sel.test_scores).all()
    assert set(sel.digests) == DIGEST_KEYS
    assert world.audit.reads == 0


def test_objective_only_runs_validation(world, zscore):
    outcomes = world.runner.objective("zscore", ZSCORE)
    assert len(outcomes) == 3
    assert all(isinstance(outcome, FoldOutcome) for outcome in outcomes)
    assert outcomes == zscore.selection.folds
    assert world.audit.reads == 0


def test_keep_models_returns_fitted_objects(world):
    kept = world.runner.select("zscore", ZSCORE, keep_models=True)
    assert kept.pipeline is not None
    assert kept.detector is not None
    assert kept.pipeline.fit_end == LAST_FIT_END
    assert kept.pipeline.state_digest() == kept.digests["pipeline"]
    assert kept.detector.state_digest() == kept.digests["detector"]


def test_runner_refuses_an_audit_the_labels_do_not_report_to():
    data = make_entity(n_train=N_TRAIN, n_test=N_TEST)
    with pytest.raises(ValueError, match="synthetic-0"):
        EntityRunner(data, LabelAudit(), CONFIG)


def test_evaluate_refuses_a_selection_of_another_entity(zscore):
    other = build_world()
    data = make_entity(n_train=N_TRAIN, n_test=N_TEST, audit=other.audit, name="synthetic-1")
    runner = EntityRunner(data, other.audit, CONFIG)
    with pytest.raises(ValueError, match="synthetic-0"):
        runner.evaluate(zscore.selection)
    assert other.audit.reads == 0


def test_select_names_the_model_it_does_not_know(world):
    with pytest.raises(ValueError, match="no_such_model"):
        world.runner.select("no_such_model", {})


class StubDetector:
    """A detector that scores a constant, or NaN where ``nan_rows`` says so."""

    name = "stub"
    input_kind = "features"
    lookback = 1

    def __init__(self, nan_when_rows: int | None) -> None:
        self._nan_when_rows = nan_when_rows

    def fit(self, X: np.ndarray) -> None:
        pass

    def score(self, X: np.ndarray, *, context: int) -> np.ndarray:
        n = len(X) - context
        return np.full(n, np.nan if self._nan_when_rows in (None, n) else 0.5)

    def state_digest(self) -> str:
        return "0" * 64


@pytest.mark.parametrize(
    ("nan_when_rows", "where"), [(None, "validation block of fold 1"), (N_TEST, "test rows")]
)
def test_non_finite_scores_name_the_entity_and_model(monkeypatch, nan_when_rows, where):
    monkeypatch.setattr(
        "tsguard.experiment.runner.build_detector", lambda *a, **k: StubDetector(nan_when_rows)
    )
    runner = build_world(RunConfig(causal_check=False)).runner
    with pytest.raises(RuntimeError, match=rf"synthetic-0.*'zscore'.*{where}"):
        runner.select("zscore", ZSCORE)


@dataclass
class ThresholdCall:
    scores: np.ndarray
    labels: np.ndarray
    phase: Phase


def spying_world(*, causal_check: bool = True) -> tuple[World, list[ThresholdCall]]:
    """Build a world whose threshold search records what it was given and the phase it ran in."""
    calls: list[ThresholdCall] = []
    audit = LabelAudit()

    def spy(scores: np.ndarray, labels: np.ndarray, params: ProtocolParams) -> float:
        calls.append(ThresholdCall(scores.copy(), labels.copy(), audit.current))
        return select_threshold(scores, labels, params)

    data = make_entity(n_train=N_TRAIN, n_test=N_TEST, audit=audit)
    runner = EntityRunner(
        data,
        audit,
        RunConfig(ae_epochs=1, threads=1, causal_check=causal_check),
        threshold_fn=spy,
    )
    return World(data, audit, runner), calls


def injected_labels(data: EntityData, fold: Fold, seed: int = 0) -> np.ndarray:
    """Re-derive the synthetic labels of a fold's validation block from the documented seeds."""
    train = data.train
    _, labels, _ = inject(
        train[: fold.val_end],
        start=fold.val_start,
        end=fold.val_end,
        sigma=reference_sigma(train[: fold.fit_end]),
        rng=np.random.default_rng(derive_seed(seed, data.name, "inject", fold.index)),
        config=InjectionConfig(),
    )
    return labels


def test_threshold_is_chosen_from_the_last_fold_block():
    spied, calls = spying_world(causal_check=False)
    selection = spied.runner.select("zscore", ZSCORE)
    folds = walk_forward_folds(N_TRAIN)
    (call,) = calls
    # Three different objectives, so equality with the last one cannot hold for another fold.
    assert len({fold.objective for fold in selection.folds}) == 3
    assert event_level_auc_pr(call.scores, call.labels, CONFIG.protocol) == (
        selection.folds[-1].objective
    )
    assert np.array_equal(call.labels, injected_labels(spied.data, folds[-1]))
    assert not np.array_equal(call.labels, injected_labels(spied.data, folds[0]))
    assert len(call.scores) == folds[-1].val_end - folds[-1].val_start


@pytest.mark.parametrize(("model", "params"), [("zscore", ZSCORE), ("cnn_ae", CNN_AE)])
def test_test_scores_come_from_the_kept_last_fold_models(world, model, params):
    kept = world.runner.select(model, params, keep_models=True)
    assert kept.pipeline is not None
    assert kept.detector is not None
    assert kept.pipeline.fit_end == LAST_FIT_END
    context = kept.detector.lookback - 1
    rows = kept.pipeline.transform(world.data.full, start_index=N_TRAIN, context=context)
    rescored = kept.detector.score(rows, context=context)
    assert rescored.tobytes() == kept.test_scores.tobytes()


def fresh_runner() -> EntityRunner:
    return build_world(RunConfig(ae_epochs=1, threads=1, causal_check=False)).runner


def assert_same_selection(warm: Selection, fresh: Selection) -> None:
    assert warm.digests == fresh.digests
    assert warm.objective == fresh.objective
    assert warm.folds == fresh.folds
    assert warm.threshold.value == fresh.threshold.value
    assert warm.test_scores.tobytes() == fresh.test_scores.tobytes()


def test_a_warm_runner_selects_a_new_seed_like_a_fresh_one(world, zscore):
    warm = world.runner.select("zscore", ZSCORE, seed=1)
    assert_same_selection(warm, fresh_runner().select("zscore", ZSCORE, seed=1))


def test_a_warm_runner_selects_a_new_window_like_a_fresh_one(world):
    world.runner.select("cnn_ae", CNN_AE)
    wider = {**CNN_AE, "window": 32}
    warm = world.runner.select("cnn_ae", wider)
    assert_same_selection(warm, fresh_runner().select("cnn_ae", wider))


class TrackedDetector:
    """Wraps a detector and records how many rows each ``score`` call returned."""

    def __init__(self, inner: Detector, scored: list[int]) -> None:
        self._inner = inner
        self._scored = scored
        self.name = inner.name
        self.input_kind = inner.input_kind
        self.lookback = inner.lookback

    def fit(self, X: np.ndarray) -> None:
        self._inner.fit(X)

    def score(self, X: np.ndarray, *, context: int) -> np.ndarray:
        scores = self._inner.score(X, context=context)
        self._scored.append(len(scores))
        return scores

    def state_digest(self) -> str:
        return self._inner.state_digest()


def test_causal_checks_run_once_per_input_kind_and_once_per_model(monkeypatch):
    full_rows: list[int] = []
    scored: list[int] = []

    class Tracked(FeaturePipeline):
        def transform(
            self, X: np.ndarray, *, start_index: int, context: int = 0, offset: int = 0
        ) -> np.ndarray:
            full_rows.append(len(X))
            return super().transform(X, start_index=start_index, context=context, offset=offset)

    monkeypatch.setattr(
        "tsguard.experiment.runner.build_detector",
        lambda *a, **k: TrackedDetector(build_detector(*a, **k), scored),
    )

    def count() -> tuple[int, int]:
        return full_rows.count(N_TRAIN + N_TEST), scored.count(N_TEST)

    runner = build_world(pipeline_factory=lambda specs: Tracked(specs)).runner
    runner.select("zscore", ZSCORE)
    # The test rows are transformed and scored once; each causal check does it twice more.
    assert count() == (3, 3)
    runner.select("zscore", {"aggregation": "mean", "k": 8})
    assert count() == (4, 4)  # same kind and same model: only the test rows
    runner.select("iforest", {"n_estimators": 20, "max_samples": 256})
    assert count() == (5, 7)  # same kind, new model: its own score check only


def test_select_runs_in_the_selection_phase_inside_an_outer_block():
    spied, calls = spying_world(causal_check=False)
    with spied.audit.phase(Phase.FINAL_EVALUATION):
        spied.runner.select("zscore", ZSCORE)
        assert spied.audit.current is Phase.FINAL_EVALUATION
    assert [call.phase for call in calls] == [Phase.SELECTION]
    assert spied.audit.reads == 0


def test_drop_seed_caches_forgets_one_seed_only(monkeypatch):
    injected: list[int] = []
    validation_rows: list[int] = []

    def counting_inject(*args: Any, **kwargs: Any) -> Any:
        injected.append(1)
        return inject(*args, **kwargs)

    class Tracked(FeaturePipeline):
        def transform(
            self, X: np.ndarray, *, start_index: int, context: int = 0, offset: int = 0
        ) -> np.ndarray:
            if len(X) <= N_TRAIN:  # a validation block, not the test rows
                validation_rows.append(len(X))
            return super().transform(X, start_index=start_index, context=context, offset=offset)

    monkeypatch.setattr("tsguard.experiment.runner.inject", counting_inject)
    config = RunConfig(ae_epochs=1, threads=1, causal_check=False)
    runner = build_world(config, pipeline_factory=lambda specs: Tracked(specs)).runner
    first = runner.select("zscore", ZSCORE, seed=1)
    runner.select("zscore", ZSCORE, seed=2)
    assert (len(injected), len(validation_rows)) == (6, 6)
    runner.select("zscore", ZSCORE, seed=1)
    assert (len(injected), len(validation_rows)) == (6, 6)  # cached
    runner.drop_seed_caches(1)
    runner.drop_seed_caches(99)  # a seed that was never used is not an error
    again = runner.select("zscore", ZSCORE, seed=1)
    assert (len(injected), len(validation_rows)) == (9, 9)  # seed 1 was rebuilt
    runner.select("zscore", ZSCORE, seed=2)
    assert (len(injected), len(validation_rows)) == (9, 9)  # seed 2 was kept
    assert_same_selection(again, first)
