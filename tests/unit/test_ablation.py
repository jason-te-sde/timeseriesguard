from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.conftest import make_entity
from tsguard.data.labels import LabelAudit, Phase
from tsguard.data.smd import EntityData
from tsguard.eval.protocols import ProtocolScore
from tsguard.experiment import ablation
from tsguard.experiment.ablation import AblationSummary, run_ablation
from tsguard.experiment.runner import EntityRunner, RunConfig, evaluate_selection
from tsguard.features.specs import FEATURE_SPECS, FeatureSpec
from tsguard.models import Params
from tsguard.models.space import DEFAULT_PARAMS

N_TRAIN = 3000
N_TEST = 2000
ENTITIES = ["synthetic-0", "synthetic-1"]
SEEDS = (0, 1)
CONFIG = RunConfig(causal_check=False, threads=1)
PARAMS: dict[str, Params] = {"zscore": DEFAULT_PARAMS["zscore"]}
FFT_SPECS = 10
FIXED_F1 = 0.5


def load_synthetic(name: str, audit: LabelAudit) -> EntityData:
    index = int(name.rsplit("-", 1)[1])
    return make_entity(seed=index, name=name, n_train=N_TRAIN, n_test=N_TEST, audit=audit)


@dataclass
class RunnerRecord:
    data: EntityData
    audit: LabelAudit
    specs: tuple[FeatureSpec, ...]
    seeds: list[int | None] = field(default_factory=list)
    reads_at_select: list[int] = field(default_factory=list)
    reads_before: list[int] = field(default_factory=list)


@dataclass
class LoadRecord:
    name: str
    audit: LabelAudit
    previous_runners_dead: bool


@dataclass
class Spied:
    summaries: dict[str, AblationSummary]
    runners: list[RunnerRecord]
    loads: list[LoadRecord]
    events: list[tuple[str, str]]  # (entity, "select" | "evaluate"), in call order


@pytest.fixture(scope="module")
def spied() -> Spied:
    """Run the ablation once on two synthetic entities, recording what it built and did.

    The runners and the label reads are real. The F1 handed on to the ablation is replaced by
    a constant: the F1 of a synthetic entity at its frozen threshold is a threshold coincidence,
    and a zero would make the ablation raise. The arithmetic is tested with stubs below.
    """
    runners: list[RunnerRecord] = []
    loads: list[LoadRecord] = []
    events: list[tuple[str, str]] = []
    refs: list[weakref.ReferenceType[EntityRunner]] = []

    class SpyRunner(EntityRunner):
        def __init__(self, data, audit, config=CONFIG, *, specs=FEATURE_SPECS):
            super().__init__(data, audit, config, specs=specs)
            self.record = RunnerRecord(data, audit, tuple(specs))
            runners.append(self.record)
            refs.append(weakref.ref(self))

        def select(self, model, params, *, seed=None, keep_models=False):
            self.record.seeds.append(seed)
            self.record.reads_at_select.append(self.record.audit.reads)
            events.append((self.entity, "select"))
            selection = super().select(model, params, seed=seed, keep_models=keep_models)
            self.record.reads_before.append(selection.threshold.reads_before)
            return selection

    def spy_evaluate(selection, labels, audit, params):
        events.append((selection.entity, "evaluate"))
        real = evaluate_selection(selection, labels, audit, params).protocols["event_level"]
        score = ProtocolScore(auc_pr=real.auc_pr, f1=FIXED_F1, f1_oracle=real.f1_oracle)
        return SimpleNamespace(protocols={"event_level": score})

    def loader(name, audit):
        # No gc.collect() here: run_ablation itself must have released the last entity.
        loads.append(LoadRecord(name, audit, all(ref() is None for ref in refs)))
        return load_synthetic(name, audit)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ablation, "EntityRunner", SpyRunner)
        patch.setattr(ablation, "evaluate_selection", spy_evaluate)
        summaries = run_ablation(
            ENTITIES,
            loader=loader,
            params=PARAMS,
            seeds=SEEDS,
            config=CONFIG,
            models=("zscore",),
        )
    return Spied(summaries, runners, loads, events)


def entity_runners(spied: Spied) -> list[tuple[RunnerRecord, RunnerRecord]]:
    """Pair each entity's full-registry runner with its FFT-less runner, in entity order."""
    assert len(spied.runners) == 2 * len(ENTITIES)
    return [(spied.runners[i], spied.runners[i + 1]) for i in range(0, len(spied.runners), 2)]


def test_summary_keys_and_lengths(spied):
    assert list(spied.summaries) == ["zscore"]
    summary = spied.summaries["zscore"]
    assert summary.model == "zscore"
    assert summary.seeds == SEEDS
    for values in (
        summary.f1_with,
        summary.f1_without,
        summary.auc_pr_with,
        summary.auc_pr_without,
    ):
        assert isinstance(values, tuple)
        assert len(values) == len(SEEDS)
        assert all(isinstance(v, float) for v in values)


def test_arms_differ_only_in_fft(spied):
    for full, reduced in entity_runners(spied):
        assert len(full.specs) == 63
        assert len(reduced.specs) == 63 - FFT_SPECS == 53
        assert full.specs == tuple(FEATURE_SPECS)
        assert not any(spec.family == "fft" for spec in reduced.specs)
        removed = [spec for spec in full.specs if spec not in reduced.specs]
        assert len(removed) == FFT_SPECS
        assert all(spec.family == "fft" for spec in removed)
        assert reduced.specs == tuple(spec for spec in full.specs if spec.family != "fft")


def test_both_arms_share_the_entity_and_its_audit(spied):
    assert [load.name for load in spied.loads] == ENTITIES
    assert spied.loads[0].audit is not spied.loads[1].audit
    for load, (full, reduced) in zip(spied.loads, entity_runners(spied), strict=True):
        assert full.data is reduced.data
        assert full.data.name == load.name
        assert full.audit is reduced.audit is load.audit
        assert full.data.labels.audit is load.audit
        assert full.seeds == reduced.seeds == list(SEEDS)
        # one label read per selection and arm: 2 seeds x 1 model x 2 arms
        assert load.audit.reads == 4
        assert {caller for _, caller, _ in load.audit.accesses} == {"evaluate[zscore]"}


def test_every_selection_is_made_before_any_label_is_read(spied):
    for runner in (record for pair in entity_runners(spied) for record in pair):
        assert runner.reads_at_select == [0] * len(SEEDS)
        assert runner.reads_before == [0] * len(SEEDS)
    for entity in ENTITIES:
        kinds = [kind for name, kind in spied.events if name == entity]
        assert kinds == ["select"] * 4 + ["evaluate"] * 4


def test_runners_are_released_before_the_next_entity_loads(spied):
    assert [load.previous_runners_dead for load in spied.loads] == [True, True]
    assert len(spied.runners) == 4


def make_fake_world(
    patch: pytest.MonkeyPatch,
    f1_of: Callable[[str, int, int], float],
    *,
    leaky_entity: str | None = None,
    reads_before: int = 0,
) -> None:
    """Replace the runner and the evaluation with stubs whose scores ``f1_of`` dictates.

    ``f1_of(entity, seed, n_specs)`` gives the event-level F1; the AUC-PR is ``f1 / 2``. A
    stub selection reports that its threshold was frozen after ``reads_before`` label reads;
    the stub runner of ``leaky_entity`` reads the test labels during selection.
    """

    class FakeRunner:
        def __init__(self, data, audit, config, *, specs):
            self.data = data
            self.audit = audit
            self.n_specs = len(specs)

        def select(self, model, params, *, seed):
            if self.data.name == leaky_entity:
                with self.audit.phase(Phase.FINAL_EVALUATION):
                    self.data.labels.read("leak")
            return SimpleNamespace(
                entity=self.data.name,
                seed=seed,
                n_specs=self.n_specs,
                threshold=SimpleNamespace(reads_before=reads_before),
            )

        def drop_seed_caches(self, seed):
            pass

    def fake_evaluate(selection, labels, audit, params):
        f1 = f1_of(selection.entity, selection.seed, selection.n_specs)
        score = ProtocolScore(auc_pr=f1 / 2, f1=f1, f1_oracle=1.0)
        return SimpleNamespace(protocols={"event_level": score})

    patch.setattr(ablation, "EntityRunner", FakeRunner)
    patch.setattr(ablation, "evaluate_selection", fake_evaluate)


def tiny_loader(name: str, audit: LabelAudit) -> EntityData:
    return make_entity(name=name, n_train=100, n_test=100, audit=audit, anomalies=((10, 5),))


def run_stubbed(
    entities: list[str],
    seeds: tuple[int, ...],
    f1_of: Callable[[str, int, int], float],
    **options: Any,
) -> AblationSummary:
    with pytest.MonkeyPatch.context() as patch:
        make_fake_world(patch, f1_of, **options)
        return run_ablation(
            entities,
            loader=tiny_loader,
            params=PARAMS,
            seeds=seeds,
            config=CONFIG,
            models=("zscore",),
        )["zscore"]


def test_per_seed_values_are_macro_means_over_entities():
    f1_table = {
        # (entity, seed, with-arm?) -> F1
        ("a", 0, True): 0.6,
        ("b", 0, True): 0.4,
        ("a", 0, False): 0.3,
        ("b", 0, False): 0.5,
        ("a", 1, True): 0.8,
        ("b", 1, True): 0.6,
        ("a", 1, False): 0.4,
        ("b", 1, False): 0.4,
    }
    messages: list[str] = []
    with pytest.MonkeyPatch.context() as patch:
        make_fake_world(
            patch, lambda entity, seed, n_specs: f1_table[(entity, seed, n_specs == 63)]
        )
        summary = run_ablation(
            ["a", "b"],
            loader=tiny_loader,
            params=PARAMS,
            seeds=(0, 1),
            config=CONFIG,
            models=("zscore",),
            progress=messages.append,
        )["zscore"]
    assert summary.model == "zscore"
    assert summary.seeds == (0, 1)
    assert summary.f1_with == pytest.approx((0.5, 0.7))
    assert summary.f1_without == pytest.approx((0.4, 0.4))
    assert summary.auc_pr_with == pytest.approx((0.25, 0.35))
    assert summary.auc_pr_without == pytest.approx((0.2, 0.2))
    assert summary.delta_f1_points_mean == pytest.approx((10 + 30) / 2)
    assert summary.delta_f1_points_std == pytest.approx(np.std([10, 30], ddof=1))
    assert summary.delta_f1_relative_mean == pytest.approx((25 + 75) / 2)
    assert summary.delta_f1_relative_std == pytest.approx(np.std([25, 75], ddof=1))
    assert summary.delta_auc_pr_mean == pytest.approx((0.05 + 0.15) / 2)
    assert summary.delta_auc_pr_std == pytest.approx(np.std([0.05, 0.15], ddof=1))
    assert len(messages) == 2 * 2 * 1  # entities x seeds x models
    assert all(name in " ".join(messages) for name in ("a", "b"))


def test_single_seed_std_is_zero():
    summary = run_stubbed(["a"], (3,), lambda entity, seed, n_specs: 0.6 if n_specs == 63 else 0.3)
    assert summary.seeds == (3,)
    assert summary.f1_with == pytest.approx((0.6,))
    assert summary.delta_f1_points_mean == pytest.approx(30.0)
    assert summary.delta_f1_relative_mean == pytest.approx(100.0)
    assert summary.delta_auc_pr_mean == pytest.approx(0.15)
    assert summary.delta_f1_points_std == 0.0
    assert summary.delta_f1_relative_std == 0.0
    assert summary.delta_auc_pr_std == 0.0


def test_zero_f1_without_fft_is_an_error_naming_model_and_seed():
    with pytest.raises(ValueError, match=r"zscore.*seed 7"):
        run_stubbed(["a"], (7,), lambda entity, seed, n_specs: 0.5 if n_specs == 63 else 0.0)


def test_a_label_read_during_selection_is_an_error_naming_the_entity():
    with pytest.raises(RuntimeError, match=r"\bb\b.*read"):
        run_stubbed(["a", "b"], (0,), lambda *_: 0.5, leaky_entity="b")


def test_a_threshold_frozen_after_a_read_is_an_error_naming_the_entity():
    with pytest.raises(RuntimeError, match=r"\ba\b.*frozen after 2 test-label reads"):
        run_stubbed(["a"], (0,), lambda *_: 0.5, reads_before=2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"entities": []}, "entities"),
        ({"seeds": ()}, "seeds"),
        ({"models": ()}, "models"),
        ({"models": ("iforest",)}, "iforest"),  # no parameters given for it
        ({"models": ("zscore", "cnn_ae"), "params": {**PARAMS, "cnn_ae": {}}}, "cnn_ae"),
    ],
)
def test_bad_arguments_fail_before_anything_is_loaded(kwargs, message):
    def loader(name, audit):
        raise AssertionError("the loader must not be called")

    arguments = {
        "entities": ["synthetic-0"],
        "params": PARAMS,
        "seeds": (0,),
        "models": ("zscore",),
    } | kwargs
    with pytest.raises(ValueError, match=message):
        run_ablation(loader=loader, config=CONFIG, **arguments)
