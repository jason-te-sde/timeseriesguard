from __future__ import annotations

import dataclasses
import re
import subprocess
import sys
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from tests.conftest import (
    TINY,
    TINY_N_TEST,
    TINY_N_TRAIN,
    TINY_SEGMENTS_PER_ENTITY,
    load_synthetic,
    make_entity,
)
from tsguard.data.labels import LabelAudit, Phase
from tsguard.data.smd import ENTITIES, SMD_COMMIT, EntityData
from tsguard.eval.protocols import PROTOCOLS
from tsguard.experiment import reproduce as reproduce_module
from tsguard.experiment.ablation import run_ablation
from tsguard.experiment.reproduce import (
    FULL,
    PROFILES,
    SMOKE,
    Profile,
    git_revision,
    label_reads_at_selection,
    reproduce,
    require_experiment_extra,
)
from tsguard.experiment.runner import EntityRunner, RunConfig, Selection
from tsguard.experiment.tracking import MemoryTracker
from tsguard.experiment.tuning import FINAL_EXPERIMENT, TUNING_EXPERIMENT
from tsguard.models import MODEL_NAMES
from tsguard.models.space import DEFAULT_PARAMS
from tsguard.repro.report import Metrics, best_model, read_metrics


@dataclass
class TinyRun:
    metrics: Metrics
    tracker: MemoryTracker
    out_dir: Path
    lines: list[str]
    # for each load, whether every runner built before it was already freed
    runners_freed_at_load: list[bool] = field(default_factory=list)
    ablation_calls: list[dict[str, object]] = field(default_factory=list)
    # test-label reads over every audit, at the moment each progress line was logged
    reads_at_line: list[int] = field(default_factory=list)
    # (objectives given, model returned, reads over every audit) for each call of best_model
    best_model_calls: list[tuple[dict[str, float], str, int]] = field(default_factory=list)
    # what the output directory held when the ablation started
    files_at_ablation: list[str] = field(default_factory=list)
    partial_at_ablation: Metrics | None = None


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> TinyRun:
    """Run the whole pipeline once on two synthetic entities; the slow tests share it."""
    out_dir = tmp_path_factory.mktemp("tiny")
    tracker = MemoryTracker()
    refs: list[weakref.ReferenceType[EntityRunner]] = []
    audits: list[LabelAudit] = []
    run = TinyRun(Metrics.model_construct(), tracker, out_dir, [])

    def total_reads() -> int:
        return sum(audit.reads for audit in audits)

    class SpyRunner(EntityRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            refs.append(weakref.ref(self))

    def loader(name: str, audit: LabelAudit) -> EntityData:
        # No gc.collect() here: reproduce itself must have released the earlier runners.
        run.runners_freed_at_load.append(all(ref() is None for ref in refs))
        audits.append(audit)
        return load_synthetic(name, audit)

    def progress(line: str) -> None:
        run.lines.append(line)
        run.reads_at_line.append(total_reads())

    def spy_best_model(objectives):
        chosen = best_model(objectives)
        run.best_model_calls.append((dict(objectives), chosen, total_reads()))
        return chosen

    def spy_ablation(entities, **kwargs):
        run.ablation_calls.append({"entities": tuple(entities), **kwargs})
        run.files_at_ablation = sorted(path.name for path in out_dir.iterdir())
        partial_path = out_dir / "metrics.partial.json"
        run.partial_at_ablation = read_metrics(partial_path) if partial_path.exists() else None
        return run_ablation(entities, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(reproduce_module, "EntityRunner", SpyRunner)
        patch.setattr(reproduce_module, "run_ablation", spy_ablation)
        patch.setattr(reproduce_module, "best_model", spy_best_model)
        run.metrics = reproduce(
            TINY, loader=loader, tracker=tracker, out_dir=out_dir, progress=progress
        )
    return run


@pytest.mark.slow
def test_tiny_reproduce_writes_valid_metrics(tiny_run):
    m = tiny_run.metrics
    assert read_metrics(tiny_run.out_dir / "metrics.json") == m
    assert set(m.models) == set(MODEL_NAMES)
    assert all(set(s.macro) == set(PROTOCOLS) for s in m.models.values())
    assert m.headline.label_reads_at_selection == 0
    assert m.headline.model in MODEL_NAMES
    assert set(m.ablation) == {"zscore", "iforest"}
    assert m.feature_specs == 63


@pytest.mark.slow
def test_tiny_reproduce_describes_the_run(tiny_run):
    m = tiny_run.metrics
    assert (m.profile, m.seed, m.schema_version) == ("tiny", 0, 1)
    assert m.dataset.source_commit == SMD_COMMIT
    assert (m.dataset.entities, m.dataset.channels) == (2, 38)
    assert (m.dataset.train_steps, m.dataset.test_steps) == (2 * TINY_N_TRAIN, 2 * TINY_N_TEST)
    assert m.dataset.anomaly_segments == 2 * TINY_SEGMENTS_PER_ENTITY
    assert m.protocol == {"chunk": 60, "cooldown": 60, "k_percent": 20}
    assert m.tuning.entities == ["synthetic-0"]
    assert m.tuning.n_trials == 1
    assert set(m.tuning.best) == set(MODEL_NAMES)
    assert m.ablation["zscore"].seeds == [0]
    for name, summary in m.models.items():
        # one trial is the default configuration, so the tuned params are the defaults
        assert summary.params == dict(DEFAULT_PARAMS[name])
        assert [e.entity for e in summary.per_entity] == ["synthetic-0", "synthetic-1"]
        assert set(summary.sensitivity_macro) == {
            f"L{chunk}_C{cooldown}" for chunk in (30, 60, 120) for cooldown in (0, 60)
        }
    keys = set(m.environment)
    assert {"platform", "python", "numpy", "pandas", "scikit-learn", "torch"} <= keys
    assert {"optuna", "mlflow", "torch_threads", "git_sha", "generated_at"} <= keys
    assert m.environment["torch_threads"] == "1"
    assert m.environment["generated_at"].endswith("Z")


@pytest.mark.slow
def test_macro_scores_are_the_means_over_entities(tiny_run):
    for summary in tiny_run.metrics.models.values():
        for protocol in PROTOCOLS:
            for score in ("auc_pr", "f1", "f1_oracle"):
                per_entity = [getattr(e.protocols[protocol], score) for e in summary.per_entity]
                macro = getattr(summary.macro[protocol], score)
                assert macro == pytest.approx(np.mean(per_entity), abs=1e-12), (protocol, score)
        assert set(summary.sensitivity_macro) == set(summary.per_entity[0].sensitivity)
        for key, macro in summary.sensitivity_macro.items():
            per_entity = [e.sensitivity[key] for e in summary.per_entity]
            assert macro == pytest.approx(np.mean(per_entity), abs=1e-12), key
        objectives = [e.validation_objective for e in summary.per_entity]
        assert summary.validation_objective == pytest.approx(np.mean(objectives), abs=1e-12)


@pytest.mark.slow
def test_tracker_gets_trial_and_final_runs(tiny_run):
    runs = tiny_run.tracker.runs
    assert len(runs) == 4 * 1 + 4 * 2
    trials = [r for r in runs if r["experiment"] == TUNING_EXPERIMENT]
    finals = [r for r in runs if r["experiment"] == FINAL_EXPERIMENT]
    assert (len(trials), len(finals)) == (4, 8)
    assert sorted(str(r["run_name"]) for r in finals) == sorted(
        f"{model}-synthetic-{i}" for model in MODEL_NAMES for i in (0, 1)
    )
    for run in finals:
        tags = dict(run["tags"])
        assert tags == {"kind": "final", "model": str(run["run_name"]).split("-synthetic")[0]}
        assert set(run["metrics"]) == {"objective"}
        assert run["params"] == tiny_run.metrics.models[tags["model"]].params


@pytest.mark.slow
def test_headline_is_the_best_validation_objective(tiny_run):
    m = tiny_run.metrics
    objective = {name: s.validation_objective for name, s in m.models.items()}
    best = max(objective.values())
    winners = [name for name in MODEL_NAMES if objective[name] == best]
    assert m.headline.model == winners[0]  # a tie goes to the earlier model in MODEL_NAMES
    assert m.headline.protocol == "event_level"
    assert m.headline.selected_by == "validation_objective"
    assert m.headline.auc_pr == m.models[m.headline.model].macro["event_level"].auc_pr
    low, high = m.headline.ci95
    scores = [e.protocols["event_level"].auc_pr for e in m.models[m.headline.model].per_entity]
    assert low <= np.mean(scores) <= high


@pytest.mark.slow
def test_each_runner_is_freed_before_the_next_entity_is_loaded(tiny_run):
    # loads: tuning synthetic-0, selection synthetic-0 and synthetic-1, ablation 0 and 1
    assert tiny_run.runners_freed_at_load == [True] * 5


@pytest.mark.slow
def test_ablation_gets_the_tuned_feature_params_and_skips_the_causal_check(tiny_run):
    (call,) = tiny_run.ablation_calls
    assert call["entities"] == TINY.entities
    assert call["seeds"] == TINY.ablation_seeds
    assert call["params"] == {name: dict(DEFAULT_PARAMS[name]) for name in ("zscore", "iforest")}
    config = call["config"]
    assert config == dataclasses.replace(TINY.run, causal_check=False)
    assert TINY.run.causal_check  # the selection phase keeps the check on


HEADLINE_LINE = re.compile(
    r"^headline chosen before any test label: (\w+) \(validation objective (\d\.\d{4})\)$"
)


def line_index(lines: list[str], prefix: str) -> int:
    return next(i for i, line in enumerate(lines) if line.startswith(prefix))


@pytest.mark.slow
def test_the_headline_is_chosen_once_before_any_test_label_is_read(tiny_run):
    m = tiny_run.metrics
    ((objectives, chosen, reads),) = tiny_run.best_model_calls  # chosen once, not recomputed
    assert reads == 0
    assert chosen == m.headline.model
    assert objectives == {name: s.validation_objective for name, s in m.models.items()}


@pytest.mark.slow
def test_the_headline_line_comes_after_the_blind_check_and_before_evaluation(tiny_run):
    lines, m = tiny_run.lines, tiny_run.metrics
    at = line_index(lines, "headline chosen")
    match = HEADLINE_LINE.match(lines[at])
    assert match, lines[at]
    assert match.group(1) == m.headline.model
    assert match.group(2) == f"{m.models[m.headline.model].validation_objective:.4f}"
    assert tiny_run.reads_at_line[at] == 0
    assert line_index(lines, "selection done: 0 test-label reads") < at
    first_evaluation = line_index(lines, "evaluate ")
    assert at < first_evaluation
    assert tiny_run.reads_at_line[first_evaluation] > 0  # the reads did start there
    assert at < next(i for i, line in enumerate(lines) if " seed=0 " in line)  # before the ablation


@pytest.mark.slow
def test_a_checkpoint_without_the_ablation_exists_while_the_ablation_runs(tiny_run):
    m, partial = tiny_run.metrics, tiny_run.partial_at_ablation
    assert tiny_run.files_at_ablation == ["metrics.partial.json"]
    assert partial is not None
    assert partial.profile == "tiny-partial"
    assert partial.ablation == {}
    rest = {"profile", "ablation", "environment"}
    assert partial.model_dump(exclude=rest) == m.model_dump(exclude=rest)


@pytest.mark.slow
def test_a_successful_run_leaves_only_metrics_json(tiny_run):
    assert sorted(path.name for path in tiny_run.out_dir.iterdir()) == ["metrics.json"]


@pytest.mark.slow
def test_a_failing_ablation_leaves_the_checkpoint_and_no_metrics_json(tmp_path, monkeypatch):
    def failing_ablation(entities, **kwargs):
        raise RuntimeError("the ablation failed")

    monkeypatch.setattr(reproduce_module, "run_ablation", failing_ablation)
    with pytest.raises(RuntimeError, match="the ablation failed"):
        reproduce(TINY, loader=load_synthetic, tracker=MemoryTracker(), out_dir=tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["metrics.partial.json"]
    partial = read_metrics(tmp_path / "metrics.partial.json")
    assert partial.profile == "tiny-partial"
    assert partial.ablation == {}
    assert set(partial.models) == set(MODEL_NAMES)
    assert partial.headline.label_reads_at_selection == 0
    assert partial.headline.model in MODEL_NAMES


@pytest.mark.slow
def test_progress_has_one_line_per_phase_entity_and_model(tiny_run):
    lines = tiny_run.lines
    assert sum(line.startswith("tune ") for line in lines) == len(MODEL_NAMES)
    for phase in ("select", "evaluate"):
        for entity in ("synthetic-0", "synthetic-1"):
            for model in MODEL_NAMES:
                assert sum(line.startswith(f"{phase} {entity} {model}:") for line in lines) == 1
    ablation_lines = [line for line in lines if " seed=0 " in line]
    assert len(ablation_lines) == 2 * 1 * 2  # entities x seeds x feature models


def test_profiles_are_the_documented_ones():
    assert isinstance(PROFILES, MappingProxyType)
    assert dict(PROFILES) == {"full": FULL, "smoke": SMOKE}
    assert (
        Profile(
            "full",
            ENTITIES,
            ("machine-1-1", "machine-2-1", "machine-3-1"),
            8,
            (0, 1, 2, 3, 4),
            RunConfig(),
        )
        == FULL
    )
    assert (
        Profile(
            "smoke",
            ("machine-1-1", "machine-2-1"),
            ("machine-1-1",),
            2,
            (0,),
            RunConfig(ae_epochs=1),
        )
        == SMOKE
    )
    assert len(FULL.entities) == 28


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("entities", ()),
        ("entities", ("a", "a")),
        ("tuning_entities", ()),
        ("n_trials", 0),
        ("ablation_seeds", ()),
        ("ablation_seeds", (0, 0)),
    ],
)
def test_a_profile_that_cannot_finish_is_refused_at_construction(field, value):
    valid = Profile("ok", ("a",), ("a",), 1, (0,), RunConfig())
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(valid, **{field: value})


def test_optuna_is_imported_before_any_runner_exists(monkeypatch, tmp_path):
    """Optuna's first import keeps a traceback alive for the whole process.

    The traceback pins the frame that triggered the import. Imported lazily inside ``tune``,
    that frame holds the tuning runners, and their 2 to 3 GB of cached features each would
    survive every later phase. A test of the memory itself passes whenever another test module
    has already imported Optuna, so this pins the order instead.
    """
    events: list[str] = []

    class RunnerBuiltError(Exception):
        pass

    def stub_runner(*args, **kwargs):
        events.append("runner")
        raise RunnerBuiltError

    monkeypatch.setattr(
        reproduce_module, "require_experiment_extra", lambda: events.append("import")
    )
    monkeypatch.setattr(reproduce_module, "EntityRunner", stub_runner)
    with pytest.raises(RunnerBuiltError):
        reproduce(TINY, loader=load_synthetic, tracker=MemoryTracker(), out_dir=tmp_path)
    assert events == ["import", "runner"]


def test_a_missing_experiment_extra_is_an_import_error_that_names_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(ImportError, match=r"experiment"):
        require_experiment_extra()


def test_selection_that_read_no_label_is_clean():
    audit = LabelAudit()
    assert label_reads_at_selection([audit, LabelAudit()], []) == 0


def test_a_label_read_during_selection_is_a_runtime_error():
    audit = LabelAudit()
    data = make_entity(audit=audit)
    with audit.phase(Phase.FINAL_EVALUATION):
        data.labels.read("peek")
    with pytest.raises(RuntimeError, match=r"1 test-label read"):
        label_reads_at_selection([LabelAudit(), audit], [])


def test_a_threshold_frozen_after_a_read_is_a_runtime_error():
    other = LabelAudit()
    data = make_entity(audit=other, name="synthetic-9")
    with other.phase(Phase.FINAL_EVALUATION):
        data.labels.read("peek")
    late = Selection(
        entity="synthetic-9",
        model="zscore",
        params={},
        seed=0,
        folds=(),
        objective=0.0,
        threshold=other.freeze_threshold(0.5, caller="test"),
        test_scores=np.zeros(1),
        digests={},
    )
    with pytest.raises(RuntimeError, match=r"synthetic-9.*zscore.*1 test-label read"):
        label_reads_at_selection([LabelAudit()], [late])


def git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def test_git_revision_outside_a_repository_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_revision(tmp_path) == "unknown"


def test_git_revision_marks_a_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    git(tmp_path, "init", "-q")
    git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty", "-m", "first")
    sha = git(tmp_path, "rev-parse", "HEAD")
    assert git_revision(tmp_path) == sha
    (tmp_path / "new.txt").write_text("x")
    assert git_revision(tmp_path) == f"{sha}+dirty"
