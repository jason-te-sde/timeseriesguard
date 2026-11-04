from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import optuna
import pytest

from tsguard.experiment.runner import FoldOutcome
from tsguard.experiment.tracking import MemoryTracker, MlflowTracker, export_runs
from tsguard.experiment.tuning import TUNING_EXPERIMENT, ObjectiveSource, tune
from tsguard.models import Params
from tsguard.models.space import DEFAULT_PARAMS

# MLflow's SQLAlchemy store triggers a deprecation warning inside SQLAlchemy on first use.
pytestmark = pytest.mark.filterwarnings("ignore:The ``noload`` loader strategy")

CSV_HEADER = ["run_id", "experiment", "run_name", "kind", "model", "objective", "params_json"]


class FakeSource:
    """Objective 1 - |k - 64| / 256 for zscore params, the same on every fold."""

    def __init__(self, entity: str = "fake", *, folds: int = 3, shift: float = 0.0) -> None:
        self.entity = entity
        self.folds = folds
        self.shift = shift
        self.calls: list[tuple[str, dict[str, int | float | str], int | None]] = []
        self.verbosity: list[int] = []

    def objective(
        self, model: str, params: Params, *, seed: int | None = None
    ) -> tuple[FoldOutcome, ...]:
        self.calls.append((model, dict(params), seed))
        self.verbosity.append(optuna.logging.get_verbosity())
        value = 1 - abs(float(params["k"]) - 64) / 256 + self.shift
        return tuple(FoldOutcome(fold, value, 5) for fold in range(1, self.folds + 1))


class FoldVaryingSource:
    """Fold objectives 0.2, 0.4 and 0.9 whatever the parameters."""

    entity = "varying"

    def objective(
        self, model: str, params: Params, *, seed: int | None = None
    ) -> tuple[FoldOutcome, ...]:
        return tuple(FoldOutcome(i, v, 5) for i, v in enumerate((0.2, 0.4, 0.9), start=1))


class ExplodingSource:
    entity = "boom"

    def objective(
        self, model: str, params: Params, *, seed: int | None = None
    ) -> tuple[FoldOutcome, ...]:
        raise RuntimeError("boom")


def run_tune(
    n_trials: int = 3,
    seed: int = 0,
    sources: Sequence[ObjectiveSource] | None = None,
    tracker: MemoryTracker | None = None,
):
    return tune(
        "zscore",
        [FakeSource()] if sources is None else sources,
        n_trials=n_trials,
        seed=seed,
        tracker=MemoryTracker() if tracker is None else tracker,
    )


def test_first_trial_is_the_default():
    result = run_tune()
    assert result.trials[0].params == DEFAULT_PARAMS["zscore"]


def test_best_is_the_max_trial():
    result = run_tune(n_trials=6)
    assert result.best_value == max(t.value for t in result.trials)
    best = next(t for t in result.trials if t.value == result.best_value)
    assert result.best_params == best.params


def test_the_earliest_trial_wins_a_tie():
    result = run_tune(n_trials=4, sources=[FoldVaryingSource()])
    assert len({t.value for t in result.trials}) == 1
    assert result.best_params == result.trials[0].params == DEFAULT_PARAMS["zscore"]


def test_best_params_is_a_valid_configuration_of_the_model():
    result = run_tune(n_trials=4)
    assert result.model == "zscore"
    assert set(result.best_params) == {"aggregation", "k"}


def test_every_trial_is_logged_once():
    tracker = MemoryTracker()
    result = run_tune(tracker=tracker)
    assert [r["run_name"] for r in tracker.runs] == [f"zscore-trial-0{i}" for i in range(3)]
    assert all(r["tags"] == {"kind": "trial", "model": "zscore"} for r in tracker.runs)
    assert all(r["experiment"] == TUNING_EXPERIMENT for r in tracker.runs)
    assert [r["params"] for r in tracker.runs] == [t.params for t in result.trials]
    assert [t.run_id for t in result.trials] == [r["run_id"] for r in tracker.runs]
    assert [t.number for t in result.trials] == [0, 1, 2]


def test_trial_metrics_hold_the_objective_and_one_per_entity():
    tracker = MemoryTracker()
    result = run_tune(sources=[FakeSource("a"), FakeSource("b", shift=0.1)], tracker=tracker)
    for trial, run in zip(result.trials, tracker.runs, strict=True):
        metrics = run["metrics"]
        assert isinstance(metrics, dict)
        assert set(metrics) == {"objective", "objective.a", "objective.b"}
        assert metrics["objective"] == trial.value
        assert metrics["objective.b"] == pytest.approx(metrics["objective.a"] + 0.1)


def test_trial_value_is_the_mean_over_sources_of_the_mean_fold_objective():
    tracker = MemoryTracker()
    result = run_tune(
        n_trials=1, sources=[FoldVaryingSource(), FakeSource("flat", shift=0.0)], tracker=tracker
    )
    flat = 1 - abs(32 - 64) / 256  # the default k is 32
    varying = (0.2 + 0.4 + 0.9) / 3
    assert result.trials[0].value == pytest.approx((varying + flat) / 2)
    metrics = tracker.runs[0]["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["objective.varying"] == pytest.approx(varying)
    assert metrics["objective.flat"] == pytest.approx(flat)


def test_tuning_is_deterministic():
    first = run_tune(n_trials=8, seed=3)
    second = run_tune(n_trials=8, seed=3)
    assert [t.params for t in first.trials] == [t.params for t in second.trials]
    assert [t.value for t in first.trials] == [t.value for t in second.trials]


def test_the_seed_changes_the_sampled_configurations():
    first = run_tune(n_trials=8, seed=0)
    second = run_tune(n_trials=8, seed=1)
    assert first.trials[0].params == second.trials[0].params  # the enqueued default
    assert [t.params for t in first.trials[1:]] != [t.params for t in second.trials[1:]]


def test_the_seed_and_the_model_reach_every_source():
    source = FakeSource()
    run_tune(n_trials=3, seed=7, sources=[source])
    assert [(model, seed) for model, _, seed in source.calls] == [("zscore", 7)] * 3


def test_optuna_is_quiet_during_the_search_and_restored_after():
    source = FakeSource()
    before = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    try:
        run_tune(sources=[source])
        assert set(source.verbosity) == {optuna.logging.WARNING}
        assert optuna.logging.get_verbosity() == optuna.logging.INFO
    finally:
        optuna.logging.set_verbosity(before)


def test_optuna_verbosity_is_restored_when_a_trial_fails():
    before = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            run_tune(sources=[ExplodingSource()])
        assert optuna.logging.get_verbosity() == optuna.logging.DEBUG
    finally:
        optuna.logging.set_verbosity(before)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_trials": 0}, "n_trials"),
        ({"sources": []}, "source"),
        ({"sources": [FakeSource("same"), FakeSource("same")]}, "same"),
    ],
)
def test_tune_rejects_an_unusable_request(kwargs, message):
    tracker = MemoryTracker()
    with pytest.raises(ValueError, match=message):
        run_tune(tracker=tracker, **kwargs)
    assert tracker.runs == []


def test_tune_rejects_a_model_without_a_search_space():
    with pytest.raises(ValueError, match="nope"):
        tune("nope", [FakeSource()], n_trials=1, seed=0, tracker=MemoryTracker())


def test_a_source_without_folds_is_named():
    with pytest.raises(RuntimeError, match="empty"):
        run_tune(sources=[FakeSource("empty", folds=0)])


def test_memory_tracker_numbers_its_runs():
    tracker = MemoryTracker()
    ids = [
        tracker.log_run(experiment="x", run_name=n, params={}, metrics={}, tags={})
        for n in ("a", "b")
    ]
    assert ids == ["mem-0", "mem-1"]
    assert [r["run_id"] for r in tracker.runs] == ids


def test_memory_tracker_keeps_a_copy_of_what_it_was_given():
    tracker = MemoryTracker()
    params: dict[str, object] = {"k": 1}
    tracker.log_run(experiment="x", run_name="a", params=params, metrics={}, tags={})
    params["k"] = 2
    assert tracker.runs[0]["params"] == {"k": 1}


def test_mlflow_tracker_round_trip(tmp_path: Path):
    uri = f"sqlite:///{tmp_path}/mlruns.db"
    t = MlflowTracker(uri)
    t.log_run(
        experiment="x",
        run_name="b",
        params={"k": 1},
        metrics={"objective": 0.5},
        tags={"kind": "trial", "model": "zscore"},
    )
    t.log_run(
        experiment="x",
        run_name="a",
        params={"k": 2},
        metrics={"objective": 0.6},
        tags={"kind": "final", "model": "zscore"},
    )
    assert export_runs(uri, tmp_path / "runs.csv", experiments=["x"]) == 2
    with open(tmp_path / "runs.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["run_name"] for r in rows] == ["a", "b"]
    assert list(rows[0]) == CSV_HEADER
    assert [(r["kind"], r["model"], r["experiment"]) for r in rows] == [
        ("final", "zscore", "x"),
        ("trial", "zscore", "x"),
    ]
    assert [float(r["objective"]) for r in rows] == [0.6, 0.5]


def test_export_restores_the_types_of_the_params(tmp_path: Path):
    uri = f"sqlite:///{tmp_path}/mlruns.db"
    params = {"lr": 0.001, "k": 32, "aggregation": "topk"}
    MlflowTracker(uri).log_run(
        experiment="x",
        run_name="a",
        params=params,
        metrics={"objective": 0.5},
        tags={"kind": "trial", "model": "zscore"},
    )
    export_runs(uri, tmp_path / "runs.csv", experiments=["x"])
    with open(tmp_path / "runs.csv", newline="") as f:
        (row,) = csv.DictReader(f)
    assert row["params_json"] == '{"aggregation":"topk","k":32,"lr":0.001}'
    assert json.loads(row["params_json"]) == params


def test_export_sorts_across_experiments_and_reports_the_row_count(tmp_path: Path):
    uri = f"sqlite:///{tmp_path}/mlruns.db"
    t = MlflowTracker(uri)
    for experiment, name in [("b", "r0"), ("a", "r1"), ("b", "r2"), ("a", "r0")]:
        t.log_run(
            experiment=experiment,
            run_name=name,
            params={"k": 1},
            metrics={"objective": 0.1},
            tags={"kind": "trial", "model": "zscore"},
        )
    assert export_runs(uri, tmp_path / "runs.csv", experiments=["b", "a"]) == 4
    with open(tmp_path / "runs.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["experiment"], r["run_name"]) for r in rows] == [
        ("a", "r0"),
        ("a", "r1"),
        ("b", "r0"),
        ("b", "r2"),
    ]


def test_two_trackers_on_one_store_share_an_experiment(tmp_path: Path):
    uri = f"sqlite:///{tmp_path}/mlruns.db"
    for name in ("a", "b"):
        MlflowTracker(uri).log_run(
            experiment="x",
            run_name=name,
            params={"k": 1},
            metrics={"objective": 0.1},
            tags={"kind": "trial", "model": "zscore"},
        )
    assert export_runs(uri, tmp_path / "runs.csv", experiments=["x"]) == 2


def test_export_names_an_experiment_that_does_not_exist(tmp_path: Path):
    uri = f"sqlite:///{tmp_path}/mlruns.db"
    MlflowTracker(uri)
    with pytest.raises(ValueError, match="missing-experiment"):
        export_runs(uri, tmp_path / "runs.csv", experiments=["missing-experiment"])
    assert not (tmp_path / "runs.csv").exists()


def test_export_names_a_run_that_this_tracker_did_not_write(tmp_path: Path):
    import mlflow

    uri = f"sqlite:///{tmp_path}/mlruns.db"
    client = mlflow.MlflowClient(tracking_uri=uri)
    exp_id = client.create_experiment("x")
    client.create_run(exp_id, run_name="foreign")
    with pytest.raises(ValueError, match="foreign"):
        export_runs(uri, tmp_path / "runs.csv", experiments=["x"])


def test_a_run_that_cannot_be_logged_is_marked_failed(tmp_path: Path):
    import mlflow

    uri = f"sqlite:///{tmp_path}/mlruns.db"
    tracker = MlflowTracker(uri)
    with pytest.raises(mlflow.exceptions.MlflowException):
        tracker.log_run(
            experiment="x",
            run_name="bad-metric",
            params={"k": 1},
            metrics={"not a valid name?": 1.0},
            tags={"kind": "trial", "model": "zscore"},
        )
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("x")
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FAILED"


def test_params_that_mlflow_would_truncate_are_refused_before_a_run_exists(tmp_path: Path):
    import mlflow

    uri = f"sqlite:///{tmp_path}/mlruns.db"
    tracker = MlflowTracker(uri)
    with pytest.raises(ValueError, match="too-long"):
        tracker.log_run(
            experiment="x",
            run_name="too-long",
            params={"payload": "z" * 7000},
            metrics={},
            tags={"kind": "trial", "model": "zscore"},
        )
    client = mlflow.MlflowClient(tracking_uri=uri)
    assert client.get_experiment_by_name("x") is None


def test_a_param_named_params_json_is_refused(tmp_path: Path):
    tracker = MlflowTracker(f"sqlite:///{tmp_path}/mlruns.db")
    with pytest.raises(ValueError, match="params_json"):
        tracker.log_run(
            experiment="x", run_name="a", params={"params_json": 1}, metrics={}, tags={}
        )


def test_mlflow_tracker_does_not_touch_the_global_tracking_uri(tmp_path: Path):
    import mlflow

    before = mlflow.get_tracking_uri()
    MlflowTracker(f"sqlite:///{tmp_path}/mlruns.db").log_run(
        experiment="x", run_name="a", params={}, metrics={}, tags={}
    )
    assert mlflow.get_tracking_uri() == before


def test_tuning_into_mlflow_records_every_trial(tmp_path: Path):
    uri = f"sqlite:///{tmp_path}/mlruns.db"
    result = tune("zscore", [FakeSource()], n_trials=3, seed=0, tracker=MlflowTracker(uri))
    assert export_runs(uri, tmp_path / "runs.csv", experiments=[TUNING_EXPERIMENT]) == 3
    with open(tmp_path / "runs.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["run_name"] for r in rows] == [f"zscore-trial-0{i}" for i in range(3)]
    assert [r["run_id"] for r in rows] == [t.run_id for t in result.trials]
    assert [json.loads(r["params_json"]) for r in rows] == [t.params for t in result.trials]
    assert [float(r["objective"]) for r in rows] == [t.value for t in result.trials]
