"""The reproduce pipeline: tune, select without labels, evaluate once, ablate, and report.

`reproduce` is the one function that turns a `Profile` into ``metrics.json``. Its phases run in
a fixed order, and the order is what the leakage claims rest on:

1. **Tuning.** One runner per tuning entity is kept for all four models; each model is tuned
   on the synthetic validation objective of those runners, which touches train rows only.
2. **Selection**, entity-major. Each entity gets a fresh `LabelAudit` and its own runner, and
   all four models are selected with their tuned parameters. A runner holds 2 to 3 GB of
   cached features on a real entity, so it is dropped before the next entity is loaded. What
   stays is small: the selections (without fitted models), the entity's guarded labels and its
   audit.
3. **The blind check.** Every audit must have logged zero reads, and every frozen threshold
   must predate every read, or the run stops with a `RuntimeError` before a single test label
   is used.
4. **Evaluation.** Each selection is scored on the test labels, once.
5. **Ablation.** The FFT features are removed and re-added on the frozen selections of the
   feature models, over several seeds (see `tsguard.experiment.ablation`). It runs with
   ``causal_check=False``: both arms use specs the selection phase already verified causal
   (the 53-spec arm is a subset of the same registry and pipeline), so repeating the
   future-perturbation tests would only add about 20 minutes to a full run.

Progress goes to a callback, one line per phase, entity and model step, so a run of several
hours can be followed from a log file.
"""

from __future__ import annotations

import dataclasses
import gc
import importlib
import importlib.metadata
import platform
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
import torch

from tsguard.data.labels import AuditedLabels, LabelAudit, Phase
from tsguard.data.smd import ENTITIES, SMD_COMMIT
from tsguard.eval.protocols import PROTOCOLS, ProtocolScore, segments
from tsguard.experiment.ablation import AblationSummary, EntityLoader, run_ablation
from tsguard.experiment.runner import (
    EntityRunner,
    Evaluation,
    RunConfig,
    Selection,
    evaluate_selection,
)
from tsguard.experiment.tracking import Tracker
from tsguard.experiment.tuning import FINAL_EXPERIMENT, TuningResult, tune
from tsguard.features.specs import FEATURE_SPECS
from tsguard.models import FEATURE_MODELS, MODEL_NAMES
from tsguard.repro.report import (
    AblationModel,
    DatasetModel,
    EntityResultModel,
    HeadlineModel,
    Metrics,
    ModelSummaryModel,
    ProtocolScoreModel,
    TuningModel,
    bootstrap_ci,
    write_metrics,
)
from tsguard.repro.seeding import configure_runtime

HEADLINE_PROTOCOL: Final = "event_level"
UNKNOWN_REVISION: Final = "unknown"
_PACKAGES: Final = ("numpy", "pandas", "scikit-learn", "torch", "optuna", "mlflow")
_NOT_INSTALLED: Final = "not installed"


@dataclass(frozen=True)
class Profile:
    """How much of the benchmark one run covers.

    Attributes:
        name: Profile name, recorded in ``metrics.json``.
        entities: Entities the models are selected on and evaluated on.
        tuning_entities: Entities the hyper-parameter search scores trials on.
        n_trials: Tuning trials per model, the default configuration included.
        ablation_seeds: Root seeds of the FFT ablation.
        run: The settings of every runner; its ``seed`` is the root seed of tuning, selection
            and the bootstrap.
    """

    name: str
    entities: tuple[str, ...]
    tuning_entities: tuple[str, ...]
    n_trials: int
    ablation_seeds: tuple[int, ...]
    run: RunConfig

    def __post_init__(self) -> None:
        """Refuse a profile that would fail only after hours of work.

        Raises:
            ValueError: ``entities``, ``tuning_entities`` or ``ablation_seeds`` is empty or
                repeats a value, or ``n_trials`` is below 1. The message names the field.
        """
        for field in ("entities", "tuning_entities", "ablation_seeds"):
            values = getattr(self, field)
            if not values:
                raise ValueError(f"profile {self.name!r}: {field} must not be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"profile {self.name!r}: {field} repeats a value: {values}")
        if self.n_trials < 1:
            raise ValueError(f"profile {self.name!r}: n_trials must be at least 1")


FULL: Final = Profile(
    "full", ENTITIES, ("machine-1-1", "machine-2-1", "machine-3-1"), 8, (0, 1, 2, 3, 4), RunConfig()
)
"""All 28 machines, 8 tuning trials per model, five ablation seeds."""

SMOKE: Final = Profile(
    "smoke", ("machine-1-1", "machine-2-1"), ("machine-1-1",), 2, (0,), RunConfig(ae_epochs=1)
)
"""Two machines and the shortest settings that still run every phase; what CI executes."""

PROFILES: Final[Mapping[str, Profile]] = MappingProxyType({"full": FULL, "smoke": SMOKE})
"""The profiles ``tsguard reproduce`` accepts, by name."""


@dataclass(frozen=True, eq=False)
class _EntityRecord:
    """What the selection phase keeps of one entity: no features, no fitted state."""

    name: str
    labels: AuditedLabels
    audit: LabelAudit
    n_train: int
    n_test: int
    channels: int
    selections: dict[str, Selection]


def label_reads_at_selection(audits: Iterable[LabelAudit], selections: Iterable[Selection]) -> int:
    """Count the test-label reads that happened before evaluation, and refuse if any did.

    Args:
        audits: Every audit the run created: one per entity of tuning and of selection.
        selections: Every selection made.

    Returns:
        The number of reads, which is 0.

    Raises:
        RuntimeError: An audit logged a read, or a threshold was frozen after a read. Either
            means a label was used while models and thresholds were still being chosen.
    """
    reads = sum(audit.reads for audit in audits)
    if reads:
        raise RuntimeError(
            f"{reads} test-label read{'s' if reads != 1 else ''} happened during selection; "
            f"the headline would not be label-free"
        )
    for selection in selections:
        if selection.threshold.reads_before:
            raise RuntimeError(
                f"{selection.entity}: the threshold of model {selection.model!r} was frozen "
                f"after {selection.threshold.reads_before} test-label "
                f"read{'s' if selection.threshold.reads_before != 1 else ''}"
            )
    return reads


def git_revision(cwd: Path) -> str:
    """Return the commit the code is at, marked when the working tree has changes.

    Args:
        cwd: A directory inside the checkout.

    Returns:
        The full sha of ``HEAD``, followed by ``+dirty`` when ``git status --porcelain`` lists
        anything (untracked files included); ``"unknown"`` when ``cwd`` is not in a git
        checkout with a commit, or git is not installed.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN_REVISION
    return f"{sha}+dirty" if status.strip() else sha


def require_experiment_extra() -> None:
    """Import Optuna, which the tuning phase needs, before any runner exists.

    The first import of Optuna leaves a traceback in module state for good, and that traceback
    pins the frame that triggered the import. Left to `tune`, which imports Optuna lazily, that
    frame holds the tuning runners, and each one's 2 to 3 GB of cached features would survive
    every later phase. Importing here, with no runner built yet, keeps the pinned frame
    harmless. It also stops a run that lacks the extra before it loads any data.

    Raises:
        ImportError: Optuna is not installed; the message names the extra to install.
    """
    try:
        importlib.import_module("optuna")
    except ImportError as exc:
        raise ImportError(
            "reproduce needs the 'experiment' extra: uv sync --extra experiment"
        ) from exc


def reproduce(
    profile: Profile,
    *,
    loader: EntityLoader,
    tracker: Tracker,
    out_dir: Path,
    progress: Callable[[str], None] = lambda _: None,
) -> Metrics:
    """Run every phase of a profile and write ``metrics.json``.

    Args:
        profile: What to run.
        loader: Loads an entity whose labels report to the audit it is given; called once per
            entity in each phase that needs the entity, so a phase never holds more than it
            uses.
        tracker: Where the tuning trials and the final runs are recorded.
        out_dir: Directory for ``metrics.json``; created if missing.
        progress: Called with one line per phase, entity and model step.

    Returns:
        The metrics that were written.

    Raises:
        ImportError: The ``experiment`` extra (Optuna) is not installed.
        RuntimeError: A test label was read during selection, or a model produced a
            non-finite score.
        ValueError: The ablation is undefined (some model's F1 without the FFT features is 0).
        LeakageError: A leakage guard fired.
    """
    require_experiment_extra()
    configure_runtime(threads=profile.run.threads)
    revision = git_revision(Path(__file__).resolve().parent)
    tuned, tuning_audits = _tune_all(profile, loader, tracker, progress)
    gc.collect()  # the tuning runners refer to themselves; free their caches before selecting
    records: list[_EntityRecord] = []
    for name in profile.entities:
        records.append(_select_entity(name, profile, loader, tuned, tracker, progress))
        gc.collect()
    selections = [s for record in records for s in record.selections.values()]
    reads = label_reads_at_selection([*tuning_audits, *(r.audit for r in records)], selections)
    progress(f"selection done: {reads} test-label reads")
    evaluations, n_segments = _evaluate(records, profile, progress)
    ablation = run_ablation(
        profile.entities,
        loader=loader,
        params={model: tuned[model].best_params for model in FEATURE_MODELS},
        seeds=profile.ablation_seeds,
        config=dataclasses.replace(profile.run, causal_check=False),
        progress=progress,
    )
    models = {
        model: _summarize_model(model, tuned[model], records, evaluations) for model in MODEL_NAMES
    }
    headline = _headline(profile, models, reads)
    metrics = Metrics(
        schema_version=1,
        profile=profile.name,
        seed=profile.run.seed,
        dataset=DatasetModel(
            source_commit=SMD_COMMIT,
            entities=len(records),
            channels=records[0].channels,
            train_steps=sum(r.n_train for r in records),
            test_steps=sum(r.n_test for r in records),
            anomaly_segments=n_segments,
        ),
        protocol=dataclasses.asdict(profile.run.protocol),
        feature_specs=len(FEATURE_SPECS),
        models=models,
        headline=headline,
        ablation={model: _ablation_model(summary) for model, summary in ablation.items()},
        tuning=TuningModel(
            entities=list(profile.tuning_entities),
            n_trials=profile.n_trials,
            best={
                model: {"params": result.best_params, "value": result.best_value}
                for model, result in tuned.items()
            },
        ),
        environment=_environment(revision),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(out_dir / "metrics.json", metrics)
    progress(f"wrote {out_dir / 'metrics.json'}")
    return metrics


def _tune_all(
    profile: Profile,
    loader: EntityLoader,
    tracker: Tracker,
    progress: Callable[[str], None],
) -> tuple[dict[str, TuningResult], list[LabelAudit]]:
    """Tune every model on the tuning entities; the runners are locals, so they die on return."""
    audits: list[LabelAudit] = []
    runners: list[EntityRunner] = []
    for name in profile.tuning_entities:
        audit = LabelAudit()
        audits.append(audit)
        runners.append(EntityRunner(loader(name, audit), audit, profile.run))
    tuned: dict[str, TuningResult] = {}
    for model in MODEL_NAMES:
        tuned[model] = tune(
            model,
            runners,
            n_trials=profile.n_trials,
            seed=profile.run.seed,
            tracker=tracker,
        )
        progress(
            f"tune {model}: best objective {tuned[model].best_value:.4f} over "
            f"{profile.n_trials} trials on {', '.join(profile.tuning_entities)}"
        )
    return tuned, audits


def _select_entity(
    name: str,
    profile: Profile,
    loader: EntityLoader,
    tuned: Mapping[str, TuningResult],
    tracker: Tracker,
    progress: Callable[[str], None],
) -> _EntityRecord:
    """Select all four models on one entity and log one final run per selection."""
    audit = LabelAudit()
    data = loader(name, audit)
    runner = EntityRunner(data, audit, profile.run)
    selections: dict[str, Selection] = {}
    for model in MODEL_NAMES:
        selection = runner.select(model, tuned[model].best_params, seed=profile.run.seed)
        tracker.log_run(
            experiment=FINAL_EXPERIMENT,
            run_name=f"{model}-{name}",
            params=selection.params,
            metrics={"objective": selection.objective},
            tags={"kind": "final", "model": model},
        )
        selections[model] = selection
        progress(f"select {name} {model}: validation objective {selection.objective:.4f}")
    return _EntityRecord(
        name=name,
        labels=data.labels,
        audit=audit,
        n_train=len(data.train),
        n_test=data.n_test,
        channels=int(data.train.shape[1]),
        selections=selections,
    )


def _evaluate(
    records: Sequence[_EntityRecord], profile: Profile, progress: Callable[[str], None]
) -> tuple[dict[tuple[str, str], Evaluation], int]:
    """Score every selection on its entity's test labels; count the anomaly segments."""
    evaluations: dict[tuple[str, str], Evaluation] = {}
    n_segments = 0
    for record in records:
        for model, selection in record.selections.items():
            evaluation = evaluate_selection(
                selection, record.labels, record.audit, profile.run.protocol
            )
            evaluations[(record.name, model)] = evaluation
            score = evaluation.protocols[HEADLINE_PROTOCOL].auc_pr
            progress(f"evaluate {record.name} {model}: event-level AUC-PR {score:.4f}")
        with record.audit.phase(Phase.FINAL_EVALUATION):
            n_segments += len(segments(record.labels.read("count_segments")))
    return evaluations, n_segments


def _mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean as a Python float."""
    return float(np.mean(list(values)))


def _protocol_model(score: ProtocolScore) -> ProtocolScoreModel:
    return ProtocolScoreModel(auc_pr=score.auc_pr, f1=score.f1, f1_oracle=score.f1_oracle)


def _summarize_model(
    model: str,
    tuning: TuningResult,
    records: Sequence[_EntityRecord],
    evaluations: Mapping[tuple[str, str], Evaluation],
) -> ModelSummaryModel:
    """Build one model's per-entity results and their macro means over the entities."""
    per_entity = []
    for record in records:
        selection = record.selections[model]
        evaluation = evaluations[(record.name, model)]
        per_entity.append(
            EntityResultModel(
                entity=record.name,
                validation_objective=selection.objective,
                threshold=selection.threshold.value,
                protocols={name: _protocol_model(evaluation.protocols[name]) for name in PROTOCOLS},
                sensitivity=dict(evaluation.sensitivity),
                digests=dict(selection.digests),
            )
        )
    return ModelSummaryModel(
        params=dict(tuning.best_params),
        validation_objective=_mean(r.validation_objective for r in per_entity),
        macro={
            name: ProtocolScoreModel(
                auc_pr=_mean(r.protocols[name].auc_pr for r in per_entity),
                f1=_mean(r.protocols[name].f1 for r in per_entity),
                f1_oracle=_mean(r.protocols[name].f1_oracle for r in per_entity),
            )
            for name in PROTOCOLS
        },
        sensitivity_macro={
            key: _mean(r.sensitivity[key] for r in per_entity) for key in per_entity[0].sensitivity
        },
        per_entity=per_entity,
    )


def best_model(validation_objective: Mapping[str, float]) -> str:
    """Return the model with the highest validation objective.

    Args:
        validation_objective: Macro validation objective by model name; it must hold every
            name in `MODEL_NAMES`.

    Returns:
        The winner. A tie goes to the model that comes first in `MODEL_NAMES`, whatever the
        order of the mapping.
    """
    return max(MODEL_NAMES, key=lambda model: validation_objective[model])


def _headline(
    profile: Profile, models: Mapping[str, ModelSummaryModel], reads: int
) -> HeadlineModel:
    """Report the event-level score of the model with the best macro validation objective."""
    best = best_model({model: summary.validation_objective for model, summary in models.items()})
    chosen = models[best]
    per_entity = [e.protocols[HEADLINE_PROTOCOL].auc_pr for e in chosen.per_entity]
    low, high = bootstrap_ci(per_entity, seed=profile.run.seed)
    return HeadlineModel(
        model=best,
        protocol=HEADLINE_PROTOCOL,
        auc_pr=chosen.macro[HEADLINE_PROTOCOL].auc_pr,
        ci95=(low, high),
        selected_by="validation_objective",
        label_reads_at_selection=reads,
    )


def _ablation_model(summary: AblationSummary) -> AblationModel:
    return AblationModel(
        model=summary.model,
        seeds=list(summary.seeds),
        f1_with=list(summary.f1_with),
        f1_without=list(summary.f1_without),
        auc_pr_with=list(summary.auc_pr_with),
        auc_pr_without=list(summary.auc_pr_without),
        delta_f1_points_mean=summary.delta_f1_points_mean,
        delta_f1_points_std=summary.delta_f1_points_std,
        delta_f1_relative_mean=summary.delta_f1_relative_mean,
        delta_f1_relative_std=summary.delta_f1_relative_std,
        delta_auc_pr_mean=summary.delta_auc_pr_mean,
        delta_auc_pr_std=summary.delta_auc_pr_std,
    )


def _environment(revision: str) -> dict[str, str]:
    """Describe the machine and the code: versions, thread count, commit and time."""
    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = _NOT_INSTALLED
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        **versions,
        "torch_threads": str(torch.get_num_threads()),
        "git_sha": revision,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
