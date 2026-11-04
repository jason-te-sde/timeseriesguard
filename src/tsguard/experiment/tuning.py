"""Hyper-parameter tuning: an Optuna search over the synthetic validation objective.

A trial is scored by the entities' runners, never by test labels: its value is the mean over
the sources (one per entity) of the mean fold objective, where the fold objective is the
event-level AUC-PR against anomalies injected into the train rows. The first trial is always
the model's default configuration, so the search can never end up worse than not tuning, and
every trial is recorded through a `Tracker`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from tsguard.experiment.runner import FoldOutcome
from tsguard.experiment.tracking import Tracker
from tsguard.models import Params
from tsguard.models.space import DEFAULT_PARAMS, suggest_params
from tsguard.repro.seeding import derive_seed

if TYPE_CHECKING:
    import optuna

TUNING_EXPERIMENT: Final = "tsguard-tuning"
"""Name of the tracker experiment that holds the tuning trials."""

FINAL_EXPERIMENT: Final = "tsguard-final"
"""Name of the tracker experiment that holds the final runs of the chosen configurations."""


class ObjectiveSource(Protocol):
    """Anything that scores a configuration on one entity; `EntityRunner` is one.

    Attributes:
        entity: Name of the entity, used to name the per-entity metric of each trial.
    """

    entity: str

    def objective(
        self, model: str, params: Params, *, seed: int | None = None
    ) -> tuple[FoldOutcome, ...]:
        """Score ``params`` of ``model`` on every validation fold of the entity.

        Args:
            model: Registry name of the model.
            params: Hyper-parameters of the model.
            seed: Root seed of the run.

        Returns:
            One outcome per fold.
        """
        ...


@dataclass(frozen=True)
class TrialRecord:
    """One finished trial.

    Attributes:
        number: Trial number, counting from 0; trial 0 is the default configuration.
        params: The configuration the trial scored.
        value: Mean over the sources of the mean fold objective.
        run_id: Id of the run the tracker recorded for the trial.
    """

    number: int
    params: dict[str, int | float | str]
    value: float
    run_id: str


@dataclass(frozen=True)
class TuningResult:
    """The outcome of a search.

    Attributes:
        model: Registry name of the tuned model.
        best_params: The configuration of the best trial; the earliest one if several tie.
        best_value: Value of the best trial.
        trials: Every trial, in order.
    """

    model: str
    best_params: dict[str, int | float | str]
    best_value: float
    trials: tuple[TrialRecord, ...]


def tune(
    model: str,
    sources: Sequence[ObjectiveSource],
    *,
    n_trials: int,
    seed: int,
    tracker: Tracker,
) -> TuningResult:
    """Search the hyper-parameters of ``model`` for the best synthetic validation objective.

    The search is a TPE sampler seeded from ``seed``, so equal arguments give the same trials.
    The sampler's first trial is the model's default configuration. Each trial is logged to
    the `TUNING_EXPERIMENT` experiment as the run ``<model>-trial-<NN>``, with the tags
    ``kind=trial`` and ``model=<model>``, the metric ``objective`` and one metric
    ``objective.<entity>`` per source. Optuna's own logging is silenced to warnings for the
    duration of the call.

    Args:
        model: Registry name of a model with a search space.
        sources: One objective source per entity; each is called with ``model`` and the
            trial's parameters and with ``seed`` as the root seed.
        n_trials: Number of trials, the default configuration included.
        seed: Root seed; the sampler's seed is derived from it and the model name.
        tracker: Where the trials are recorded.

    Returns:
        The best configuration and every trial.

    Raises:
        ValueError: ``model`` has no search space, ``n_trials`` is below 1, there is no
            source, or two sources share an entity name.
        RuntimeError: A source returned no fold outcomes or a non-finite objective.
    """
    import optuna

    if model not in DEFAULT_PARAMS:
        valid = ", ".join(DEFAULT_PARAMS)
        raise ValueError(f"no search space for model {model!r}; valid names: {valid}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if not sources:
        raise ValueError("tune needs at least one objective source")
    entities = [source.entity for source in sources]
    if len(set(entities)) != len(entities):
        duplicated = sorted({e for e in entities if entities.count(e) > 1})
        raise ValueError(f"objective sources must have distinct entities; repeated: {duplicated}")

    records: list[TrialRecord] = []

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(model, trial)
        per_entity = {
            source.entity: _entity_value(source, model, params, seed) for source in sources
        }
        value = sum(per_entity.values()) / len(per_entity)
        run_id = tracker.log_run(
            experiment=TUNING_EXPERIMENT,
            run_name=f"{model}-trial-{trial.number:02d}",
            params=params,
            metrics={
                "objective": value,
                **{f"objective.{entity}": v for entity, v in per_entity.items()},
            },
            tags={"kind": "trial", "model": model},
        )
        records.append(TrialRecord(trial.number, params, value, run_id))
        return value

    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=derive_seed(seed, "optuna", model)),
        )
        study.enqueue_trial(dict(DEFAULT_PARAMS[model]))
        study.optimize(objective, n_trials=n_trials)
    finally:
        optuna.logging.set_verbosity(previous)

    best = max(records, key=lambda record: (record.value, -record.number))
    return TuningResult(
        model=model,
        best_params=dict(best.params),
        best_value=best.value,
        trials=tuple(records),
    )


def _entity_value(source: ObjectiveSource, model: str, params: Params, seed: int) -> float:
    """Return the mean fold objective of one source, which must be finite."""
    outcomes = source.objective(model, params, seed=seed)
    if not outcomes:
        raise RuntimeError(f"{source.entity}: model {model!r} was scored on no validation fold")
    value = sum(outcome.objective for outcome in outcomes) / len(outcomes)
    if not math.isfinite(value):
        raise RuntimeError(
            f"{source.entity}: model {model!r} has a non-finite objective {value} "
            f"for {dict(params)}"
        )
    return value
