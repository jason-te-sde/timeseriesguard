"""The FFT ablation: how much the spectral features add, measured on the frozen selection.

Every entity is run twice with the same data, the same audit and the same seeds: once with the
full 63-spec registry and once with the ``fft`` family removed. Nothing else differs between the
two arms, so the gap in event-level F1 (at the threshold frozen in selection) and in event-level
AUC-PR is the contribution of the FFT features. A seed's value is the macro mean over entities;
the spread reported is the sample standard deviation over seeds.

The loop is entity-major. A real entity's runner caches its fitted pipelines and their features
(2 to 3 GB), so the runners of one entity are released before the next entity is loaded.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from tsguard.data.labels import LabelAudit
from tsguard.data.smd import EntityData
from tsguard.experiment.runner import EntityRunner, RunConfig, evaluate_selection
from tsguard.features.specs import FEATURE_SPECS, specs_without
from tsguard.models import FEATURE_MODELS, Params

ABLATED_FAMILY: Final = "fft"
"""The feature family the ablation removes."""

EntityLoader = Callable[[str, LabelAudit], EntityData]
"""Loads the named entity with its labels reporting to the given audit."""


@dataclass(frozen=True)
class AblationSummary:
    """What removing the FFT features did to one model, over several seeds.

    The per-seed tuples are indexed like ``seeds``. Each value is the mean over the entities
    of the entity's event-level score under that seed.

    Attributes:
        model: Registry name of the model.
        seeds: The root seeds the ablation ran with.
        f1_with: Event-level F1 at the frozen threshold, with the FFT features.
        f1_without: The same without them.
        auc_pr_with: Event-level AUC-PR with the FFT features.
        auc_pr_without: The same without them.
        delta_f1_points_mean: Mean over seeds of ``100 * (with - without)`` for F1.
        delta_f1_points_std: Sample standard deviation of the same; 0.0 for one seed.
        delta_f1_relative_mean: Mean over seeds of ``100 * (with - without) / without`` for F1.
        delta_f1_relative_std: Sample standard deviation of the same; 0.0 for one seed.
        delta_auc_pr_mean: Mean over seeds of ``with - without`` for AUC-PR.
        delta_auc_pr_std: Sample standard deviation of the same; 0.0 for one seed.
    """

    model: str
    seeds: tuple[int, ...]
    f1_with: tuple[float, ...]
    f1_without: tuple[float, ...]
    auc_pr_with: tuple[float, ...]
    auc_pr_without: tuple[float, ...]
    delta_f1_points_mean: float
    delta_f1_points_std: float
    delta_f1_relative_mean: float
    delta_f1_relative_std: float
    delta_auc_pr_mean: float
    delta_auc_pr_std: float


@dataclass(frozen=True)
class _Score:
    """The two numbers the ablation compares, for one entity, seed, model and arm."""

    f1: float
    auc_pr: float


_Cell = tuple[str, int]
"""``(model, seed)``: the unit the entities' scores are collected under."""


def run_ablation(
    entities: Sequence[str],
    *,
    loader: EntityLoader,
    params: Mapping[str, Params],
    seeds: Sequence[int],
    config: RunConfig,
    models: Sequence[str] = FEATURE_MODELS,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, AblationSummary]:
    """Measure the FFT features' contribution to each model over several seeds.

    For each entity, in order: load it once under a fresh `LabelAudit`, build one runner with
    the full feature registry and one without the ``fft`` family (both on that entity's data
    and audit), then for each seed and model select a configuration and score it on the test
    labels in both arms. The runners are released before the next entity is loaded.

    Args:
        entities: Names of the entities to run, each passed to ``loader``.
        loader: Loads an entity whose labels report to the audit it is given.
        params: Hyper-parameters of each model in ``models``.
        seeds: Root seeds; each is a full selection and evaluation in both arms.
        config: Settings of the runs; its ``seed`` is not used, the ``seeds`` are.
        models: Models to ablate: only models that read the engineered features, because the
            others never see the FFT features and both arms would be the same run.
        progress: Called with one line after each entity, seed and model.

    Returns:
        One summary per model, keyed by its name, in the order of ``models``.

    Raises:
        ValueError: ``entities``, ``seeds`` or ``models`` is empty, a model is not a feature
            model or has no entry in ``params``, or the F1 without the FFT features of some
            model and seed is 0, which leaves the relative change undefined.
    """
    entity_names, seed_list, model_list = tuple(entities), tuple(seeds), tuple(models)
    _check_arguments(entity_names, seed_list, model_list, params)
    scores: dict[_Cell, list[tuple[_Score, _Score]]] = {
        (model, seed): [] for model in model_list for seed in seed_list
    }
    for name in entity_names:
        per_cell = _run_entity(
            name,
            loader=loader,
            params=params,
            seeds=seed_list,
            config=config,
            models=model_list,
            progress=progress,
        )
        for cell, pair in per_cell.items():
            scores[cell].append(pair)
        # A runner refers to itself through its fold builder, so reference counting alone
        # does not free its cached features; collect them before the next entity is loaded.
        gc.collect()
    return {model: _summarize(model, seed_list, scores) for model in model_list}


def _check_arguments(
    entities: tuple[str, ...],
    seeds: tuple[int, ...],
    models: tuple[str, ...],
    params: Mapping[str, Params],
) -> None:
    """Raise ValueError for an argument that would make the ablation empty or meaningless."""
    if not entities:
        raise ValueError("the ablation needs at least one entity, got none in entities")
    if not seeds:
        raise ValueError("the ablation needs at least one seed, got none in seeds")
    if not models:
        raise ValueError("the ablation needs at least one model, got none in models")
    for model in models:
        if model not in FEATURE_MODELS:
            raise ValueError(
                f"model {model!r} does not read the engineered features, so removing the FFT "
                f"features would change nothing; the feature models are {list(FEATURE_MODELS)}"
            )
        if model not in params:
            raise ValueError(f"no hyper-parameters given for model {model!r}")


def _run_entity(
    name: str,
    *,
    loader: EntityLoader,
    params: Mapping[str, Params],
    seeds: tuple[int, ...],
    config: RunConfig,
    models: tuple[str, ...],
    progress: Callable[[str], None],
) -> dict[_Cell, tuple[_Score, _Score]]:
    """Run both arms for every seed and model on one entity.

    The runners are locals of this function, so they are unreachable once it returns.

    Returns:
        The ``(with FFT, without FFT)`` scores of each ``(model, seed)``.
    """
    audit = LabelAudit()
    data = loader(name, audit)
    arms = (
        EntityRunner(data, audit, config, specs=FEATURE_SPECS),
        EntityRunner(data, audit, config, specs=specs_without(ABLATED_FAMILY)),
    )
    out: dict[_Cell, tuple[_Score, _Score]] = {}
    for seed in seeds:
        for model in models:
            with_fft, without_fft = (
                _score(runner, data, audit, config, model, params[model], seed) for runner in arms
            )
            out[(model, seed)] = (with_fft, without_fft)
            progress(
                f"{name} seed={seed} {model}: F1 {with_fft.f1:.4f} with FFT, "
                f"{without_fft.f1:.4f} without"
            )
    return out


def _score(
    runner: EntityRunner,
    data: EntityData,
    audit: LabelAudit,
    config: RunConfig,
    model: str,
    params: Params,
    seed: int,
) -> _Score:
    """Select without labels, then read the test labels once and return the event-level scores."""
    selection = runner.select(model, params, seed=seed)
    evaluation = evaluate_selection(selection, data.labels, audit, config.protocol)
    event = evaluation.protocols["event_level"]
    return _Score(f1=event.f1, auc_pr=event.auc_pr)


def _summarize(
    model: str, seeds: tuple[int, ...], scores: dict[_Cell, list[tuple[_Score, _Score]]]
) -> AblationSummary:
    """Macro-average the entities per seed, then take the deltas and their spread over seeds."""

    def per_seed(arm: int, metric: Callable[[_Score], float]) -> tuple[float, ...]:
        return tuple(
            float(np.mean([metric(pair[arm]) for pair in scores[(model, seed)]])) for seed in seeds
        )

    f1_with = per_seed(0, lambda s: s.f1)
    f1_without = per_seed(1, lambda s: s.f1)
    auc_with = per_seed(0, lambda s: s.auc_pr)
    auc_without = per_seed(1, lambda s: s.auc_pr)
    for seed, without in zip(seeds, f1_without, strict=True):
        if without == 0.0:
            raise ValueError(
                f"model {model!r}, seed {seed}: the F1 without the FFT features is 0, so the "
                f"relative change in F1 is undefined"
            )
    f1_gain = np.array(f1_with) - np.array(f1_without)
    points_mean, points_std = _mean_and_std(100.0 * f1_gain)
    relative_mean, relative_std = _mean_and_std(100.0 * f1_gain / np.array(f1_without))
    auc_mean, auc_std = _mean_and_std(np.array(auc_with) - np.array(auc_without))
    return AblationSummary(
        model=model,
        seeds=seeds,
        f1_with=f1_with,
        f1_without=f1_without,
        auc_pr_with=auc_with,
        auc_pr_without=auc_without,
        delta_f1_points_mean=points_mean,
        delta_f1_points_std=points_std,
        delta_f1_relative_mean=relative_mean,
        delta_f1_relative_std=relative_std,
        delta_auc_pr_mean=auc_mean,
        delta_auc_pr_std=auc_std,
    )


def _mean_and_std(values: np.ndarray) -> tuple[float, float]:
    """Return the mean and the sample standard deviation (``ddof=1``); 0.0 for one value."""
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return float(np.mean(values)), std
