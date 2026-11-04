"""The schema of ``metrics.json``, the one file every published number is derived from.

``reproduce`` writes a `Metrics` object; ``claims.json`` and the README block are computed from
it and from nothing else, so a number that appears in the README can always be traced to a field
here. Every model forbids unknown fields and is frozen: a typo in a field name fails at
validation, and a result cannot be edited after it is built.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

PERCENTILES = (2.5, 97.5)
"""The percentiles that bound a 95 % bootstrap interval."""


class _Strict(BaseModel):
    """Base of every metrics model: unknown fields are errors and instances are frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProtocolScoreModel(_Strict):
    """The three numbers every protocol reports for one detector.

    Attributes:
        auc_pr: Average precision under the protocol.
        f1: F1 at the threshold fixed on validation data.
        f1_oracle: The best F1 any threshold could reach (an upper bound, not a result).
    """

    auc_pr: float
    f1: float
    f1_oracle: float


class EntityResultModel(_Strict):
    """One detector's result on one entity.

    Attributes:
        entity: Entity name, such as ``machine-1-1``.
        validation_objective: The objective the hyperparameters were selected on.
        threshold: The frozen decision threshold.
        protocols: Scores by protocol name.
        sensitivity: Event-level AUC-PR under changed protocol constants, by setting name.
        digests: Content digests of the fitted state and the score arrays, by name.
    """

    entity: str
    validation_objective: float
    threshold: float
    protocols: dict[str, ProtocolScoreModel]
    sensitivity: dict[str, float]
    digests: dict[str, str]


class ModelSummaryModel(_Strict):
    """One detector's results over all entities.

    Attributes:
        params: The tuned hyperparameters.
        validation_objective: Macro mean of the entities' validation objective.
        macro: Macro mean over entities of each protocol's scores.
        sensitivity_macro: Macro mean over entities of the sensitivity scores.
        per_entity: The per-entity results, in entity order.
    """

    params: dict[str, int | float | str]
    validation_objective: float
    macro: dict[str, ProtocolScoreModel]
    sensitivity_macro: dict[str, float]
    per_entity: list[EntityResultModel]


class HeadlineModel(_Strict):
    """The single number the project reports.

    Attributes:
        model: The detector with the highest macro validation objective.
        protocol: The protocol the number is reported under.
        auc_pr: Macro mean over entities of that detector's event-level AUC-PR.
        ci95: Bootstrap 95 % interval of that mean, over entities.
        selected_by: What the detector was chosen by.
        label_reads_at_selection: Labels read while selecting; the guards require 0.
    """

    model: str
    protocol: Literal["event_level"]
    auc_pr: float
    ci95: tuple[float, float]
    selected_by: Literal["validation_objective"]
    label_reads_at_selection: int


class AblationModel(_Strict):
    """What removing the FFT features changes, for one detector, over several seeds.

    The per-seed lists are aligned with ``seeds``. Standard deviations are over seeds (sample
    standard deviation; a single seed gives 0.0).

    Attributes:
        model: Detector name.
        seeds: The seeds the ablation ran with.
        f1_with: Macro event-level F1 per seed with all feature specs.
        f1_without: The same without the FFT specs.
        auc_pr_with: Macro event-level AUC-PR per seed with all feature specs.
        auc_pr_without: The same without the FFT specs.
        delta_f1_points_mean: Mean of ``100 * (with - without)`` for F1.
        delta_f1_points_std: Standard deviation of the same.
        delta_f1_relative_mean: Mean of ``100 * (with - without) / without`` for F1.
        delta_f1_relative_std: Standard deviation of the same.
        delta_auc_pr_mean: Mean of ``with - without`` for AUC-PR.
        delta_auc_pr_std: Standard deviation of the same.
    """

    model: str
    seeds: list[int]
    f1_with: list[float]
    f1_without: list[float]
    auc_pr_with: list[float]
    auc_pr_without: list[float]
    delta_f1_points_mean: float
    delta_f1_points_std: float
    delta_f1_relative_mean: float
    delta_f1_relative_std: float
    delta_auc_pr_mean: float
    delta_auc_pr_std: float


class TuningModel(_Strict):
    """How the hyperparameters were chosen.

    Attributes:
        entities: The entities the search ran on.
        n_trials: Trials per detector.
        best: For each detector, ``{"params": ..., "value": ...}`` of the best trial.
    """

    entities: list[str]
    n_trials: int
    best: dict[str, dict[str, object]]


class DatasetModel(_Strict):
    """The data the run used.

    Attributes:
        source_commit: The pinned commit of the dataset repository.
        entities: Number of entities.
        channels: Number of channels per entity.
        train_steps: Training rows over all entities.
        test_steps: Test rows over all entities.
        anomaly_segments: Anomaly segments inside the final (evaluation) phase.
    """

    source_commit: str
    entities: int
    channels: int
    train_steps: int
    test_steps: int
    anomaly_segments: int


class Metrics(_Strict):
    """Everything a ``reproduce`` run measured: the content of ``metrics.json``.

    Attributes:
        schema_version: Version of this schema; bumped when a field changes meaning.
        profile: Name of the profile that produced the run (``full`` or ``smoke``).
        seed: The run's root seed.
        dataset: What data the run covered.
        protocol: The protocol constants (chunk, cooldown, K), by name.
        feature_specs: Number of registered feature specs.
        models: Per-detector results, by detector name.
        headline: The reported number.
        ablation: FFT ablation results, by detector name.
        tuning: The hyperparameter search.
        environment: Machine- and time-dependent facts (versions, commit, timestamp). The only
            field that two runs of the same code may legitimately disagree on.
    """

    schema_version: Literal[1]
    profile: str
    seed: int
    dataset: DatasetModel
    protocol: dict[str, int]
    feature_specs: int
    models: dict[str, ModelSummaryModel]
    headline: HeadlineModel
    ablation: dict[str, AblationModel]
    tuning: TuningModel
    environment: dict[str, str]


def write_metrics(path: Path, metrics: Metrics) -> None:
    """Write metrics as canonical JSON: sorted keys, two-space indent, trailing newline.

    The text is built before the file is opened, so a refused value leaves no file behind.

    Args:
        path: Destination file.
        metrics: The metrics to write.

    Raises:
        ValueError: A value is NaN or infinite, which JSON cannot represent.
    """
    text = json.dumps(metrics.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def read_metrics(path: Path) -> Metrics:
    """Read and validate a ``metrics.json``.

    Args:
        path: The file to read.

    Returns:
        The validated metrics.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: The file is not valid JSON or does not match the schema (pydantic's
            ``ValidationError`` is a ``ValueError``); the message names the offending fields.
    """
    return Metrics.model_validate_json(path.read_text(encoding="utf-8"))


def deterministic_view(metrics: Metrics) -> dict[str, object]:
    """Return the metrics as a JSON-ready dict without the ``environment`` field.

    Two runs of the same code on the same data must produce equal views; ``environment`` holds
    versions and a timestamp, which differ between machines and moments.

    Args:
        metrics: The metrics to view.

    Returns:
        The ``model_dump(mode="json")`` of ``metrics`` minus ``environment``.
    """
    dump = metrics.model_dump(mode="json")
    del dump["environment"]
    return dump


def bootstrap_ci(values: Sequence[float], *, seed: int, n: int = 1000) -> tuple[float, float]:
    """Return the percentile bootstrap 95 % interval of the mean of ``values``.

    Args:
        values: The sample, such as one score per entity.
        seed: Seed of the resampling generator.
        n: Number of resamples.

    Returns:
        The 2.5th and 97.5th percentiles of the means of ``n`` resamples with replacement.

    Raises:
        ValueError: ``values`` is empty or ``n`` is below 1.
    """
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size == 0:
        raise ValueError(f"bootstrap_ci needs a non-empty 1-D sample, got shape {sample.shape}")
    if n < 1:
        raise ValueError(f"bootstrap_ci needs at least one resample, got n={n}")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, sample.size, size=(n, sample.size))
    low, high = np.percentile(sample[picks].mean(axis=1), PERCENTILES)
    return float(low), float(high)
