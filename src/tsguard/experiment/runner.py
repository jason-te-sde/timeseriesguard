"""The entity runner: the one place that wires data, features, models and protocols.

`EntityRunner.select` chooses nothing from test labels. It fits one feature pipeline per
walk-forward fold, plants seeded synthetic anomalies in each validation block, scores them,
freezes a decision threshold from the last fold, and scores the test rows. Along the way it
calls every leakage guard itself, so a leaky component handed to the runner is stopped by the
guard that exists for it (the tests in ``tests/leakage/test_runner_wiring.py`` pin each call).

`EntityRunner.evaluate` is the only place the project's normal flow reads test labels: it opens
the final-evaluation phase, reads the labels once under the name ``evaluate[<model>]``, and
scores the frozen selection under every protocol.

Pipeline fits and injected validation blocks are cached on the runner, because they depend on
the entity, the fold and the seed but not on the model: a search over several models or
hyper-parameters fits each fold's pipeline once. The seed-keyed caches grow with every seed
used; `EntityRunner.drop_seed_caches` forgets one seed's.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from tsguard.data.labels import AuditedLabels, FrozenThreshold, LabelAudit, Phase
from tsguard.data.smd import EntityData
from tsguard.data.splits import (
    Fold,
    check_fold_boundaries,
    check_split_monotonic,
    chronological_split,
    walk_forward_folds,
)
from tsguard.errors import LeakageError
from tsguard.eval.inject import InjectionConfig, inject, reference_sigma
from tsguard.eval.protocols import (
    ProtocolName,
    ProtocolParams,
    ProtocolScore,
    event_level_auc_pr,
    select_threshold,
)
from tsguard.eval.protocols import evaluate as evaluate_protocols
from tsguard.features.causality import assert_causal, assert_fit_ignores_future
from tsguard.features.pipeline import RAW_SPECS, FeaturePipeline
from tsguard.features.specs import FEATURE_SPECS, FeatureSpec
from tsguard.models import Params, build_detector
from tsguard.models.base import Detector, InputKind
from tsguard.repro.hashing import array_digest
from tsguard.repro.seeding import derive_seed

SENSITIVITY_GRID: Final[tuple[tuple[int, int], ...]] = (
    (30, 0),
    (30, 60),
    (60, 0),
    (60, 60),
    (120, 0),
    (120, 60),
)
"""The ``(chunk, cooldown)`` pairs the event-level AUC-PR is re-computed for, as a check that
the reported number does not hinge on the two protocol constants."""

PipelineFactory = Callable[[Sequence[FeatureSpec]], FeaturePipeline]
FoldFn = Callable[[int], tuple[Fold, ...]]
SplitFn = Callable[[int, int], tuple[np.ndarray, np.ndarray]]
ThresholdFn = Callable[[np.ndarray, np.ndarray, ProtocolParams], float]


@dataclass(frozen=True)
class RunConfig:
    """The settings of one run.

    Attributes:
        seed: Root seed; every random choice of the run is derived from it.
        folds: Number of walk-forward folds.
        initial_frac: Fraction of the train rows the first fit window covers.
        purge: Rows dropped between a fit window and its validation window; also the purge
            the fold guard demands.
        protocol: The constants of the evaluation protocols.
        injection: How the synthetic validation anomalies are drawn.
        causal_check: Whether `EntityRunner.select` runs the future-perturbation tests: the
            feature transform and the fitting once per input kind, and each model's scoring
            once per input kind.
        ae_epochs: Training epochs of the autoencoders.
        threads: Threads of the models that fit in parallel.
    """

    seed: int = 0
    folds: int = 3
    initial_frac: float = 0.5
    purge: int = 64
    protocol: ProtocolParams = field(default_factory=ProtocolParams)
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    causal_check: bool = True
    ae_epochs: int = 8
    threads: int = 4


@dataclass(frozen=True)
class FoldOutcome:
    """How one model did on the synthetic anomalies of one validation block.

    Attributes:
        fold: 1-based fold number.
        objective: Event-level AUC-PR of the scores against the injected labels.
        n_segments: Number of anomalies injected into the block.
    """

    fold: int
    objective: float
    n_segments: int


@dataclass(frozen=True, eq=False)
class Selection:
    """A model configuration chosen without test labels, with its scores on the test rows.

    Equality is by identity (``eq=False``): the default field-by-field comparison would
    raise on the ambiguous truth value of an array comparison.

    Attributes:
        entity: Entity name.
        model: Registry name of the model.
        params: Hyper-parameters the model was built with.
        seed: Root seed the run used.
        folds: One outcome per validation fold.
        objective: Mean of the fold objectives.
        threshold: Decision threshold frozen from the last fold's synthetic labels.
        test_scores: One finite score per test row, read-only.
        digests: ``"pipeline"``, ``"detector"`` and ``"test_scores"`` digests of the last
            fold's fitted pipeline, its fitted detector and the test scores.
        pipeline: The last fold's fitted pipeline; kept only when ``keep_models`` was set. It is
            the runner's cached instance, which later selections share: do not refit it.
        detector: The last fold's fitted detector; kept only when ``keep_models`` was set.
    """

    entity: str
    model: str
    params: dict[str, int | float | str]
    seed: int
    folds: tuple[FoldOutcome, ...]
    objective: float
    threshold: FrozenThreshold
    test_scores: np.ndarray
    digests: dict[str, str]
    pipeline: FeaturePipeline | None = None
    detector: Detector | None = None


@dataclass(frozen=True)
class Evaluation:
    """A selection scored on the test labels.

    Attributes:
        selection: What was evaluated.
        protocols: Average precision, F1 at the frozen threshold and best F1 under each
            protocol.
        sensitivity: Event-level AUC-PR for each pair of `SENSITIVITY_GRID`, keyed
            ``"L<chunk>_C<cooldown>"``.
    """

    selection: Selection
    protocols: dict[ProtocolName, ProtocolScore]
    sensitivity: dict[str, float]


@dataclass(frozen=True, eq=False)
class _FoldRun:
    """Everything one fold produced that `EntityRunner.select` goes on to use."""

    outcome: FoldOutcome
    fold: Fold
    pipeline: FeaturePipeline
    detector: Detector
    scores: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class _Plan:
    """What a model needs before any fold is fitted."""

    kind: InputKind
    specs: Sequence[FeatureSpec]
    folds: tuple[Fold, ...]
    lookback: int


def _read_only(a: np.ndarray) -> np.ndarray:
    """Mark ``a`` read-only and return it, so a cached array cannot be changed by a user."""
    a.flags.writeable = False
    return a


def evaluate_selection(
    selection: Selection, labels: AuditedLabels, audit: LabelAudit, params: ProtocolParams
) -> Evaluation:
    """Score a selection on the test labels: the only label read of the normal flow.

    Args:
        selection: The selection to score.
        labels: The test labels of the selection's entity.
        audit: The audit ``labels`` report to; it is put in the final-evaluation phase for
            the read only.
        params: The protocol constants.

    Returns:
        The selection with its protocol scores and its sensitivity to the event-level
        constants.

    Raises:
        LabelAccessError: ``audit`` is not the audit of ``labels``, which then refuses the
            read.
        TypeError: The threshold of the selection is not a `FrozenThreshold`.
        ValueError: The labels have no anomaly or do not match the test scores in length.
    """
    with audit.phase(Phase.FINAL_EVALUATION):
        y = labels.read(f"evaluate[{selection.model}]")
    protocols = evaluate_protocols(selection.test_scores, y, selection.threshold, params)
    sensitivity = {
        f"L{chunk}_C{cooldown}": event_level_auc_pr(
            selection.test_scores, y, dataclasses.replace(params, chunk=chunk, cooldown=cooldown)
        )
        for chunk, cooldown in SENSITIVITY_GRID
    }
    return Evaluation(selection=selection, protocols=protocols, sensitivity=sensitivity)


class EntityRunner:
    """Selects a model on one entity without test labels, and evaluates it once.

    Components that decide how a leak could happen (the feature specs, the pipeline class,
    the fold and split builders and the threshold search) are constructor arguments, so a test
    can hand the runner a leaky one and see which guard stops it.
    """

    def __init__(
        self,
        data: EntityData,
        audit: LabelAudit,
        config: RunConfig = RunConfig(),  # noqa: B008
        *,
        specs: Sequence[FeatureSpec] = FEATURE_SPECS,
        pipeline_factory: PipelineFactory = FeaturePipeline,
        fold_fn: FoldFn | None = None,
        split_fn: SplitFn = chronological_split,
        threshold_fn: ThresholdFn = select_threshold,
    ) -> None:
        """Create a runner.

        Args:
            data: The entity.
            audit: The audit the entity's labels report to.
            config: The settings of the run.
            specs: Feature specs for the detectors that read features.
            pipeline_factory: Builds an unfitted pipeline from feature specs.
            fold_fn: Maps the number of train rows to walk-forward folds; by default
                `walk_forward_folds` with the settings of ``config``.
            split_fn: Maps ``(n_train, n_test)`` to the train and test row indices.
            threshold_fn: Picks the decision threshold from synthetic validation scores
                and labels.

        Raises:
            ValueError: ``audit`` is not the audit ``data.labels`` reports to; selecting
                under another audit would say nothing about the labels' guard.
        """
        if data.labels.audit is not audit:
            raise ValueError(
                f"{data.name}: the audit given to the runner is not the audit the entity's "
                f"labels report to, so the runner's selection phase would not guard them"
            )
        self._data = data
        self._audit = audit
        self._config = config
        self._specs = tuple(specs)
        self._pipeline_factory = pipeline_factory
        self._fold_fn: FoldFn = fold_fn if fold_fn is not None else self._default_folds
        self._split_fn = split_fn
        self._threshold_fn = threshold_fn
        self._lookbacks: dict[InputKind, int] = {}
        self._fits: dict[tuple[InputKind, Fold], tuple[FeaturePipeline, np.ndarray]] = {}
        self._injections: dict[tuple[int, Fold], tuple[np.ndarray, np.ndarray, int]] = {}
        self._validation_rows: dict[tuple[InputKind, Fold, int, int], np.ndarray] = {}
        self._causal_done: set[tuple[str, ...]] = set()

    @property
    def entity(self) -> str:
        """Name of the entity this runner works on."""
        return self._data.name

    def objective(
        self, model: str, params: Params, *, seed: int | None = None
    ) -> tuple[FoldOutcome, ...]:
        """Score a configuration on the synthetic anomalies of every validation fold.

        This is the quantity a hyper-parameter search maximizes. It touches the train rows
        only, and no label.

        Args:
            model: Registry name of the model.
            params: Hyper-parameters of the model.
            seed: Root seed; the one in the run config if omitted.

        Returns:
            One outcome per fold.

        Raises:
            LeakageError: A fold or feature guard fired; the error carries a note naming
                the entity and the model.
            ValueError: ``model`` or ``params`` is not valid.
            RuntimeError: The model produced a non-finite validation score.
        """
        root = self._config.seed if seed is None else seed
        with self._annotated(model):
            plan = self._plan(model, params, root)
            return tuple(run.outcome for run in self._run_folds(model, params, root, plan))

    def select(
        self,
        model: str,
        params: Params,
        *,
        seed: int | None = None,
        keep_models: bool = False,
    ) -> Selection:
        """Fit a configuration on the train rows, freeze its threshold, and score the test rows.

        No test label is read. The whole call runs in the selection phase of the audit, even
        inside an outer final-evaluation block, so a threshold search that reaches for the
        labels is refused. The threshold comes from the last fold's synthetic labels, and the
        test rows are scored with the pipeline and detector of the last fold.

        Args:
            model: Registry name of the model.
            params: Hyper-parameters of the model.
            seed: Root seed; the one in the run config if omitted.
            keep_models: Keep the last fold's fitted pipeline and detector in the result. The
                pipeline is the runner's cached instance, which later selections share, so it
                must not be refitted.

        Returns:
            The selection.

        Raises:
            LeakageError: The split, a fold, the feature pipeline, a detector or the threshold
                search broke a leakage guard; the error's ``guard`` names it and a note names
                the entity and the model.
            ValueError: ``model`` or ``params`` is not valid.
            RuntimeError: The model produced a non-finite score.
        """
        root = self._config.seed if seed is None else seed
        with self._annotated(model), self._audit.phase(Phase.SELECTION):
            return self._select(model, params, root, keep_models)

    def drop_seed_caches(self, seed: int) -> None:
        """Forget the injected validation blocks and rows cached for one seed.

        The pipeline fits do not depend on the seed and stay. A run over several seeds on one
        runner calls this after each seed: on a real entity a seed holds about 150 MB.

        Args:
            seed: The root seed whose caches to drop; a seed never used is ignored.
        """
        for key in [key for key in self._injections if key[0] == seed]:
            del self._injections[key]
        for validation_key in [key for key in self._validation_rows if key[2] == seed]:
            del self._validation_rows[validation_key]

    def evaluate(self, selection: Selection) -> Evaluation:
        """Score a selection of this entity on its test labels.

        Args:
            selection: A selection made for this entity.

        Returns:
            The protocol scores and the sensitivity grid.

        Raises:
            ValueError: ``selection`` belongs to another entity, or the test labels have no
                anomaly.
        """
        if selection.entity != self.entity:
            raise ValueError(
                f"selection of entity {selection.entity!r} cannot be evaluated on the "
                f"labels of entity {self.entity!r}"
            )
        return evaluate_selection(selection, self._data.labels, self._audit, self._config.protocol)

    @contextmanager
    def _annotated(self, model: str) -> Iterator[None]:
        """Add the entity and the model to the notes of any leakage error raised inside."""
        try:
            yield
        except LeakageError as error:
            error.add_note(f"entity {self._data.name}, model {model}")
            raise

    def _select(self, model: str, params: Params, seed: int, keep_models: bool) -> Selection:
        data, config = self._data, self._config
        train_idx, test_idx = self._split_fn(data.train_end, data.n_test)
        check_split_monotonic(train_idx, test_idx)
        plan = self._plan(model, params, seed)
        runs = self._run_folds(model, params, seed, plan)
        last = runs[-1]
        threshold = self._audit.freeze_threshold(
            self._threshold_fn(last.scores, last.labels, config.protocol),
            caller=f"select[{model}]",
        )
        context = last.detector.lookback - 1
        full = data.full
        rows = last.pipeline.transform(full, start_index=data.train_end, context=context)
        test_scores = np.array(last.detector.score(rows, context=context), dtype=np.float64)
        if not np.isfinite(test_scores).all():
            raise RuntimeError(
                f"{data.name}: model {model!r} produced {int((~np.isfinite(test_scores)).sum())} "
                f"non-finite scores on the test rows"
            )
        _read_only(test_scores)
        self._check_causal(model, plan, last, full, rows, seed)
        folds = tuple(run.outcome for run in runs)
        return Selection(
            entity=data.name,
            model=model,
            params=dict(params),
            seed=seed,
            folds=folds,
            objective=float(np.mean([fold.objective for fold in folds])),
            threshold=threshold,
            test_scores=test_scores,
            digests={
                "pipeline": last.pipeline.state_digest(),
                "detector": last.detector.state_digest(),
                "test_scores": array_digest(test_scores),
            },
            pipeline=last.pipeline if keep_models else None,
            detector=last.detector if keep_models else None,
        )

    def _default_folds(self, n: int) -> tuple[Fold, ...]:
        config = self._config
        return walk_forward_folds(
            n, k=config.folds, initial_frac=config.initial_frac, purge=config.purge
        )

    def _pipeline_lookback(self, kind: InputKind, specs: Sequence[FeatureSpec]) -> int:
        """Return how many rows the pipeline of ``kind`` reads, from an unfitted instance."""
        if kind not in self._lookbacks:
            self._lookbacks[kind] = self._pipeline_factory(specs).lookback
        return self._lookbacks[kind]

    def _plan(self, model: str, params: Params, seed: int) -> _Plan:
        """Learn the model's input kind and lookback, and check the folds against them."""
        config, entity = self._config, self._data.name
        probe = build_detector(
            model,
            params,
            seed=derive_seed(seed, entity, model, 0),
            threads=config.threads,
            epochs=config.ae_epochs,
        )
        kind = probe.input_kind
        specs: Sequence[FeatureSpec] = self._specs if kind == "features" else RAW_SPECS
        folds = self._fold_fn(self._data.train_end)
        lookback = max(self._pipeline_lookback(kind, specs), probe.lookback)
        check_fold_boundaries(folds, lookback=lookback, purge=config.purge)
        return _Plan(kind=kind, specs=specs, folds=folds, lookback=lookback)

    def _fitted(self, plan: _Plan, fold: Fold) -> tuple[FeaturePipeline, np.ndarray]:
        """Return the pipeline fitted on the fold's fit rows and its in-sample features."""
        key = (plan.kind, fold)
        if key not in self._fits:
            pipeline = self._pipeline_factory(plan.specs)
            in_sample = pipeline.fit_transform(self._data.train, end_index=fold.fit_end)
            self._fits[key] = (pipeline, _read_only(in_sample))
        return self._fits[key]

    def _injected(self, seed: int, fold: Fold) -> tuple[np.ndarray, np.ndarray, int]:
        """Return the train rows up to the fold's validation end with anomalies planted."""
        key = (seed, fold)
        if key not in self._injections:
            config, train = self._config, self._data.train
            sigma = reference_sigma(train[: fold.fit_end], fallback=config.injection.fallback_sigma)
            rng = np.random.default_rng(derive_seed(seed, self._data.name, "inject", fold.index))
            x_inj, y_inj, segments = inject(
                train[: fold.val_end],
                start=fold.val_start,
                end=fold.val_end,
                sigma=sigma,
                rng=rng,
                config=config.injection,
            )
            self._injections[key] = (_read_only(x_inj), _read_only(y_inj), len(segments))
        return self._injections[key]

    def _validation(
        self, plan: _Plan, fold: Fold, seed: int, pipeline: FeaturePipeline, context: int
    ) -> np.ndarray:
        """Return the standardized validation rows, preceded by ``context`` history rows."""
        key = (plan.kind, fold, seed, context)
        if key not in self._validation_rows:
            x_inj, _, _ = self._injected(seed, fold)
            rows = pipeline.transform(x_inj, start_index=fold.val_start, context=context)
            self._validation_rows[key] = _read_only(rows)
        return self._validation_rows[key]

    def _run_folds(self, model: str, params: Params, seed: int, plan: _Plan) -> list[_FoldRun]:
        """Fit and score the model on every fold's synthetic validation block."""
        config, entity = self._config, self._data.name
        runs: list[_FoldRun] = []
        for fold in plan.folds:
            pipeline, in_sample = self._fitted(plan, fold)
            _, y_inj, n_segments = self._injected(seed, fold)
            detector = build_detector(
                model,
                params,
                seed=derive_seed(seed, entity, model, fold.index),
                threads=config.threads,
                epochs=config.ae_epochs,
            )
            context = detector.lookback - 1
            val_rows = self._validation(plan, fold, seed, pipeline, context)
            detector.fit(in_sample)
            scores = detector.score(val_rows, context=context)
            if not np.isfinite(scores).all():
                raise RuntimeError(
                    f"{entity}: model {model!r} produced non-finite scores on the validation "
                    f"block of fold {fold.index}"
                )
            objective = event_level_auc_pr(scores, y_inj, config.protocol)
            runs.append(
                _FoldRun(
                    outcome=FoldOutcome(fold.index, objective, n_segments),
                    fold=fold,
                    pipeline=pipeline,
                    detector=detector,
                    scores=scores,
                    labels=y_inj,
                )
            )
        return runs

    def _check_causal(
        self,
        model: str,
        plan: _Plan,
        last: _FoldRun,
        full: np.ndarray,
        test_rows: np.ndarray,
        seed: int,
    ) -> None:
        """Run the future-perturbation tests that have not run yet for this model.

        Per input kind, once: the pipeline's transform of the test rows must not read later
        rows, and its fit on the fit rows must not depend on the rows after them. Per input
        kind and model, once: the detector's scores over the standardized test rows must not
        read later rows. The last two do not recompute any feature.
        """
        if not self._config.causal_check:
            return
        data, kind, pipeline, detector = self._data, plan.kind, last.pipeline, last.detector
        label = (seed, data.name, "causal")
        if ("transform", kind) not in self._causal_done:
            assert_causal(
                lambda X: pipeline.transform(X, start_index=data.train_end),
                full,
                t0=data.train_end + data.n_test // 2,
                start=data.train_end,
                rng=np.random.default_rng(derive_seed(*label)),
            )
            self._causal_done.add(("transform", kind))
        if ("fit", kind) not in self._causal_done:

            def fit_digest(X: np.ndarray) -> str:
                fitted = self._pipeline_factory(plan.specs).fit(X, end_index=last.fold.fit_end)
                return fitted.state_digest()

            assert_fit_ignores_future(
                fit_digest,
                full,
                end_index=last.fold.fit_end,
                rng=np.random.default_rng(derive_seed(*label)),
            )
            self._causal_done.add(("fit", kind))
        if ("score", kind, model) not in self._causal_done:
            context = detector.lookback - 1
            assert_causal(
                lambda X: detector.score(X, context=context),
                test_rows,
                t0=context + data.n_test // 2,
                start=context,
                rng=np.random.default_rng(derive_seed(*label, model)),
            )
            self._causal_done.add(("score", kind, model))
