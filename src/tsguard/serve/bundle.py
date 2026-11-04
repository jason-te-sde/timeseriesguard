"""The model bundle: a fitted pipeline and detector in one checksummed file, and scoring with it.

A bundle is what ``EntityRunner.select`` left after fitting the last walk-forward fold: the
fitted feature pipeline, the fitted detector and the frozen threshold, for one entity. Nothing
else travels with it: not the training rows, and not a label (selection reads none).

The file is a joblib pickle of a plain dict, not of `Bundle`, because a fitted `FeaturePipeline`
cannot be pickled: its feature specs are closures. The dict holds the pipeline's fitted
statistics and which spec registry it was built from, and `load_bundle` rebuilds the pipeline
from that registry, then checks the rebuilt state against the digest recorded at save time.

The sha256 sidecar (``<bundle>.sha256``) detects a damaged or swapped file before anything is
unpickled. It is not a signature: whoever can write the bundle can write its sidecar, and
unpickling runs code. Load only bundles you built.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np

from tsguard.data.labels import LabelAudit
from tsguard.data.smd import SMD_COMMIT, EntityData
from tsguard.errors import ChecksumError, ContractError, TsguardError
from tsguard.experiment.runner import EntityRunner, RunConfig, Selection
from tsguard.features.pipeline import RAW_SPECS, FeaturePipeline
from tsguard.features.specs import FEATURE_SPECS, FeatureSpec
from tsguard.models import Params
from tsguard.models.base import Detector, InputKind
from tsguard.models.iforest import IsolationForestDetector

FORMAT_VERSION: Final = 1
"""Version of the dict that `save_bundle` pickles; bumped when a key changes meaning."""

SIDECAR_SUFFIX: Final = ".sha256"
_DIGEST: Final = re.compile(r"[0-9a-f]{64}")


class HistoryTooShortError(TsguardError, ValueError):
    """The history holds fewer rows than the bundle needs to score its last row.

    Attributes:
        required_rows: Rows the bundle needs.
        received_rows: Rows the caller supplied.
    """

    def __init__(self, required_rows: int, received_rows: int) -> None:
        """Create the error.

        Args:
            required_rows: Rows the bundle needs.
            received_rows: Rows the caller supplied.
        """
        super().__init__(f"needs {required_rows} rows of history, got {received_rows}")
        self.required_rows = required_rows
        self.received_rows = received_rows


@dataclass(frozen=True, eq=False)
class Bundle:
    """A fitted pipeline and detector for one entity, with the threshold frozen for them.

    Attributes:
        entity: Entity the models were fitted on.
        model: Registry name of the detector.
        params: Hyper-parameters the detector was built with.
        threshold: Decision threshold frozen from the last fold's synthetic labels.
        pipeline: The fitted feature pipeline. Do not refit it.
        detector: The fitted detector.
        source_commit: The dataset commit the entity was read from.
    """

    entity: str
    model: str
    params: dict[str, int | float | str]
    threshold: float
    pipeline: FeaturePipeline
    detector: Detector
    source_commit: str

    def __post_init__(self) -> None:
        """Refuse a bundle whose pipeline was never fitted.

        Raises:
            ValueError: ``pipeline`` is not fitted.
        """
        if self.pipeline.fit_end is None:
            raise ValueError(f"{self.entity}: the pipeline of bundle {self.model!r} is not fitted")

    @property
    def required_rows(self) -> int:
        """Rows of history a request must carry to score its last row."""
        return self.pipeline.required_history + self.detector.lookback - 1

    @property
    def fit_end(self) -> int:
        """Number of rows the pipeline was fitted on; no row before it can be scored."""
        fit_end = self.pipeline.fit_end
        assert fit_end is not None  # checked in __post_init__
        return fit_end


def bundle_from_selection(selection: Selection) -> Bundle:
    """Turn a selection that kept its models into a bundle.

    The Isolation Forest is set to one thread, in place on the selection's detector: a service
    must not fan one request out across the machine, and the thread count changes no score.

    Args:
        selection: A selection made with ``keep_models=True``.

    Returns:
        The bundle.

    Raises:
        ValueError: ``selection`` did not keep its pipeline and detector.
    """
    pipeline, detector = selection.pipeline, selection.detector
    if pipeline is None or detector is None:
        raise ValueError(
            f"{selection.entity}: the selection of {selection.model!r} kept no fitted models; "
            f"select with keep_models=True to bundle it"
        )
    if isinstance(detector, IsolationForestDetector):
        detector.set_n_jobs(1)
    return Bundle(
        entity=selection.entity,
        model=selection.model,
        params=dict(selection.params),
        threshold=float(selection.threshold.value),
        pipeline=pipeline,
        detector=detector,
        source_commit=SMD_COMMIT,
    )


def build_bundle(
    data: EntityData, audit: LabelAudit, model: str, params: Params, *, config: RunConfig
) -> Bundle:
    """Select a model on one entity and bundle what it fitted.

    This is the selection of the experiment, run once: it reads no label, and the bundle's
    pipeline and detector are the ones whose scores the evaluation of the same configuration
    reports.

    Args:
        data: The entity.
        audit: The audit ``data.labels`` reports to.
        model: Registry name of the detector.
        params: Hyper-parameters of the detector.
        config: The settings of the run.

    Returns:
        The bundle.

    Raises:
        LeakageError: A leakage guard fired during selection.
        ValueError: ``model`` or ``params`` is not valid, or ``audit`` is not the audit of
            ``data.labels``.
    """
    selection = EntityRunner(data, audit, config).select(model, params, keep_models=True)
    return bundle_from_selection(selection)


def score_rows(bundle: Bundle, history: np.ndarray, *, first_index: int) -> float:
    """Score the last row of ``history``.

    Only the trailing ``required_rows`` rows are read, so the result does not depend on how much
    older history the caller sent, and it equals the score the batch evaluation gave the row.

    Args:
        bundle: The bundle to score with.
        history: Raw channel values ``(n, 38)``, in time order.
        first_index: Absolute row index of ``history[0]``.

    Returns:
        The anomaly score of absolute row ``first_index + len(history) - 1``; higher is more
        anomalous.

    Raises:
        HistoryTooShortError: ``history`` has fewer than ``bundle.required_rows`` rows (a
            `ValueError`).
        LeakageError: The scored row is before ``bundle.fit_end`` (guard ``fit_end``).
        ValueError: ``first_index`` is negative, or ``history`` has the wrong number of
            channels.
    """
    rows = np.asarray(history, dtype=np.float64)
    if first_index < 0:
        raise ValueError(f"first_index must not be negative, got {first_index}")
    if len(rows) < bundle.required_rows:
        raise HistoryTooShortError(bundle.required_rows, len(rows))
    row = first_index + len(rows) - 1
    context = bundle.detector.lookback - 1
    features = bundle.pipeline.transform(rows, start_index=row, context=context, offset=first_index)
    return float(bundle.detector.score(features, context=context)[-1])


def save_bundle(bundle: Bundle, path: Path) -> str:
    """Write a bundle and its checksum sidecar.

    The digest is taken from the bytes that are written, and the sidecar is written last, so a
    crash between the two files leaves a bundle that `load_bundle` refuses.

    Args:
        bundle: The bundle to write.
        path: Destination of the bundle; ``<path>.sha256`` holds its digest. Missing parent
            directories are created.

    Returns:
        The sha256 hex digest of the bundle file.
    """
    buffer = io.BytesIO()
    joblib.dump(_payload(bundle), buffer)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    _sidecar(path).write_text(digest + "\n", encoding="utf-8")
    return digest


def load_bundle(path: Path) -> tuple[Bundle, str]:
    """Read a bundle, after checking its bytes against the sidecar.

    The file is read once; the digest is taken from those bytes, and the same bytes are
    unpickled, so the file cannot change between the check and the load.

    Args:
        path: The bundle file; ``<path>.sha256`` must sit next to it.

    Returns:
        The bundle and its sha256 hex digest.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ChecksumError: The sidecar is missing or is not a sha256 digest, or the file does not
            match it.
        ContractError: The file is not a bundle of this format, or the rebuilt pipeline or
            detector does not reproduce the state recorded at save time.
    """
    data = path.read_bytes()
    expected = _expected_digest(path)
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ChecksumError(
            f"{path}: sha256 is {actual} but {_sidecar(path).name} says {expected}; the file was "
            f"changed or damaged after it was saved, so it is not loaded"
        )
    payload = joblib.load(io.BytesIO(data))
    return _from_payload(payload, path), actual


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + SIDECAR_SUFFIX)


def _expected_digest(path: Path) -> str:
    sidecar = _sidecar(path)
    try:
        text = sidecar.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise ChecksumError(
            f"{path}: no checksum file {sidecar}; a bundle without one is not loaded"
        ) from None
    if not _DIGEST.fullmatch(text):
        raise ChecksumError(f"{sidecar}: not a lowercase hex sha256 digest: {text[:80]!r}")
    return text


def _specs_for(kind: InputKind) -> tuple[FeatureSpec, ...]:
    """Return the registry a pipeline for detectors of ``kind`` is built from."""
    return FEATURE_SPECS if kind == "features" else RAW_SPECS


def _payload(bundle: Bundle) -> dict[str, Any]:
    pipeline = bundle.pipeline
    # The pipeline has no public accessor for its fitted statistics; state_digest, checked on
    # load, is what stops this from drifting out of step with the class.
    return {
        "format": FORMAT_VERSION,
        "entity": bundle.entity,
        "model": bundle.model,
        "params": dict(bundle.params),
        "threshold": bundle.threshold,
        "source_commit": bundle.source_commit,
        "input_kind": bundle.detector.input_kind,
        "pipeline": {
            "columns": pipeline.columns,
            "fit_end": bundle.fit_end,
            "mean": np.array(pipeline._mean),
            "scale": np.array(pipeline._scale),
            "digest": pipeline.state_digest(),
        },
        "detector": bundle.detector,
        "detector_digest": bundle.detector.state_digest(),
    }


def _from_payload(payload: object, path: Path) -> Bundle:
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_VERSION:
        found = payload.get("format") if isinstance(payload, dict) else type(payload).__name__
        raise ContractError(
            f"{path}: not a tsguard bundle of format {FORMAT_VERSION} (found {found!r})"
        )
    try:
        kind: InputKind = payload["input_kind"]
        pipeline = _restore_pipeline(payload["pipeline"], kind, path)
        detector: Detector = payload["detector"]
        if detector.state_digest() != payload["detector_digest"]:
            raise ContractError(
                f"{path}: the detector does not reproduce the state recorded when it was saved"
            )
        return Bundle(
            entity=payload["entity"],
            model=payload["model"],
            params=dict(payload["params"]),
            threshold=float(payload["threshold"]),
            pipeline=pipeline,
            detector=detector,
            source_commit=payload["source_commit"],
        )
    except KeyError as missing:
        raise ContractError(f"{path}: bundle has no field {missing}") from None


def _restore_pipeline(state: dict[str, Any], kind: InputKind, path: Path) -> FeaturePipeline:
    pipeline = FeaturePipeline(_specs_for(kind))
    if pipeline.columns != tuple(state["columns"]):
        raise ContractError(
            f"{path}: the bundle was fitted on {len(state['columns'])} feature columns that do "
            f"not match this version's {len(pipeline.columns)}; rebuild the bundle"
        )
    mean, scale = np.array(state["mean"]), np.array(state["scale"])
    mean.flags.writeable = False
    scale.flags.writeable = False
    pipeline._fit_end, pipeline._mean, pipeline._scale = int(state["fit_end"]), mean, scale
    if pipeline.state_digest() != state["digest"]:
        raise ContractError(
            f"{path}: the rebuilt feature pipeline does not reproduce the state recorded when "
            f"the bundle was saved; rebuild the bundle"
        )
    return pipeline
