"""The two commands that build and run a bundle, as plain functions the command line calls.

`bundle_command` is the entry point that fits (it sets the torch runtime like every entry
point that trains), and `serve_command` the one that listens.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import uvicorn

from tsguard.data.labels import LabelAudit
from tsguard.data.smd import load_entity
from tsguard.experiment.runner import RunConfig
from tsguard.models import MODEL_NAMES, Params
from tsguard.models.space import DEFAULT_PARAMS
from tsguard.repro.report import read_metrics
from tsguard.repro.seeding import configure_runtime
from tsguard.serve.app import create_app
from tsguard.serve.bundle import build_bundle, save_bundle

FALLBACK_MODEL: Final = "iforest"
"""The model bundled when no metrics exist to name a headline."""

SERVE_THREADS: Final = 1
"""Torch threads of a served process: one request must not fan out across the machine."""


def resolve_configuration(
    *, model: str | None, metrics_path: Path, default_params: bool
) -> tuple[str, Params]:
    """Choose the model and hyper-parameters a bundle is built with.

    The model is the one given, else the headline of the metrics if the file exists, else
    `FALLBACK_MODEL`. The parameters are the registry defaults if ``default_params``, else the
    ones the metrics recorded for that model.

    Args:
        model: Registry name, or ``None`` to take the headline.
        metrics_path: A ``metrics.json``; it need not exist when ``default_params`` is set.
        default_params: Use the registry defaults instead of the tuned parameters.

    Returns:
        The model name and its hyper-parameters.

    Raises:
        FileNotFoundError: The tuned parameters are wanted and ``metrics_path`` does not exist.
        ValueError: ``model`` is not a registered model, or the metrics hold no entry for it,
            or the metrics file is invalid.
    """
    exists = metrics_path.exists()
    if not exists and not default_params:
        raise FileNotFoundError(
            f"{metrics_path} does not exist, so there are no tuned parameters to build the "
            f"bundle with; run the experiment first or use the default parameters"
        )
    metrics = read_metrics(metrics_path) if exists else None
    chosen = model or (metrics.headline.model if metrics is not None else FALLBACK_MODEL)
    if default_params:
        return chosen, _defaults(chosen)
    assert metrics is not None  # not exists and not default_params was refused above
    if chosen not in metrics.models:
        raise ValueError(
            f"{metrics_path} has no model {chosen!r}; it holds {', '.join(sorted(metrics.models))}"
        )
    return chosen, dict(metrics.models[chosen].params)


def _defaults(model: str) -> Mapping[str, int | float | str]:
    if model not in DEFAULT_PARAMS:
        raise ValueError(f"unknown model {model!r}; valid names: {', '.join(MODEL_NAMES)}")
    return dict(DEFAULT_PARAMS[model])


def bundle_command(
    *,
    entity: str,
    model: str | None,
    metrics_path: Path,
    default_params: bool,
    out: Path,
    data_dir: Path | None = None,
) -> str:
    """Fit a model on one entity's train rows and write it as a checksummed bundle.

    Labels are never read: the entity is loaded with a fresh audit that stays in its selection
    phase, and the bundle is the selection of the experiment.

    Args:
        entity: SMD machine name, such as ``machine-1-1``.
        model: Registry name, or ``None`` for the headline of the metrics (``iforest`` if
            there are none).
        metrics_path: A ``metrics.json``; it need not exist when ``default_params`` is set.
        default_params: Use the registry defaults instead of the tuned parameters.
        out: Destination of the bundle; ``<out>.sha256`` is written beside it.
        data_dir: SMD data directory; the default cache if omitted.

    Returns:
        A one-line summary naming the file, the entity, the model and the sha256 digest.

    Raises:
        FileNotFoundError: See `resolve_configuration`.
        ValueError: See `resolve_configuration`.
        ChecksumError: A data file does not match the pinned manifest.
    """
    chosen, params = resolve_configuration(
        model=model, metrics_path=metrics_path, default_params=default_params
    )
    config = RunConfig()
    configure_runtime(threads=config.threads)
    audit = LabelAudit()
    data = load_entity(entity, audit=audit, dest=data_dir)
    bundle = build_bundle(data, audit, chosen, params, config=config)
    digest = save_bundle(bundle, out)
    return f"wrote {out}: entity {entity}, model {chosen}, sha256 {digest}"


def serve_command(*, bundle: Path, host: str, port: int) -> None:
    """Serve a bundle over HTTP until the process is stopped.

    The bundle is loaded and verified before the server starts listening.

    Args:
        bundle: A bundle file with its sidecar.
        host: Address to bind.
        port: Port to bind.

    Raises:
        FileNotFoundError: ``bundle`` does not exist.
        ChecksumError: The bundle does not match its sidecar.
        ContractError: The file is not a usable bundle.
    """
    app = create_app(bundle)
    configure_runtime(threads=SERVE_THREADS)
    uvicorn.run(app, host=host, port=port)
