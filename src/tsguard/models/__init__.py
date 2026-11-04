"""Detectors: the shared protocol, the feature-based models, and the registry that builds them."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal

from tsguard.models.autoencoders import AutoencoderDetector
from tsguard.models.base import Detector
from tsguard.models.iforest import IsolationForestDetector
from tsguard.models.space import DEFAULT_PARAMS
from tsguard.models.zscore import ZScoreDetector

Params = Mapping[str, int | float | str]

MODEL_NAMES: Final[tuple[str, ...]] = ("zscore", "iforest", "cnn_ae", "lstm_ae")
FEATURE_MODELS: Final = ("zscore", "iforest")
AE_MODELS: Final = ("cnn_ae", "lstm_ae")


def build_detector(
    name: str, params: Params, *, seed: int, threads: int = 4, epochs: int = 8
) -> Detector:
    """Build an unfitted detector from its registry name and hyperparameters.

    Parameters the model has a default for may be left out of ``params``.

    Args:
        name: One of `MODEL_NAMES`.
        params: Hyperparameters of the model, keyed by the names in
            `tsguard.models.space.DEFAULT_PARAMS`.
        seed: Seed of the model's random state. Models without random state ignore it.
        threads: Threads for models that fit in parallel: the Isolation Forest's ``n_jobs``.
            Other models ignore it; the autoencoders use torch's thread count, which
            `tsguard.repro.seeding.configure_runtime` sets once per process.
        epochs: Training epochs for the autoencoders. The feature models ignore it.

    Returns:
        The detector, ready for ``fit``.

    Raises:
        ValueError: ``name`` is not a registered model, ``params`` holds a key the model does
            not take, or the model rejects a value.
    """
    kwargs: dict[str, Any] = dict(params)
    match name:
        case "zscore":
            _reject_unknown_keys(name, params, ("aggregation", "k"))
            return ZScoreDetector(**kwargs)
        case "iforest":
            _reject_unknown_keys(name, params, ("n_estimators", "max_samples", "max_features"))
            return IsolationForestDetector(**kwargs, seed=seed, n_jobs=threads)
        case "cnn_ae":
            _reject_unknown_keys(name, params, ("window", "width", "latent", "lr"))
            return _build_autoencoder("cnn", kwargs, "width", seed=seed, epochs=epochs)
        case "lstm_ae":
            _reject_unknown_keys(name, params, ("window", "hidden", "latent", "lr"))
            return _build_autoencoder("lstm", kwargs, "hidden", seed=seed, epochs=epochs)
        case _:
            valid = ", ".join(MODEL_NAMES)
            raise ValueError(f"unknown model {name!r}; valid names: {valid}")


def _build_autoencoder(
    kind: Literal["cnn", "lstm"], kwargs: dict[str, Any], size_key: str, *, seed: int, epochs: int
) -> AutoencoderDetector:
    """Build an autoencoder from registry parameters, mapping ``size_key`` to ``size``."""
    merged: dict[str, Any] = {**DEFAULT_PARAMS[f"{kind}_ae"], **kwargs}
    size = merged.pop(size_key)
    return AutoencoderDetector(kind=kind, size=size, seed=seed, epochs=epochs, **merged)


def _reject_unknown_keys(name: str, params: Params, allowed: tuple[str, ...]) -> None:
    """Raise ValueError naming every key of ``params`` that model ``name`` does not take."""
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        raise ValueError(
            f"model {name!r} has no parameter {', '.join(map(repr, unknown))}; "
            f"it takes: {', '.join(allowed)}"
        )
