"""Detectors and the registry that builds them.

Four models sit behind the `Detector` protocol: the feature models ``zscore`` (a robust z-score
baseline) and ``iforest`` (an Isolation Forest), and the raw-window autoencoders ``cnn_ae`` and
``lstm_ae``. `build_detector` constructs any of them from its registry name and hyperparameters.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, get_args

from tsguard.models.autoencoders import AutoencoderDetector, Kind
from tsguard.models.base import Detector
from tsguard.models.iforest import IsolationForestDetector
from tsguard.models.space import DEFAULT_PARAMS
from tsguard.models.zscore import Aggregation, ZScoreDetector

Params = Mapping[str, int | float | str]

MODEL_NAMES: Final[tuple[str, ...]] = ("zscore", "iforest", "cnn_ae", "lstm_ae")
FEATURE_MODELS: Final = ("zscore", "iforest")
AE_MODELS: Final = ("cnn_ae", "lstm_ae")

_PARAM_TYPES: Final[Mapping[str, Mapping[str, type | tuple[str, ...]]]] = MappingProxyType(
    {
        "zscore": MappingProxyType({"aggregation": get_args(Aggregation), "k": int}),
        "iforest": MappingProxyType(
            {"n_estimators": int, "max_samples": int, "max_features": float}
        ),
        "cnn_ae": MappingProxyType({"window": int, "width": int, "latent": int, "lr": float}),
        "lstm_ae": MappingProxyType({"window": int, "hidden": int, "latent": int, "lr": float}),
    }
)
"""Per model, each parameter's type: ``int``, ``float`` (any real number) or allowed strings."""


def build_detector(
    name: str, params: Params, *, seed: int, threads: int = 4, epochs: int = 8
) -> Detector:
    """Build an unfitted detector from its registry name and hyperparameters.

    Parameters the model has a default for may be left out of ``params``. The Isolation Forest
    also takes ``max_features`` (default 1.0, no feature bagging), which the search space does
    not tune.

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
            not take, a value has the wrong type (an integer parameter must be an ``int``, not
            a ``float`` such as ``8.0`` and not a ``bool``; a real parameter must be a number;
            a name parameter must be one of the allowed strings), or the model rejects a value
            of the right type.
    """
    kwargs: dict[str, Any] = dict(params)
    match name:
        case "zscore":
            _check_params(name, params)
            return ZScoreDetector(**kwargs)
        case "iforest":
            _check_params(name, params)
            return IsolationForestDetector(**kwargs, seed=seed, n_jobs=threads)
        case "cnn_ae":
            _check_params(name, params)
            return _build_autoencoder("cnn", kwargs, "width", seed=seed, epochs=epochs)
        case "lstm_ae":
            _check_params(name, params)
            return _build_autoencoder("lstm", kwargs, "hidden", seed=seed, epochs=epochs)
        case _:
            valid = ", ".join(MODEL_NAMES)
            raise ValueError(f"unknown model {name!r}; valid names: {valid}")


def _build_autoencoder(
    kind: Kind, kwargs: dict[str, Any], size_key: str, *, seed: int, epochs: int
) -> AutoencoderDetector:
    """Build an autoencoder from registry parameters, mapping ``size_key`` to ``size``."""
    merged: dict[str, Any] = {**DEFAULT_PARAMS[f"{kind}_ae"], **kwargs}
    size = merged.pop(size_key)
    return AutoencoderDetector(kind=kind, size=size, seed=seed, epochs=epochs, **merged)


def _check_params(name: str, params: Params) -> None:
    """Raise ValueError unless ``params`` holds only keys model ``name`` takes, each well typed."""
    types = _PARAM_TYPES[name]
    unknown = sorted(set(params) - set(types))
    if unknown:
        raise ValueError(
            f"model {name!r} has no parameter {', '.join(map(repr, unknown))}; "
            f"it takes: {', '.join(types)}"
        )
    for key, value in params.items():
        wanted = _wanted(types[key], value)
        if wanted is not None:
            raise ValueError(
                f"model {name!r} parameter {key!r} must be {wanted}, got "
                f"{type(value).__name__} {value!r}"
            )


def _wanted(expected: type | tuple[str, ...], value: object) -> str | None:
    """Return what ``value`` should have been, or None when it is acceptable."""
    if isinstance(expected, tuple):
        if not isinstance(value, str):
            return "a string"
        return None if value in expected else f"one of {expected}"
    if isinstance(value, bool):
        return "an int" if expected is int else "a real number"
    if expected is int:
        return None if isinstance(value, int) else "an int"
    return None if isinstance(value, numbers.Real) else "a real number"
