"""Hyperparameter defaults and search spaces, one entry per registered model.

The defaults are what a run uses when nothing is tuned. The search space is a function of an
Optuna trial, so the sampler and the pruner stay out of this module; it only asks the trial for
values, which also lets a `FixedTrial` replay a stored configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import optuna

DEFAULT_PARAMS: Final[Mapping[str, Mapping[str, int | float | str]]] = MappingProxyType(
    {
        "zscore": MappingProxyType({"aggregation": "topk", "k": 32}),
        "iforest": MappingProxyType({"n_estimators": 200, "max_samples": 1024}),
        "cnn_ae": MappingProxyType({"window": 32, "width": 64, "latent": 8, "lr": 1e-3}),
        "lstm_ae": MappingProxyType({"window": 32, "hidden": 64, "latent": 8, "lr": 1e-3}),
    }
)


def suggest_params(name: str, trial: optuna.trial.BaseTrial) -> dict[str, int | float | str]:
    """Ask ``trial`` for one configuration of the model ``name``.

    Every parameter of the model is suggested on every trial, including ones the chosen
    configuration makes irrelevant (``k`` when ``aggregation`` is not ``"topk"``), so the set of
    suggested names never depends on an earlier suggestion.

    Args:
        name: A model with a search space: any name in `DEFAULT_PARAMS`.
        trial: The Optuna trial to draw from. Only its ``suggest_*`` methods are called.

    Returns:
        A parameter dict that `tsguard.models.build_detector` accepts.

    Raises:
        ValueError: ``name`` has no search space.
    """
    match name:
        case "zscore":
            return {
                "aggregation": trial.suggest_categorical("aggregation", ["mean", "max", "topk"]),
                "k": trial.suggest_int("k", 8, 256, log=True),
            }
        case "iforest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_samples": trial.suggest_int("max_samples", 256, 4096, log=True),
            }
        case "cnn_ae":
            return {
                "window": trial.suggest_categorical("window", [16, 32, 64]),
                "width": trial.suggest_categorical("width", [32, 64, 128]),
                "latent": trial.suggest_categorical("latent", [4, 8, 16]),
                "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            }
        case "lstm_ae":
            return {
                "window": trial.suggest_categorical("window", [16, 32, 64]),
                "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
                "latent": trial.suggest_categorical("latent", [4, 8, 16]),
                "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            }
        case _:
            valid = ", ".join(DEFAULT_PARAMS)
            raise ValueError(f"no search space for model {name!r}; valid names: {valid}")
