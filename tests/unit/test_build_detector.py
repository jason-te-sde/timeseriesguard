from __future__ import annotations

import numpy as np
import pytest

import tsguard.models
from tsguard.errors import ConfigurationError
from tsguard.models import MODEL_NAMES, build_detector
from tsguard.models.autoencoders import AutoencoderDetector
from tsguard.models.iforest import IsolationForestDetector
from tsguard.models.space import DEFAULT_PARAMS
from tsguard.models.zscore import ZScoreDetector


@pytest.mark.parametrize(
    ("model", "key", "bad", "expected"),
    [
        pytest.param("zscore", "k", 8.0, "an int, got float 8.0", id="zscore-k-float"),
        pytest.param("zscore", "k", True, "an int, got bool True", id="zscore-k-bool"),
        pytest.param("zscore", "k", "8", "an int, got str '8'", id="zscore-k-str"),
        pytest.param("zscore", "k", None, "an int, got NoneType None", id="zscore-k-none"),
        pytest.param("zscore", "aggregation", "median", "one of", id="zscore-aggregation-name"),
        pytest.param(
            "zscore", "aggregation", 3, "a string, got int 3", id="zscore-aggregation-int"
        ),
        pytest.param("iforest", "n_estimators", 100.0, "an int, got float", id="iforest-trees"),
        pytest.param("iforest", "max_samples", True, "an int, got bool", id="iforest-samples"),
        pytest.param("iforest", "max_features", "1.0", "a real number, got str", id="iforest-feat"),
        pytest.param(
            "iforest", "max_features", False, "a real number, got bool", id="iforest-bool"
        ),
        pytest.param("cnn_ae", "window", 32.0, "an int, got float", id="cnn-window"),
        pytest.param("cnn_ae", "width", "64", "an int, got str", id="cnn-width"),
        pytest.param("cnn_ae", "latent", 8.5, "an int, got float", id="cnn-latent"),
        pytest.param("cnn_ae", "lr", "1e-3", "a real number, got str", id="cnn-lr"),
        pytest.param("lstm_ae", "hidden", 64.0, "an int, got float", id="lstm-hidden"),
        pytest.param("lstm_ae", "window", True, "an int, got bool", id="lstm-window"),
        pytest.param("lstm_ae", "lr", None, "a real number, got NoneType", id="lstm-lr"),
    ],
)
def test_a_parameter_of_the_wrong_type_is_refused_at_build_time(model, key, bad, expected):
    params = {**DEFAULT_PARAMS[model], key: bad}
    with pytest.raises(ConfigurationError, match=rf"model '{model}' parameter '{key}'.*{expected}"):
        build_detector(model, params, seed=0)


def test_a_zscore_k_of_eight_point_zero_fails_in_build_not_in_score():
    with pytest.raises(ConfigurationError, match=r"model 'zscore' parameter 'k' must be an int"):
        build_detector("zscore", {"aggregation": "topk", "k": 8.0}, seed=0)


def test_an_unknown_aggregation_lists_the_allowed_names():
    with pytest.raises(
        ConfigurationError, match=r"'aggregation' must be one of .*'mean'.*'max'.*'topk'"
    ):
        build_detector("zscore", {"aggregation": "median"}, seed=0)


def test_an_unknown_model_is_a_configuration_error_that_lists_the_valid_names():
    with pytest.raises(ConfigurationError, match=r"unknown model 'nope'.*zscore.*lstm_ae"):
        build_detector("nope", {}, seed=0)


def test_an_unknown_key_is_still_reported_before_a_type_problem():
    with pytest.raises(ConfigurationError, match="has no parameter 'q'"):
        build_detector("zscore", {"q": 1, "k": 8.0}, seed=0)


def test_an_int_is_accepted_where_a_real_number_is_wanted():
    ae = build_detector("cnn_ae", {"lr": 1}, seed=0)
    assert isinstance(ae, AutoencoderDetector)
    forest = build_detector("iforest", {"max_features": 1}, seed=0)
    assert isinstance(forest, IsolationForestDetector)


def test_numpy_floats_are_real_numbers():
    forest = build_detector("iforest", {"max_features": np.float64(0.5)}, seed=0)
    assert isinstance(forest, IsolationForestDetector)
    assert forest.max_features == 0.5


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_the_defaults_pass_the_type_check(model):
    build_detector(model, DEFAULT_PARAMS[model], seed=0)


def test_valid_zscore_parameters_still_reach_the_detector():
    d = build_detector("zscore", {"aggregation": "max", "k": 9}, seed=0)
    assert isinstance(d, ZScoreDetector)
    assert (d.aggregation, d.k) == ("max", 9)


def test_the_models_docstring_mentions_every_registered_model():
    doc = tsguard.models.__doc__ or ""
    for name in MODEL_NAMES:
        assert name in doc
