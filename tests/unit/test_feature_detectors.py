from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import optuna
import pytest

from tsguard.models import FEATURE_MODELS, MODEL_NAMES, build_detector
from tsguard.models.base import Detector
from tsguard.models.iforest import IsolationForestDetector
from tsguard.models.space import DEFAULT_PARAMS, suggest_params
from tsguard.models.zscore import ZScoreDetector

# -- z-score ----------------------------------------------------------------------------------


def test_zscore_topk_exact():
    d = ZScoreDetector(aggregation="topk", k=2)
    d.fit(np.array([[0.0, 0, 0], [2, 2, 2], [4, 4, 4]]))
    # median 2, MAD 2 -> scale 2.9652; row [8, 2, 5]: z = [2.0235, 0, 1.0117] -> top-2 mean 1.5176
    assert d.score(np.array([[8.0, 2, 5]]), context=0)[0] == pytest.approx(1.5176, abs=1e-4)


def test_zscore_mean_and_max():
    train = np.array([[0.0, 0, 0], [2, 2, 2], [4, 4, 4]])
    row = np.array([[8.0, 2, 5]])
    scale = 1.4826 * 2.0
    mean = ZScoreDetector(aggregation="mean")
    mean.fit(train)
    assert mean.score(row, context=0)[0] == pytest.approx((6 + 0 + 3) / 3 / scale)
    peak = ZScoreDetector(aggregation="max")
    peak.fit(train)
    assert peak.score(row, context=0)[0] == pytest.approx(6 / scale)


def test_zscore_topk_with_k_above_columns_averages_every_column():
    train = np.array([[0.0, 0, 0], [2, 2, 2], [4, 4, 4]])
    row = np.array([[8.0, 2, 5]])
    wide = ZScoreDetector(aggregation="topk", k=100)
    wide.fit(train)
    mean = ZScoreDetector(aggregation="mean")
    mean.fit(train)
    assert wide.score(row, context=0)[0] == pytest.approx(mean.score(row, context=0)[0])


def test_zscore_zero_mad_column_is_finite():
    d = ZScoreDetector()
    X = np.ones((50, 4))
    d.fit(X)
    assert np.isfinite(d.score(X + 1.0, context=0)).all()


def test_zscore_zero_mad_column_scores_its_raw_distance():
    d = ZScoreDetector(aggregation="max")
    d.fit(np.ones((50, 4)))
    # the zero MAD leaves the scale at its floor of 1, so |3 - 1| / 1 = 2
    assert d.score(np.full((1, 4), 3.0), context=0)[0] == pytest.approx(2.0)


def test_zscore_scale_is_floored_at_one_training_standard_deviation():
    # Median 0 and MAD 0.001: the robust scale would be 0.0015, so a shift of 3 would score 2000.
    # Inputs are standardized to unit training standard deviation, so the scale stops at 1.
    d = ZScoreDetector(aggregation="max")
    d.fit(np.tile(np.array([[-0.001], [0.0], [0.001]]), (50, 1)))
    assert d.score(np.array([[3.0]]), context=0)[0] == 3.0


def test_zscore_floor_applies_per_column_and_leaves_wider_columns_alone():
    # Column 0 has MAD 2, scale 2.9652 (above the floor, unchanged); column 1 has MAD 0.001.
    X = np.array([[0.0, -0.001], [2.0, 0.0], [4.0, 0.001]])
    d = ZScoreDetector(aggregation="mean")
    d.fit(np.tile(X, (50, 1)))
    row = np.array([[8.0, 3.0]])
    expected = (6.0 / (1.4826 * 2.0) + 3.0) / 2
    assert d.score(row, context=0)[0] == pytest.approx(expected)


def test_zscore_context_drops_rows(rng):
    X = rng.standard_normal((20, 5))
    d = ZScoreDetector()
    d.fit(X)
    full = d.score(X, context=0)
    tail = d.score(X, context=3)
    assert len(tail) == len(X) - 3
    np.testing.assert_array_equal(tail, full[3:])


def test_zscore_context_must_leave_rows_to_score(rng):
    X = rng.standard_normal((5, 3))
    d = ZScoreDetector()
    d.fit(X)
    with pytest.raises(ValueError, match="zscore"):
        d.score(X, context=5)


def test_zscore_float32_input_gives_float64_scores(rng):
    X = rng.standard_normal((100, 6)).astype(np.float32)
    d = ZScoreDetector()
    d.fit(X)
    scores = d.score(X, context=0)
    assert scores.dtype == np.float64
    assert np.isfinite(scores).all()


def test_zscore_digest_follows_config_and_fit(rng):
    X = rng.standard_normal((100, 6))

    def fitted(data: np.ndarray, **config: Any) -> str:
        d = ZScoreDetector(**config)
        d.fit(data)
        return d.state_digest()

    base = fitted(X)
    assert fitted(X.copy()) == base
    assert fitted(X, aggregation="mean") != base
    assert fitted(X, k=33) != base
    assert fitted(X + 1.0) != base


def test_zscore_rejects_bad_config():
    with pytest.raises(ValueError, match="aggregation"):
        ZScoreDetector(aggregation="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k"):
        ZScoreDetector(k=0)


def test_zscore_refuses_unfitted_mismatched_and_non_finite_input(rng):
    d = ZScoreDetector()
    with pytest.raises(RuntimeError, match="zscore"):
        d.score(np.zeros((3, 2)), context=0)
    X = rng.standard_normal((10, 4))
    bad = X.copy()
    bad[2, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        d.fit(bad)
    d.fit(X)
    with pytest.raises(ValueError, match="columns"):
        d.score(np.zeros((3, 2)), context=0)
    with pytest.raises(ValueError, match="finite"):
        d.score(bad, context=0)


# -- Isolation Forest -------------------------------------------------------------------------


def test_iforest_is_deterministic_per_seed(rng):
    X = rng.standard_normal((400, 8))
    a, b, c = (IsolationForestDetector(n_estimators=50, seed=s) for s in (3, 3, 4))
    for d in (a, b, c):
        d.fit(X)
    np.testing.assert_array_equal(a.score(X, context=0), b.score(X, context=0))
    assert a.state_digest() == b.state_digest()
    assert a.state_digest() != c.state_digest()


def test_iforest_ranks_an_outlier_above_inliers(rng):
    inliers = rng.standard_normal((2000, 20))
    d = IsolationForestDetector(seed=0)
    d.fit(inliers)
    outlier = np.full((1, 20), 10.0)
    assert d.score(outlier, context=0)[0] > np.percentile(d.score(inliers, context=0), 99)


def test_iforest_trees_see_every_column_by_default(rng):
    # Isolation Forest already draws a random feature at each split; bagging columns per tree
    # only adds a column copy per tree, so the default gives every tree all of them.
    X = rng.standard_normal((200, 9))
    d = IsolationForestDetector(n_estimators=10, seed=0, n_jobs=1)
    assert d.max_features == 1.0
    d.fit(X)
    assert d._model is not None
    for subset in d._model.estimators_features_:
        np.testing.assert_array_equal(subset, np.arange(9))


def test_iforest_clips_max_samples_to_rows(rng):
    X = rng.standard_normal((300, 5))
    d = IsolationForestDetector(max_samples=1024, n_estimators=20, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d.fit(X)
        d.score(X, context=0)


def test_iforest_float32_input_gives_float64_scores(rng):
    X = rng.standard_normal((300, 5)).astype(np.float32)
    d = IsolationForestDetector(n_estimators=20, seed=0)
    d.fit(X)
    scores = d.score(X, context=0)
    assert scores.dtype == np.float64
    assert scores.shape == (300,)


def test_iforest_context_drops_rows(rng):
    X = rng.standard_normal((300, 5))
    d = IsolationForestDetector(n_estimators=20, seed=0)
    d.fit(X)
    np.testing.assert_array_equal(d.score(X, context=7), d.score(X, context=0)[7:])
    with pytest.raises(ValueError, match="iforest"):
        d.score(X, context=300)


def test_iforest_digest_ignores_the_thread_count(rng):
    X = rng.standard_normal((400, 8))
    one = IsolationForestDetector(n_estimators=40, seed=5, n_jobs=1)
    four = IsolationForestDetector(n_estimators=40, seed=5, n_jobs=4)
    one.fit(X)
    four.fit(X)
    assert one.state_digest() == four.state_digest()
    four.set_n_jobs(1)
    np.testing.assert_array_equal(one.score(X, context=0), four.score(X, context=0))


def _single_tree(
    X: np.ndarray,
) -> tuple[IsolationForestDetector, tuple[list[int], list[float]], list[int]]:
    d = IsolationForestDetector(n_estimators=1, max_features=1.0, seed=0, n_jobs=1)
    d.fit(X)
    assert d._model is not None
    tree = d._model.estimators_[0].tree_
    return d, (tree.feature.tolist(), tree.threshold.tolist()), tree.n_node_samples.tolist()


def test_iforest_digest_sees_the_leaf_counts(rng):
    # Moving one point across a split leaves every feature and threshold of the tree unchanged
    # but changes how many rows reach two leaves, and a row's path length is corrected by the
    # rows left in its leaf. A digest of splits alone would call the two forests equal.
    X = rng.standard_normal((300, 1))
    base, base_splits, base_counts = _single_tree(X)
    for row in range(len(X)):
        moved = X.copy()
        moved[row] += 0.2
        other, splits, counts = _single_tree(moved)
        if splits == base_splits and counts != base_counts:
            assert other.state_digest() != base.state_digest()
            assert not np.array_equal(other.score(X, context=0), base.score(X, context=0))
            return
    pytest.fail("no single-row move kept the splits and changed the leaf counts")


def test_iforest_refuses_unfitted_mismatched_and_bad_config(rng):
    d = IsolationForestDetector(n_estimators=10, seed=0)
    with pytest.raises(RuntimeError, match="iforest"):
        d.score(np.zeros((3, 2)), context=0)
    with pytest.raises(RuntimeError, match="iforest"):
        d.state_digest()
    d.fit(rng.standard_normal((100, 4)))
    with pytest.raises(ValueError, match="columns"):
        d.score(np.zeros((3, 2)), context=0)
    with pytest.raises(ValueError, match="max_features"):
        IsolationForestDetector(max_features=0.0, seed=0)
    with pytest.raises(ValueError, match="n_jobs"):
        d.set_n_jobs(0)


# -- registry and search spaces ---------------------------------------------------------------


def test_build_detector_and_unknown_model():
    assert build_detector("zscore", DEFAULT_PARAMS["zscore"], seed=0).name == "zscore"
    with pytest.raises(ValueError, match="iforest"):
        build_detector("nope", {}, seed=0)


def test_unknown_param_key_is_value_error():
    with pytest.raises(ValueError, match="'q'"):
        build_detector("zscore", {"q": 1}, seed=0)
    with pytest.raises(ValueError, match="'q'"):
        build_detector("iforest", {"q": 1}, seed=0)


def test_every_registered_model_builds_a_detector_from_its_defaults():
    assert MODEL_NAMES == ("zscore", "iforest", "cnn_ae", "lstm_ae")
    assert FEATURE_MODELS == ("zscore", "iforest")
    for name in MODEL_NAMES:
        detector = build_detector(name, DEFAULT_PARAMS[name], seed=0)
        assert isinstance(detector, Detector)
        assert detector.name == name
    for name in FEATURE_MODELS:
        detector = build_detector(name, DEFAULT_PARAMS[name], seed=0)
        assert detector.input_kind == "features"
        assert detector.lookback == 1


def test_build_detector_passes_params_seed_and_threads_to_iforest():
    d = build_detector(
        "iforest", {"n_estimators": 60, "max_features": 0.3}, seed=11, threads=2, epochs=1
    )
    assert isinstance(d, IsolationForestDetector)
    assert (d.n_estimators, d.max_samples, d.max_features) == (60, 1024, 0.3)
    assert (d.seed, d.n_jobs) == (11, 2)


def test_build_detector_defaults_iforest_to_every_column_and_still_takes_max_features():
    assert dict(DEFAULT_PARAMS["iforest"]) == {"n_estimators": 200, "max_samples": 1024}
    default = build_detector("iforest", DEFAULT_PARAMS["iforest"], seed=0)
    assert isinstance(default, IsolationForestDetector)
    assert (default.n_estimators, default.max_samples, default.max_features) == (200, 1024, 1.0)
    bagged = build_detector("iforest", {"max_features": 0.25}, seed=0)
    assert isinstance(bagged, IsolationForestDetector)
    assert bagged.max_features == 0.25


def test_build_detector_passes_params_to_zscore():
    d = build_detector("zscore", {"aggregation": "max", "k": 9}, seed=0)
    assert isinstance(d, ZScoreDetector)
    assert (d.aggregation, d.k) == ("max", 9)


def test_suggest_params_covers_defaults_keys():
    for name in FEATURE_MODELS:
        trial = optuna.trial.FixedTrial(dict(DEFAULT_PARAMS[name]))
        assert suggest_params(name, trial) == dict(DEFAULT_PARAMS[name])


def test_iforest_search_space_does_not_tune_feature_bagging():
    params = suggest_params(
        "iforest", optuna.trial.FixedTrial({"n_estimators": 150, "max_samples": 512})
    )
    assert params == {"n_estimators": 150, "max_samples": 512}


def test_suggest_params_stays_in_range_and_always_suggests_every_key():
    seen: dict[str, list[dict[str, object]]] = {name: [] for name in FEATURE_MODELS}

    def objective(trial: optuna.Trial) -> float:
        name = trial.suggest_categorical("model", list(FEATURE_MODELS))
        params = suggest_params(name, trial)
        build_detector(name, params, seed=0)
        seen[name].append(dict(params))
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(objective, n_trials=40)
    for name in FEATURE_MODELS:
        assert seen[name], f"the sampler never drew {name}"
        for params in seen[name]:
            assert set(params) == set(DEFAULT_PARAMS[name])
    for z in seen["zscore"]:
        assert z["aggregation"] in ("mean", "max", "topk")
        assert isinstance(z["k"], int)
        assert 8 <= z["k"] <= 256
    for f in seen["iforest"]:
        assert isinstance(f["n_estimators"], int)
        assert f["n_estimators"] in range(100, 401, 50)
        assert isinstance(f["max_samples"], int)
        assert 256 <= f["max_samples"] <= 4096


def test_suggest_params_unknown_model_names_the_valid_ones():
    with pytest.raises(ValueError, match="zscore"):
        suggest_params("nope", optuna.trial.FixedTrial({}))


def test_default_params_are_read_only():
    with pytest.raises(TypeError):
        DEFAULT_PARAMS["zscore"]["k"] = 1  # type: ignore[index]
