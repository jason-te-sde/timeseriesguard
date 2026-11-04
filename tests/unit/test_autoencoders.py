from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import optuna
import pytest
import torch

from tests.conftest import make_entity
from tsguard.models import AE_MODELS, MODEL_NAMES, build_detector
from tsguard.models.autoencoders import AutoencoderDetector, CnnAutoencoder, LstmAutoencoder
from tsguard.models.base import Detector
from tsguard.models.space import DEFAULT_PARAMS, suggest_params
from tsguard.repro.seeding import configure_runtime

WINDOW = 16
KINDS = ["cnn", "lstm"]


@pytest.fixture(scope="module", autouse=True)
def single_thread_runtime() -> Iterator[None]:
    """Fix torch to one thread and deterministic ops, then restore what the process had."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    configure_runtime(threads=1)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


@pytest.fixture(scope="module")
def entity():
    return make_entity(n_train=1200, n_test=400, anomalies=((100, 20), (300, 5)))


def small(kind: str, **overrides: Any) -> AutoencoderDetector:
    config: dict[str, Any] = {
        "kind": kind,
        "window": WINDOW,
        "latent": 4,
        "size": 16,
        "lr": 3e-3,
        "seed": 5,
        "epochs": 2,
        "batch_size": 64,
    }
    config.update(overrides)
    return AutoencoderDetector(**config)


@pytest.mark.parametrize("kind", KINDS)
def test_forward_shape(kind):
    model = CnnAutoencoder(38, 16, 4) if kind == "cnn" else LstmAutoencoder(38, 16, 4)
    x = torch.zeros(8, WINDOW, 38)
    assert model(x).shape == (8, WINDOW, 38)


@pytest.mark.parametrize("kind", KINDS)
def test_training_is_deterministic(kind, entity):
    first, second = small(kind), small(kind)
    first.fit(entity.train)
    second.fit(entity.train)
    assert first.state_digest() == second.state_digest()
    a = first.score(entity.test, context=WINDOW - 1)
    b = second.score(entity.test, context=WINDOW - 1)
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("kind", KINDS)
def test_seed_changes_weights(kind, entity):
    first, second = small(kind, seed=5), small(kind, seed=6)
    first.fit(entity.train)
    second.fit(entity.train)
    assert first.state_digest() != second.state_digest()


@pytest.mark.parametrize("kind", KINDS)
def test_seed_sets_the_initial_weights(kind, entity):
    # One window is one batch whatever the seed, so only the initialization can differ.
    one_window = entity.train[:WINDOW]
    first, second = small(kind, seed=5, epochs=1), small(kind, seed=6, epochs=1)
    first.fit(one_window)
    second.fit(one_window)
    assert first.state_digest() != second.state_digest()


@pytest.mark.parametrize("kind", KINDS)
def test_a_batch_shorter_than_batch_size_still_trains(kind, entity):
    one_window = entity.train[:WINDOW]
    once, twice = small(kind, epochs=1), small(kind, epochs=2)
    once.fit(one_window)
    twice.fit(one_window)
    assert once.state_digest() != twice.state_digest()


def test_training_actually_fits_the_data(entity):
    # Without an optimizer step the weights would stay at their initial values, which the
    # determinism tests cannot tell apart from a trained model.
    untrained, trained = small("cnn", epochs=1), small("cnn", epochs=4)
    untrained.fit(entity.train[: WINDOW + 1])  # a single window, so one step
    trained.fit(entity.train)
    window_rows = entity.test[:200]
    context = WINDOW - 1
    assert (
        trained.score(window_rows, context=context).mean()
        < untrained.score(window_rows, context=context).mean()
    )


@pytest.mark.parametrize("kind", KINDS)
def test_score_length_and_context(kind, entity):
    detector = small(kind)
    detector.fit(entity.train)
    assert len(detector.score(entity.test, context=15)) == len(entity.test) - 15
    assert detector.score(entity.test, context=15).dtype == np.float64
    with pytest.raises(ValueError, match=f"{kind}_ae"):
        detector.score(entity.test, context=14)
    with pytest.raises(ValueError, match=f"{kind}_ae"):
        detector.score(entity.test, context=len(entity.test))


@pytest.mark.parametrize("kind", KINDS)
def test_scores_are_the_last_step_error_of_each_window_across_batches(kind, rng):
    detector = small(kind)
    detector.fit(rng.random((300, 6)).astype(np.float32))
    X = rng.random((2200, 6)).astype(np.float32)  # more rows than one batch of 2,048
    model = detector._model
    assert model is not None
    for context in (WINDOW - 1, 40):
        starts = np.arange(context, len(X)) - (WINDOW - 1)
        windows = torch.from_numpy(np.stack([X[s : s + WINDOW] for s in starts]))
        with torch.no_grad():
            error = (model(windows)[:, -1, :] - windows[:, -1, :]) ** 2
        expected = error.mean(dim=1).numpy().astype(np.float64)
        np.testing.assert_allclose(
            detector.score(X, context=context), expected, rtol=1e-4, atol=1e-6
        )


def test_global_torch_rng_is_untouched(entity):
    detector = small("lstm")
    before = torch.get_rng_state()
    detector.fit(entity.train)
    assert torch.equal(before, torch.get_rng_state())


def test_injected_spike_scores_above_median(entity):
    for kind in KINDS:
        detector = small(kind, epochs=4, batch_size=32)
        detector.fit(entity.train)
        clean = np.asarray(entity.test, dtype=np.float32)
        spiked = clean.copy()
        row = 200
        spiked[row, :8] += 0.5
        context = WINDOW - 1
        baseline = detector.score(clean, context=context)
        assert detector.score(spiked, context=context)[row - context] > np.median(baseline)


def test_fit_and_score_refuse_bad_input(entity):
    detector = small("cnn")
    with pytest.raises(RuntimeError, match="cnn_ae"):
        detector.score(entity.test, context=WINDOW - 1)
    with pytest.raises(RuntimeError, match="cnn_ae"):
        detector.state_digest()
    with pytest.raises(ValueError, match="window"):
        detector.fit(entity.train[: WINDOW - 1])
    bad = np.array(entity.train)
    bad[3, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        detector.fit(bad)
    with pytest.raises(ValueError, match="2-D"):
        detector.fit(entity.train[:, 0])
    detector.fit(entity.train)
    with pytest.raises(ValueError, match="columns"):
        detector.score(entity.test[:, :10], context=WINDOW - 1)
    with pytest.raises(ValueError, match="finite"):
        detector.score(bad, context=WINDOW - 1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kind": "gru"}, "kind"),
        ({"window": 0}, "window"),
        ({"latent": 0}, "latent"),
        ({"size": 0}, "size"),
        ({"lr": 0.0}, "lr"),
        ({"epochs": 0}, "epochs"),
        ({"stride": 0}, "stride"),
        ({"batch_size": 0}, "batch_size"),
    ],
)
def test_rejects_bad_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        small(**{"kind": "cnn", **overrides})


def test_registry_builds_both():
    assert AE_MODELS == ("cnn_ae", "lstm_ae")
    assert MODEL_NAMES == ("zscore", "iforest", "cnn_ae", "lstm_ae")
    for name, size_key in (("cnn_ae", "width"), ("lstm_ae", "hidden")):
        detector = build_detector(name, DEFAULT_PARAMS[name], seed=0, epochs=1)
        assert isinstance(detector, Detector)
        assert isinstance(detector, AutoencoderDetector)
        assert (detector.name, detector.input_kind, detector.lookback) == (name, "raw", 32)
        assert (detector.window, detector.latent, detector.lr) == (32, 8, 1e-3)
        assert detector.size == DEFAULT_PARAMS[name][size_key] == 64
        assert (detector.epochs, detector.stride, detector.seed) == (1, 4, 0)


def test_registry_maps_params_and_rejects_the_other_models_keys():
    d = build_detector("lstm_ae", {"hidden": 32, "window": 16}, seed=3, epochs=2)
    assert isinstance(d, AutoencoderDetector)
    assert (d.kind, d.size, d.window, d.latent, d.seed, d.epochs) == ("lstm", 32, 16, 8, 3, 2)
    with pytest.raises(ValueError, match="'width'"):
        build_detector("lstm_ae", {"width": 32}, seed=0)
    with pytest.raises(ValueError, match="'hidden'"):
        build_detector("cnn_ae", {"hidden": 32}, seed=0)


def test_suggest_params_round_trips_defaults_for_aes():
    for name in AE_MODELS:
        trial = optuna.trial.FixedTrial(dict(DEFAULT_PARAMS[name]))
        assert suggest_params(name, trial) == dict(DEFAULT_PARAMS[name])


def test_suggest_params_for_aes_stays_in_the_space():
    size_key = {"cnn_ae": "width", "lstm_ae": "hidden"}
    seen: list[tuple[str, dict[str, Any]]] = []

    def objective(trial: optuna.Trial) -> float:
        name = trial.suggest_categorical("model", list(AE_MODELS))
        params = suggest_params(name, trial)
        build_detector(name, params, seed=0)
        seen.append((name, dict(params)))
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study.optimize(objective, n_trials=30)
    assert {name for name, _ in seen} == set(AE_MODELS)
    for name, params in seen:
        assert set(params) == {"window", size_key[name], "latent", "lr"}
        assert params["window"] in (16, 32, 64)
        assert params[size_key[name]] in (32, 64, 128)
        assert params["latent"] in (4, 8, 16)
        assert 1e-4 <= params["lr"] <= 3e-3
