from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.conftest import make_entity
from tests.unit.test_report_and_claims import tiny_metrics
from tsguard.data.labels import LabelAudit
from tsguard.data.smd import SMD_COMMIT, EntityData
from tsguard.data.splits import walk_forward_folds
from tsguard.errors import ChecksumError, ContractError, LeakageError
from tsguard.experiment.runner import EntityRunner, RunConfig, Selection
from tsguard.features.pipeline import FeaturePipeline
from tsguard.models.iforest import IsolationForestDetector
from tsguard.models.space import DEFAULT_PARAMS
from tsguard.repro.report import write_metrics
from tsguard.repro.seeding import configure_runtime
from tsguard.serve import commands
from tsguard.serve.app import create_app
from tsguard.serve.bundle import (
    Bundle,
    HistoryTooShortError,
    build_bundle,
    bundle_from_selection,
    load_bundle,
    save_bundle,
    score_rows,
)
from tsguard.serve.schemas import ScoreRequest, ScoreResponse

N_TRAIN = 3000
N_TEST = 2000
N_CHANNELS = 38
CONFIG = RunConfig(ae_epochs=1, threads=1, causal_check=False)
IFOREST: dict[str, int | float | str] = {**DEFAULT_PARAMS["iforest"], "n_estimators": 100}
CNN_AE: dict[str, int | float | str] = {"window": 16, "width": 16, "latent": 4, "lr": 1e-3}
PARAMS: dict[str, dict[str, int | float | str]] = {"iforest": IFOREST, "cnn_ae": CNN_AE}
REQUIRED_ROWS = {"iforest": 64, "cnn_ae": 16}
LAST_FIT_END = walk_forward_folds(N_TRAIN, k=3, initial_frac=0.5, purge=64)[-1].fit_end
MODELS = ["iforest", "cnn_ae"]


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
def entity() -> EntityData:
    return make_entity(n_train=N_TRAIN, n_test=N_TEST)


@dataclass
class Served:
    model: str
    selection: Selection
    bundle: Bundle
    path: Path
    digest: str
    client: TestClient


@pytest.fixture(scope="module")
def served(entity: EntityData, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Served]:
    """Fit each model once, save its bundle, and put a client in front of the saved file."""
    out: dict[str, Served] = {}
    for model in MODELS:
        runner = EntityRunner(entity, entity.labels.audit, CONFIG)
        selection = runner.select(model, PARAMS[model], keep_models=True)
        bundle = bundle_from_selection(selection)
        path = tmp_path_factory.mktemp(model) / "bundle.joblib"
        digest = save_bundle(bundle, path)
        out[model] = Served(model, selection, bundle, path, digest, TestClient(create_app(path)))
    return out


@pytest.fixture(scope="module")
def iforest(served: dict[str, Served]) -> Served:
    return served["iforest"]


def _request(e: EntityData, r: int, n: int) -> dict[str, Any]:
    """Build a score request for the ``n`` rows ending at absolute row ``r``."""
    return {
        "entity": e.name,
        "first_index": r - n + 1,
        "history": e.full[r - n + 1 : r + 1].tolist(),
    }


def test_score_ok(iforest, entity):
    r = N_TRAIN + 100
    response = iforest.client.post("/v1/score", json=_request(entity, r, 64))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["entity"] == entity.name
    assert body["model"] == "iforest"
    assert body["row_index"] == r
    assert body["threshold"] == iforest.bundle.threshold
    assert body["is_anomaly"] == (body["score"] >= body["threshold"])
    sidecar = Path(str(iforest.path) + ".sha256").read_text()
    assert body["bundle_sha256"] == sidecar.strip()
    assert body["bundle_sha256"] == iforest.digest


def test_row_index_follows_first_index_and_history_length(iforest, entity):
    r = N_TRAIN + 300
    body = iforest.client.post("/v1/score", json=_request(entity, r, 100)).json()
    assert body["row_index"] == r


def test_is_anomaly_is_true_at_the_threshold(iforest, entity):
    """The comparison is `score >= threshold`: a score equal to the threshold is an anomaly."""
    r = N_TRAIN + 100
    score = iforest.client.post("/v1/score", json=_request(entity, r, 64)).json()["score"]
    at_score = dataclasses.replace(iforest.bundle, threshold=score)
    app = _app_for(at_score, iforest.path.with_name("at_threshold.joblib"))
    body = TestClient(app).post("/v1/score", json=_request(entity, r, 64)).json()
    assert body["score"] == score
    assert body["is_anomaly"] is True


def _app_for(bundle: Bundle, path: Path) -> FastAPI:
    save_bundle(bundle, path)
    return create_app(path)


@pytest.mark.parametrize("which", MODELS)
def test_service_score_matches_batch_score(which, served, entity, rng):
    s = served[which]
    n = REQUIRED_ROWS[which]
    # The first test row reaches back into the train rows for its history; the last is the
    # end of the series; the rest are drawn.
    drawn = rng.choice(np.arange(1, N_TEST - 1), size=18, replace=False)
    rows = [0, *sorted(int(i) for i in drawn), N_TEST - 1]
    assert len(rows) == 20
    got = [
        s.client.post("/v1/score", json=_request(entity, N_TRAIN + i, n)).json()["score"]
        for i in rows
    ]
    expected = s.selection.test_scores[rows]
    np.testing.assert_allclose(got, expected, rtol=1e-6)


@pytest.mark.parametrize("which", MODELS)
def test_extra_history_changes_no_score(which, served, entity):
    s = served[which]
    r = N_TRAIN + 700
    short = s.client.post("/v1/score", json=_request(entity, r, REQUIRED_ROWS[which])).json()
    long = s.client.post("/v1/score", json=_request(entity, r, 500)).json()
    assert long["score"] == pytest.approx(short["score"], rel=1e-6)


@pytest.mark.parametrize("which", MODELS)
def test_history_one_row_short_is_422_and_exact_length_is_accepted(which, served, entity):
    s = served[which]
    need = REQUIRED_ROWS[which]
    r = N_TRAIN + 50
    short = s.client.post("/v1/score", json=_request(entity, r, need - 1))
    assert short.status_code == 422
    detail = short.json()["detail"]
    assert detail["required_rows"] == need
    assert detail["received_rows"] == need - 1
    assert str(need) in detail["message"]
    assert s.client.post("/v1/score", json=_request(entity, r, need)).status_code == 200


def test_short_history_is_422_with_required_rows(iforest, entity):
    response = iforest.client.post("/v1/score", json=_request(entity, N_TRAIN + 50, 10))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["required_rows"] == 64
    assert detail["received_rows"] == 10


def test_wrong_width_is_422(iforest, entity):
    body = _request(entity, N_TRAIN + 50, 64)
    body["history"][3] = body["history"][3][:37]
    response = iforest.client.post("/v1/score", json=body)
    assert response.status_code == 422
    assert "history[3] has 37 values; expected 38" in response.text


def test_schema_error_names_the_problem_and_does_not_echo_the_rows(iforest, entity):
    body = _request(entity, N_TRAIN + 50, 64)
    body["history"][3] = body["history"][3][:37]
    response = iforest.client.post("/v1/score", json=body)
    detail = response.json()["detail"]
    assert [set(item) for item in detail] == [{"type", "loc", "msg"}]
    assert len(response.text) < 500, f"422 body has {len(response.text)} bytes: {response.text}"


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_value_is_422(token, iforest, entity):
    # Python's json module accepts these tokens, so the schema is the only line of defence.
    rows = _request(entity, N_TRAIN + 50, 64)["history"]
    rows[10][4] = 0.5
    text = json.dumps({"entity": entity.name, "first_index": N_TRAIN - 13, "history": rows})
    text = text.replace("0.5,", f"{token},", 1)
    assert token in text
    response = iforest.client.post(
        "/v1/score", content=text, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("value", [1e300, -1e300, 1.0000001e6, -1.0000001e6])
def test_value_beyond_any_plausible_channel_is_422(value, iforest, entity):
    # The channels are scaled to [0, 1]; a magnitude this large overflows the fourth-power
    # features and used to surface as a 500.
    body = _request(entity, LAST_FIT_END, 64)
    body["history"][5][7] = value
    response = iforest.client.post("/v1/score", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"] == ["body", "history", 5, 7]


@pytest.mark.parametrize("which", MODELS)
def test_largest_accepted_magnitude_scores_without_error(which, served, entity):
    s = served[which]
    need = REQUIRED_ROWS[which]
    swing = [[1e6 if i % 2 else -1e6] * 38 for i in range(need)]
    body = {"entity": entity.name, "first_index": LAST_FIT_END, "history": swing}
    response = s.client.post("/v1/score", json=body)
    assert response.status_code == 200, response.text
    assert np.isfinite(response.json()["score"])


def test_row_inside_training_range_is_422(iforest, entity):
    assert iforest.bundle.fit_end == LAST_FIT_END
    r = LAST_FIT_END - 1
    response = iforest.client.post("/v1/score", json=_request(entity, r, 64))
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["fit_end"] == LAST_FIT_END
    assert str(LAST_FIT_END) in detail["message"]


def test_first_row_after_the_fit_range_is_accepted(iforest, entity):
    """The refusal stops exactly at ``fit_end``: row ``fit_end`` itself is scoreable."""
    response = iforest.client.post("/v1/score", json=_request(entity, LAST_FIT_END, 64))
    assert response.status_code == 200, response.text
    assert response.json()["row_index"] == LAST_FIT_END


def test_window_of_an_autoencoder_may_reach_into_the_fit_range(served, entity):
    s = served["cnn_ae"]
    response = s.client.post("/v1/score", json=_request(entity, LAST_FIT_END, 16))
    assert response.status_code == 200, response.text


def test_unknown_entity_is_404(iforest, entity):
    body = _request(entity, N_TRAIN + 50, 64)
    body["entity"] = "machine-9-9"
    response = iforest.client.post("/v1/score", json=body)
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["served_entity"] == entity.name
    assert "machine-9-9" in detail["message"]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda b: b.update(first_index=-1), id="negative_first_index"),
        pytest.param(lambda b: b.update(history=[]), id="empty_history"),
        pytest.param(lambda b: b.update(extra=1), id="unknown_field"),
        pytest.param(lambda b: b.pop("entity"), id="missing_entity"),
        pytest.param(lambda b: b.update(first_index="x"), id="non_integer_first_index"),
    ],
)
def test_schema_violations_are_422(mutate, iforest, entity):
    body = _request(entity, N_TRAIN + 50, 64)
    mutate(body)
    assert iforest.client.post("/v1/score", json=body).status_code == 422


@pytest.mark.parametrize("which", MODELS)
def test_model_info(which, served):
    s = served[which]
    response = s.client.get("/v1/model")
    assert response.status_code == 200
    info = response.json()
    assert info["entity"] == "synthetic-0"
    assert info["model"] == which
    assert info["params"] == PARAMS[which]
    assert info["threshold"] == s.selection.threshold.value
    assert info["required_rows"] == REQUIRED_ROWS[which]
    assert info["fit_end"] == LAST_FIT_END
    assert info["feature_columns"] == len(s.bundle.pipeline.columns)
    assert info["bundle_sha256"] == s.digest
    assert info["source_commit"] == SMD_COMMIT


def test_feature_columns_of_the_two_input_kinds(served):
    assert served["iforest"].client.get("/v1/model").json()["feature_columns"] == 59 * 38 + 4
    assert served["cnn_ae"].client.get("/v1/model").json()["feature_columns"] == 38


def test_healthz(iforest):
    response = iforest.client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_tampered_bundle_is_refused(iforest, tmp_path):
    path = tmp_path / "bundle.joblib"
    data = bytearray(iforest.path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    Path(str(path) + ".sha256").write_text(iforest.digest + "\n")
    with pytest.raises(ChecksumError, match=r"bundle\.joblib"):
        load_bundle(path)


def test_create_app_refuses_a_tampered_bundle(iforest, tmp_path):
    path = tmp_path / "bundle.joblib"
    path.write_bytes(iforest.path.read_bytes() + b"\x00")
    Path(str(path) + ".sha256").write_text(iforest.digest + "\n")
    with pytest.raises(ChecksumError):
        create_app(path)


def test_missing_sidecar_is_refused(iforest, tmp_path):
    path = tmp_path / "bundle.joblib"
    path.write_bytes(iforest.path.read_bytes())
    with pytest.raises(ChecksumError, match=r"\.sha256"):
        load_bundle(path)


@pytest.mark.parametrize("text", ["", "not a digest\n", "ab" * 31 + "\n"])
def test_malformed_sidecar_is_refused(text, iforest, tmp_path):
    path = tmp_path / "bundle.joblib"
    path.write_bytes(iforest.path.read_bytes())
    Path(str(path) + ".sha256").write_text(text)
    with pytest.raises(ChecksumError, match=r"\.sha256"):
        load_bundle(path)


class _Marker:
    """A pickle payload that leaves a file behind when it is unpickled."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[Any, ...]:
        return (Path.touch, (self.marker,))


def test_checksum_is_verified_before_unpickling(tmp_path):
    marker = tmp_path / "unpickled"
    path = tmp_path / "evil.joblib"
    sidecar = Path(str(path) + ".sha256")
    import joblib

    joblib.dump(_Marker(marker), path)
    sidecar.write_text("0" * 64 + "\n")
    with pytest.raises(ChecksumError):
        load_bundle(path)
    assert not marker.exists(), "load_bundle unpickled a file whose checksum did not match"

    # The same file with a matching sidecar does get unpickled, so the check above can fail.
    sidecar.write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    with pytest.raises(ContractError):
        load_bundle(path)
    assert marker.exists()


def test_save_writes_the_digest_of_the_file_and_a_newline(iforest):
    sidecar = Path(str(iforest.path) + ".sha256")
    expected = hashlib.sha256(iforest.path.read_bytes()).hexdigest()
    assert iforest.digest == expected
    assert sidecar.read_text() == expected + "\n"


def test_save_creates_the_parent_directory(iforest, tmp_path):
    path = tmp_path / "nested" / "dir" / "bundle.joblib"
    digest = save_bundle(iforest.bundle, path)
    assert load_bundle(path)[1] == digest


@pytest.mark.parametrize("which", MODELS)
def test_loaded_bundle_reproduces_the_fitted_state_and_the_batch_scores(which, served, entity):
    s = served[which]
    loaded, digest = load_bundle(s.path)
    assert digest == s.digest
    assert (loaded.entity, loaded.model, loaded.params) == ("synthetic-0", which, PARAMS[which])
    assert loaded.threshold == s.selection.threshold.value
    assert loaded.source_commit == SMD_COMMIT
    assert loaded.pipeline.fit_end == LAST_FIT_END
    assert loaded.pipeline.state_digest() == s.selection.digests["pipeline"]
    assert loaded.detector.state_digest() == s.selection.digests["detector"]
    context = loaded.detector.lookback - 1
    rows = loaded.pipeline.transform(entity.full, start_index=N_TRAIN, context=context)
    scores = loaded.detector.score(rows, context=context)
    np.testing.assert_array_equal(scores, s.selection.test_scores)


def test_bundle_pins_one_thread_for_the_isolation_forest(iforest):
    assert isinstance(iforest.bundle.detector, IsolationForestDetector)
    assert iforest.bundle.detector.n_jobs == 1
    loaded, _ = load_bundle(iforest.path)
    assert isinstance(loaded.detector, IsolationForestDetector)
    assert loaded.detector.n_jobs == 1


def test_required_rows_and_fit_end(served):
    for which, s in served.items():
        assert s.bundle.required_rows == REQUIRED_ROWS[which]
        assert s.bundle.fit_end == LAST_FIT_END


def test_build_bundle_is_the_selection_without_reading_labels():
    audit = LabelAudit()
    data = make_entity(n_train=N_TRAIN, n_test=N_TEST, audit=audit)
    # Four fit threads, so the one-thread pin below is the bundle's doing and not the config's.
    # The digests are thread-independent, so the one-thread selection is the reference.
    threaded = RunConfig(ae_epochs=1, threads=4, causal_check=False)
    probe = EntityRunner(data, audit, threaded).select("iforest", IFOREST, keep_models=True)
    assert isinstance(probe.detector, IsolationForestDetector)
    assert probe.detector.n_jobs == 4
    bundle = build_bundle(data, audit, "iforest", IFOREST, config=threaded)
    selection = EntityRunner(data, audit, CONFIG).select("iforest", IFOREST)
    assert audit.reads == 0
    assert bundle.pipeline.state_digest() == selection.digests["pipeline"]
    assert bundle.detector.state_digest() == selection.digests["detector"]
    assert bundle.threshold == selection.threshold.value
    assert isinstance(bundle.detector, IsolationForestDetector)
    assert bundle.detector.n_jobs == 1
    assert (bundle.entity, bundle.model, bundle.params) == (data.name, "iforest", IFOREST)


def test_bundle_from_selection_needs_the_kept_models(iforest):
    bare = dataclasses.replace(iforest.selection, pipeline=None, detector=None)
    with pytest.raises(ValueError, match="keep_models"):
        bundle_from_selection(bare)


def test_bundle_refuses_an_unfitted_pipeline(iforest):
    with pytest.raises(ValueError, match="not fitted"):
        dataclasses.replace(iforest.bundle, pipeline=FeaturePipeline())


def test_score_rows_names_the_rows_it_needs(iforest, entity):
    r = N_TRAIN + 50
    history = entity.full[r - 9 : r + 1]
    with pytest.raises(HistoryTooShortError, match="needs 64 rows of history, got 10") as caught:
        score_rows(iforest.bundle, history, first_index=r - 9)
    assert (caught.value.required_rows, caught.value.received_rows) == (64, 10)
    assert isinstance(caught.value, ValueError)


def test_score_rows_refuses_a_row_inside_the_fit_range(iforest, entity):
    r = LAST_FIT_END - 1
    history = entity.full[r - 63 : r + 1]
    with pytest.raises(LeakageError, match=str(LAST_FIT_END)):
        score_rows(iforest.bundle, history, first_index=r - 63)


@pytest.mark.parametrize("which", MODELS)
def test_concurrent_requests_score_like_sequential_ones(which, served, entity):
    """The service handles requests on a thread pool; the shared model must not mind."""
    s = served[which]
    need = REQUIRED_ROWS[which]
    full = entity.full
    rows = list(range(N_TRAIN, N_TRAIN + 160, 4))

    def score(r: int) -> float:
        return score_rows(s.bundle, full[r - need + 1 : r + 1], first_index=r - need + 1)

    sequential = [score(r) for r in rows]
    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = list(pool.map(score, rows))
    assert len(concurrent) == len(rows) == 40
    assert concurrent == sequential


class TestScoreRequestSchema:
    def test_accepts_integers_as_floats(self):
        request = ScoreRequest(entity="e", first_index=0, history=[[1] * N_CHANNELS])
        assert request.history[0][0] == 1.0

    def test_names_the_first_row_with_the_wrong_width(self):
        rows = [[0.0] * N_CHANNELS, [0.0] * 5]
        with pytest.raises(ValueError, match=r"history\[1\] has 5 values; expected 38"):
            ScoreRequest(entity="e", first_index=0, history=rows)

    def test_response_rejects_non_finite_score(self):
        with pytest.raises(ValueError, match="finite"):
            ScoreResponse(
                entity="e",
                model="m",
                row_index=0,
                score=float("nan"),
                threshold=0.0,
                is_anomaly=False,
                bundle_sha256="0" * 64,
            )


# ---------------------------------------------------------------- command functions


@pytest.fixture
def metrics_file(tmp_path: Path) -> Path:
    metrics = tiny_metrics()
    models = dict(metrics.models)
    models["iforest"] = models["iforest"].model_copy(
        update={"params": {"n_estimators": 150, "max_samples": 512}}
    )
    models["cnn_ae"] = models["cnn_ae"].model_copy(update={"params": dict(CNN_AE)})
    path = tmp_path / "reports" / "metrics.json"
    path.parent.mkdir()
    write_metrics(path, metrics.model_copy(update={"models": models}))
    return path


def test_resolve_without_metrics_defaults_to_iforest_with_default_params(tmp_path):
    model, params = commands.resolve_configuration(
        model=None, metrics_path=tmp_path / "missing.json", default_params=True
    )
    assert model == "iforest"
    assert params == dict(DEFAULT_PARAMS["iforest"])


def test_resolve_takes_the_headline_and_its_tuned_params_from_the_metrics(metrics_file):
    model, params = commands.resolve_configuration(
        model=None, metrics_path=metrics_file, default_params=False
    )
    assert (model, params) == ("cnn_ae", CNN_AE)


def test_resolve_default_params_ignores_the_tuned_ones(metrics_file):
    model, params = commands.resolve_configuration(
        model=None, metrics_path=metrics_file, default_params=True
    )
    assert model == "cnn_ae"
    assert params == dict(DEFAULT_PARAMS["cnn_ae"])


def test_resolve_explicit_model_takes_its_own_tuned_params(metrics_file):
    model, params = commands.resolve_configuration(
        model="iforest", metrics_path=metrics_file, default_params=False
    )
    assert (model, params) == ("iforest", {"n_estimators": 150, "max_samples": 512})


def test_resolve_explicit_model_beats_the_headline_with_default_params(metrics_file):
    model, params = commands.resolve_configuration(
        model="iforest", metrics_path=metrics_file, default_params=True
    )
    assert (model, params) == ("iforest", dict(DEFAULT_PARAMS["iforest"]))


def test_resolve_tuned_params_need_a_metrics_file(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match=r"missing\.json") as caught:
        commands.resolve_configuration(model="iforest", metrics_path=missing, default_params=False)
    assert "default" in str(caught.value)


def test_resolve_refuses_a_model_the_metrics_do_not_hold(metrics_file):
    with pytest.raises(ValueError, match="lstm_ae_v2"):
        commands.resolve_configuration(
            model="lstm_ae_v2", metrics_path=metrics_file, default_params=False
        )


def test_resolve_refuses_an_unknown_model_with_default_params(tmp_path):
    with pytest.raises(ValueError, match=r"unknown model 'nope'.*iforest"):
        commands.resolve_configuration(
            model="nope", metrics_path=tmp_path / "missing.json", default_params=True
        )


def test_bundle_command_writes_a_loadable_bundle(tmp_path, monkeypatch):
    loaded_with: dict[str, Any] = {}

    def fake_load_entity(name: str, *, audit: LabelAudit, dest: Path | None = None) -> EntityData:
        loaded_with.update(name=name, dest=dest, audit=audit)
        return make_entity(n_train=N_TRAIN, n_test=N_TEST, audit=audit, name=name)

    monkeypatch.setattr(commands, "load_entity", fake_load_entity)
    out = tmp_path / "artifacts" / "bundle" / "bundle.joblib"
    data_dir = tmp_path / "data"
    summary = commands.bundle_command(
        entity="machine-1-1",
        model="iforest",
        metrics_path=tmp_path / "missing.json",
        default_params=True,
        out=out,
        data_dir=data_dir,
    )
    assert loaded_with["name"] == "machine-1-1"
    assert loaded_with["dest"] == data_dir
    assert loaded_with["audit"].reads == 0
    bundle, digest = load_bundle(out)
    assert (bundle.entity, bundle.model) == ("machine-1-1", "iforest")
    assert bundle.params == dict(DEFAULT_PARAMS["iforest"])
    assert digest in summary
    assert str(out) in summary


def test_serve_command_hands_the_app_to_uvicorn(iforest, monkeypatch):
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(
        uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)), raising=True
    )
    commands.serve_command(bundle=iforest.path, host="127.0.0.1", port=8123)
    assert len(calls) == 1
    app, kwargs = calls[0]
    assert isinstance(app, FastAPI)
    assert (kwargs["host"], kwargs["port"]) == ("127.0.0.1", 8123)


def test_serve_command_does_not_start_a_server_for_a_tampered_bundle(
    iforest, tmp_path, monkeypatch
):
    calls: list[object] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append(a), raising=True)
    path = tmp_path / "bundle.joblib"
    path.write_bytes(iforest.path.read_bytes()[:-1])
    Path(str(path) + ".sha256").write_text(iforest.digest + "\n")
    with pytest.raises(ChecksumError):
        commands.serve_command(bundle=path, host="127.0.0.1", port=8123)
    assert calls == []
