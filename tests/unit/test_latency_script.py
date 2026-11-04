from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from tests.conftest import make_entity
from tsguard.data.labels import LabelAudit, Phase
from tsguard.data.smd import EntityData
from tsguard.serve.schemas import ModelInfo, ScoreRequest, ScoreResponse

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "latency.py"
URL = "http://stub.test"
N_TRAIN = 300
N_TEST = 120
REQUIRED_ROWS = 64
FIT_END = 150
RESULT_KEYS = {
    "n",
    "warmup",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "max_ms",
    "budget_ms",
    "url",
    "environment",
}


@pytest.fixture(scope="module")
def latency() -> Iterator[ModuleType]:
    """Import scripts/latency.py, which is not part of the package."""
    spec = importlib.util.spec_from_file_location("latency_script", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module")
def entity() -> EntityData:
    return make_entity(n_train=N_TRAIN, n_test=N_TEST, anomalies=((10, 5),), name="machine-1-1")


class StubService:
    """A fake service and a fake clock that agree on how long each score request took.

    ``/v1/model`` answers instantly. Every ``/v1/score`` request moves the clock by
    ``step(i)`` seconds, ``i`` being its position among the score requests, so the client
    measures exactly that.
    """

    def __init__(
        self,
        entity: EntityData,
        step: Callable[[int], float],
        *,
        required_rows: int = REQUIRED_ROWS,
        fit_end: int = FIT_END,
        fail_at: int | None = None,
    ) -> None:
        self.entity = entity
        self.step = step
        self.required_rows = required_rows
        self.fit_end = fit_end
        self.fail_at = fail_at
        self.now = 0.0
        self.requests: list[ScoreRequest] = []
        self.transport = httpx.MockTransport(self._handle)

    def clock(self) -> float:
        return self.now

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/model":
            info = ModelInfo(
                entity=self.entity.name,
                model="iforest",
                params={},
                threshold=0.9,
                required_rows=self.required_rows,
                fit_end=self.fit_end,
                feature_columns=100,
                bundle_sha256="0" * 64,
                source_commit="stub",
            )
            return httpx.Response(200, content=info.model_dump_json())
        assert request.method == "POST"
        assert request.url.path == "/v1/score"
        assert request.headers["content-type"] == "application/json"
        parsed = ScoreRequest.model_validate_json(request.content)
        index = len(self.requests)
        self.requests.append(parsed)
        self.now += self.step(index)
        if index == self.fail_at:
            return httpx.Response(500, json={"detail": "boom"})
        answer = ScoreResponse(
            entity=parsed.entity,
            model="iforest",
            row_index=parsed.first_index + len(parsed.history) - 1,
            score=0.5,
            threshold=0.9,
            is_anomaly=False,
            bundle_sha256="0" * 64,
        )
        return httpx.Response(200, content=answer.model_dump_json())


def run_script(
    latency: ModuleType,
    stub: StubService,
    *,
    n: int = 50,
    warmup: int = 10,
    budget_ms: float = 50.0,
    out: Path | None = None,
) -> int:
    code: int = latency.run(
        url=URL,
        n=n,
        warmup=warmup,
        budget_ms=budget_ms,
        out=out,
        loader=lambda name: stub.entity,
        transport=stub.transport,
        clock=stub.clock,
    )
    return code


def test_percentile_ms(latency):
    assert latency.percentile_ms([0.001] * 99 + [0.1], 95) == pytest.approx(1.0)
    assert latency.percentile_ms([0.001] * 99 + [0.1], 100) == pytest.approx(100.0)
    assert latency.percentile_ms([0.002], 50) == pytest.approx(2.0)


def test_percentile_ms_of_nothing_is_an_error(latency):
    with pytest.raises(ValueError, match="no samples"):
        latency.percentile_ms([], 95)


def test_budget_breach_returns_1(latency, entity, capsys, tmp_path):
    stub = StubService(entity, lambda i: 0.060)
    out = tmp_path / "latency.json"
    assert run_script(latency, stub, out=out) == 1
    assert "p95" in capsys.readouterr().err
    # The numbers are still written: a breach is a measurement, not a missing one.
    assert json.loads(out.read_text())["p95_ms"] == pytest.approx(60.0)


def test_within_budget_returns_0_and_writes_the_result(latency, entity, capsys, tmp_path):
    stub = StubService(entity, lambda i: 0.005)
    out = tmp_path / "nested" / "latency.json"
    assert run_script(latency, stub, out=out) == 0
    result = json.loads(out.read_text())
    assert set(result) == RESULT_KEYS
    assert result["p95_ms"] == 5.0
    assert result["p50_ms"] == 5.0
    assert result["p99_ms"] == 5.0
    assert result["max_ms"] == 5.0
    assert (result["n"], result["warmup"]) == (50, 10)
    assert result["budget_ms"] == 50.0
    assert result["url"] == URL
    assert set(result["environment"]) == {"platform", "python", "cpu_count"}
    assert json.loads(capsys.readouterr().out) == result


def test_the_budget_is_exclusive(latency, entity):
    stub = StubService(entity, lambda i: 0.050)
    assert run_script(latency, stub, budget_ms=50.0) == 1
    stub = StubService(entity, lambda i: 0.050)
    assert run_script(latency, stub, budget_ms=50.001) == 0


def test_percentiles_cover_the_measured_requests_and_not_the_warmup(latency, entity, tmp_path):
    warmup = 10

    def step(i: int) -> float:
        if i < warmup:
            return 0.500  # a cold start, which must not reach the numbers
        k = i - warmup
        return 0.100 if k == 99 else 0.020 if k >= 94 else 0.001

    out = tmp_path / "latency.json"
    stub = StubService(entity, step)
    assert run_script(latency, stub, n=100, warmup=warmup, out=out) == 0
    result = json.loads(out.read_text())
    assert result["p50_ms"] == 1.0
    assert result["p95_ms"] == 20.0
    assert result["p99_ms"] == pytest.approx(20.8)
    assert result["max_ms"] == 100.0


def test_requests_end_at_successive_test_rows_and_wrap_around_the_test_range(latency, entity):
    n, warmup = 130, 10  # 140 requests over 120 test rows: the last ones wrap
    stub = StubService(entity, lambda i: 0.001)
    assert run_script(latency, stub, n=n, warmup=warmup) == 0
    assert len(stub.requests) == n + warmup
    full = entity.full
    for i, request in enumerate(stub.requests):
        last = N_TRAIN + i % N_TEST
        assert request.entity == "machine-1-1"
        assert request.first_index + len(request.history) - 1 == last
        assert len(request.history) == REQUIRED_ROWS
        assert request.history == full[last - REQUIRED_ROWS + 1 : last + 1].tolist()
    assert stub.requests[-1].first_index + REQUIRED_ROWS - 1 == N_TRAIN + 139 % N_TEST


def test_the_entity_is_loaded_by_the_name_the_service_reports(latency, entity: EntityData):
    asked: list[str] = []

    def loader(name: str) -> EntityData:
        asked.append(name)
        return entity

    stub = StubService(entity, lambda i: 0.001)
    code = latency.run(
        url=URL,
        n=5,
        warmup=0,
        budget_ms=50.0,
        out=None,
        loader=loader,
        transport=stub.transport,
        clock=stub.clock,
    )
    assert code == 0
    assert asked == ["machine-1-1"]


def test_a_failed_response_returns_1_and_names_the_request(latency, entity, capsys, tmp_path):
    out = tmp_path / "latency.json"
    stub = StubService(entity, lambda i: 0.001, fail_at=12)
    assert run_script(latency, stub, n=50, warmup=10, out=out) == 1
    err = capsys.readouterr().err
    assert "500" in err
    assert "request 12" in err
    assert "boom" in err
    assert len(stub.requests) == 13  # it stops at the first failure
    assert not out.exists()  # a run with a failed request has no latency to report


def test_an_unreachable_service_returns_1(latency, entity, capsys):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    code = latency.run(
        url=URL,
        n=5,
        warmup=0,
        budget_ms=50.0,
        out=None,
        loader=lambda name: entity,
        transport=httpx.MockTransport(refuse),
        clock=lambda: 0.0,
    )
    assert code == 1
    err = capsys.readouterr().err
    assert URL in err
    assert "connection refused" in err


@pytest.mark.parametrize(
    ("required_rows", "fit_end", "message"),
    [
        (N_TRAIN + 1, FIT_END, "needs 301 rows"),
        (REQUIRED_ROWS, N_TRAIN + 1, "fitted on 301 rows"),
    ],
)
def test_a_service_built_for_other_data_returns_1(
    latency, entity, capsys, required_rows, fit_end, message
):
    stub = StubService(entity, lambda i: 0.001, required_rows=required_rows, fit_end=fit_end)
    assert run_script(latency, stub) == 1
    assert message in capsys.readouterr().err
    assert stub.requests == []


def test_main_wires_the_flags_into_run(latency, monkeypatch, tmp_path):
    seen: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> int:
        seen.update(kwargs)
        return 7

    monkeypatch.setattr(latency, "run", fake_run)
    assert latency.main([]) == 7
    assert (seen["url"], seen["n"], seen["warmup"], seen["budget_ms"], seen["out"]) == (
        "http://127.0.0.1:8000",
        2000,
        100,
        50.0,
        None,
    )
    out = tmp_path / "result.json"
    argv = ["--url", "http://h:9", "--n", "5", "--warmup", "0", "--budget-ms", "12.5"]
    assert latency.main([*argv, "--out", str(out), "--data-dir", str(tmp_path)]) == 7
    assert (seen["url"], seen["n"], seen["warmup"], seen["budget_ms"], seen["out"]) == (
        "http://h:9",
        5,
        0,
        12.5,
        out,
    )


def test_main_loads_the_entity_with_a_fresh_audit_that_never_reads_labels(
    latency, monkeypatch, tmp_path, entity: EntityData
):
    captured: dict[str, Any] = {}

    def fake_load_entity(name: str, *, audit: LabelAudit, dest: Path | None = None) -> EntityData:
        captured.update(name=name, audit=audit, dest=dest)
        return entity

    loaders: list[Callable[[str], EntityData]] = []
    monkeypatch.setattr(latency, "load_entity", fake_load_entity)

    def fake_run(**kwargs: Any) -> int:
        loaders.append(kwargs["loader"])
        return 0

    monkeypatch.setattr(latency, "run", fake_run)
    assert latency.main(["--data-dir", str(tmp_path)]) == 0
    assert loaders[0]("machine-1-1") is entity
    assert captured["name"] == "machine-1-1"
    assert captured["dest"] == tmp_path
    assert captured["audit"].current is Phase.SELECTION
    assert captured["audit"].reads == 0


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--n", "0"), ("--n", "-1"), ("--warmup", "-1"), ("--budget-ms", "0"), ("--budget-ms", "-1")],
)
def test_main_rejects_a_count_or_budget_that_cannot_be_measured(latency, flag, value):
    with pytest.raises(SystemExit) as raised:
        latency.main([flag, value])
    assert raised.value.code == 2
