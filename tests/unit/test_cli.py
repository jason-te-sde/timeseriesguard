from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from tests.unit.test_report_and_claims import README_TEXT, tiny_metrics, write_runs_csv
from tsguard import cli
from tsguard.cli import main
from tsguard.data.labels import LabelAudit
from tsguard.errors import ChecksumError, ConfigurationError, ContractError, FetchError
from tsguard.experiment.reproduce import SMOKE
from tsguard.experiment.tuning import FINAL_EXPERIMENT, TUNING_EXPERIMENT
from tsguard.repro.report import write_metrics

# MLflow's SQLAlchemy store triggers a deprecation warning inside SQLAlchemy on first use.
pytestmark = pytest.mark.filterwarnings("ignore:The ``noload`` loader strategy")

SMOKE_RUNS = 16  # 4 models x 2 trials + 4 models x 2 entities
STAMP = re.compile(r"^\[\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ \+\d+\.\ds\] (.*)$")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    reports = tmp_path / "reports"
    reports.mkdir()
    write_metrics(reports / "metrics.json", tiny_metrics())
    write_runs_csv(reports / "mlflow_runs.csv", 32)
    (tmp_path / "README.md").write_text(README_TEXT)
    return tmp_path


def test_claims_check_exit_codes(repo, capsys):
    root = ["--root", str(repo)]
    assert main(["claims", "--write", *root]) == 0
    assert main(["claims", "--check", *root]) == 0
    assert capsys.readouterr().err == ""
    path = repo / "reports" / "claims.json"
    data = json.loads(path.read_text())
    data["headline_auc_pr"] = 0.80
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    capsys.readouterr()
    assert main(["claims", "--check", *root]) == 1
    assert (
        "claims.json: headline_auc_pr is 0.80, metrics.json gives 0.82" in capsys.readouterr().out
    )


def test_claims_with_broken_readme_markers_exit_1_with_the_message(repo, capsys):
    (repo / "README.md").write_text("# Demo\n\nno markers here\n")
    for flag in ("--check", "--write"):
        assert main(["claims", flag, "--root", str(repo)]) == 1
        err = capsys.readouterr().err
        assert err.startswith("tsguard: error: ")
        assert "claims:begin" in err
        assert "Traceback" not in err


def test_claims_refuse_a_run_that_is_not_publishable_with_one_message_per_problem(repo, capsys):
    metrics = tiny_metrics()
    smoke = metrics.model_copy(update={"profile": "smoke"})
    write_metrics(repo / "reports" / "metrics.json", smoke)
    assert main(["claims", "--write", "--root", str(repo)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("tsguard: error: reports/metrics.json is not publishable: profile is ")
    assert "Traceback" not in err
    assert not (repo / "reports" / "claims.json").exists()
    assert main(["claims", "--check", "--root", str(repo)]) == 1
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "reports/metrics.json: not publishable: profile is 'smoke', not 'full': "
        "only a full run is publishable"
    ]


def test_claims_write_without_reports_exits_1(tmp_path, capsys):
    assert main(["claims", "--write", "--root", str(tmp_path)]) == 1
    assert "metrics.json" in capsys.readouterr().err


def test_claims_needs_exactly_one_mode():
    for argv in (["claims"], ["claims", "--check", "--write"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 2


def test_unknown_entity_exit_code(capsys):
    assert main(["data", "verify", "--entities", "machine-9-9"]) == 1
    err = capsys.readouterr().err
    assert "machine-9-9" in err
    assert "machine-1-1" in err  # the message lists the valid names


def test_usage_error_is_2():
    with pytest.raises(SystemExit) as e:
        main(["nope"])
    assert e.value.code == 2


def test_no_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2


def test_unknown_profile_is_a_usage_error():
    with pytest.raises(SystemExit) as e:
        main(["reproduce", "--profile", "huge"])
    assert e.value.code == 2


def test_data_fetch_passes_entities_and_destination(monkeypatch, tmp_path, capsys):
    seen: dict[str, Any] = {}

    def fake_fetch(entities, *, dest):
        seen.update(entities=tuple(entities), dest=dest)
        return [tmp_path / "a", tmp_path / "b"]

    monkeypatch.setattr(cli, "fetch", fake_fetch)
    argv = ["data", "fetch", "--entities", "machine-1-1", "machine-1-2", "--dest", str(tmp_path)]
    assert main(argv) == 0
    assert seen == {"entities": ("machine-1-1", "machine-1-2"), "dest": tmp_path}
    assert "2 files" in capsys.readouterr().out


def test_data_fetch_defaults_to_every_entity(monkeypatch):
    seen: dict[str, Any] = {}

    def fake_fetch(entities, *, dest):
        seen.update(entities=tuple(entities), dest=dest)
        return []

    monkeypatch.setattr(cli, "fetch", fake_fetch)
    assert main(["data", "fetch"]) == 0
    assert len(seen["entities"]) == 28
    assert seen["dest"] is None


@pytest.mark.parametrize(
    "error", [FetchError("train/machine-1-1.txt: offline"), ChecksumError("bad")]
)
def test_data_errors_are_a_message_and_exit_1(monkeypatch, capsys, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "fetch", fail)
    monkeypatch.setattr(cli, "verify", fail)
    for command in ("fetch", "verify"):
        assert main(["data", command]) == 1
        captured = capsys.readouterr()
        assert captured.err == f"tsguard: error: {error}\n"
        assert captured.out == ""


def test_data_verify_reports_what_it_checked(monkeypatch, capsys):
    monkeypatch.setattr(cli, "verify", lambda entities, *, dest: None)
    assert main(["data", "verify", "--entities", "machine-1-1"]) == 0
    assert "machine-1-1" in capsys.readouterr().out


def test_reproduce_wires_profile_and_export(monkeypatch, tmp_path):
    seen: dict[str, Any] = {}

    def fake_reproduce(profile, *, loader, tracker, out_dir, progress):
        seen.update(profile=profile, out_dir=out_dir, loader=loader, tracker=tracker)

    def fake_export(uri, path, *, experiments):
        seen.update(uri=uri, csv=path, experiments=list(experiments))
        return SMOKE_RUNS

    monkeypatch.setattr(cli, "reproduce", fake_reproduce)
    monkeypatch.setattr(cli, "export_runs", fake_export)
    assert main(["reproduce", "--profile", "smoke", "--out", str(tmp_path)]) == 0
    assert seen["profile"] == SMOKE
    assert seen["out_dir"] == tmp_path
    assert seen["csv"] == tmp_path / "mlflow_runs.csv"
    assert seen["experiments"] == [TUNING_EXPERIMENT, FINAL_EXPERIMENT]
    assert seen["uri"] == f"sqlite:///{tmp_path / 'mlruns.db'}"  # the default lives in --out
    assert type(seen["tracker"]).__name__ == "MlflowTracker"


def test_reproduce_entities_override_only_the_evaluated_entities(monkeypatch, tmp_path):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli, "reproduce", lambda profile, **kw: seen.update(profile=profile))
    monkeypatch.setattr(cli, "export_runs", lambda uri, path, *, experiments: 4 * 2 + 4 * 1)
    argv = ["reproduce", "--profile", "smoke", "--out", str(tmp_path), "--entities", "machine-1-2"]
    assert main(argv) == 0
    assert seen["profile"].entities == ("machine-1-2",)
    assert seen["profile"].tuning_entities == SMOKE.tuning_entities
    assert seen["profile"].n_trials == SMOKE.n_trials


def test_an_entities_override_renames_the_profile_so_its_metrics_cannot_pass_for_the_real_run(
    monkeypatch, tmp_path
):
    names: list[str] = []
    monkeypatch.setattr(cli, "reproduce", lambda profile, **kw: names.append(profile.name))
    monkeypatch.setattr(cli, "export_runs", lambda uri, path, *, experiments: SMOKE_RUNS)
    assert main(["reproduce", "--profile", "smoke", "--out", str(tmp_path / "a")]) == 0
    argv = ["reproduce", "--profile", "smoke", "--out", str(tmp_path / "b"), "--entities"]
    assert main([*argv, "machine-1-1", "machine-2-1"]) == 0  # the same entities, still a custom run
    assert names == ["smoke", "smoke-custom"]


def test_a_wrong_exported_run_count_is_an_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "reproduce", lambda profile, **kw: None)
    monkeypatch.setattr(cli, "export_runs", lambda uri, path, *, experiments: SMOKE_RUNS + 4)
    assert main(["reproduce", "--profile", "smoke", "--out", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    # MLflow's own INFO lines may precede ours the first time a client is built in a process
    assert captured.err.strip().splitlines()[-1].startswith("tsguard: error: ")
    assert f"{SMOKE_RUNS + 4} runs" in captured.err
    assert f"expected {SMOKE_RUNS}" in captured.err
    assert "mlflow_runs.csv" in captured.err
    assert "--tracking-uri" in captured.err
    assert captured.out == ""  # no success line for a run whose export is wrong


def test_the_expected_run_count_follows_the_evaluated_entities(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "reproduce", lambda profile, **kw: None)
    monkeypatch.setattr(cli, "export_runs", lambda uri, path, *, experiments: SMOKE_RUNS)
    argv = ["reproduce", "--profile", "smoke", "--out", str(tmp_path), "--entities", "machine-1-2"]
    assert main(argv) == 1  # one entity gives 4 x 2 + 4 x 1 = 12 runs, not 16
    assert "expected 12" in capsys.readouterr().err


def test_reproduce_rejects_an_unknown_entity_before_any_work(monkeypatch, tmp_path, capsys):
    def never(*args, **kwargs):
        raise AssertionError("reproduce must not start")

    monkeypatch.setattr(cli, "reproduce", never)
    argv = ["reproduce", "--profile", "smoke", "--out", str(tmp_path), "--entities", "nope"]
    assert main(argv) == 1
    assert "nope" in capsys.readouterr().err


def test_reproduce_rejects_a_repeated_entity(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "reproduce", lambda *a, **k: None)
    names = ["machine-1-1", "machine-1-1"]
    argv = ["reproduce", "--profile", "smoke", "--out", str(tmp_path), "--entities", *names]
    assert main(argv) == 1
    assert "machine-1-1" in capsys.readouterr().err


def test_reproduce_refuses_an_existing_tracking_store(monkeypatch, tmp_path, capsys):
    def never(*args, **kwargs):
        raise AssertionError("reproduce must not start")

    monkeypatch.setattr(cli, "reproduce", never)
    store = tmp_path / "mlruns.db"
    store.write_bytes(b"")
    assert main(["reproduce", "--profile", "smoke", "--out", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert str(store) in err
    assert "delete it or pass another --tracking-uri" in err
    assert store.read_bytes() == b""  # the refusal touches nothing


def test_reproduce_creates_the_directory_of_a_new_tracking_store(monkeypatch, tmp_path):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli, "reproduce", lambda profile, **kw: None)
    monkeypatch.setattr(
        cli, "export_runs", lambda uri, path, *, experiments: seen.update(uri=uri) or SMOKE_RUNS
    )
    uri = f"sqlite:///{tmp_path / 'deep' / 'er' / 'runs.db'}"
    argv = [
        "reproduce",
        "--profile",
        "smoke",
        "--out",
        str(tmp_path / "out"),
        "--tracking-uri",
        uri,
    ]
    assert main(argv) == 0
    assert seen["uri"] == uri
    assert (tmp_path / "deep" / "er").is_dir()


def test_reproduce_progress_lines_carry_a_utc_stamp_and_elapsed_seconds(
    monkeypatch, tmp_path, capsys
):
    def fake_reproduce(profile, *, loader, tracker, out_dir, progress):
        progress("tune zscore: best objective 0.5")
        progress("select machine-1-1 zscore: validation objective 0.4")

    monkeypatch.setattr(cli, "reproduce", fake_reproduce)
    monkeypatch.setattr(cli, "export_runs", lambda uri, path, *, experiments: SMOKE_RUNS)
    assert main(["reproduce", "--profile", "smoke", "--out", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    stamped = [line for line in captured.err.splitlines() if STAMP.match(line)]
    assert [STAMP.match(line).group(1) for line in stamped] == [  # type: ignore[union-attr]
        "tune zscore: best objective 0.5",
        "select machine-1-1 zscore: validation objective 0.4",
    ]
    assert "tune zscore" not in captured.out  # progress is on stderr only


def test_reproduce_loads_entities_from_the_data_directory(monkeypatch, tmp_path):
    seen: dict[str, Any] = {}
    loaded: list[tuple[str, LabelAudit]] = []

    def fake_reproduce(profile, *, loader, tracker, out_dir, progress):
        seen["loader"] = loader

    def fake_load_entity(name, *, audit, dest=None):
        loaded.append((name, audit))
        assert dest is None  # TSGUARD_DATA_DIR and the default directory decide
        return "entity"

    monkeypatch.setattr(cli, "reproduce", fake_reproduce)
    monkeypatch.setattr(cli, "export_runs", lambda uri, path, *, experiments: SMOKE_RUNS)
    monkeypatch.setattr(cli, "load_entity", fake_load_entity)
    assert main(["reproduce", "--profile", "smoke", "--out", str(tmp_path)]) == 0
    audit = LabelAudit()
    assert seen["loader"]("machine-1-1", audit) == "entity"
    assert loaded == [("machine-1-1", audit)]


BUNDLE_ARGV = ["bundle", "--entity", "machine-1-1", "--out", "artifacts/bundle/bundle.joblib"]


def test_bundle_defaults_take_the_headline_and_the_committed_metrics(monkeypatch, capsys):
    seen: dict[str, Any] = {}

    def fake_bundle_command(**kwargs):
        seen.update(kwargs)
        return "wrote the bundle"

    monkeypatch.setattr(cli, "bundle_command", fake_bundle_command)
    assert main(BUNDLE_ARGV) == 0
    assert seen == {
        "entity": "machine-1-1",
        "model": None,
        "metrics_path": Path("reports/metrics.json"),
        "default_params": False,
        "out": Path("artifacts/bundle/bundle.joblib"),
    }
    assert capsys.readouterr().out == "wrote the bundle\n"


def test_bundle_passes_every_flag_through(monkeypatch, tmp_path):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli, "bundle_command", lambda **kwargs: seen.update(kwargs) or "ok")
    argv = [
        "bundle",
        "--entity",
        "machine-2-1",
        "--model",
        "cnn_ae",
        "--metrics",
        str(tmp_path / "m.json"),
        "--default-params",
        "--out",
        str(tmp_path / "b.joblib"),
    ]
    assert main(argv) == 0
    assert seen == {
        "entity": "machine-2-1",
        "model": "cnn_ae",
        "metrics_path": tmp_path / "m.json",
        "default_params": True,
        "out": tmp_path / "b.joblib",
    }


def test_bundle_rejects_an_unknown_entity_before_any_work(monkeypatch, capsys):
    def never(**kwargs):
        raise AssertionError("nothing may be fitted for an entity that does not exist")

    monkeypatch.setattr(cli, "bundle_command", never)
    assert main(["bundle", "--entity", "machine-9-9", "--out", "b.joblib"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("tsguard: error: ")
    assert "machine-9-9" in err
    assert "machine-1-1" in err  # the message lists the valid names


@pytest.mark.parametrize(
    "argv",
    [
        ["bundle", "--out", "b.joblib"],
        ["bundle", "--entity", "machine-1-1"],
        ["bundle", *BUNDLE_ARGV[1:], "--model", "gpt"],
        ["serve"],
        ["serve", "--bundle", "b.joblib", "--port", "eighty"],
    ],
)
def test_bundle_and_serve_usage_errors_are_2(argv):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2


def test_serve_defaults_to_loopback_on_8000(monkeypatch):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli, "serve_command", lambda **kwargs: seen.update(kwargs))
    assert main(["serve", "--bundle", "artifacts/bundle/bundle.joblib"]) == 0
    assert seen == {
        "bundle": Path("artifacts/bundle/bundle.joblib"),
        "host": "127.0.0.1",
        "port": 8000,
    }


def test_serve_passes_host_and_port_through(monkeypatch, tmp_path):
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli, "serve_command", lambda **kwargs: seen.update(kwargs))
    argv = ["serve", "--bundle", str(tmp_path / "b.joblib"), "--host", "0.0.0.0", "--port", "9001"]
    assert main(argv) == 0
    assert seen == {"bundle": tmp_path / "b.joblib", "host": "0.0.0.0", "port": 9001}


@pytest.mark.parametrize(
    "error",
    [
        ChecksumError("bundle.joblib: sha256 does not match its sidecar"),
        ContractError("bundle.joblib: not a tsguard bundle"),
        FileNotFoundError("reports/metrics.json does not exist, so there is nothing to tune from"),
        ConfigurationError("unknown model 'nope'; valid names: zscore, iforest"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_bundle_and_serve_failures_are_a_message_and_exit_1(monkeypatch, capsys, error):
    def fail(**kwargs):
        raise error

    monkeypatch.setattr(cli, "bundle_command", fail)
    monkeypatch.setattr(cli, "serve_command", fail)
    for argv in (BUNDLE_ARGV, ["serve", "--bundle", "b.joblib"]):
        assert main(argv) == 1
        err = capsys.readouterr().err
        assert err == f"tsguard: error: {error}\n"


def sklearn_invalid_parameter_error() -> ValueError:
    """Return the ValueError scikit-learn raises for a parameter it rejects at fit time."""
    try:
        IsolationForest(n_estimators=-1).fit(np.zeros((4, 2)))
    except ValueError as error:
        return error
    raise AssertionError("scikit-learn accepted n_estimators=-1")


@pytest.mark.parametrize("command", ["bundle", "serve"])
@pytest.mark.parametrize(
    "bug",
    [ValueError("a bug"), sklearn_invalid_parameter_error()],
    ids=["ValueError", "InvalidParameterError"],
)
def test_a_value_error_that_is_not_a_tsguard_error_is_a_bug_and_keeps_its_traceback(
    monkeypatch, command, bug
):
    def broken(**kwargs):
        raise bug

    monkeypatch.setattr(cli, f"{command}_command", broken)
    argv = BUNDLE_ARGV if command == "bundle" else ["serve", "--bundle", "b.joblib"]
    with pytest.raises(ValueError, match=re.escape(str(bug))) as caught:
        main(argv)
    assert caught.value is bug


def test_a_bug_in_a_command_keeps_its_traceback(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("a bug")

    monkeypatch.setattr(cli, "serve_command", broken)
    with pytest.raises(RuntimeError, match="a bug"):
        main(["serve", "--bundle", "b.joblib"])


def test_the_command_line_and_the_service_import_without_the_experiment_extra():
    # The container installs the runtime dependencies only, so `tsguard serve` must start
    # without MLflow and Optuna. A fresh interpreter, because this process already has them.
    code = "\n".join(
        [
            "import sys",
            "for name in ('mlflow', 'optuna'):",
            "    sys.modules[name] = None",
            "try:",
            "    import mlflow",
            "except ImportError:",
            "    pass",
            "else:",
            "    raise SystemExit('the stub does not block mlflow')",
            "import tsguard.cli, tsguard.serve.app, tsguard.serve.commands",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
