from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from tsguard.data.smd import ENTITIES
from tsguard.eval.protocols import PROTOCOLS
from tsguard.models import FEATURE_MODELS, MODEL_NAMES
from tsguard.repro import claims as claims_module
from tsguard.repro import report as report_module
from tsguard.repro.claims import (
    BEGIN,
    END,
    P95_BUDGET_MS,
    check_claims,
    derive_claims,
    publishable_problems,
    render_readme_block,
    write_claims,
)
from tsguard.repro.report import (
    AblationModel,
    DatasetModel,
    EntityResultModel,
    HeadlineModel,
    Metrics,
    ModelSummaryModel,
    ProtocolScoreModel,
    TuningModel,
    best_model,
    bootstrap_ci,
    deterministic_view,
    read_metrics,
    write_metrics,
)

# Macro event-level AUC-PR per model; the other three columns follow `PROTOCOLS` order.
MACRO_AUC_PR = {
    "lstm_ae": (0.3902, 0.9521, 0.4410, 0.7046),
    "zscore": (0.1123, 0.6234, 0.2045, 0.4012),
    "cnn_ae": (0.4149, 0.9734, 0.4851, 0.8234),
    "iforest": (0.1876, 0.7011, 0.3302, 0.5538),
}
# Macro validation objective per model; the headline is the one with the highest.
VALIDATION_OBJECTIVE = {"lstm_ae": 0.63, "zscore": 0.41, "cnn_ae": 0.77, "iforest": 0.52}
HEADLINE = "cnn_ae"
STALE_README = "README block differs from metrics.json; run `tsguard claims --write`"
README_TEXT = f"# Demo\n\nIntro.\n\n{BEGIN}\n{END}\n\n## After\n\nFooter.\n"
CSV_HEADER = ["run_id", "experiment", "run_name", "kind", "model", "objective", "params_json"]


def _scores(values: tuple[float, ...], shift: float = 0.0) -> dict[str, ProtocolScoreModel]:
    return {
        name: ProtocolScoreModel(
            auc_pr=round(v + shift, 4), f1=round(0.8 * v, 4), f1_oracle=round(0.9 * v, 4)
        )
        for name, v in zip(PROTOCOLS, values, strict=True)
    }


def _entity(
    name: str, values: tuple[float, ...], objective: float, shift: float
) -> EntityResultModel:
    return EntityResultModel(
        entity=name,
        validation_objective=objective + shift,
        threshold=0.9,
        protocols=_scores(values, shift),
        sensitivity={"chunk_30": 0.7, "chunk_120": 0.8},
        digests={"state": "a" * 64},
    )


def _summary(values: tuple[float, ...], objective: float) -> ModelSummaryModel:
    return ModelSummaryModel(
        params={"window": 32, "learning_rate": 0.001, "kind": "tiny"},
        validation_objective=objective,
        macro=_scores(values),
        sensitivity_macro={"chunk_30": 0.7, "chunk_120": 0.8},
        per_entity=[
            _entity("machine-1-1", values, objective, 0.01),
            _entity("machine-2-1", values, objective, -0.01),
        ],
    )


def _ablation(
    model: str, points: tuple[float, float], relative: tuple[float, float], auc: tuple[float, float]
) -> AblationModel:
    return AblationModel(
        model=model,
        seeds=[0, 1, 2],
        f1_with=[0.61, 0.60, 0.62],
        f1_without=[0.58, 0.57, 0.58],
        auc_pr_with=[0.50, 0.51, 0.52],
        auc_pr_without=[0.49, 0.50, 0.50],
        delta_f1_points_mean=points[0],
        delta_f1_points_std=points[1],
        delta_f1_relative_mean=relative[0],
        delta_f1_relative_std=relative[1],
        delta_auc_pr_mean=auc[0],
        delta_auc_pr_std=auc[1],
    )


def tiny_metrics() -> Metrics:
    """Build a valid, publishable Metrics: 4 models (out of report order), a full-run shape.

    The model tables hold two entities, not 28: nothing checks their number against
    ``dataset.entities``, and two keep the fixture small.
    """
    return Metrics(
        schema_version=1,
        profile="full",
        seed=0,
        dataset=DatasetModel(
            source_commit="7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4",
            entities=len(ENTITIES),
            channels=38,
            train_steps=708405,
            test_steps=708420,
            anomaly_segments=7,
        ),
        protocol={"chunk": 60, "cooldown": 60, "k_percent": 20},
        feature_specs=63,
        models={
            name: _summary(values, VALIDATION_OBJECTIVE[name])
            for name, values in MACRO_AUC_PR.items()
        },
        headline=HeadlineModel(
            model=HEADLINE,
            protocol="event_level",
            auc_pr=0.8234,
            ci95=(0.7149, 0.9087),
            selected_by="validation_objective",
            label_reads_at_selection=0,
        ),
        ablation={
            "zscore": _ablation("zscore", (3.14, 0.42), (12.46, 1.53), (0.0123, 0.0041)),
            "iforest": _ablation("iforest", (-0.04, 0.3), (7.04, 0.96), (-0.0057, 0.0032)),
        },
        tuning=TuningModel(
            entities=["machine-1-1"],
            n_trials=2,
            best={"zscore": {"params": {"window": 32}, "value": 0.5}},
        ),
        environment={"platform": "test", "generated_at": "2026-11-03T00:00:00Z"},
    )


def write_latency(path: Path, p95_ms: float = 12.3, *, model: str = HEADLINE) -> None:
    """Write a ``latency.json`` the way scripts/latency.py does, for the fields claims reads."""
    path.write_text(json.dumps({"p95_ms": p95_ms, "model": model, "bundle_sha256": "0" * 64}))


def write_runs_csv(path: Path, rows: int) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for i in range(rows):
            # params_json holds commas, so a line count would be wrong for quoted fields.
            params = json.dumps({"window": 8, "learning_rate": 0.01})
            writer.writerow([f"run{i}", "final", f"zscore-{i}", "final", "zscore", "0.5", params])


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    reports = tmp_path / "reports"
    reports.mkdir()
    write_metrics(reports / "metrics.json", tiny_metrics())
    write_runs_csv(reports / "mlflow_runs.csv", 32)
    write_latency(reports / "latency.json")
    (tmp_path / "README.md").write_text(README_TEXT)
    return tmp_path


def _derive(root: Path, *, latency: bool = True):
    reports = root / "reports"
    return derive_claims(
        read_metrics(reports / "metrics.json"),
        mlflow_csv=reports / "mlflow_runs.csv",
        latency_json=reports / "latency.json" if latency else None,
    )


def test_metrics_round_trip(tmp_path):
    path = tmp_path / "metrics.json"
    metrics = tiny_metrics()
    write_metrics(path, metrics)
    assert read_metrics(path) == metrics


def test_metrics_file_is_canonical_json(tmp_path):
    path = tmp_path / "metrics.json"
    metrics = tiny_metrics()
    write_metrics(path, metrics)
    expected = json.dumps(metrics.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    assert path.read_text() == expected


def test_write_metrics_refuses_nan_and_leaves_no_file(tmp_path):
    metrics = tiny_metrics()
    bad = metrics.model_copy(
        update={"headline": metrics.headline.model_copy(update={"auc_pr": math.nan})}
    )
    path = tmp_path / "metrics.json"
    with pytest.raises(ValueError, match="JSON"):
        write_metrics(path, bad)
    assert not path.exists()


@pytest.mark.parametrize("where", ["top", "dataset", "headline"])
def test_extra_field_is_rejected(where):
    raw = tiny_metrics().model_dump(mode="json")
    target = raw if where == "top" else raw[where]
    target["surprise"] = 1
    with pytest.raises(ValidationError, match="surprise"):
        Metrics.model_validate(raw)


def test_metrics_are_frozen():
    metrics = tiny_metrics()
    with pytest.raises(ValidationError):
        metrics.seed = 1


def test_deterministic_view_drops_environment():
    metrics = tiny_metrics()
    other = metrics.model_copy(update={"environment": {"platform": "elsewhere"}})
    view = deterministic_view(metrics)
    assert "environment" not in view
    assert view == deterministic_view(other)
    assert view == {k: v for k, v in metrics.model_dump(mode="json").items() if k != "environment"}


def test_bootstrap_ci_of_constant_values():
    assert bootstrap_ci([0.5] * 10, seed=0) == (0.5, 0.5)


def test_bootstrap_ci_is_seeded():
    values = [0.2, 0.4, 0.5, 0.9, 0.3, 0.8]
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)
    assert bootstrap_ci(values, seed=3) != bootstrap_ci(values, seed=4)


def test_bootstrap_ci_brackets_the_mean_with_the_expected_width():
    values = [float(v) for v in range(100)]  # mean 49.5, std 28.9: width ~ 2 * 1.96 * 28.9 / 10
    lo, hi = bootstrap_ci(values, seed=0)
    assert lo < 49.5 < hi
    assert 9.0 < hi - lo < 14.0


def test_bootstrap_ci_refuses_empty_input():
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci([], seed=0)


def test_derive_claims_values(repo):
    claims = _derive(repo)
    assert claims.mlflow_runs == 32
    assert claims.p95_ms == 12.3
    assert claims.p95_budget_ms == P95_BUDGET_MS == 50.0
    assert (claims.train_steps, claims.test_steps) == (708405, 708420)
    assert (claims.channels, claims.feature_specs, claims.models) == (38, 63, 4)
    assert claims.headline_model == HEADLINE
    assert claims.headline_auc_pr == 0.82
    assert claims.headline_ci95 == (0.71, 0.91)
    assert claims.fft_gain["zscore"] == {
        "f1_points": 3.1,
        "f1_points_std": 0.4,
        "f1_relative_pct": 12.5,
        "f1_relative_pct_std": 1.5,
    }


def test_derive_claims_turns_negative_zero_into_zero(repo):
    claims = _derive(repo)
    gain = claims.fft_gain["iforest"]["f1_points"]  # round(-0.04, 1) is -0.0
    assert gain == 0.0
    assert math.copysign(1.0, gain) == 1.0


def test_derive_claims_without_latency(repo):
    assert _derive(repo, latency=False).p95_ms is None


def test_derive_claims_counts_quoted_rows_not_lines(repo):
    path = repo / "reports" / "mlflow_runs.csv"
    with path.open("a", newline="") as f:
        csv.writer(f).writerow(["x", "final", "multi\nline", "final", "zscore", "0.5", "{}"])
    assert _derive(repo).mlflow_runs == 33


def test_derive_claims_rejects_a_csv_without_header(repo):
    path = repo / "reports" / "mlflow_runs.csv"
    path.write_text("r1,final,zscore-1,final,zscore,0.5,{}\n")
    with pytest.raises(ValueError, match="run_id"):
        _derive(repo)


def test_derive_claims_rejects_a_latency_file_without_p95(repo):
    (repo / "reports" / "latency.json").write_text(json.dumps({"p50_ms": 3.0}))
    with pytest.raises(ValueError, match="p95_ms"):
        _derive(repo)


def test_derive_claims_refuses_a_latency_measured_on_another_model(repo):
    write_latency(repo / "reports" / "latency.json", model="iforest")
    message = (
        "latency.json: measured on model 'iforest', but the headline model is 'cnn_ae'; "
        "serve the headline model and rerun scripts/latency.py"
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        _derive(repo)


def test_derive_claims_refuses_a_latency_file_that_names_no_model(repo):
    (repo / "reports" / "latency.json").write_text(json.dumps({"p95_ms": 12.3}))
    with pytest.raises(ValueError, match=r"latency\.json: records no model.*'cnn_ae'"):
        _derive(repo)


def test_a_latency_measured_on_another_model_is_a_message_for_check_and_an_error_for_write(repo):
    write_claims(repo)
    write_latency(repo / "reports" / "latency.json", model="zscore")
    assert check_claims(repo) == [
        "reports/latency.json: measured on model 'zscore', but the headline model is 'cnn_ae'; "
        "serve the headline model and rerun scripts/latency.py"
    ]
    with pytest.raises(ValueError, match="measured on model 'zscore'"):
        write_claims(repo)


def test_block_is_deterministic_and_ordered_by_report_order(repo):
    metrics = tiny_metrics()
    claims = _derive(repo)
    block = render_readme_block(claims, metrics)
    assert block == render_readme_block(claims, metrics)
    assert block.startswith(BEGIN + "\n")
    assert block.endswith("\n" + END)
    first_cells = [
        line.split("|")[1].strip(" *") for line in block.splitlines() if line.startswith("|")
    ]
    order = [cell for cell in first_cells if cell in MACRO_AUC_PR]
    assert order[:4] == ["zscore", "iforest", "cnn_ae", "lstm_ae"]  # the macro AUC-PR table


def test_the_model_table_shows_what_the_headline_was_chosen_by(repo):
    lines = render_readme_block(_derive(repo), tiny_metrics()).splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("| Model | Validation"))
    table = lines[header + 2 : lines.index("", header)]  # the rows under the header and its rule
    rows = {line.split("|")[1].strip(" *"): line.split("|")[2].strip(" *") for line in table}
    assert rows == {name: f"{value:.3f}" for name, value in VALIDATION_OBJECTIVE.items()}
    assert max(rows, key=lambda name: rows[name]) == HEADLINE  # the bold row has the highest


def test_block_claims_table_and_headline_row(repo):
    block = render_readme_block(_derive(repo), tiny_metrics())
    assert "| Training steps | 708,405 |" in block
    assert "| Test steps | 708,420 |" in block
    assert "| Headline event-level AUC-PR | 0.82 (95% CI 0.71 to 0.91) |" in block
    assert "| MLflow runs | 32 |" in block
    assert "| p95 HTTP round trip, client-side (ms) | 12.3 (budget 50) |" in block
    header = (
        "| Model | Validation objective | point_wise | point_adjust | pa_percent_k | event_level |"
    )
    assert header in block
    assert "| zscore | 0.410 | 0.11 | 0.62 | 0.20 | 0.40 |" in block
    assert "| **cnn_ae** | **0.770** | **0.41** | **0.97** | **0.49** | **0.82** |" in block
    assert block.count("**") == 12  # only the headline row is bold


def test_block_says_so_when_latency_was_not_measured(repo):
    block = render_readme_block(_derive(repo, latency=False), tiny_metrics())
    assert "| p95 HTTP round trip, client-side (ms) | not measured (budget 50) |" in block
    assert "inference latency" not in block


def test_block_ablation_table(repo):
    block = render_readme_block(_derive(repo), tiny_metrics())
    assert "| zscore | +3.1 ± 0.4 | +12.5 ± 1.5 | +0.012 ± 0.004 |" in block
    assert "| iforest | +0.0 ± 0.3 | +7.0 ± 1.0 | -0.006 ± 0.003 |" in block


def test_block_refuses_a_model_that_lacks_a_protocol(repo):
    metrics = tiny_metrics()
    zscore = metrics.models["zscore"]
    macro = {k: v for k, v in zscore.macro.items() if k != "event_level"}
    broken = metrics.model_copy(
        update={"models": {**metrics.models, "zscore": zscore.model_copy(update={"macro": macro})}}
    )
    with pytest.raises(ValueError, match=r"zscore.*event_level"):
        render_readme_block(_derive(repo), broken)


def test_write_then_check_is_clean(repo):
    write_claims(repo)
    assert check_claims(repo) == []


def test_write_claims_replaces_only_the_block(repo):
    write_claims(repo)
    text = (repo / "README.md").read_text()
    assert text.startswith("# Demo\n\nIntro.\n\n" + BEGIN + "\n| Claim | Value |")
    assert text.endswith(END + "\n\n## After\n\nFooter.\n")


def test_write_claims_is_idempotent(repo):
    write_claims(repo)
    first = ((repo / "README.md").read_text(), (repo / "reports" / "claims.json").read_text())
    write_claims(repo)
    assert first == (
        (repo / "README.md").read_text(),
        (repo / "reports" / "claims.json").read_text(),
    )


def test_claims_json_is_canonical(repo):
    write_claims(repo)
    text = (repo / "reports" / "claims.json").read_text()
    data = json.loads(text)
    assert text == json.dumps(data, indent=2, sort_keys=True) + "\n"
    assert data["headline_auc_pr"] == 0.82
    assert data["headline_ci95"] == [0.71, 0.91]


def test_write_then_check_is_clean_without_latency(repo):
    (repo / "reports" / "latency.json").unlink()
    write_claims(repo)
    assert json.loads((repo / "reports" / "claims.json").read_text())["p95_ms"] is None
    assert check_claims(repo) == []


def test_tampered_claims_json_is_named(repo):
    write_claims(repo)
    path = repo / "reports" / "claims.json"
    data = json.loads(path.read_text())
    data["headline_auc_pr"] = 0.80
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    assert check_claims(repo) == ["claims.json: headline_auc_pr is 0.80, metrics.json gives 0.82"]


def test_tampered_nested_claim_is_named_by_its_path(repo):
    write_claims(repo)
    path = repo / "reports" / "claims.json"
    data = json.loads(path.read_text())
    data["fft_gain"]["zscore"]["f1_points"] = 9.9
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    messages = check_claims(repo)
    assert len(messages) == 1
    assert "fft_gain.zscore.f1_points" in messages[0]


def test_missing_and_unexpected_claims_fields_are_named(repo):
    write_claims(repo)
    path = repo / "reports" / "claims.json"
    data = json.loads(path.read_text())
    del data["channels"]
    data["surprise"] = 1
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    assert check_claims(repo) == [
        "claims.json: channels is missing, metrics.json gives 38",
        "claims.json: unexpected field surprise",
    ]


def test_unparseable_claims_json_is_a_message(repo):
    write_claims(repo)
    (repo / "reports" / "claims.json").write_text("{not json")
    messages = check_claims(repo)
    assert len(messages) == 1
    assert messages[0].startswith("reports/claims.json: not valid JSON")


def test_missing_claims_json_is_reported(repo):
    messages = check_claims(repo)  # nothing written yet: the README block is empty too
    assert any("claims.json: not found" in m for m in messages)
    assert any("README block differs" in m for m in messages)


def test_tampered_readme_is_reported(repo):
    write_claims(repo)
    readme = repo / "README.md"
    text = readme.read_text()
    assert "0.82" in text
    readme.write_text(text.replace("0.82", "0.83", 1))
    assert check_claims(repo) == [
        "README block differs from metrics.json; run `tsguard claims --write`"
    ]


def test_text_outside_the_block_is_not_checked(repo):
    write_claims(repo)
    readme = repo / "README.md"
    readme.write_text(readme.read_text().replace("Footer.", "A different footer, 0.99."))
    assert check_claims(repo) == []


def test_too_few_mlflow_runs_is_reported(repo):
    write_claims(repo)
    write_runs_csv(repo / "reports" / "mlflow_runs.csv", 29)
    messages = check_claims(repo)
    assert any("29" in m and "30" in m for m in messages)
    assert any(
        m.startswith("claims.json: mlflow_runs is 32, mlflow_runs.csv gives 29") for m in messages
    )


def test_exactly_thirty_runs_is_enough(repo):
    write_runs_csv(repo / "reports" / "mlflow_runs.csv", 30)
    write_claims(repo)
    assert check_claims(repo) == []


@pytest.mark.parametrize(
    "readme",
    [
        "# No markers\n",
        f"{BEGIN}\nonly a begin\n",
        f"{END}\nonly an end\n",
        f"{END}\n{BEGIN}\nreversed\n",
        f"{BEGIN}\n{END}\n{BEGIN}\n{END}\n",
    ],
    ids=["none", "begin-only", "end-only", "reversed", "twice"],
)
def test_missing_markers_is_value_error(repo, readme):
    (repo / "README.md").write_text(readme)
    with pytest.raises(ValueError, match="claims:"):
        write_claims(repo)
    assert not (repo / "reports" / "claims.json").exists()  # a refused write leaves no half state
    assert (repo / "README.md").read_text() == readme
    with pytest.raises(ValueError, match="claims:"):
        check_claims(repo)


def test_markers_inside_prose_are_not_markers(repo):
    prose = f"Fill the block between `{BEGIN}` and `{END}`.\n\n{BEGIN}\n{END}\n"
    (repo / "README.md").write_text(prose)
    write_claims(repo)
    assert check_claims(repo) == []
    assert (repo / "README.md").read_text().startswith(f"Fill the block between `{BEGIN}`")


def test_marker_lines_tolerate_trailing_whitespace_but_not_trailing_text(repo):
    (repo / "README.md").write_text(f"{BEGIN}  \n{END}\t\n")
    write_claims(repo)
    assert check_claims(repo) == []
    (repo / "README.md").write_text(f"{BEGIN} and more\n{END}\n")
    with pytest.raises(ValueError, match="claims:begin"):
        write_claims(repo)


def test_missing_metrics_is_a_message_not_an_exception(repo):
    (repo / "reports" / "metrics.json").unlink()
    messages = check_claims(repo)
    assert len(messages) == 1
    assert "metrics.json: not found" in messages[0]


def test_invalid_metrics_is_a_message_not_an_exception(repo):
    path = repo / "reports" / "metrics.json"
    data = json.loads(path.read_text())
    data["surprise"] = 1
    path.write_text(json.dumps(data))
    messages = check_claims(repo)
    assert len(messages) == 1
    assert "metrics.json: invalid" in messages[0]
    assert "surprise" in messages[0]


def test_missing_mlflow_csv_is_a_message_not_an_exception(repo):
    (repo / "reports" / "mlflow_runs.csv").unlink()
    messages = check_claims(repo)
    assert len(messages) == 1
    assert "mlflow_runs.csv: not found" in messages[0]


def _raw() -> dict[str, Any]:
    return tiny_metrics().model_dump(mode="json")


def _headline_auc_disagrees(raw: dict[str, Any]) -> None:
    raw["headline"]["auc_pr"] = 0.90


def _no_models(raw: dict[str, Any]) -> None:
    raw["models"] = {}


def _headline_model_unknown(raw: dict[str, Any]) -> None:
    raw["headline"]["model"] = "ghost"


def _headline_protocol_missing(raw: dict[str, Any]) -> None:
    del raw["models"][HEADLINE]["macro"]["event_level"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            _headline_auc_disagrees,
            "headline.auc_pr 0.9 differs from models['cnn_ae'].macro['event_level'].auc_pr 0.8234",
            id="headline-auc-pr-disagrees",
        ),
        pytest.param(_no_models, "models must not be empty", id="no-models"),
        pytest.param(
            _headline_model_unknown, "headline.model 'ghost' is not in models", id="unknown-model"
        ),
        pytest.param(
            _headline_protocol_missing,
            "models['cnn_ae'] has no macro score for headline.protocol 'event_level'",
            id="headline-protocol-missing",
        ),
    ],
)
def test_contradictory_metrics_are_rejected(mutate, message):
    raw = _raw()
    mutate(raw)
    with pytest.raises(ValidationError) as excinfo:
        Metrics.model_validate(raw)
    assert message in str(excinfo.value)


def test_headline_auc_pr_must_match_to_a_tight_tolerance():
    raw = _raw()
    raw["headline"]["auc_pr"] += 1e-12
    Metrics.model_validate(raw)  # float noise is fine
    raw["headline"]["auc_pr"] += 1e-6
    with pytest.raises(ValidationError, match=r"headline\.auc_pr"):
        Metrics.model_validate(raw)


def test_the_headline_must_be_the_model_with_the_highest_validation_objective():
    raw = _raw()
    raw["models"]["zscore"]["validation_objective"] = 0.99
    with pytest.raises(ValidationError) as excinfo:
        Metrics.model_validate(raw)
    assert "headline.model 'cnn_ae' is not the model with the highest validation objective" in str(
        excinfo.value
    )
    assert "'zscore' (0.99)" in str(excinfo.value)


def test_a_tie_for_the_best_validation_objective_goes_to_the_earlier_registry_name():
    raw = _raw()
    raw["models"]["lstm_ae"]["validation_objective"] = raw["models"][HEADLINE][
        "validation_objective"
    ]
    Metrics.model_validate(raw)  # cnn_ae and lstm_ae tie, and cnn_ae comes first
    raw["headline"]["model"] = "lstm_ae"
    raw["headline"]["auc_pr"] = raw["models"]["lstm_ae"]["macro"]["event_level"]["auc_pr"]
    with pytest.raises(ValidationError, match="not the model with the highest validation"):
        Metrics.model_validate(raw)


def test_a_negative_count_of_label_reads_is_rejected():
    raw = _raw()
    raw["headline"]["label_reads_at_selection"] = -1
    with pytest.raises(ValidationError, match=r"label_reads_at_selection must not be negative"):
        Metrics.model_validate(raw)


def test_best_model_is_the_highest_validation_objective():
    objective = {"zscore": 0.2, "iforest": 0.3, "cnn_ae": 0.9, "lstm_ae": 0.5}
    assert best_model(objective) == "cnn_ae"


def test_a_tie_for_the_best_model_goes_to_the_earlier_registry_name():
    tied = {"lstm_ae": 0.7, "cnn_ae": 0.7, "iforest": 0.1, "zscore": 0.2}  # not in registry order
    assert best_model(tied) == "cnn_ae"
    assert best_model({name: 0.5 for name in reversed(MODEL_NAMES)}) == MODEL_NAMES[0]


def test_best_model_ranks_a_name_outside_the_registry_after_the_registered_ones():
    assert best_model({"zzz": 0.5, "aaa": 0.5, "lstm_ae": 0.5}) == "lstm_ae"
    assert best_model({"zzz": 0.5, "aaa": 0.5}) == "aaa"
    assert best_model({"zzz": 0.9, "lstm_ae": 0.5}) == "zzz"


def test_check_claims_reports_a_self_contradictory_metrics_file_as_invalid(repo):
    raw = _raw()
    _headline_auc_disagrees(raw)
    (repo / "reports" / "metrics.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=r"headline\.auc_pr"):
        read_metrics(repo / "reports" / "metrics.json")
    messages = check_claims(repo)
    assert len(messages) == 1
    assert messages[0].startswith("reports/metrics.json: invalid")
    assert "headline.auc_pr" in messages[0]


def _nan_threshold(raw: dict[str, Any], bad: float) -> None:
    raw["models"]["zscore"]["per_entity"][0]["threshold"] = bad


def _nan_in_a_list(raw: dict[str, Any], bad: float) -> None:
    raw["ablation"]["zscore"]["f1_with"][0] = bad


def _nan_in_the_interval(raw: dict[str, Any], bad: float) -> None:
    raw["headline"]["ci95"][1] = bad


def _nan_in_params(raw: dict[str, Any], bad: float) -> None:
    raw["models"]["zscore"]["params"]["learning_rate"] = bad


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize(
    "put",
    [_nan_threshold, _nan_in_a_list, _nan_in_the_interval, _nan_in_params],
    ids=["float-field", "list-item", "tuple-item", "params-union"],
)
def test_non_finite_numbers_fail_validation_with_a_clear_message(put, bad):
    raw = _raw()
    put(raw, bad)
    with pytest.raises(ValidationError, match="finite number"):
        Metrics.model_validate(raw)


def test_a_nan_in_metrics_json_is_reported_as_invalid(repo):
    raw = _raw()
    _nan_threshold(raw, math.nan)
    (repo / "reports" / "metrics.json").write_text(json.dumps(raw))  # json writes the token NaN
    messages = check_claims(repo)
    assert len(messages) == 1
    assert "metrics.json: invalid" in messages[0]
    assert "finite number" in messages[0]


def test_bootstrap_ci_of_two_points_is_hand_checkable():
    # The mean of a resample of [0, 1] is 0, 0.5 or 1 with probability 1/4, 1/2, 1/4, so the
    # 2.5th percentile of 1000 resampled means is 0 and the 97.5th is 1.
    assert bootstrap_ci([0.0, 1.0], seed=0) == (0.0, 1.0)


def test_bootstrap_ci_is_the_2_5_and_97_5_percentile_of_1000_resampled_means():
    values = [1.0, 2.0, 4.0]
    picks = np.random.default_rng(5).integers(0, len(values), size=(1000, len(values)))
    means = np.sort(np.asarray(values)[picks].mean(axis=1))

    def at(percent: float) -> float:  # linear interpolation between order statistics, written out
        rank = percent / 100 * (len(means) - 1)
        below = int(rank)
        return float(means[below] + (rank - below) * (means[below + 1] - means[below]))

    assert bootstrap_ci(values, seed=5, n=1000) == pytest.approx((at(2.5), at(97.5)), rel=1e-12)
    assert bootstrap_ci(values, seed=5) == bootstrap_ci(values, seed=5, n=1000)


def _stale_metrics(root):
    raw = json.loads((root / "reports" / "metrics.json").read_text())
    raw["headline"]["auc_pr"] = raw["models"][HEADLINE]["macro"]["event_level"]["auc_pr"] = 0.9123
    write_metrics(root / "reports" / "metrics.json", Metrics.model_validate(raw))


def _stale_latency(root):
    write_latency(root / "reports" / "latency.json", 45.67)


def _removed_latency(root):
    (root / "reports" / "latency.json").unlink()


def _more_runs(root):
    write_runs_csv(root / "reports" / "mlflow_runs.csv", 33)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        pytest.param(
            _stale_metrics,
            ["claims.json: headline_auc_pr is 0.82, metrics.json gives 0.91", STALE_README],
            id="metrics-json",
        ),
        pytest.param(
            _stale_latency,
            ["claims.json: p95_ms is 12.30, latency.json gives 45.67", STALE_README],
            id="latency-json",
        ),
        pytest.param(
            _removed_latency,
            ["claims.json: p95_ms is 12.30, latency.json gives null", STALE_README],
            id="latency-json-removed",
        ),
        pytest.param(
            _more_runs,
            ["claims.json: mlflow_runs is 32, mlflow_runs.csv gives 33", STALE_README],
            id="mlflow-runs-csv",
        ),
    ],
)
def test_a_source_changed_after_write_claims_is_reported(repo, change, expected):
    write_claims(repo)
    assert check_claims(repo) == []
    change(repo)
    assert check_claims(repo) == expected


def _with(metrics: Metrics, **changes: Any) -> Metrics:
    """Copy ``metrics`` with fields replaced and NOT validated, as only a hand edit could."""
    return metrics.model_copy(update=changes)


def _smoke_profile(metrics: Metrics) -> Metrics:
    return _with(metrics, profile="smoke")


def _two_entities(metrics: Metrics) -> Metrics:
    return _with(metrics, dataset=metrics.dataset.model_copy(update={"entities": 2}))


def _one_ablation(metrics: Metrics) -> Metrics:
    return _with(metrics, ablation={"zscore": metrics.ablation["zscore"]})


def _extra_ablation(metrics: Metrics) -> Metrics:
    return _with(metrics, ablation={**metrics.ablation, "cnn_ae": metrics.ablation["zscore"]})


def _labels_read(metrics: Metrics) -> Metrics:
    headline = metrics.headline.model_copy(update={"label_reads_at_selection": 3})
    return _with(metrics, headline=headline)


def _wrong_headline(metrics: Metrics) -> Metrics:
    return _with(metrics, headline=metrics.headline.model_copy(update={"model": "zscore"}))


PUBLISHABLE_PROBLEMS = [
    pytest.param(
        _smoke_profile, "profile is 'smoke', not 'full': only a full run is publishable", id="smoke"
    ),
    pytest.param(
        _two_entities,
        f"dataset.entities is 2, a full run covers all {len(ENTITIES)}",
        id="entities",
    ),
    pytest.param(
        _one_ablation,
        f"ablation covers ['zscore'], a full run covers {sorted(FEATURE_MODELS)}",
        id="ablation-missing",
    ),
    pytest.param(
        _extra_ablation,
        f"ablation covers ['cnn_ae', 'iforest', 'zscore'], a full run covers "
        f"{sorted(FEATURE_MODELS)}",
        id="ablation-extra",
    ),
    pytest.param(
        _labels_read,
        "headline.label_reads_at_selection is 3, not 0: a test label was read before the headline "
        "was chosen",
        id="label-reads",
    ),
    pytest.param(
        _wrong_headline,
        "headline.model is 'zscore' but 'cnn_ae' has the highest validation objective",
        id="headline-not-the-argmax",
    ),
]


def test_a_full_run_has_no_publishable_problems():
    assert publishable_problems(tiny_metrics()) == []


@pytest.mark.parametrize(("break_it", "problem"), PUBLISHABLE_PROBLEMS)
def test_each_unpublishable_condition_is_named_alone(break_it, problem):
    assert publishable_problems(break_it(tiny_metrics())) == [problem]


def test_every_problem_is_listed():
    metrics = tiny_metrics()
    for break_it in (_smoke_profile, _two_entities, _one_ablation, _labels_read):
        metrics = break_it(metrics)
    assert len(publishable_problems(metrics)) == 4


# the argmax rule is also a validation error, so a file that breaks it cannot reach check_claims
FILE_PROBLEMS = [param for param in PUBLISHABLE_PROBLEMS if param.id != "headline-not-the-argmax"]


@pytest.mark.parametrize(("break_it", "problem"), FILE_PROBLEMS)
def test_write_claims_refuses_an_unpublishable_run_and_writes_nothing(repo, break_it, problem):
    write_metrics(repo / "reports" / "metrics.json", break_it(tiny_metrics()))
    with pytest.raises(ValueError, match="not publishable") as raised:
        write_claims(repo)
    assert problem in str(raised.value)
    assert not (repo / "reports" / "claims.json").exists()
    assert (repo / "README.md").read_text() == README_TEXT


@pytest.mark.parametrize(("break_it", "problem"), FILE_PROBLEMS)
def test_check_claims_names_an_unpublishable_run(repo, break_it, problem):
    write_metrics(repo / "reports" / "metrics.json", break_it(tiny_metrics()))
    assert check_claims(repo) == [f"reports/metrics.json: not publishable: {problem}"]


def test_check_claims_gives_one_message_per_problem(repo):
    metrics = tiny_metrics()
    for break_it in (_smoke_profile, _two_entities, _labels_read):
        metrics = break_it(metrics)
    write_metrics(repo / "reports" / "metrics.json", metrics)
    messages = check_claims(repo)
    assert len(messages) == 3
    assert all(m.startswith("reports/metrics.json: not publishable: ") for m in messages)


def test_a_published_file_that_is_not_the_best_model_is_invalid_not_publishable(repo):
    write_metrics(repo / "reports" / "metrics.json", _wrong_headline(tiny_metrics()))
    messages = check_claims(repo)
    assert len(messages) == 1
    assert messages[0].startswith("reports/metrics.json: invalid")
    assert "highest validation objective" in messages[0]


@pytest.mark.parametrize("module", [report_module, claims_module], ids=["report", "claims"])
def test_modules_hold_no_mutable_global_state(module):
    mutable = [
        name
        for name, value in vars(module).items()
        if not name.startswith("__") and isinstance(value, dict | list | set)
    ]
    assert mutable == []
