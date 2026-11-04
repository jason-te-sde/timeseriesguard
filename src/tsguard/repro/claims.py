"""Every number the README states, derived from ``reports/`` and checked against it.

``reports/metrics.json`` (written by ``reproduce``), ``reports/mlflow_runs.csv`` and the optional
``reports/latency.json`` are the sources. `derive_claims` reduces them to a small `Claims`
record; `write_claims` stores that record as ``reports/claims.json`` and renders it into the
README between two marker lines; `check_claims` recomputes everything and names each place where
a stored file disagrees. A number is therefore never typed into the README by hand, and CI can
fail when one drifts.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Collection
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from tsguard.eval.protocols import PROTOCOLS
from tsguard.repro.report import AblationModel, Metrics, read_metrics

BEGIN: Final = "<!-- claims:begin -->"
"""The README line that opens the generated block."""

END: Final = "<!-- claims:end -->"
"""The README line that closes the generated block."""

P95_BUDGET_MS: Final = 50.0
"""The per-request p95 latency the serving claim is measured against, in milliseconds."""

MIN_MLFLOW_RUNS: Final = 30
"""The fewest tracked runs a reported experiment may rest on."""

_MODEL_ORDER: Final = ("zscore", "iforest", "cnn_ae", "lstm_ae")
_REPORTS: Final = "reports"
_RUN_ID_COLUMN: Final = "run_id"
_STALE_README: Final = "README block differs from metrics.json; run `tsguard claims --write`"
_SOURCE_OF: Final = {
    "mlflow_runs": "mlflow_runs.csv",
    "p95_ms": "latency.json",
    "p95_budget_ms": "P95_BUDGET_MS",
}
_BEGIN_LINE: Final = re.compile(rf"^{re.escape(BEGIN)}(?=[ \t]*$)", re.MULTILINE)
_END_LINE: Final = re.compile(rf"^{re.escape(END)}(?=[ \t]*$)", re.MULTILINE)


@dataclass(frozen=True)
class Claims:
    """The numbers the README states.

    Attributes:
        train_steps: Training rows over all entities.
        test_steps: Test rows over all entities.
        channels: Channels per entity.
        feature_specs: Registered feature specs.
        models: Number of detectors compared.
        headline_model: The detector reported as the headline.
        headline_auc_pr: Its macro event-level AUC-PR, rounded to 2 decimals.
        headline_ci95: The bootstrap 95 % interval of that number, rounded to 2 decimals.
        fft_gain: Per detector, the effect of the FFT features on event-level F1 over seeds:
            ``f1_points`` and ``f1_relative_pct`` (means) with their ``_std`` counterparts, each
            rounded to 1 decimal.
        mlflow_runs: Tracked runs in ``mlflow_runs.csv``.
        p95_ms: The measured p95 latency, or None when ``latency.json`` is absent.
        p95_budget_ms: The latency budget the measurement is compared with.
    """

    train_steps: int
    test_steps: int
    channels: int
    feature_specs: int
    models: int
    headline_model: str
    headline_auc_pr: float
    headline_ci95: tuple[float, float]
    fft_gain: dict[str, dict[str, float]]
    mlflow_runs: int
    p95_ms: float | None
    p95_budget_ms: float


def _round(value: float, digits: int) -> float:
    """Round, and turn ``-0.0`` into ``0.0`` so that a tiny negative never prints a sign."""
    return round(value, digits) + 0.0


def _fft_gain(ablation: AblationModel) -> dict[str, float]:
    return {
        "f1_points": _round(ablation.delta_f1_points_mean, 1),
        "f1_points_std": _round(ablation.delta_f1_points_std, 1),
        "f1_relative_pct": _round(ablation.delta_f1_relative_mean, 1),
        "f1_relative_pct_std": _round(ablation.delta_f1_relative_std, 1),
    }


def _count_runs(path: Path) -> int:
    """Count the data rows of an exported MLflow CSV (quoted fields may span lines)."""
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None or _RUN_ID_COLUMN not in header:
                raise ValueError(
                    f"{path.name}: expected a header row with a {_RUN_ID_COLUMN} column"
                )
            return sum(1 for row in reader if any(cell.strip() for cell in row))
    except csv.Error as exc:
        raise ValueError(f"{path.name}: not a readable CSV: {exc}") from exc


def _read_p95(path: Path) -> float:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name}: not valid JSON: {exc}") from exc
    value = data.get("p95_ms") if isinstance(data, dict) else None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{path.name}: expected a finite number under p95_ms, got {value!r}")
    return float(value)


def derive_claims(metrics: Metrics, *, mlflow_csv: Path, latency_json: Path | None) -> Claims:
    """Reduce the report files to the numbers the README states.

    Args:
        metrics: The validated ``metrics.json``.
        mlflow_csv: The exported MLflow runs; every row after the header is one run.
        latency_json: ``{"p95_ms": ...}`` from the latency benchmark. None, or a path that does
            not exist, means the benchmark was not run and gives ``p95_ms = None``.

    Returns:
        The claims.

    Raises:
        ValueError: The CSV has no header row with a ``run_id`` column, or the latency file is
            not a JSON object with a finite ``p95_ms``; the message names the file.
        FileNotFoundError: ``mlflow_csv`` does not exist.
    """
    low, high = metrics.headline.ci95
    p95_ms = None
    if latency_json is not None and latency_json.exists():
        p95_ms = _read_p95(latency_json)
    return Claims(
        train_steps=metrics.dataset.train_steps,
        test_steps=metrics.dataset.test_steps,
        channels=metrics.dataset.channels,
        feature_specs=metrics.feature_specs,
        models=len(metrics.models),
        headline_model=metrics.headline.model,
        headline_auc_pr=_round(metrics.headline.auc_pr, 2),
        headline_ci95=(_round(low, 2), _round(high, 2)),
        fft_gain={name: _fft_gain(metrics.ablation[name]) for name in sorted(metrics.ablation)},
        mlflow_runs=_count_runs(mlflow_csv),
        p95_ms=p95_ms,
        p95_budget_ms=P95_BUDGET_MS,
    )


def _ordered(names: Collection[str]) -> list[str]:
    """Return the known detectors in report order, then any others sorted."""
    known = [name for name in _MODEL_ORDER if name in names]
    return known + sorted(set(names) - set(_MODEL_ORDER))


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _signed(value: float, digits: int) -> str:
    return f"{_round(value, digits):+.{digits}f}"


def _claims_rows(claims: Claims) -> list[str]:
    low, high = claims.headline_ci95
    if claims.p95_ms is None:
        latency = f"not measured (budget {claims.p95_budget_ms:g} ms)"
    else:
        latency = f"{claims.p95_ms:.1f} ms (budget {claims.p95_budget_ms:g} ms)"
    return [
        _row(["Claim", "Value"]),
        _row(["---", "---"]),
        _row(["Training steps", f"{claims.train_steps:,}"]),
        _row(["Test steps", f"{claims.test_steps:,}"]),
        _row(["Channels", str(claims.channels)]),
        _row(["Feature specs", str(claims.feature_specs)]),
        _row(["Models compared", str(claims.models)]),
        _row(["Headline model", claims.headline_model]),
        _row(
            [
                "Headline event-level AUC-PR",
                f"{claims.headline_auc_pr:.2f} (95% CI {low:.2f} to {high:.2f})",
            ]
        ),
        _row(["MLflow runs", str(claims.mlflow_runs)]),
        _row(["p95 inference latency", latency]),
    ]


def _protocol_rows(claims: Claims, metrics: Metrics) -> list[str]:
    rows = [_row(["Model", *PROTOCOLS]), _row(["---", *["---:"] * len(PROTOCOLS)])]
    for name in _ordered(metrics.models):
        macro = metrics.models[name].macro
        cells = [name]
        for protocol in PROTOCOLS:
            if protocol not in macro:
                raise ValueError(f"model {name!r} has no macro score for protocol {protocol!r}")
            cells.append(f"{macro[protocol].auc_pr:.2f}")
        if name == claims.headline_model:
            cells = [f"**{cell}**" for cell in cells]
        rows.append(_row(cells))
    return rows


def _ablation_rows(claims: Claims, metrics: Metrics) -> list[str]:
    rows = [
        _row(["Model", "ΔF1 (points)", "ΔF1 (relative %)", "ΔAUC-PR"]),
        _row(["---", "---:", "---:", "---:"]),
    ]
    for name in _ordered(claims.fft_gain):
        gain = claims.fft_gain[name]
        ablation = metrics.ablation.get(name)
        if ablation is None:
            raise ValueError(f"claims name an FFT ablation for {name!r}, metrics.json has none")
        rows.append(
            _row(
                [
                    name,
                    f"{_signed(gain['f1_points'], 1)} ± {gain['f1_points_std']:.1f}",
                    f"{_signed(gain['f1_relative_pct'], 1)} ± {gain['f1_relative_pct_std']:.1f}",
                    f"{_signed(ablation.delta_auc_pr_mean, 3)} ± {ablation.delta_auc_pr_std:.3f}",
                ]
            )
        )
    return rows


def render_readme_block(claims: Claims, metrics: Metrics) -> str:
    """Render the generated README block, from the begin marker to the end marker.

    The block holds three tables: the claims (claim, value), the macro AUC-PR of every model
    under every protocol with the headline model's row in bold, and the FFT ablation. The output
    depends only on the arguments, so rendering twice gives the same text.

    Args:
        claims: The derived claims.
        metrics: The metrics ``claims`` was derived from.

    Returns:
        The block, starting with `BEGIN` and ending with `END`, without a trailing newline.

    Raises:
        ValueError: A model lacks a score for one of the protocols, or the claims name an
            ablation that ``metrics`` does not have.
    """
    lines = [
        BEGIN,
        *_claims_rows(claims),
        "",
        "Macro-mean AUC-PR over entities under each evaluation protocol (headline model in bold):",
        "",
        *_protocol_rows(claims, metrics),
        "",
        "Effect of the FFT features on event-level F1 and AUC-PR (mean ± sample std over seeds):",
        "",
        *_ablation_rows(claims, metrics),
        END,
    ]
    return "\n".join(lines)


def _block_span(text: str) -> tuple[int, int]:
    """Locate the generated block: from the start of the begin line to the end of the end line.

    The markers count only when they are alone on their line, so prose that mentions them in
    backticks does not confuse the lookup.
    """
    begins = list(_BEGIN_LINE.finditer(text))
    ends = list(_END_LINE.finditer(text))
    for marker, found in ((BEGIN, begins), (END, ends)):
        if len(found) != 1:
            raise ValueError(
                f"README.md: expected exactly one line holding {marker}, found {len(found)}"
            )
    if ends[0].start() < begins[0].end():
        raise ValueError(f"README.md: {END} comes before {BEGIN}")
    return begins[0].start(), ends[0].end()


def _claims_json(claims: Claims) -> str:
    return json.dumps(asdict(claims), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _derive_from_reports(reports: Path, metrics: Metrics) -> Claims:
    return derive_claims(
        metrics,
        mlflow_csv=reports / "mlflow_runs.csv",
        latency_json=reports / "latency.json",
    )


def write_claims(root: Path) -> None:
    """Write ``reports/claims.json`` and refresh the README block.

    Nothing is written unless both can be: the README is checked for its markers first.

    Args:
        root: The repository root, holding ``README.md`` and ``reports/``.

    Raises:
        ValueError: The README lacks the begin or end marker line, has one twice, or has them
            in the wrong order; or an input file is malformed (see `derive_claims`).
        FileNotFoundError: ``reports/metrics.json``, ``reports/mlflow_runs.csv`` or the README
            does not exist.
    """
    reports = root / _REPORTS
    metrics = read_metrics(reports / "metrics.json")
    claims = _derive_from_reports(reports, metrics)
    readme_path = root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    start, end = _block_span(readme)
    updated = readme[:start] + render_readme_block(claims, metrics) + readme[end:]
    (reports / "claims.json").write_text(_claims_json(claims), encoding="utf-8")
    readme_path.write_text(updated, encoding="utf-8")


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    """Flatten nested dicts to dotted paths; lists and scalars are leaves."""
    if not isinstance(value, dict):
        return {prefix: value}
    flat: dict[str, object] = {}
    for key in sorted(value):
        flat.update(_flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
    return flat


def _show(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}" if round(value, 2) == value else repr(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_show(item) for item in value) + "]"
    return json.dumps(value)


def _compare_claims_json(path: Path, claims: Claims) -> list[str]:
    if not path.exists():
        return [f"{_REPORTS}/claims.json: not found; run `tsguard claims --write`"]
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{_REPORTS}/claims.json: not valid JSON: {exc}"]
    if not isinstance(stored, dict):
        return [f"{_REPORTS}/claims.json: expected a JSON object, got {type(stored).__name__}"]
    got = _flatten(stored)
    want = _flatten(json.loads(json.dumps(asdict(claims))))
    messages = []
    for field in sorted(got.keys() | want.keys()):
        source = _SOURCE_OF.get(field.split(".")[0], "metrics.json")
        if field not in want:
            messages.append(f"claims.json: unexpected field {field}")
        elif field not in got:
            messages.append(f"claims.json: {field} is missing, {source} gives {_show(want[field])}")
        elif got[field] != want[field]:
            messages.append(
                f"claims.json: {field} is {_show(got[field])}, {source} gives {_show(want[field])}"
            )
    return messages


def _compare_readme(path: Path, claims: Claims, metrics: Metrics) -> list[str]:
    if not path.exists():
        return ["README.md: not found"]
    text = path.read_text(encoding="utf-8")
    start, end = _block_span(text)
    try:
        block = render_readme_block(claims, metrics)
    except ValueError as exc:
        return [f"README.md: the block cannot be rendered from metrics.json: {_one_line(exc)}"]
    return [] if text[start:end] == block else [_STALE_README]


def _one_line(exc: Exception) -> str:
    return " ".join(str(exc).split())


def check_claims(root: Path) -> list[str]:
    """Compare ``reports/claims.json`` and the README block with what the reports give.

    A mismatch is a message, never an exception: the caller decides what to do with it. A source
    file that is missing or malformed ends the check with a message naming it, since nothing can
    be derived without it.

    Args:
        root: The repository root, holding ``README.md`` and ``reports/``.

    Returns:
        One message per problem, empty when everything agrees: each field of ``claims.json``
        that differs (``"claims.json: headline_auc_pr is 0.80, metrics.json gives 0.82"``),
        a stale README block, and fewer than `MIN_MLFLOW_RUNS` tracked runs.

    Raises:
        ValueError: The README lacks the begin or end marker line, has one twice, or has them
            in the wrong order.
    """
    reports = root / _REPORTS
    metrics_path = reports / "metrics.json"
    csv_path = reports / "mlflow_runs.csv"
    missing = [
        f"{_REPORTS}/{path.name}: not found; run `tsguard reproduce`"
        for path in (metrics_path, csv_path)
        if not path.exists()
    ]
    if missing:
        return missing
    try:
        metrics = read_metrics(metrics_path)
    except ValueError as exc:
        return [f"{_REPORTS}/metrics.json: invalid: {_one_line(exc)}"]
    try:
        claims = _derive_from_reports(reports, metrics)
    except ValueError as exc:
        return [f"{_REPORTS}/{exc}"]
    messages = []
    if claims.mlflow_runs < MIN_MLFLOW_RUNS:
        messages.append(
            f"{_REPORTS}/mlflow_runs.csv: {claims.mlflow_runs} runs, "
            f"fewer than the {MIN_MLFLOW_RUNS} the claims require"
        )
    messages += _compare_claims_json(reports / "claims.json", claims)
    return messages + _compare_readme(root / "README.md", claims, metrics)
