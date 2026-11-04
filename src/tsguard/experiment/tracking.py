"""Experiment tracking: where tuning trials and final runs are recorded.

`Tracker` is the one-method interface the search code logs through. `MemoryTracker` keeps runs
in a list, so a test can read back what was logged. `MlflowTracker` writes to an MLflow store
and `export_runs` flattens that store into a CSV that the report reads, so the report never
depends on MLflow being installed.

MLflow is imported when it is first needed, so a `MemoryTracker` works without the
``experiment`` extra. The tracker talks to the store through a client bound to its own URI and
never calls the process-global ``mlflow.set_tracking_uri``.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from mlflow import MlflowClient
    from mlflow.entities import Run

PARAMS_JSON: Final = "params_json"
"""Name of the MLflow param that holds all the parameters of a run as canonical JSON, so their
types survive the string-only param store."""

CSV_COLUMNS: Final[tuple[str, ...]] = (
    "run_id",
    "experiment",
    "run_name",
    "kind",
    "model",
    "objective",
    PARAMS_JSON,
)
"""Columns of the CSV `export_runs` writes."""

_PAGE_SIZE: Final = 500


class Tracker(Protocol):
    """Records one run of an experiment."""

    def log_run(
        self,
        *,
        experiment: str,
        run_name: str,
        params: Mapping[str, object],
        metrics: Mapping[str, float],
        tags: Mapping[str, str],
    ) -> str:
        """Record a finished run.

        Args:
            experiment: Name of the experiment the run belongs to; created on demand.
            run_name: Human-readable name of the run.
            params: Hyper-parameters of the run.
            metrics: Final metric values of the run.
            tags: Labels of the run, such as ``kind`` and ``model``.

        Returns:
            The id of the recorded run.
        """
        ...


class MemoryTracker:
    """A `Tracker` that keeps its runs in memory.

    Attributes:
        runs: One dict per run, in logging order, with the keys ``run_id``, ``experiment``,
            ``run_name``, ``params``, ``metrics`` and ``tags``.
    """

    def __init__(self) -> None:
        """Create a tracker with no runs."""
        self.runs: list[dict[str, object]] = []

    def log_run(
        self,
        *,
        experiment: str,
        run_name: str,
        params: Mapping[str, object],
        metrics: Mapping[str, float],
        tags: Mapping[str, str],
    ) -> str:
        """Append a copy of the run to `runs`.

        Args:
            experiment: Name of the experiment the run belongs to.
            run_name: Human-readable name of the run.
            params: Hyper-parameters of the run.
            metrics: Final metric values of the run.
            tags: Labels of the run.

        Returns:
            The run id, ``"mem-<n>"`` for the n-th run (counting from 0).
        """
        run_id = f"mem-{len(self.runs)}"
        self.runs.append(
            {
                "run_id": run_id,
                "experiment": experiment,
                "run_name": run_name,
                "params": dict(params),
                "metrics": dict(metrics),
                "tags": dict(tags),
            }
        )
        return run_id


def _client(tracking_uri: str) -> MlflowClient:
    """Return a client bound to ``tracking_uri``, without touching MLflow's global state."""
    try:
        from mlflow import MlflowClient
    except ImportError as exc:
        raise ImportError(
            "MLflow tracking needs the 'experiment' extra: uv sync --extra experiment"
        ) from exc
    return MlflowClient(tracking_uri=tracking_uri)


def _canonical_json(value: object) -> str:
    """Serialize ``value`` as JSON with sorted keys and no insignificant whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class MlflowTracker:
    """A `Tracker` that writes runs to an MLflow tracking store.

    Each run keeps every param as a string for the MLflow UI, plus one param, ``params_json``,
    with all params as canonical JSON, which `export_runs` reads back with the original types.
    """

    def __init__(self, tracking_uri: str) -> None:
        """Connect to a tracking store.

        Args:
            tracking_uri: MLflow tracking URI, such as ``sqlite:///mlruns.db``.
        """
        self._client = _client(tracking_uri)
        self._experiment_ids: dict[str, str] = {}

    def log_run(
        self,
        *,
        experiment: str,
        run_name: str,
        params: Mapping[str, object],
        metrics: Mapping[str, float],
        tags: Mapping[str, str],
    ) -> str:
        """Create a run, log it in one batch and close it.

        A run that cannot be logged (a param over MLflow's length limit, for instance) is
        closed as FAILED before the error propagates, so the store never shows it as running.

        Args:
            experiment: Name of the experiment the run belongs to; created if it does not
                exist yet.
            run_name: Human-readable name of the run.
            params: Hyper-parameters of the run; values must be JSON-serializable.
            metrics: Final metric values of the run.
            tags: Labels of the run.

        Returns:
            The MLflow run id.

        Raises:
            ValueError: ``params`` has a param named ``params_json``, which the tracker
                reserves, or serializes to more characters than MLflow keeps (it would
                truncate them without an error). No run is created in either case.
        """
        from mlflow.entities import Metric, Param
        from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH

        if PARAMS_JSON in params:
            raise ValueError(
                f"run {run_name!r}: {PARAMS_JSON!r} is reserved for the JSON copy of the "
                f"params and cannot be a param name"
            )
        params_json = _canonical_json(dict(params))
        if len(params_json) > MAX_PARAM_VAL_LENGTH:
            raise ValueError(
                f"run {run_name!r}: the params serialize to {len(params_json)} characters, "
                f"over the {MAX_PARAM_VAL_LENGTH} MLflow keeps; it would truncate them silently"
            )
        run = self._client.create_run(
            self._experiment_id(experiment), run_name=run_name, tags=dict(tags)
        )
        run_id: str = run.info.run_id
        stamp = int(time.time() * 1000)
        try:
            self._client.log_batch(
                run_id,
                metrics=[Metric(key, float(value), stamp, 0) for key, value in metrics.items()],
                params=[
                    *(Param(key, str(value)) for key, value in params.items()),
                    Param(PARAMS_JSON, params_json),
                ],
            )
        except BaseException:
            self._client.set_terminated(run_id, status="FAILED")
            raise
        self._client.set_terminated(run_id, status="FINISHED")
        return run_id

    def _experiment_id(self, name: str) -> str:
        """Return the id of the experiment ``name``, creating it on first use."""
        if name not in self._experiment_ids:
            existing = self._client.get_experiment_by_name(name)
            if existing is not None:
                self._experiment_ids[name] = existing.experiment_id
            else:
                self._experiment_ids[name] = self._client.create_experiment(name)
        return self._experiment_ids[name]


def export_runs(tracking_uri: str, path: Path, *, experiments: Sequence[str]) -> int:
    """Write the runs of some experiments to a CSV file.

    The columns are `CSV_COLUMNS`. Rows are sorted by experiment name and then run name, so
    the file does not depend on the order the runs were logged or the store returns them in.
    ``params_json`` is canonical JSON of the params as they were logged, with their types.

    Args:
        tracking_uri: MLflow tracking URI of the store to read.
        path: The CSV file to write; its directory is created if missing.
        experiments: Names of the experiments to export.

    Returns:
        The number of rows written.

    Raises:
        ValueError: An experiment does not exist, or a run lacks the ``kind`` and ``model``
            tags, the ``objective`` metric or the ``params_json`` param that `MlflowTracker`
            logs. Nothing is written in either case.
    """
    client = _client(tracking_uri)
    rows: list[dict[str, str]] = []
    for name in experiments:
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            raise ValueError(f"experiment {name!r} does not exist in the store {tracking_uri!r}")
        token: str | None = None
        while True:
            page = client.search_runs(
                [experiment.experiment_id],
                order_by=["attributes.start_time ASC"],
                max_results=_PAGE_SIZE,
                page_token=token,
            )
            for run in page:
                rows.append(_row(name, run))
            token = page.token
            if not token:
                break
    rows.sort(key=lambda row: (row["experiment"], row["run_name"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _row(experiment: str, run: Run) -> dict[str, str]:
    """Flatten one MLflow run into a CSV row, naming the run if anything is missing."""
    info, data = run.info, run.data
    name = info.run_name
    missing = [
        f"{kind} {key!r}"
        for kind, key, present in (
            ("tag", "kind", "kind" in data.tags),
            ("tag", "model", "model" in data.tags),
            ("metric", "objective", "objective" in data.metrics),
            ("param", PARAMS_JSON, PARAMS_JSON in data.params),
        )
        if not present
    ]
    if missing:
        raise ValueError(
            f"run {name!r} ({info.run_id}) of experiment {experiment!r} was not logged by "
            f"MlflowTracker: it has no {', no '.join(missing)}"
        )
    return {
        "run_id": info.run_id,
        "experiment": experiment,
        "run_name": name,
        "kind": data.tags["kind"],
        "model": data.tags["model"],
        "objective": repr(float(data.metrics["objective"])),
        PARAMS_JSON: _canonical_json(json.loads(data.params[PARAMS_JSON])),
    }
