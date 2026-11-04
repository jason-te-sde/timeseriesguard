"""Command-line entry point for tsguard.

Five command groups: ``data`` fetches and verifies the SMD files, ``reproduce`` runs a profile
end to end and exports its MLflow runs, ``claims`` checks or rewrites the numbers the README
quotes, ``bundle`` fits one model on one machine and writes it as a checksummed file, and
``serve`` serves such a file over HTTP. Every failure the package raises on purpose is printed as
one line on stderr with exit code 1, and so is a missing ``experiment`` extra (an `ImportError`
from ``reproduce`` that names the extra to install); a usage error exits with 2 (argparse), and
anything else is a bug and keeps its traceback.

Nothing imported here may need MLflow or Optuna when it is imported: the container installs the
runtime dependencies only and starts ``tsguard serve``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from tsguard.data.labels import LabelAudit
from tsguard.data.smd import ENTITIES, EntityData, fetch, load_entity, verify
from tsguard.errors import TsguardError
from tsguard.experiment.reproduce import PROFILES, reproduce
from tsguard.experiment.tracking import MlflowTracker, export_runs
from tsguard.experiment.tuning import FINAL_EXPERIMENT, TUNING_EXPERIMENT
from tsguard.models import MODEL_NAMES
from tsguard.repro.claims import check_claims, write_claims
from tsguard.serve.commands import bundle_command, serve_command

PROG: Final = "tsguard"
SQLITE_PREFIX: Final = "sqlite:///"
EXIT_ERROR: Final = 1
DEFAULT_METRICS: Final = Path("reports") / "metrics.json"
DEFAULT_HOST: Final = "127.0.0.1"
"""Loopback: serving beyond this machine is a choice the caller makes with ``--host``."""
DEFAULT_PORT: Final = 8000


class CliError(TsguardError):
    """An argument that parsed but cannot be used: an unknown entity, a store already there."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tsguard command line.

    Args:
        argv: The arguments after the program name; ``sys.argv[1:]`` if omitted.

    Returns:
        The process exit code: 0 on success, 1 when a command failed with a message (a
        `TsguardError`, or ``claims`` finding a mismatch), and 2 on a usage error, which
        argparse reports by raising ``SystemExit``.
    """
    args = _build_parser().parse_args(argv)
    try:
        code: int = args.handler(args)
    except TsguardError as error:
        print(f"{PROG}: error: {error}", file=sys.stderr)
        return EXIT_ERROR
    return code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG, description="TimeSeriesGuard: anomaly detection on SMD under a fixed protocol."
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    data = commands.add_parser("data", help="fetch or verify the SMD files")
    data_commands = data.add_subparsers(dest="data_command", required=True, metavar="action")
    for action, handler, text in (
        ("fetch", _data_fetch, "download what is missing and verify every file"),
        ("verify", _data_verify, "check the files against the manifest"),
    ):
        sub = data_commands.add_parser(action, help=text)
        sub.add_argument("--entities", nargs="+", metavar="NAME", help="machines; default all 28")
        sub.add_argument("--dest", type=Path, help="data directory; default $TSGUARD_DATA_DIR")
        sub.set_defaults(handler=handler)

    run = commands.add_parser("reproduce", help="run a profile and write metrics.json")
    run.add_argument("--profile", required=True, choices=sorted(PROFILES))
    run.add_argument("--out", type=Path, default=Path("reports"), help="output directory")
    run.add_argument(
        "--tracking-uri",
        help="MLflow tracking URI; default sqlite:///<out>/mlruns.db. A sqlite store that "
        "already exists is refused, because the export reads the whole store",
    )
    run.add_argument(
        "--entities", nargs="+", metavar="NAME", help="evaluate these instead of the profile's"
    )
    run.set_defaults(handler=_reproduce)

    claims = commands.add_parser("claims", help="check or rewrite the numbers the README quotes")
    mode = claims.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="exit 1 if anything is stale")
    mode.add_argument("--write", action="store_true", help="rewrite claims.json and the README")
    claims.add_argument("--root", type=Path, default=Path("."), help="the repository root")
    claims.set_defaults(handler=_claims)

    bundle = commands.add_parser(
        "bundle", help="fit a model on one machine's train rows and write it as a bundle"
    )
    bundle.add_argument("--entity", required=True, metavar="NAME", help="machine, e.g. machine-1-1")
    bundle.add_argument(
        "--model",
        choices=MODEL_NAMES,
        help="detector to fit; default the headline of --metrics, or iforest if there is none",
    )
    bundle.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
        help=f"metrics.json that names the headline and holds the tuned parameters "
        f"(default: {DEFAULT_METRICS})",
    )
    bundle.add_argument(
        "--default-params",
        action="store_true",
        help="use the registry's default hyper-parameters instead of the tuned ones",
    )
    bundle.add_argument(
        "--out", type=Path, required=True, help="bundle file; <out>.sha256 is written beside it"
    )
    bundle.set_defaults(handler=_bundle)

    serve = commands.add_parser("serve", help="serve a bundle over HTTP")
    serve.add_argument(
        "--bundle", type=Path, required=True, help="bundle file, with its .sha256 beside it"
    )
    serve.add_argument(
        "--host", default=DEFAULT_HOST, help=f"address to bind (default: {DEFAULT_HOST})"
    )
    serve.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"port to bind (default: {DEFAULT_PORT})"
    )
    serve.set_defaults(handler=_serve)
    return parser


def _entities(names: Sequence[str] | None) -> tuple[str, ...]:
    """Return the named machines, or all 28; refuse a name that is not one or is repeated."""
    if not names:
        return ENTITIES
    unknown = [name for name in names if name not in ENTITIES]
    if unknown:
        raise CliError(
            f"unknown SMD entity {', '.join(repr(name) for name in unknown)}; "
            f"valid names: {', '.join(ENTITIES)}"
        )
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise CliError(f"--entities repeats {', '.join(repr(name) for name in repeated)}")
    return tuple(names)


def _data_fetch(args: argparse.Namespace) -> int:
    entities = _entities(args.entities)
    paths = fetch(entities, dest=args.dest)
    print(f"{len(paths)} files of {len(entities)} entities are present and verified")
    return 0


def _data_verify(args: argparse.Namespace) -> int:
    entities = _entities(args.entities)
    verify(entities, dest=args.dest)
    print(f"{len(entities)} entities match the manifest: {', '.join(entities)}")
    return 0


def _tracking_uri(uri: str | None, out: Path) -> str:
    """Return the tracking URI to use, refusing a sqlite store that already holds runs."""
    chosen = uri if uri is not None else f"{SQLITE_PREFIX}{out / 'mlruns.db'}"
    if chosen.startswith(SQLITE_PREFIX):
        store = Path(chosen[len(SQLITE_PREFIX) :])
        if store.exists():
            raise CliError(
                f"the tracking store {store} already exists; reproduce exports every run in "
                f"the store, so a second run would duplicate rows: delete it or pass another "
                f"--tracking-uri"
            )
        store.parent.mkdir(parents=True, exist_ok=True)
    return chosen


def _stderr_progress(started: float) -> Callable[[str], None]:
    """Return a callback that prints ``[<UTC time> +<seconds since start>s] <line>`` to stderr."""

    def progress(line: str) -> None:
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[{stamp} +{time.monotonic() - started:.1f}s] {line}", file=sys.stderr, flush=True)

    return progress


def _reproduce(args: argparse.Namespace) -> int:
    profile = PROFILES[args.profile]
    if args.entities:
        # A run on other entities is not the profile's run: its metrics must not pass for it.
        profile = replace(profile, name=f"{profile.name}-custom", entities=_entities(args.entities))
    out: Path = args.out
    uri = _tracking_uri(args.tracking_uri, out)
    started = time.monotonic()

    def loader(name: str, audit: LabelAudit) -> EntityData:
        return load_entity(name, audit=audit)

    csv_path = out / "mlflow_runs.csv"
    try:
        reproduce(
            profile,
            loader=loader,
            tracker=MlflowTracker(uri),
            out_dir=out,
            progress=_stderr_progress(started),
        )
        rows = export_runs(uri, csv_path, experiments=[TUNING_EXPERIMENT, FINAL_EXPERIMENT])
    except ImportError as error:
        # Optuna and MLflow are imported on first use, so a missing `experiment` extra shows up
        # here, as an ImportError whose message names the extra to install.
        raise CliError(" ".join(str(error).split())) from error
    expected = len(MODEL_NAMES) * (profile.n_trials + len(profile.entities))
    if rows != expected:
        raise CliError(
            f"{csv_path} has {rows} runs, expected {expected} ({len(MODEL_NAMES)} models x "
            f"{profile.n_trials} trials + {len(MODEL_NAMES)} models x {len(profile.entities)} "
            f"entities): the tracking store {uri} holds runs from another run, or lost some, "
            f"so the CSV does not describe {out / 'metrics.json'}; use a fresh --tracking-uri"
        )
    print(f"wrote {out / 'metrics.json'} and {csv_path} ({rows} runs)")
    return 0


def _claims(args: argparse.Namespace) -> int:
    root: Path = args.root
    try:
        if args.write:
            write_claims(root)
            print(f"wrote {root / 'reports' / 'claims.json'} and the block in {root / 'README.md'}")
            return 0
        messages = check_claims(root)
    except (ValueError, OSError) as error:
        raise CliError(" ".join(str(error).split())) from error
    for message in messages:
        print(message)
    return EXIT_ERROR if messages else 0


def _bundle(args: argparse.Namespace) -> int:
    (entity,) = _entities([args.entity])
    try:
        summary = bundle_command(
            entity=entity,
            model=args.model,
            metrics_path=args.metrics,
            default_params=args.default_params,
            out=args.out,
        )
    except OSError as error:
        # A TsguardError reaches main as it is; a bare ValueError is a bug and keeps its
        # traceback. What is left to catch here is the file system: no metrics file, an
        # unwritable output, a data file that is not there.
        raise CliError(" ".join(str(error).split())) from error
    print(summary)
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        serve_command(bundle=args.bundle, host=args.host, port=args.port)
    except OSError as error:
        raise CliError(" ".join(str(error).split())) from error
    return 0
