"""Command-line entry point for tsguard.

Three command groups: ``data`` fetches and verifies the SMD files, ``reproduce`` runs a profile
end to end and exports its MLflow runs, and ``claims`` checks or rewrites the numbers the README
quotes. Every failure the package raises on purpose is printed as one line on stderr with exit
code 1; a usage error exits with 2 (argparse), and anything else is a bug and keeps its
traceback.
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
from tsguard.repro.claims import check_claims, write_claims

PROG: Final = "tsguard"
SQLITE_PREFIX: Final = "sqlite:///"
EXIT_ERROR: Final = 1


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
        profile = replace(profile, entities=_entities(args.entities))
    out: Path = args.out
    uri = _tracking_uri(args.tracking_uri, out)
    started = time.monotonic()

    def loader(name: str, audit: LabelAudit) -> EntityData:
        return load_entity(name, audit=audit)

    reproduce(
        profile,
        loader=loader,
        tracker=MlflowTracker(uri),
        out_dir=out,
        progress=_stderr_progress(started),
    )
    csv_path = out / "mlflow_runs.csv"
    rows = export_runs(uri, csv_path, experiments=[TUNING_EXPERIMENT, FINAL_EXPERIMENT])
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
