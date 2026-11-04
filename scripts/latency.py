"""Measure the request latency of a running tsguard service and check it against a budget.

Start the service (``make container-up``, or ``tsguard serve --bundle ...``), then::

    uv run python scripts/latency.py --budget-ms 50 --out artifacts/latency.json

The script asks ``GET /v1/model`` which machine is served and how many rows a request needs,
loads that machine's test rows, and sends ``POST /v1/score`` requests over one keep-alive
connection. Request ``i`` carries the rows that end at row ``train_end + i`` of the machine's
series, wrapping around the test range, so a long run walks through the whole test split. Each
request is timed from the moment the body is ready until the response has been read. The
warm-up requests are sent and timed like the others but left out of the numbers.

The test rows are loaded with a fresh label audit that never leaves its selection phase: the
script reads no label.

The exit code is 1 when the P95 is not below the budget, when any response is not a 200, or
when the service cannot be reached; otherwise 0.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

import httpx
import numpy as np

from tsguard.data.labels import LabelAudit
from tsguard.data.smd import EntityData, load_entity
from tsguard.errors import TsguardError
from tsguard.serve.schemas import ModelInfo

DEFAULT_URL: Final = "http://127.0.0.1:8000"
DEFAULT_N: Final = 2000
DEFAULT_WARMUP: Final = 100
DEFAULT_BUDGET_MS: Final = 50.0
MODEL_PATH: Final = "/v1/model"
SCORE_PATH: Final = "/v1/score"
REQUEST_TIMEOUT_S: Final = 10.0
ERROR_BODY_CHARS: Final = 200
MS_PER_S: Final = 1000.0
RESULT_DECIMALS: Final = 3
"""Milliseconds are reported to the microsecond, finer than a request over a socket can tell."""

Loader = Callable[[str], EntityData]
"""Loads the named machine; injected so a test never reads the dataset."""


class LatencyError(TsguardError):
    """The service answered, or the data it needs is, in a way that makes a run meaningless."""


def percentile_ms(samples_s: Sequence[float], q: float) -> float:
    """Return a percentile of durations given in seconds, in milliseconds.

    Args:
        samples_s: Durations in seconds; at least one.
        q: Percentile from 0 to 100.

    Returns:
        The percentile by linear interpolation between the closest ranks, in milliseconds.

    Raises:
        ValueError: ``samples_s`` is empty.
    """
    if len(samples_s) == 0:
        raise ValueError("no samples to take a percentile of")
    return float(np.percentile(np.asarray(samples_s, dtype=np.float64), q, method="linear")) * (
        MS_PER_S
    )


def run(
    *,
    url: str,
    n: int,
    warmup: int,
    budget_ms: float,
    out: Path | None,
    loader: Loader,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], float] | None = None,
) -> int:
    """Measure a running service and report the percentiles.

    Args:
        url: Base URL of the service.
        n: Requests that count.
        warmup: Requests sent before them whose durations are dropped.
        budget_ms: The P95 must be below this many milliseconds.
        out: Where to write the result as JSON; the result is printed either way.
        loader: Loads a machine's rows by the name the service reports.
        transport: The HTTP transport; the default is a real connection. Tests pass a mock.
        clock: Seconds from an arbitrary origin; ``time.perf_counter`` if omitted.

    Returns:
        0 if the P95 is below the budget, 1 if it is not or the run could not be completed (a
        non-200 response, an unreachable service, a service built for other data); the reason
        is on stderr.
    """
    now = clock if clock is not None else time.perf_counter
    try:
        with httpx.Client(
            base_url=url,
            transport=transport,
            timeout=REQUEST_TIMEOUT_S,
            trust_env=False,  # a proxy in the path would be measured as part of the service
        ) as client:
            samples = _measure(client, n=n, warmup=warmup, loader=loader, clock=now)
    except httpx.HTTPError as error:
        print(f"latency: {url}: {error}", file=sys.stderr)
        return 1
    except (LatencyError, TsguardError, ValueError, OSError) as error:
        print(f"latency: {error}", file=sys.stderr)
        return 1

    result: dict[str, Any] = {
        "n": n,
        "warmup": warmup,
        "p50_ms": _rounded(percentile_ms(samples, 50)),
        "p95_ms": _rounded(percentile_ms(samples, 95)),
        "p99_ms": _rounded(percentile_ms(samples, 99)),
        "max_ms": _rounded(max(samples) * MS_PER_S),
        "budget_ms": budget_ms,
        "url": url,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
    }
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    if result["p95_ms"] >= budget_ms:
        print(
            f"latency: p95 is {result['p95_ms']} ms, not below the {budget_ms} ms budget",
            file=sys.stderr,
        )
        return 1
    return 0


def _rounded(value: float) -> float:
    return round(value, RESULT_DECIMALS)


def _measure(
    client: httpx.Client,
    *,
    n: int,
    warmup: int,
    loader: Loader,
    clock: Callable[[], float],
) -> list[float]:
    """Send ``warmup + n`` score requests and return the durations of the last ``n``, in seconds."""
    response = client.get(MODEL_PATH)
    response.raise_for_status()
    info = ModelInfo.model_validate_json(response.content)
    entity = loader(info.entity)
    train_end = entity.train_end
    if info.required_rows > train_end:
        raise LatencyError(
            f"the service needs {info.required_rows} rows per request but {info.entity} has "
            f"only {train_end} train rows to precede its test rows"
        )
    if info.fit_end > train_end:
        raise LatencyError(
            f"the service's bundle was fitted on {info.fit_end} rows but {info.entity} has "
            f"only {train_end} train rows: it was built from other data"
        )
    full = entity.full
    samples: list[float] = []
    for i in range(warmup + n):
        last = train_end + i % entity.n_test
        content = _score_body(info.entity, full, last, info.required_rows)
        started = clock()
        response = client.post(
            SCORE_PATH, content=content, headers={"Content-Type": "application/json"}
        )
        elapsed = clock() - started
        if response.status_code != 200:
            raise LatencyError(
                f"request {i} (rows ending at {last}) got status {response.status_code}: "
                f"{response.text[:ERROR_BODY_CHARS]}"
            )
        if i >= warmup:
            samples.append(elapsed)
    return samples


def _score_body(entity: str, full: np.ndarray, last: int, rows: int) -> bytes:
    """Encode the request for the ``rows`` rows ending at absolute row ``last``."""
    first = last - rows + 1
    payload: dict[str, Any] = {
        "entity": entity,
        "first_index": first,
        "history": full[first : last + 1].tolist(),
    }
    return json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _at_least(minimum: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"{value} is below the minimum of {minimum}")
        return value

    return parse


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not value > 0:  # also refuses NaN
        raise argparse.ArgumentTypeError(f"{text} is not above 0")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line, load the machine the service serves, and measure it.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` if omitted.

    Returns:
        0 if the P95 is below the budget; 1 otherwise (see `run`).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--url", default=DEFAULT_URL, help="base URL of the service")
    parser.add_argument(
        "--n", type=_at_least(1), default=DEFAULT_N, help="requests that count (default: 2000)"
    )
    parser.add_argument(
        "--warmup",
        type=_at_least(0),
        default=DEFAULT_WARMUP,
        help="requests sent first and left out of the numbers (default: 100)",
    )
    parser.add_argument(
        "--budget-ms",
        type=_positive_float,
        default=DEFAULT_BUDGET_MS,
        help="the P95 must be below this many milliseconds (default: 50)",
    )
    parser.add_argument("--out", type=Path, help="also write the result here as JSON")
    parser.add_argument("--data-dir", type=Path, help="SMD data directory; default the cache")
    args = parser.parse_args(argv)
    data_dir: Path | None = args.data_dir

    def loader(name: str) -> EntityData:
        # A fresh audit stays in its selection phase for good, so no label can be read.
        return load_entity(name, audit=LabelAudit(), dest=data_dir)

    return run(
        url=args.url,
        n=args.n,
        warmup=args.warmup,
        budget_ms=args.budget_ms,
        out=args.out,
        loader=loader,
    )


if __name__ == "__main__":
    raise SystemExit(main())
