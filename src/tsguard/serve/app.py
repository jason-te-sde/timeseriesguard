"""The scoring service: one bundle, three routes.

`create_app` loads the bundle once, when the app is created, so a missing, damaged or tampered
bundle stops the process before it listens. Handlers are plain functions, which FastAPI runs on
a thread pool; they share the bundle's pipeline and detector, which only read their fitted state
while scoring.

Every refusal says what the caller should send instead: a short history reports the rows
needed, a row inside the fit range reports where scoring starts, a wrong entity reports the
one served. A schema violation is reported as FastAPI reports it, without the rejected input:
the default body echoes it, and an echoed NaN cannot be written as JSON, which would turn the
422 into a 500.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from tsguard.errors import GUARD_FIT_END, LeakageError
from tsguard.serve.bundle import HistoryTooShortError, load_bundle, score_rows
from tsguard.serve.schemas import (
    EntityMismatch,
    EntityMismatchBody,
    FieldError,
    FitRange,
    FitRangeBody,
    ModelInfo,
    SchemaErrorBody,
    ScoreRequest,
    ScoreResponse,
    ShortHistory,
    ShortHistoryBody,
)


def create_app(bundle_path: Path) -> FastAPI:
    """Create the service for one bundle file.

    Args:
        bundle_path: A bundle written by `tsguard.serve.bundle.save_bundle`, with its sidecar.

    Returns:
        The application.

    Raises:
        FileNotFoundError: ``bundle_path`` does not exist.
        ChecksumError: The bundle does not match its sidecar.
        ContractError: The file is not a usable bundle.
    """
    bundle, digest = load_bundle(bundle_path)
    info = ModelInfo(
        entity=bundle.entity,
        model=bundle.model,
        params=bundle.params,
        threshold=bundle.threshold,
        required_rows=bundle.required_rows,
        fit_end=bundle.fit_end,
        feature_columns=len(bundle.pipeline.columns),
        bundle_sha256=digest,
        source_commit=bundle.source_commit,
    )
    app = FastAPI(title="tsguard", summary=f"Anomaly scores for {bundle.entity}")
    score_responses: dict[int | str, dict[str, Any]] = {
        404: {
            "model": EntityMismatchBody,
            "description": "The rows are of a machine this service does not serve.",
        },
        422: {
            "model": ShortHistoryBody | FitRangeBody | SchemaErrorBody,
            "description": (
                "The history is too short, its last row is inside the rows the scaling was "
                "fitted on, or the request is not a valid score request."
            ),
        },
    }

    @app.exception_handler(RequestValidationError)
    def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        """Report each schema violation as ``type``, ``loc`` and ``msg``, never the input."""
        detail = [
            FieldError(type=item["type"], loc=list(item["loc"]), msg=item["msg"])
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content=SchemaErrorBody(detail=detail).model_dump())

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Report that the process is up and the bundle is loaded."""
        return {"status": "ok"}

    @app.get("/v1/model")
    def model() -> ModelInfo:
        """Describe the served model."""
        return info

    @app.post("/v1/score", responses=score_responses)
    def score(request: ScoreRequest) -> ScoreResponse:
        """Score the last row of the history."""
        if request.entity != bundle.entity:
            raise HTTPException(
                status_code=404,
                detail=EntityMismatch(
                    message=f"this service serves {bundle.entity!r}, not {request.entity!r}",
                    served_entity=bundle.entity,
                ).model_dump(),
            )
        try:
            value = score_rows(
                bundle,
                np.asarray(request.history, dtype=np.float64),
                first_index=request.first_index,
            )
        except HistoryTooShortError as error:
            raise HTTPException(
                status_code=422,
                detail=ShortHistory(
                    message=str(error),
                    required_rows=error.required_rows,
                    received_rows=error.received_rows,
                ).model_dump(),
            ) from None
        except LeakageError as error:
            if error.guard != GUARD_FIT_END:
                raise
            raise HTTPException(
                status_code=422,
                detail=FitRange(message=str(error), fit_end=bundle.fit_end).model_dump(),
            ) from None
        return ScoreResponse(
            entity=bundle.entity,
            model=bundle.model,
            row_index=request.first_index + len(request.history) - 1,
            score=value,
            threshold=bundle.threshold,
            is_anomaly=value >= bundle.threshold,
            bundle_sha256=digest,
        )

    return app
