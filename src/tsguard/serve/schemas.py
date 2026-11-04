"""The request and response bodies of the scoring service.

Every model forbids unknown fields and refuses NaN and infinity. Python's JSON parser accepts the
tokens ``NaN`` and ``Infinity``, so a request body that carries one reaches validation as a
float; validation is the only place it can be refused, and it is refused with a 422 before any
model sees it.
"""

from __future__ import annotations

from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsguard.data.contract import N_CHANNELS

MAX_ABS_VALUE: Final = 1e6
"""Largest magnitude a channel value may have in a request.

The data contract puts every channel in [0, 1], so the bound is a safety margin, not a range
check: it is far outside anything real, and low enough that the features' fourth-power terms
stay finite. Beyond about 1e77 they overflow to NaN, which the pipeline refuses, and which would
otherwise reach the client as a 500.
"""

_STRICT = ConfigDict(extra="forbid", allow_inf_nan=False)

ChannelValue = Annotated[float, Field(ge=-MAX_ABS_VALUE, le=MAX_ABS_VALUE)]
"""One channel value of one row: finite and within `MAX_ABS_VALUE`."""


class ScoreRequest(BaseModel):
    """The rows a client wants one anomaly score for.

    The score is for the last row of ``history``. Rows are the raw channel values, in time order,
    as they appear in the data: the service computes the features itself.

    Attributes:
        entity: Name of the machine the rows come from; must be the machine the service serves.
        first_index: Absolute row index (in the machine's full series, train rows first) of
            ``history[0]``.
        history: Rows of ``N_CHANNELS`` values each, every value within ``MAX_ABS_VALUE``; at
            least as many rows as the model needs, which ``GET /v1/model`` reports as
            ``required_rows``.
    """

    model_config = _STRICT

    entity: str
    first_index: int = Field(ge=0)
    history: list[list[ChannelValue]] = Field(min_length=1)

    @model_validator(mode="after")
    def _rows_have_one_value_per_channel(self) -> Self:
        """Reject the first row whose length is not the channel count.

        Raises:
            ValueError: ``history[<i>] has <n> values; expected 38``.
        """
        for i, row in enumerate(self.history):
            if len(row) != N_CHANNELS:
                raise ValueError(f"history[{i}] has {len(row)} values; expected {N_CHANNELS}")
        return self


class ScoreResponse(BaseModel):
    """The score of one row.

    Attributes:
        entity: The machine served.
        model: Registry name of the detector.
        row_index: Absolute index of the scored row: ``first_index + len(history) - 1``.
        score: Anomaly score; higher is more anomalous. Equal to the score the batch evaluation
            gave the same row.
        threshold: The decision threshold frozen at selection time.
        is_anomaly: ``score >= threshold``.
        bundle_sha256: Digest of the bundle file that produced the score.
    """

    model_config = _STRICT

    entity: str
    model: str
    row_index: int
    score: float
    threshold: float
    is_anomaly: bool
    bundle_sha256: str


class ModelInfo(BaseModel):
    """What the service serves.

    Attributes:
        entity: The machine served.
        model: Registry name of the detector.
        params: Hyper-parameters the detector was built with.
        threshold: The decision threshold frozen at selection time.
        required_rows: Fewest rows of ``history`` a score request needs.
        fit_end: Number of rows the feature scaling was fitted on; a request whose last row is
            before this row is refused.
        feature_columns: Number of columns the detector reads.
        bundle_sha256: Digest of the bundle file.
        source_commit: The dataset commit the bundle was fitted on.
    """

    model_config = _STRICT

    entity: str
    model: str
    params: dict[str, int | float | str]
    threshold: float
    required_rows: int
    fit_end: int
    feature_columns: int
    bundle_sha256: str
    source_commit: str
