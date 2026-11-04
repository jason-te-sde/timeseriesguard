"""Test labels that can only be read during the final evaluation, with a log of every read.

Model selection must never see test labels. Instead of trusting each caller to leave them
alone, the labels of an entity are wrapped in `AuditedLabels`, which hands out its values
only while the shared `LabelAudit` is in the ``final_evaluation`` phase and records every
successful read. A decision threshold is wrapped the same way: `LabelAudit.freeze_threshold`
is the only way to obtain a `FrozenThreshold`, and it refuses once the phase says test labels
may already have been read.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import InitVar, dataclass
from typing import Final

import numpy as np

from tsguard.errors import LabelAccessError

_FREEZE_TOKEN: Final = object()


class Phase(enum.Enum):
    """The two phases of an experiment, as far as test labels are concerned.

    Attributes:
        SELECTION: Models, features and thresholds are chosen; test labels are off limits.
        FINAL_EVALUATION: The chosen configuration is scored; test labels may be read.
    """

    SELECTION = "selection"
    FINAL_EVALUATION = "final_evaluation"


class LabelAudit:
    """The phase an experiment is in, and the log of every successful test-label read.

    A new audit starts in `Phase.SELECTION`. One audit is shared by all the
    `AuditedLabels` of a run, so a single `phase` block switches every entity at once.
    """

    def __init__(self) -> None:
        """Create an audit in the selection phase with an empty access log."""
        self._phase = Phase.SELECTION
        self._accesses: list[tuple[str, str, Phase]] = []

    @property
    def current(self) -> Phase:
        """The phase the experiment is in right now."""
        return self._phase

    @property
    def accesses(self) -> tuple[tuple[str, str, Phase], ...]:
        """Successful reads so far, oldest first, as ``(entity, caller, phase)``."""
        return tuple(self._accesses)

    @property
    def reads(self) -> int:
        """Number of successful reads so far."""
        return len(self._accesses)

    @contextmanager
    def phase(self, phase: Phase) -> Iterator[None]:
        """Run a block in ``phase`` and restore the previous phase on exit, even on error.

        Args:
            phase: The phase to enter.

        Yields:
            Nothing; the phase is active for the duration of the ``with`` block.
        """
        previous = self._phase
        self._phase = phase
        try:
            yield
        finally:
            self._phase = previous

    def freeze_threshold(self, value: float, *, caller: str) -> FrozenThreshold:
        """Freeze a decision threshold chosen before any test label was read.

        Args:
            value: The threshold; must be finite.
            caller: Name of the code freezing it, quoted in the error message.

        Returns:
            The frozen threshold.

        Raises:
            LabelAccessError: The current phase is `Phase.FINAL_EVALUATION`.
            ValueError: ``value`` is NaN or infinite.
        """
        if self._phase is Phase.FINAL_EVALUATION:
            raise LabelAccessError(
                f"threshold frozen by '{caller}' during '{self._phase.value}'; "
                "reported thresholds must be chosen before any test label is read",
                detail={"caller": caller, "phase": self._phase.value},
            )
        if not math.isfinite(value):
            raise ValueError(f"threshold frozen by '{caller}' must be finite, got {value!r}")
        return FrozenThreshold(value, _token=_FREEZE_TOKEN)

    def _log_read(self, entity: str, caller: str) -> None:
        """Record a successful read; only `AuditedLabels.read` calls this."""
        self._accesses.append((entity, caller, self._phase))


@dataclass(frozen=True)
class FrozenThreshold:
    """A decision threshold fixed during selection.

    Only `LabelAudit.freeze_threshold` creates one: the constructor demands a private
    token and raises `TypeError` for anything else.

    Attributes:
        value: The threshold.
    """

    value: float
    _token: InitVar[object]

    def __post_init__(self, _token: object) -> None:
        """Refuse construction outside `LabelAudit.freeze_threshold`.

        Args:
            _token: Must be the module-private token.

        Raises:
            TypeError: ``_token`` is not the module-private token.
        """
        if _token is not _FREEZE_TOKEN:
            raise TypeError("FrozenThreshold is created only by LabelAudit.freeze_threshold")


class AuditedLabels:
    """The binary test labels of one entity, readable only during the final evaluation.

    The values are copied on construction and exposed through no other accessor than
    `read`: there is no ``__array__`` and no public attribute, and ``repr`` shows only the
    entity and the length.
    """

    def __init__(self, values: np.ndarray, *, audit: LabelAudit, entity: str) -> None:
        """Wrap ``values``.

        Args:
            values: Labels, one per row; copied as ``int8``.
            audit: The audit that says when reading is allowed and logs each read.
            entity: Name of the entity the labels belong to, quoted in errors.
        """
        self._values = np.array(values, dtype=np.int8, copy=True)
        self._audit = audit
        self._entity = entity

    def __len__(self) -> int:
        """Number of labels."""
        return len(self._values)

    def __repr__(self) -> str:
        """Describe the labels without printing any value."""
        return f"AuditedLabels(entity={self._entity!r}, length={len(self._values)})"

    def read(self, caller: str) -> np.ndarray:
        """Return the labels as a read-only copy, and log the read.

        Args:
            caller: Name of the code reading, recorded in the audit log and in errors.

        Returns:
            A read-only ``int8`` copy of the labels.

        Raises:
            LabelAccessError: The audit is not in `Phase.FINAL_EVALUATION`.
        """
        phase = self._audit.current
        if phase is not Phase.FINAL_EVALUATION:
            raise LabelAccessError(
                f"{self._entity}: test labels read by '{caller}' during phase '{phase.value}'; "
                f"labels are readable only during '{Phase.FINAL_EVALUATION.value}'",
                detail={"entity": self._entity, "caller": caller, "phase": phase.value},
            )
        self._audit._log_read(self._entity, caller)
        out = self._values.copy()
        out.flags.writeable = False
        return out
