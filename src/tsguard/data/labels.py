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
from typing import Final, NoReturn

import numpy as np

from tsguard.errors import LabelAccessError

_FREEZE_TOKEN: Final = object()
_AUDIT_COPY_MESSAGE: Final = (
    "a LabelAudit cannot be copied or pickled; a copy would stop following the phase of "
    "the original and would log reads where nobody looks"
)


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

    An audit cannot be copied or pickled: a clone would keep the phase it had when it was
    cloned and would log into a record nobody reads.
    """

    def __init__(self) -> None:
        """Create an audit in the selection phase with an empty access log."""
        self._open: list[Phase] = []
        self._accesses: list[tuple[str, str, Phase]] = []

    @property
    def current(self) -> Phase:
        """The phase the experiment is in right now."""
        return self._open[-1] if self._open else Phase.SELECTION

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

        Blocks must nest: each one has to be left before the block it was opened in. A block
        left while a block opened inside it is still open puts the audit back to the phase
        the leaving block replaced, so the audit cannot stay in a stale phase, and then
        raises.

        Args:
            phase: The phase to enter, or the string value of one (``"final_evaluation"``).

        Yields:
            Nothing; the phase is active for the duration of the ``with`` block.

        Raises:
            ValueError: ``phase`` is neither a `Phase` nor the value of one.
            RuntimeError: On exit, a block opened inside this one is still open, or this
                block was already closed by an outer one.
        """
        entered = Phase(phase)
        depth = len(self._open)
        self._open.append(entered)
        try:
            yield
        finally:
            active = self.current
            out_of_order = len(self._open) != depth + 1
            del self._open[depth:]
            if out_of_order:
                raise RuntimeError(
                    f"phase blocks exited out of order: leaving '{entered.value}' while "
                    f"'{active.value}' is the active phase; phase blocks must nest"
                )

    def freeze_threshold(self, value: float, *, caller: str) -> FrozenThreshold:
        """Freeze a decision threshold chosen before any test label was read.

        Args:
            value: The threshold; must be finite.
            caller: Name of the code freezing it, quoted in the error message.

        Returns:
            The frozen threshold.

        Raises:
            LabelAccessError: The current phase is `Phase.FINAL_EVALUATION`.
            TypeError: ``value`` is a bool or not a real number.
            ValueError: ``value`` is NaN or infinite.
        """
        phase = self.current
        if phase is Phase.FINAL_EVALUATION:
            raise LabelAccessError(
                f"threshold frozen by '{caller}' during '{phase.value}'; "
                "reported thresholds must be chosen before any test label is read",
                detail={"caller": caller, "phase": phase.value},
            )
        if isinstance(value, bool | np.bool_):
            raise TypeError(f"threshold frozen by '{caller}' must be a number, not bool")
        if not math.isfinite(value):
            raise ValueError(f"threshold frozen by '{caller}' must be finite, got {value!r}")
        return FrozenThreshold(float(value), reads_before=self.reads, _token=_FREEZE_TOKEN)

    def _log_read(self, entity: str, caller: str, phase: Phase) -> None:
        """Record a successful read in the phase `AuditedLabels.read` checked."""
        self._accesses.append((entity, caller, phase))

    def __reduce__(self) -> NoReturn:
        """Refuse pickling."""
        raise TypeError(_AUDIT_COPY_MESSAGE)

    def __copy__(self) -> NoReturn:
        """Refuse shallow copying."""
        raise TypeError(_AUDIT_COPY_MESSAGE)

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Refuse deep copying."""
        raise TypeError(_AUDIT_COPY_MESSAGE)


@dataclass(frozen=True)
class FrozenThreshold:
    """A decision threshold fixed during selection.

    Only `LabelAudit.freeze_threshold` creates one: the constructor demands a private
    token and raises `TypeError` for anything else.

    Attributes:
        value: The threshold.
        reads_before: Number of test-label reads the audit had logged when the threshold
            was frozen. A threshold chosen during selection has 0.
    """

    value: float
    reads_before: int
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
    `read`: there is no ``__array__`` and no public attribute that holds them, and ``repr``
    shows only the entity and the length. The `audit` property gives access to the shared
    phase and read log, never to values.

    The labels cannot be copied or pickled: a clone would carry the values, or a stale
    audit, out of the guard.
    """

    def __init__(self, values: np.ndarray, *, audit: LabelAudit, entity: str) -> None:
        """Wrap ``values``.

        Args:
            values: Labels, one per row, each 0 or 1; copied as ``int8``.
            audit: The audit that says when reading is allowed and logs each read.
            entity: Name of the entity the labels belong to, quoted in errors.

        Raises:
            ValueError: ``values`` is not one-dimensional, or holds anything but 0 and 1.
        """
        raw = np.asarray(values)
        if raw.ndim != 1:
            raise ValueError(f"{entity}: test labels must be 1-D, got shape {raw.shape}")
        if not np.isin(raw, (0, 1)).all():
            raise ValueError(f"{entity}: test labels must each be 0 or 1")
        self._values = np.array(raw, dtype=np.int8, copy=True)
        self._audit = audit
        self._entity = entity

    @property
    def audit(self) -> LabelAudit:
        """The audit these labels report to; it holds the phase and the read log, no values."""
        return self._audit

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
        self._audit._log_read(self._entity, caller, phase)
        out = self._values.copy()
        out.flags.writeable = False
        return out

    def __reduce__(self) -> NoReturn:
        """Refuse pickling, which would write the values out."""
        raise TypeError(self._copy_message())

    def __copy__(self) -> NoReturn:
        """Refuse shallow copying."""
        raise TypeError(self._copy_message())

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Refuse deep copying."""
        raise TypeError(self._copy_message())

    def _copy_message(self) -> str:
        return (
            f"{self._entity}: audited labels cannot be copied or pickled; "
            "a copy would carry the values, or a stale audit, past the guard"
        )
