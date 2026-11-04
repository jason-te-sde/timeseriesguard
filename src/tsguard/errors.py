"""Exception hierarchy for tsguard.

Every exception the package raises on purpose derives from `TsguardError`, so callers can
catch the whole family with one clause. Leakage errors additionally carry the name of the
guard that fired, so a report can say which protection caught the problem.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

GUARD_FIT_END: Final = "fit_end"
GUARD_CAUSAL: Final = "assert_causal"
GUARD_FOLD: Final = "fold_boundary"
GUARD_SPLIT: Final = "split_monotonic"
GUARD_LABELS: Final = "label_access_audit"


def _restore_leakage_error(
    cls: type[LeakageError], args: tuple[Any, ...], state: dict[str, Any]
) -> LeakageError:
    """Rebuild a leakage error without calling ``__init__``, which would re-prefix the message."""
    err = cls.__new__(cls)
    err.args = args
    err.__dict__.update(state)
    return err


class TsguardError(Exception):
    """Base class of every exception tsguard raises on purpose."""


class ConfigurationError(TsguardError, ValueError):
    """A model, a hyperparameter or a metrics file the user supplied cannot be used.

    It is also a `ValueError`, so a caller that documented a `ValueError` for a bad model name
    keeps working; the command line reports it as a message instead of a traceback.
    """


class ContractError(TsguardError):
    """An input or output violated a documented data contract (shape, dtype, range, order)."""


class ChecksumError(TsguardError):
    """A file on disk does not match the checksum pinned for it."""


class FetchError(TsguardError):
    """A dataset file could not be downloaded; the message names the file and the URL."""


class LeakageError(TsguardError):
    """A leakage guard detected information flowing from the future or from labels.

    The string form is ``"[<guard>] <message>"``.

    Attributes:
        guard: Name of the guard that fired, one of the ``GUARD_*`` constants.
        detail: Structured context for the failure (row, column, phase, and so on).
    """

    def __init__(
        self,
        message: str,
        *,
        guard: str,
        detail: Mapping[str, object] | None = None,
    ) -> None:
        """Create the error.

        Args:
            message: Human-readable description naming the entity, row, column or phase.
            guard: Name of the guard that fired.
            detail: Optional structured context; copied, so later mutation of the
                argument does not change the error.
        """
        super().__init__(f"[{guard}] {message}")
        self.guard = guard
        self.detail: dict[str, object] = dict(detail) if detail is not None else {}

    def __reduce__(self) -> tuple[Any, ...]:
        """Support pickling, which joblib uses to carry exceptions out of worker processes."""
        return (_restore_leakage_error, (type(self), self.args, self.__dict__))


class FoldBoundaryError(LeakageError):
    """A walk-forward fold's fit window reaches into its evaluation window."""

    def __init__(self, message: str, *, detail: Mapping[str, object] | None = None) -> None:
        """Create the error with the ``fold_boundary`` guard.

        Args:
            message: Human-readable description.
            detail: Optional structured context.
        """
        super().__init__(message, guard=GUARD_FOLD, detail=detail)


class SplitOrderError(LeakageError):
    """Train, validation and test windows are not in strictly increasing time order."""

    def __init__(self, message: str, *, detail: Mapping[str, object] | None = None) -> None:
        """Create the error with the ``split_monotonic`` guard.

        Args:
            message: Human-readable description.
            detail: Optional structured context.
        """
        super().__init__(message, guard=GUARD_SPLIT, detail=detail)


class LabelAccessError(LeakageError):
    """Labels were read in a phase that must not see them."""

    def __init__(self, message: str, *, detail: Mapping[str, object] | None = None) -> None:
        """Create the error with the ``label_access_audit`` guard.

        Args:
            message: Human-readable description.
            detail: Optional structured context.
        """
        super().__init__(message, guard=GUARD_LABELS, detail=detail)
