"""Seed derivation and the one process-global runtime setting.

Every random choice in the package flows from one root seed. `derive_seed` turns that root and
a few labels (an entity, a purpose, an index) into an independent integer seed, so changing
one purpose never shifts the random stream of another.
"""

from __future__ import annotations

import hashlib
from typing import Final

import torch

_SEED_BYTES: Final = 4  # seeds are 32-bit, which every numpy and torch seeding API accepts
_SEPARATOR: Final = "|"


def derive_seed(root: int, *labels: str | int) -> int:
    """Derive a 32-bit seed from a root seed and labels.

    The root and the labels are written as strings, joined with ``"|"``, hashed with SHA-256,
    and the first four bytes are read as a big-endian integer. The separator keeps
    ``("a", "b")`` apart from ``("ab",)``. Labels are compared by their string form, so the
    integer ``3`` and the string ``"3"`` name the same stream.

    Args:
        root: The root seed.
        *labels: Strings or integers that name the purpose, such as an entity name, a phase
            and an index.

    Returns:
        An integer in ``[0, 2**32)``.
    """
    text = _SEPARATOR.join([str(root), *map(str, labels)])
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:_SEED_BYTES], "big")


def configure_runtime(*, threads: int) -> None:
    """Fix the torch thread count and require deterministic algorithms.

    This is the one process-global setting in the package. Entry points call it once, before
    any torch model trains: the CLI, the reproduce command, and tests that train torch models.
    Library code never calls it. Calling it again with the same arguments changes nothing.

    Args:
        threads: Number of intra-op CPU threads for torch. Must be at least 1.

    Raises:
        ValueError: ``threads`` is less than 1.
    """
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
