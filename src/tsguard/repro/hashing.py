"""Content digests for arrays and JSON-serializable values.

A digest names the exact content of a fitted state, a feature matrix or a configuration, so two
runs can be compared by equality instead of by tolerance.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def array_digest(a: np.ndarray) -> str:
    """Return the SHA-256 hex digest of an array's dtype, shape and values.

    The digest covers ``a.dtype.str`` (which includes the byte order), ``a.shape`` and the bytes
    of the array in C order. A float32 array therefore never collides with the same values as
    float64, and a (2, 2) array never collides with a (4,) array. Memory layout does not matter.

    Args:
        a: The array to digest.

    Returns:
        A 64-character lowercase hex string.

    Raises:
        ValueError: ``a`` has object dtype, whose bytes are memory addresses and not content.
    """
    if a.dtype.hasobject:
        raise ValueError(f"cannot digest an array of dtype {a.dtype}: its bytes are pointers")
    h = hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def json_digest(obj: object) -> str:
    """Return the SHA-256 hex digest of a JSON-serializable value.

    The value is serialized with sorted keys and no whitespace, so dict key order does not
    change the digest.

    Args:
        obj: A value that ``json.dumps`` accepts.

    Returns:
        A 64-character lowercase hex string.

    Raises:
        ValueError: ``obj`` contains NaN or an infinity, which have no JSON form.
        TypeError: ``obj`` contains a value that is not JSON-serializable.
    """
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()
