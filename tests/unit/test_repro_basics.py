from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch

from tsguard.models.base import Detector, InputKind, check_context
from tsguard.repro.hashing import array_digest, json_digest
from tsguard.repro.seeding import configure_runtime, derive_seed


class StubDetector:
    """The smallest object that satisfies the Detector protocol."""

    def __init__(self, name: str = "stub", lookback: int = 4) -> None:
        self.name = name
        self.input_kind: InputKind = "raw"
        self.lookback = lookback

    def fit(self, X: np.ndarray) -> None:
        pass

    def score(self, X: np.ndarray, *, context: int) -> np.ndarray:
        check_context(self, X, context)
        return np.zeros(len(X) - context)

    def state_digest(self) -> str:
        return "stub"


@pytest.fixture
def restore_runtime() -> Iterator[None]:
    """Undo configure_runtime after the test: it is a process-global setting."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


def test_derive_seed_golden_values():
    assert derive_seed(0, "machine-1-1", "inject", 3) == 525749532
    assert derive_seed(7) == 2030201243


def test_derive_seed_separates_labels():
    assert derive_seed(0, "a", "b") != derive_seed(0, "ab")  # the separator matters


def test_derive_seed_fits_in_32_bits_and_depends_on_root():
    seeds = {derive_seed(root, "x") for root in range(50)}
    assert len(seeds) == 50
    assert all(0 <= s < 2**32 for s in seeds)


def test_array_digest_sees_dtype_and_shape():
    # Each pair has identical bytes, so each comparison isolates exactly one property.
    a = np.zeros(4)
    assert array_digest(a) != array_digest(np.zeros(4, dtype=np.int64))
    assert array_digest(a) != array_digest(a.reshape(2, 2))


def test_array_digest_ignores_memory_layout():
    a = np.arange(12.0).reshape(3, 4)
    assert array_digest(a) == array_digest(np.asfortranarray(a))
    assert array_digest(a[:, ::2]) == array_digest(np.ascontiguousarray(a[:, ::2]))


def test_array_digest_sees_values():
    a = np.zeros(4)
    b = a.copy()
    b[2] = 1e-12
    assert array_digest(a) != array_digest(b)


def test_array_digest_refuses_object_arrays():
    with pytest.raises(ValueError, match="object"):
        array_digest(np.array([1, "a"], dtype=object))


def test_json_digest_is_key_order_independent():
    assert json_digest({"a": 1, "b": 2}) == json_digest({"b": 2, "a": 1})


def test_json_digest_sees_values():
    assert json_digest({"a": 1}) != json_digest({"a": 2})


def test_json_digest_refuses_nan():
    with pytest.raises(ValueError, match="JSON compliant"):
        json_digest({"a": float("nan")})


def test_check_context():
    detector = StubDetector(name="stub", lookback=4)
    with pytest.raises(ValueError, match="stub needs context >= 3 rows, got 2"):
        check_context(detector, np.zeros((10, 2)), 2)


@pytest.mark.parametrize("context", [3, 9])
def test_check_context_accepts_the_boundaries(context):
    check_context(StubDetector(lookback=4), np.zeros((10, 2)), context)


@pytest.mark.parametrize("context", [10, 11])
def test_check_context_refuses_a_context_that_leaves_nothing_to_score(context):
    with pytest.raises(ValueError, match="context"):
        check_context(StubDetector(lookback=4), np.zeros((10, 2)), context)


def test_stub_satisfies_the_detector_protocol():
    detector = StubDetector()
    assert isinstance(detector, Detector)
    assert detector.score(np.zeros((10, 2)), context=3).shape == (7,)


def test_configure_runtime_sets_threads_and_determinism(restore_runtime):
    configure_runtime(threads=2)
    assert torch.get_num_threads() == 2
    assert torch.are_deterministic_algorithms_enabled()


def test_configure_runtime_is_idempotent(restore_runtime):
    configure_runtime(threads=1)
    configure_runtime(threads=1)
    assert torch.get_num_threads() == 1
    assert torch.are_deterministic_algorithms_enabled()


def test_configure_runtime_refuses_fewer_than_one_thread(restore_runtime):
    before = torch.get_num_threads()
    with pytest.raises(ValueError, match="threads"):
        configure_runtime(threads=0)
    assert torch.get_num_threads() == before
