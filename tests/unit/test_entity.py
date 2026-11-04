from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tests.conftest import make_entity
from tsguard.data.contract import N_CHANNELS, check_entity
from tsguard.data.labels import LabelAudit, Phase
from tsguard.data.smd import EntityData
from tsguard.errors import ContractError


def labels_of(entity: EntityData) -> np.ndarray:
    with entity.labels.audit.phase(Phase.FINAL_EVALUATION):
        return entity.labels.read("test_entity")


def test_default_shape_and_derived_properties():
    e = make_entity()
    assert e.name == "synthetic-0"
    assert e.train.shape == (2000, N_CHANNELS)
    assert e.test.shape == (2000, N_CHANNELS)
    assert len(e.labels) == 2000
    assert e.train_end == 2000
    assert e.n_test == 2000
    assert e.full.shape == (4000, N_CHANNELS)


def test_full_is_train_stacked_on_test_and_computed_on_access():
    e = make_entity(n_train=1500, n_test=1200, anomalies=((100, 10),))
    assert e.train_end == 1500
    assert e.n_test == 1200
    assert np.array_equal(e.full, np.vstack([e.train, e.test]))
    first, second = e.full, e.full
    assert first is not second
    first[0, 0] = -5.0  # a caller mutating its copy must not reach the entity
    assert e.train[0, 0] != -5.0
    assert np.array_equal(second, np.vstack([e.train, e.test]))


def test_entity_data_is_frozen():
    e = make_entity()
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.name = "other"  # type: ignore[misc]


def test_arrays_are_read_only():
    e = make_entity()
    for array in (e.train, e.test):
        assert array.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = 0.5


def test_same_seed_gives_identical_entities():
    a, b = make_entity(seed=3), make_entity(seed=3)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.test, b.test)
    assert np.array_equal(labels_of(a), labels_of(b))


def test_different_seeds_give_different_signals_but_the_same_labels():
    a, b = make_entity(seed=0), make_entity(seed=1)
    assert not np.array_equal(a.train, b.train)
    assert not np.array_equal(a.test, b.test)
    assert np.array_equal(labels_of(a), labels_of(b))


def forbid_global_numpy_random(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make every ``np.random.<function>`` bound to the global ``RandomState`` raise."""
    names = [
        name
        for name in dir(np.random.RandomState)
        if not name.startswith("_") and hasattr(np.random, name)
    ]

    def refuse(name: str):
        def raiser(*args: object, **kwargs: object) -> None:
            raise AssertionError(f"np.random.{name} touches the global random state")

        return raiser

    for name in names:
        monkeypatch.setattr(np.random, name, refuse(name))
    return names


def test_the_global_numpy_random_functions_are_forbidden_while_the_tests_below_run(monkeypatch):
    names = forbid_global_numpy_random(monkeypatch)
    assert {"seed", "random", "rand", "randn", "randint", "uniform", "normal"} <= set(names)
    with pytest.raises(AssertionError, match="global random state"):
        np.random.random()
    with pytest.raises(AssertionError, match="global random state"):
        np.random.seed(0)


def test_make_entity_never_touches_the_global_numpy_random_state(monkeypatch):
    forbid_global_numpy_random(monkeypatch)
    e = make_entity(seed=9)
    assert e.train.shape == (2000, N_CHANNELS)
    assert np.array_equal(e.train, make_entity(seed=9).train)


def test_signal_channels_are_scaled_into_the_documented_band():
    e = make_entity(anomalies=())
    signal = e.full[:, :37]
    assert signal.min() >= 0.05 - 1e-12
    assert signal.max() <= 0.95 + 1e-12
    # min-max scaling over train and test together touches both ends on every channel
    assert np.allclose(signal.min(axis=0), 0.05)
    assert np.allclose(signal.max(axis=0), 0.95)
    assert np.all(e.full[:, 37] == 0.5)


def autocorrelation(x: np.ndarray, lag: int) -> np.ndarray:
    centered = x - x.mean(axis=0)
    covariance = (centered[:-lag] * centered[lag:]).sum(axis=0)
    return np.asarray(covariance / (centered * centered).sum(axis=0), dtype=np.float64)


def test_channels_are_ar1_with_coefficient_0_8_on_top_of_a_daily_cycle():
    e = make_entity(n_train=3000, n_test=3000, anomalies=())
    signal = e.full[:, :37]
    # The first difference removes the slow sinusoid; for AR(1) with phi = 0.8 its lag-1
    # autocorrelation is -(1 - phi) / 2 = -0.1, while white noise would give -0.5.
    diff_lag1 = autocorrelation(np.diff(signal, axis=0), 1)
    assert -0.2 < diff_lag1.mean() < 0.0
    # A period of 1440 rows: positively correlated a full period apart, anti-correlated
    # half a period apart.
    assert autocorrelation(signal, 1440).mean() > 0.2
    assert autocorrelation(signal, 720).mean() < -0.1


def test_channels_differ_from_each_other():
    e = make_entity()
    assert not np.array_equal(e.train[:, 0], e.train[:, 1])


def test_anomalies_label_the_segments_and_shift_the_first_five_channels():
    plain = make_entity(anomalies=())
    e = make_entity()
    labels = labels_of(e)
    expected = np.zeros(2000, dtype=np.int8)
    expected[600:620] = 1
    expected[1400:1405] = 1
    assert np.array_equal(labels, expected)
    assert not labels_of(plain).any()
    assert np.array_equal(e.train, plain.train)
    for start, length in ((600, 20), (1400, 5)):
        rows = slice(start, start + length)
        want = np.minimum(plain.test[rows, :5] + 0.3, 1.0)
        assert np.array_equal(e.test[rows, :5], want)
    outside = expected == 0
    assert np.array_equal(e.test[outside], plain.test[outside])
    assert np.array_equal(e.test[:, 5:], plain.test[:, 5:])


def test_shifted_values_are_clipped_to_one():
    e = make_entity(seed=2, anomalies=((100, 1000),))
    assert e.test[100:1100, :5].max() == 1.0
    assert e.test.max() <= 1.0


def test_custom_anomalies_and_sizes():
    e = make_entity(n_train=1200, n_test=1300, anomalies=((0, 3), (1297, 3)))
    expected = np.zeros(1300, dtype=np.int8)
    expected[:3] = 1
    expected[1297:] = 1
    assert np.array_equal(labels_of(e), expected)
    check_entity(e.name, e.train, e.test, labels_of(e))


@pytest.mark.parametrize("segment", [(-1, 5), (0, 0), (1996, 5), (2000, 1)])
def test_an_anomaly_that_does_not_fit_inside_the_test_split_is_refused(segment):
    with pytest.raises(ValueError, match="anomal"):
        make_entity(anomalies=(segment,))


def test_name_is_used():
    assert make_entity(name="machine-x").name == "machine-x"
    assert "machine-x" in repr(make_entity(name="machine-x").labels)


def test_the_given_audit_is_the_one_the_labels_report_to():
    audit = LabelAudit()
    e = make_entity(audit=audit)
    assert e.labels.audit is audit
    with audit.phase(Phase.FINAL_EVALUATION):
        e.labels.read("who")
    assert audit.accesses == (("synthetic-0", "who", Phase.FINAL_EVALUATION),)


def test_each_entity_gets_a_fresh_audit_when_none_is_given():
    a, b = make_entity(), make_entity()
    assert a.labels.audit is not b.labels.audit
    assert a.labels.audit.current is Phase.SELECTION
    labels_of(a)
    assert a.labels.audit.reads == 1
    assert b.labels.audit.reads == 0


def test_a_width_other_than_38_is_built_but_fails_the_contract():
    e = make_entity(channels=10)
    assert e.train.shape[1] == 10
    assert e.test.shape[1] == 10
    with pytest.raises(ContractError, match="38 columns"):
        check_entity(e.name, e.train, e.test, labels_of(e))


def test_fewer_than_five_channels_still_get_the_shift():
    plain = make_entity(channels=3, anomalies=())
    e = make_entity(channels=3)
    assert np.array_equal(e.test[600:620], np.minimum(plain.test[600:620] + 0.3, 1.0))


def test_entities_compare_and_hash_by_identity_not_by_array_contents():
    a, b = make_entity(), make_entity()
    assert a == a
    assert a != b  # comparing field by field would raise on the ambiguous array comparison
    assert len({a, b}) == 2
