from __future__ import annotations

import numpy as np
import pytest

from tsguard.data.labels import AuditedLabels, FrozenThreshold, LabelAudit, Phase
from tsguard.errors import LabelAccessError


def test_read_outside_final_evaluation_raises():
    a = LabelAudit()
    lab = AuditedLabels(np.array([0, 1]), audit=a, entity="m")
    with pytest.raises(LabelAccessError) as ei:
        lab.read("threshold_search")
    assert ei.value.guard == "label_access_audit"
    assert ei.value.detail == {"entity": "m", "caller": "threshold_search", "phase": "selection"}
    assert str(ei.value) == (
        "[label_access_audit] m: test labels read by 'threshold_search' during phase "
        "'selection'; labels are readable only during 'final_evaluation'"
    )


def test_read_inside_final_evaluation_returns_read_only_copy():
    a = LabelAudit()
    source = np.array([0, 1, 1, 0])
    lab = AuditedLabels(source, audit=a, entity="m")
    source[0] = 1  # mutating the caller's array after construction must not reach the labels
    assert len(lab) == 4
    with a.phase(Phase.FINAL_EVALUATION):
        result = lab.read("final_metrics")
        assert result.tolist() == [0, 1, 1, 0]
        assert result.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            result[0] = 1
        # Re-enabling writes on the returned array must not corrupt the stored labels.
        result.flags.writeable = True
        result[:] = 1
        assert lab.read("final_metrics").tolist() == [0, 1, 1, 0]


def test_phase_restores_after_exception():
    a = LabelAudit()
    seen = []

    def fail_inside_final_evaluation() -> None:
        with a.phase(Phase.FINAL_EVALUATION):
            seen.append(a.current)
            raise RuntimeError("boom")

    assert a.current is Phase.SELECTION
    with pytest.raises(RuntimeError, match="boom"):
        fail_inside_final_evaluation()
    assert seen == [Phase.FINAL_EVALUATION]
    assert a.current is Phase.SELECTION


def test_nested_phases_restore_in_order():
    a = LabelAudit()
    trace: list[Phase] = []
    with a.phase(Phase.FINAL_EVALUATION):
        trace.append(a.current)
        with a.phase(Phase.SELECTION):
            trace.append(a.current)
            with a.phase(Phase.FINAL_EVALUATION):
                trace.append(a.current)
            trace.append(a.current)
        trace.append(a.current)
    trace.append(a.current)
    final, selection = Phase.FINAL_EVALUATION, Phase.SELECTION
    assert trace == [final, selection, final, selection, final, selection]


def test_accesses_are_logged():
    a = LabelAudit()
    lab = AuditedLabels(np.array([0, 1]), audit=a, entity="m")
    with pytest.raises(LabelAccessError):
        lab.read("too_early")
    assert list(a.accesses) == []
    assert a.reads == 0
    with a.phase(Phase.FINAL_EVALUATION):
        lab.read("final_metrics")
    assert list(a.accesses) == [("m", "final_metrics", Phase.FINAL_EVALUATION)]
    assert a.reads == 1
    with pytest.raises(LabelAccessError):
        lab.read("too_late")
    assert a.reads == 1


def test_repr_hides_values():
    text = repr(AuditedLabels(np.array([1, 1, 1]), audit=LabelAudit(), entity="m"))
    assert "1" not in text.replace("m", "")


def test_numpy_conversion_is_refused():
    lab = AuditedLabels(np.array([1]), audit=LabelAudit(), entity="m")
    with pytest.raises(TypeError):
        np.asarray(lab, dtype=float)
    assert not hasattr(lab, "__array__")
    assert not hasattr(lab, "values")


def test_threshold_freezes_during_selection():
    a = LabelAudit()
    t = a.freeze_threshold(0.7, caller="select")
    assert t.value == 0.7


def test_threshold_cannot_be_frozen_during_final_evaluation():
    a = LabelAudit()
    with a.phase(Phase.FINAL_EVALUATION), pytest.raises(LabelAccessError) as ei:
        a.freeze_threshold(0.7, caller="pick_on_test")
    assert ei.value.guard == "label_access_audit"
    assert ei.value.detail == {"caller": "pick_on_test", "phase": "final_evaluation"}
    assert str(ei.value) == (
        "[label_access_audit] threshold frozen by 'pick_on_test' during 'final_evaluation'; "
        "reported thresholds must be chosen before any test label is read"
    )


def test_frozen_threshold_cannot_be_constructed_directly():
    with pytest.raises(TypeError):
        FrozenThreshold(0.5, _token=object())


def test_non_finite_threshold_is_value_error():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            LabelAudit().freeze_threshold(bad, caller="x")


def test_audit_property_returns_the_audit_the_labels_were_built_with():
    a = LabelAudit()
    lab = AuditedLabels(np.array([0, 1]), audit=a, entity="m")
    assert lab.audit is a
    with a.phase(Phase.FINAL_EVALUATION):
        lab.read("final_metrics")
    assert lab.audit.accesses == (("m", "final_metrics", Phase.FINAL_EVALUATION),)


def test_audit_property_is_read_only_and_exposes_no_values():
    lab = AuditedLabels(np.array([1, 1]), audit=LabelAudit(), entity="m")
    with pytest.raises(AttributeError):
        lab.audit = LabelAudit()  # type: ignore[misc]
    assert not hasattr(lab.audit, "values")
