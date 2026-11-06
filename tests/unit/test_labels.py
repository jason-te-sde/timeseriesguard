from __future__ import annotations

import copy
import pickle
from typing import Any

import numpy as np
import pytest

from tsguard.data.labels import AuditedLabels, FrozenThreshold, LabelAudit, Phase
from tsguard.errors import LabelAccessError

# Each class must refuse on its own: the labels' message names the entity, the audit's does not.
LABELS_REFUSAL = "machine: audited labels cannot be copied or pickled"
AUDIT_REFUSAL = "a LabelAudit cannot be copied or pickled"


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


def _mixed_labels() -> AuditedLabels:
    # Seven labels with both zeros and ones; the entity name has no digits, and neither does
    # "7", so any "0" or "1" in a rendering would come from the values.
    values = np.array([0, 1, 0, 1, 1, 0, 1])
    return AuditedLabels(values, audit=LabelAudit(), entity="machine")


def test_repr_hides_values():
    text = repr(_mixed_labels())
    assert "0" not in text
    assert "1" not in text
    assert "[" not in text
    assert "machine" in text


def test_str_and_format_hide_values():
    lab = _mixed_labels()
    for text in (str(lab), format(lab), f"{lab}", f"{lab!s}", "%s" % lab):  # noqa: UP031
        assert "0" not in text
        assert "1" not in text


def test_labels_are_not_iterable_or_indexable():
    lab: Any = _mixed_labels()
    with pytest.raises(TypeError):
        iter(lab)
    with pytest.raises(TypeError):
        lab[0]
    with pytest.raises(TypeError):
        list(lab)


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
    with pytest.raises(TypeError, match=r"only by LabelAudit\.freeze_threshold"):
        FrozenThreshold(0.5, reads_before=0, _token=object())


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


def test_audited_labels_refuse_copy_and_pickle():
    lab = _mixed_labels()
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        with pytest.raises(TypeError, match=LABELS_REFUSAL):
            pickle.dumps(lab, protocol=protocol)
    with pytest.raises(TypeError, match=LABELS_REFUSAL):
        copy.copy(lab)
    with pytest.raises(TypeError, match=LABELS_REFUSAL):
        copy.deepcopy(lab)


def test_audit_refuses_copy_and_pickle():
    a = LabelAudit()
    with a.phase(Phase.FINAL_EVALUATION):
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with pytest.raises(TypeError, match=AUDIT_REFUSAL):
                pickle.dumps(a, protocol=protocol)
        with pytest.raises(TypeError, match=AUDIT_REFUSAL):
            copy.copy(a)
        with pytest.raises(TypeError, match=AUDIT_REFUSAL):
            copy.deepcopy(a)


def test_copying_a_container_of_labels_is_refused_too():
    holder = {"labels": _mixed_labels()}
    with pytest.raises(TypeError, match=LABELS_REFUSAL):
        copy.deepcopy(holder)
    with pytest.raises(TypeError, match=LABELS_REFUSAL):
        pickle.dumps(holder)


def test_reads_before_is_zero_when_frozen_before_any_read():
    a = LabelAudit()
    assert a.freeze_threshold(0.7, caller="select").reads_before == 0


def test_reads_before_counts_reads_made_before_the_freeze():
    a = LabelAudit()
    lab = AuditedLabels(np.array([0, 1]), audit=a, entity="m")
    with a.phase(Phase.FINAL_EVALUATION):
        lab.read("final_metrics")
    assert a.freeze_threshold(0.7, caller="late").reads_before == 1


def test_phase_blocks_must_exit_in_order():
    a = LabelAudit()
    outer = a.phase(Phase.FINAL_EVALUATION)
    inner = a.phase(Phase.SELECTION)
    outer.__enter__()
    inner.__enter__()
    with pytest.raises(RuntimeError, match="out of order") as ei:
        outer.__exit__(None, None, None)
    assert "'final_evaluation'" in str(ei.value)
    assert "'selection'" in str(ei.value)
    # The late exit of the inner block is reported too, and the audit ends closed.
    with pytest.raises(RuntimeError, match="out of order"):
        inner.__exit__(None, None, None)
    trace = [a.current]
    assert trace == [Phase.SELECTION]


def test_out_of_order_exit_is_detected_for_equal_phases():
    a = LabelAudit()
    outer = a.phase(Phase.FINAL_EVALUATION)
    inner = a.phase(Phase.FINAL_EVALUATION)
    outer.__enter__()
    inner.__enter__()
    with pytest.raises(RuntimeError, match="out of order"):
        outer.__exit__(None, None, None)
    with pytest.raises(RuntimeError, match="out of order"):
        inner.__exit__(None, None, None)
    trace = [a.current]
    assert trace == [Phase.SELECTION]


def test_phase_accepts_the_value_of_a_phase_and_rejects_anything_else():
    a = LabelAudit()
    name: Any = "final_evaluation"
    with a.phase(name):
        seen = [a.current]
    assert seen == [Phase.FINAL_EVALUATION]
    bogus: Any = "finalevaluation"
    with pytest.raises(ValueError, match="finalevaluation"), a.phase(bogus):
        pass
    none: Any = None
    with pytest.raises(ValueError, match="None"), a.phase(none):
        pass


def test_threshold_is_stored_as_float_and_bool_is_rejected():
    a = LabelAudit()
    narrow: Any = np.float32(0.5)
    assert type(a.freeze_threshold(1, caller="x").value) is float
    assert type(a.freeze_threshold(narrow, caller="x").value) is float
    flags: list[Any] = [True, False, np.True_]
    for flag in flags:
        with pytest.raises(TypeError, match="bool"):
            a.freeze_threshold(flag, caller="x")
    text: Any = "0.5"
    with pytest.raises(TypeError):
        a.freeze_threshold(text, caller="x")


def test_labels_must_be_one_dimensional_and_binary():
    a = LabelAudit()
    not_one_dimensional: list[Any] = [np.zeros((3, 2)), np.int8(1), np.array([[0, 1]])]
    for bad in not_one_dimensional:
        with pytest.raises(ValueError, match=r"machine-9: .*1-D"):
            AuditedLabels(bad, audit=a, entity="machine-9")
    for bad in (np.array([0, 2]), np.array([-1, 1]), np.array([0.0, 0.5]), np.array([0.0, np.nan])):
        with pytest.raises(ValueError, match=r"machine-9: .*0 or 1"):
            AuditedLabels(bad, audit=a, entity="machine-9")


def test_binary_labels_in_any_representation_are_accepted():
    a = LabelAudit()
    for ok in (np.array([False, True, True]), np.array([0.0, 1.0, 1.0]), np.zeros(0)):
        assert len(AuditedLabels(ok, audit=a, entity="m")) == len(ok)
    with a.phase(Phase.FINAL_EVALUATION):
        for ok in (np.array([False, True, True]), np.array([0.0, 1.0, 1.0])):
            read = AuditedLabels(ok, audit=a, entity="m").read("t")
            assert read.dtype == np.int8
            assert read.tolist() == [0, 1, 1]
