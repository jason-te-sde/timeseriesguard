from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from tsguard.data.labels import FrozenThreshold, LabelAudit
from tsguard.eval.protocols import (
    PROTOCOLS,
    ProtocolName,
    ProtocolParams,
    ProtocolScore,
    adjust_pa_percent_k,
    adjust_point_adjust,
    average_precision,
    best_f1,
    evaluate,
    event_level_auc_pr,
    event_samples,
    f1_at,
    protocol_view,
    segments,
    select_threshold,
)


def _example() -> tuple[np.ndarray, np.ndarray]:
    """300 rows with one anomaly segment [100, 110) and three normal-region alarms."""
    y = np.zeros(300, dtype=int)
    y[100:110] = 1
    s = np.zeros(300)
    s[5] = 0.9
    s[105] = 0.8
    s[150] = 0.95
    s[295] = 0.4
    return s, y


def _frozen(value: float = 0.5) -> FrozenThreshold:
    return LabelAudit().freeze_threshold(value, caller="test_protocols")


# --- segments -------------------------------------------------------------------------


def test_segments():
    assert segments(np.array([0, 1, 1, 0, 1])) == [(1, 3), (4, 5)]


def test_segments_touching_both_ends():
    assert segments(np.array([1, 1, 0, 0, 1])) == [(0, 2), (4, 5)]


def test_segments_of_all_normal_is_empty():
    assert segments(np.zeros(5, dtype=int)) == []


def test_segments_of_all_anomalous_is_one_run():
    assert segments(np.ones(4, dtype=int)) == [(0, 4)]


def test_segments_of_empty_labels_is_empty():
    assert segments(np.zeros(0, dtype=int)) == []


def test_segments_returns_python_ints():
    (run,) = segments(np.array([0, 1, 1], dtype=np.int8))
    assert all(type(bound) is int for bound in run)


# --- point adjust ---------------------------------------------------------------------


def test_point_adjust():
    s = np.array([0.1, 0.5, 0.2, 0.0, 0.3])
    y = np.array([0, 1, 1, 0, 1])
    assert adjust_point_adjust(s, y).tolist() == [0.1, 0.5, 0.5, 0.0, 0.3]


def test_point_adjust_does_not_touch_its_inputs():
    s = np.array([0.1, 0.5, 0.2, 0.0, 0.3])
    y = np.array([0, 1, 1, 0, 1])
    s_before, y_before = s.copy(), y.copy()
    adjust_point_adjust(s, y)
    assert s.tolist() == s_before.tolist()
    assert y.tolist() == y_before.tolist()


def test_point_adjust_handles_a_segment_that_ends_at_the_last_row():
    s = np.array([0.9, 0.1, 0.2, 0.7])
    y = np.array([1, 0, 1, 1])
    assert adjust_point_adjust(s, y).tolist() == [0.9, 0.1, 0.7, 0.7]


# --- PA%K -----------------------------------------------------------------------------


def test_pa_percent_k_twenty_on_length_ten():
    s = np.arange(10) / 10
    y = np.ones(10, dtype=int)
    assert adjust_pa_percent_k(s, y, k_percent=20).tolist() == pytest.approx([0.7] * 8 + [0.8, 0.9])


def test_pa_percent_k_needs_strictly_more_than_k_percent_of_the_segment():
    # Ten points and K = 10: one detected point is exactly 10 %, which does not exceed K,
    # so two are needed and the segment is lifted to its second-largest score.
    s = np.arange(10) / 10
    y = np.ones(10, dtype=int)
    assert adjust_pa_percent_k(s, y, k_percent=10).tolist() == pytest.approx([0.8] * 9 + [0.9])


def test_pa_percent_k_zero_is_point_adjust_and_hundred_is_point_wise(rng):
    y = np.zeros(200, dtype=int)
    y[20:50] = 1
    y[100:101] = 1
    y[150:200] = 1
    s = rng.random(200)
    assert adjust_pa_percent_k(s, y, k_percent=0).tolist() == adjust_point_adjust(s, y).tolist()
    assert adjust_pa_percent_k(s, y, k_percent=100).tolist() == s.tolist()


def test_pa_percent_k_never_lowers_a_score_and_leaves_normal_points_alone(rng):
    y = np.zeros(120, dtype=int)
    y[10:40] = 1
    y[80:120] = 1
    s = rng.random(120)
    out = adjust_pa_percent_k(s, y, k_percent=30)
    assert (out >= s).all()
    assert out[y == 0].tolist() == s[y == 0].tolist()


def test_pa_percent_k_does_not_touch_its_inputs():
    s = np.arange(10) / 10
    y = np.ones(10, dtype=int)
    s_before = s.copy()
    adjust_pa_percent_k(s, y, k_percent=20)
    assert s.tolist() == s_before.tolist()


@pytest.mark.parametrize("k_percent", [-1, 101])
def test_pa_percent_k_rejects_a_percentage_outside_zero_to_hundred(k_percent):
    with pytest.raises(ValueError, match="k_percent"):
        adjust_pa_percent_k(np.zeros(4), np.array([0, 1, 1, 0]), k_percent=k_percent)


# --- event samples --------------------------------------------------------------------


def test_event_samples_with_cooldown():
    s, y = _example()
    es, ey = event_samples(s, y, chunk=60, cooldown=60)
    assert ey.tolist() == [0, 0, 1, 0, 0, 0]
    assert es.tolist() == [0.9, 0.0, 0.8, 0.0, 0.0, 0.4]
    assert average_precision(es, ey) == pytest.approx(0.5)


def test_event_samples_without_cooldown_count_the_tail_alarm():
    s, y = _example()
    es, ey = event_samples(s, y, chunk=60, cooldown=0)
    assert ey.tolist() == [0, 0, 1, 0, 0, 0, 0]
    assert average_precision(es, ey) == pytest.approx(1 / 3)


def test_event_samples_reject_all_normal_labels():
    with pytest.raises(ValueError, match="no anomaly"):
        event_samples(np.zeros(10), np.zeros(10, dtype=int), chunk=5, cooldown=0)


def test_event_samples_take_the_segment_max_for_the_positive():
    s = np.array([0.0, 0.2, 0.9, 0.1, 0.0])
    y = np.array([0, 1, 1, 1, 0])
    es, ey = event_samples(s, y, chunk=10, cooldown=0)
    assert ey.tolist() == [0, 1, 0]
    assert es.tolist() == [0.0, 0.9, 0.0]


def test_event_samples_start_chunks_at_the_start_of_each_run():
    # A run of 130 rows after the cooldown is cut 60 + 60 + 10 from its first row.
    s = np.zeros(300)
    y = np.zeros(300, dtype=int)
    y[100:110] = 1
    s[170] = 0.3  # first row after the cooldown: chunk 1
    s[229] = 0.4  # last row of chunk 1
    s[230] = 0.5  # first row of chunk 2
    s[299] = 0.6  # last row, in the short tail chunk
    es, ey = event_samples(s, y, chunk=60, cooldown=60)
    assert es[3:].tolist() == [0.4, 0.5, 0.6]
    assert ey[3:].tolist() == [0, 0, 0]


def test_event_samples_work_when_segments_touch_both_ends():
    s = np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8])
    y = np.array([1, 0, 0, 0, 0, 0, 0, 1])
    es, ey = event_samples(s, y, chunk=3, cooldown=1)
    # cooldown drops row 1 and the (nonexistent) rows after the last segment:
    # rows 2..6 remain, chunks [2,5) and [5,7).
    assert ey.tolist() == [1, 0, 0, 1]
    assert es.tolist() == [0.9, 0.0, 0.0, 0.8]


def test_event_samples_cooldown_after_close_segments_swallows_the_gap():
    # Segments [10, 15) and [20, 25), cooldown 10: rows 15..24 and 25..34 are dropped, so
    # the second run of normal rows starts at row 35.
    s = np.zeros(60)
    y = np.zeros(60, dtype=int)
    y[10:15] = 1
    y[20:25] = 1
    s[17] = 5.0  # inside the cooldown of the first segment
    s[30] = 6.0  # inside the cooldown of the second segment
    s[40] = 0.7
    es, ey = event_samples(s, y, chunk=100, cooldown=10)
    assert ey.tolist() == [0, 1, 1, 0]
    assert es.tolist() == [0.0, 0.0, 0.0, 0.7]


def test_event_samples_reject_a_non_positive_chunk_and_a_negative_cooldown():
    s, y = _example()
    with pytest.raises(ValueError, match="chunk"):
        event_samples(s, y, chunk=0, cooldown=0)
    with pytest.raises(ValueError, match="cooldown"):
        event_samples(s, y, chunk=60, cooldown=-1)


# --- average precision, F1 -------------------------------------------------------------


def test_average_precision_matches_a_hand_computation():
    # Ranked by score: pos, neg, pos -> precision 1 and 2/3 at the two recall steps.
    s = np.array([0.9, 0.8, 0.7])
    y = np.array([1, 0, 1])
    assert average_precision(s, y) == pytest.approx((1 + 2 / 3) / 2)


def test_average_precision_rejects_labels_without_an_anomaly():
    with pytest.raises(ValueError, match="no anomaly"):
        average_precision(np.array([0.1, 0.2]), np.array([0, 0]))


def test_best_f1_with_ties():
    assert best_f1(np.array([0.5, 0.5]), np.array([0, 1])) == pytest.approx((2 / 3, 0.5))


def test_best_f1_prefers_the_higher_threshold_on_equal_f1():
    # Thresholds 0.9 and 0.6 both give F1 = 2/3 (one hit out of one prediction, and two
    # hits out of four predictions); the higher one wins.
    s = np.array([0.9, 0.8, 0.7, 0.6])
    y = np.array([1, 0, 0, 1])
    f1, threshold = best_f1(s, y)
    assert f1 == pytest.approx(2 / 3)
    assert threshold == 0.9


def test_best_f1_finds_a_perfect_split():
    s = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    assert best_f1(s, y) == (1.0, 0.8)


def test_best_f1_of_labels_without_an_anomaly_is_zero_at_the_top_score():
    assert best_f1(np.array([0.1, 0.7]), np.array([0, 0])) == (0.0, 0.7)


def test_best_f1_rejects_empty_scores():
    with pytest.raises(ValueError, match="empty"):
        best_f1(np.zeros(0), np.zeros(0, dtype=int))


def test_f1_at_with_no_prediction_is_zero():
    assert f1_at(np.array([0.1, 0.2]), np.array([0, 1]), 0.9) == 0.0


def test_f1_at_counts_a_score_equal_to_the_threshold_as_a_prediction():
    s = np.array([0.1, 0.5, 0.5, 0.9])
    y = np.array([0, 1, 0, 1])
    # predictions: rows 1, 2, 3 -> tp 2, predicted 3, positives 2 -> F1 = 4 / 5
    assert f1_at(s, y, 0.5) == pytest.approx(0.8)


def test_f1_at_rejects_a_non_finite_threshold():
    with pytest.raises(ValueError, match="threshold"):
        f1_at(np.array([0.1, 0.2]), np.array([0, 1]), float("nan"))


# --- views, selection, evaluation -----------------------------------------------------


def test_protocol_view_point_wise_is_the_input():
    s, y = _example()
    vs, vy = protocol_view("point_wise", s, y, ProtocolParams())
    assert vs.tolist() == s.tolist()
    assert vy.tolist() == y.tolist()


def test_protocol_view_point_adjust_lifts_the_segment_and_keeps_the_labels():
    s, y = _example()
    vs, vy = protocol_view("point_adjust", s, y, ProtocolParams())
    assert vs[100:110].tolist() == [0.8] * 10
    assert vy.tolist() == y.tolist()


def test_protocol_view_pa_percent_k_uses_the_params():
    s, y = _example()
    vs, _ = protocol_view("pa_percent_k", s, y, ProtocolParams(k_percent=0))
    assert vs[100:110].tolist() == [0.8] * 10
    vs, _ = protocol_view("pa_percent_k", s, y, ProtocolParams(k_percent=100))
    assert vs.tolist() == s.tolist()


def test_protocol_view_event_level_is_the_event_samples():
    s, y = _example()
    params = ProtocolParams()
    vs, vy = protocol_view("event_level", s, y, params)
    es, ey = event_samples(s, y, chunk=params.chunk, cooldown=params.cooldown)
    assert vs.tolist() == es.tolist()
    assert vy.tolist() == ey.tolist()


def test_protocol_view_rejects_an_unknown_protocol():
    s, y = _example()
    with pytest.raises(ValueError, match="unknown protocol"):
        protocol_view("f1_pa", s, y, ProtocolParams())  # type: ignore[arg-type]


def test_event_level_auc_pr():
    s, y = _example()
    assert event_level_auc_pr(s, y, ProtocolParams()) == pytest.approx(0.5)


def test_select_threshold_is_a_sample_score():
    s, y = _example()
    t = select_threshold(s, y, ProtocolParams())
    assert t in {0.9, 0.8, 0.4, 0.0}


def test_select_threshold_maximizes_f1_over_the_event_samples():
    s, y = _example()
    assert select_threshold(s, y, ProtocolParams()) == 0.8


def test_protocol_names_are_the_four_protocols_in_report_order():
    assert PROTOCOLS == ("point_wise", "point_adjust", "pa_percent_k", "event_level")


def test_protocol_params_defaults():
    assert ProtocolParams() == ProtocolParams(chunk=60, cooldown=60, k_percent=20)


def test_evaluate_reports_all_four_protocols():
    s, y = _example()
    out = evaluate(s, y, _frozen(0.8), ProtocolParams())
    assert set(out) == set(PROTOCOLS)
    assert out["event_level"].auc_pr == pytest.approx(0.5)


def test_evaluate_scores_each_protocol_on_its_own_view():
    s, y = _example()
    out = evaluate(s, y, _frozen(0.8), ProtocolParams())
    # Event level at threshold 0.8: samples 0.9 (neg) and 0.8 (pos) predicted -> F1 = 2/3,
    # and 0.8 is also the oracle threshold there.
    event = out["event_level"]
    assert isinstance(event, ProtocolScore)
    assert event.auc_pr == pytest.approx(0.5)
    assert event.f1 == pytest.approx(2 / 3)
    assert event.f1_oracle == pytest.approx(2 / 3)
    # Point adjust at 0.8: the lifted segment (10 rows at 0.8) plus rows 5 and 150 predicted.
    assert out["point_adjust"].f1 == pytest.approx(2 * 10 / (12 + 10))


def test_evaluate_oracle_f1_is_never_below_the_frozen_threshold_f1(rng):
    y = np.zeros(2_000, dtype=int)
    y[300:340] = 1
    y[1_200:1_230] = 1
    s = rng.random(2_000) + 0.5 * y
    out = evaluate(s, y, _frozen(0.9), ProtocolParams())
    for protocol in PROTOCOLS:
        assert out[protocol].f1_oracle >= out[protocol].f1


def test_evaluate_refuses_a_bare_float():
    s, y = _example()
    with pytest.raises(TypeError, match="FrozenThreshold"):
        evaluate(s, y, 0.8, ProtocolParams())  # type: ignore[arg-type]


# --- input checks on every public function -----------------------------------------------


def _clean() -> tuple[np.ndarray, np.ndarray]:
    s = np.arange(20) / 20
    y = np.zeros(20, dtype=int)
    y[5:10] = 1
    return s, y


Call = Callable[[np.ndarray, np.ndarray], object]


def _view_call(name: ProtocolName, params: ProtocolParams) -> Call:
    return lambda s, y: protocol_view(name, s, y, params)


def _calls() -> dict[str, Call]:
    params = ProtocolParams(chunk=4, cooldown=2)
    calls: dict[str, Call] = {
        "adjust_point_adjust": lambda s, y: adjust_point_adjust(s, y),
        "adjust_pa_percent_k": lambda s, y: adjust_pa_percent_k(s, y, k_percent=20),
        "event_samples": lambda s, y: event_samples(s, y, chunk=4, cooldown=2),
        "average_precision": lambda s, y: average_precision(s, y),
        "f1_at": lambda s, y: f1_at(s, y, 0.5),
        "best_f1": lambda s, y: best_f1(s, y),
        "event_level_auc_pr": lambda s, y: event_level_auc_pr(s, y, params),
        "select_threshold": lambda s, y: select_threshold(s, y, params),
        "evaluate": lambda s, y: evaluate(s, y, _frozen(), params),
    }
    for name in PROTOCOLS:
        calls[f"protocol_view[{name}]"] = _view_call(name, params)
    return calls


CALLS = _calls()


@pytest.mark.parametrize("name", CALLS)
def test_the_clean_inputs_are_accepted_everywhere(name):
    CALLS[name](*_clean())


@pytest.mark.parametrize("name", CALLS)
def test_length_mismatch_is_rejected(name):
    s, y = _clean()
    with pytest.raises(ValueError, match="length"):
        CALLS[name](s[:-1], y)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("name", CALLS)
def test_non_finite_scores_are_rejected(name, bad):
    s, y = _clean()
    s[3] = bad
    with pytest.raises(ValueError, match="finite"):
        CALLS[name](s, y)


@pytest.mark.parametrize("bad", [2, -1, 0.5])
@pytest.mark.parametrize("name", CALLS)
def test_labels_that_are_not_zero_or_one_are_rejected(name, bad):
    s, y = _clean()
    y = y.astype(float)
    y[1] = bad
    with pytest.raises(ValueError, match="0 or 1"):
        CALLS[name](s, y)


def test_segments_reject_labels_that_are_not_zero_or_one():
    with pytest.raises(ValueError, match="0 or 1"):
        segments(np.array([0, 2, 1]))


def test_two_dimensional_inputs_are_rejected():
    with pytest.raises(ValueError, match="1-D"):
        best_f1(np.zeros((2, 2)), np.zeros((2, 2), dtype=int))
    with pytest.raises(ValueError, match="1-D"):
        segments(np.zeros((2, 2), dtype=int))
