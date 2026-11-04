import pickle

import pytest

from tsguard.errors import (
    GUARD_CAUSAL,
    GUARD_FIT_END,
    GUARD_FOLD,
    GUARD_LABELS,
    GUARD_SPLIT,
    ChecksumError,
    ContractError,
    FetchError,
    FoldBoundaryError,
    LabelAccessError,
    LeakageError,
    SplitOrderError,
    TsguardError,
)


def test_every_leakage_error_names_its_guard():
    assert FoldBoundaryError("x").guard == "fold_boundary"
    assert SplitOrderError("x").guard == "split_monotonic"
    assert LabelAccessError("x").guard == "label_access_audit"
    assert LeakageError("x", guard="assert_causal", detail={"row": 3}).detail == {"row": 3}


def test_all_errors_share_one_base():
    for cls in (
        ContractError,
        ChecksumError,
        FetchError,
        LeakageError,
        FoldBoundaryError,
        SplitOrderError,
        LabelAccessError,
    ):
        assert issubclass(cls, TsguardError)


def test_leakage_error_message_carries_guard():
    err = LeakageError("row 12 changed", guard="assert_causal")
    assert "assert_causal" in str(err)
    assert "row 12 changed" in str(err)


def test_leakage_error_str_is_bracketed_guard_then_message():
    assert str(LeakageError("row 12 changed", guard="assert_causal")) == (
        "[assert_causal] row 12 changed"
    )
    assert str(FoldBoundaryError("fit overlaps eval")) == "[fold_boundary] fit overlaps eval"


def test_guard_name_constants_are_the_published_names():
    assert (GUARD_FIT_END, GUARD_CAUSAL, GUARD_FOLD, GUARD_SPLIT, GUARD_LABELS) == (
        "fit_end",
        "assert_causal",
        "fold_boundary",
        "split_monotonic",
        "label_access_audit",
    )


def test_detail_is_a_private_copy_and_defaults_to_an_empty_dict():
    supplied = {"row": 3}
    err = LeakageError("x", guard="assert_causal", detail=supplied)
    supplied["row"] = 99
    assert err.detail == {"row": 3}

    first, second = FoldBoundaryError("a"), FoldBoundaryError("b")
    first.detail["fold"] = 1
    assert second.detail == {}


@pytest.mark.parametrize(
    "err",
    [
        LeakageError("row 12 changed", guard="assert_causal", detail={"row": 12}),
        FoldBoundaryError("fit overlaps eval", detail={"fold": 1}),
        SplitOrderError("val before train"),
        LabelAccessError("labels read during fit", detail={"phase": "fit"}),
    ],
    ids=lambda err: type(err).__name__,
)
def test_leakage_errors_survive_a_pickle_round_trip(err):
    clone = pickle.loads(pickle.dumps(err))
    assert type(clone) is type(err)
    assert str(clone) == str(err)
    assert clone.guard == err.guard
    assert clone.detail == err.detail


@pytest.mark.parametrize(
    "err",
    [
        ContractError("column 3 out of range"),
        ChecksumError("train/machine-1-1.txt: expected 1 bytes"),
        FetchError("train/machine-1-1.txt: could not download http://example.test/x"),
    ],
    ids=lambda err: type(err).__name__,
)
def test_plain_errors_survive_a_pickle_round_trip(err):
    clone = pickle.loads(pickle.dumps(err))
    assert type(clone) is type(err)
    assert str(clone) == str(err)
