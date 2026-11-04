from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import make_entity
from tsguard.data.contract import N_CHANNELS, check_entity
from tsguard.data.labels import Phase
from tsguard.errors import ContractError, TsguardError

ROWS = 1_000


def valid_inputs(rows: int = ROWS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.full((rows, N_CHANNELS), 0.25)
    test = np.full((rows, N_CHANNELS), 0.75)
    labels = np.zeros(rows, dtype=np.int8)
    labels[10:20] = 1
    return train, test, labels


def test_make_entity_passes_contract_and_has_constant_channel():
    e = make_entity()
    with e.labels.audit.phase(Phase.FINAL_EVALUATION):
        labels = e.labels.read("test_contract")
    check_entity(e.name, e.train, e.test, labels)
    assert np.ptp(e.train[:, 37]) == 0.0


def test_valid_entity_passes():
    train, test, labels = valid_inputs()
    check_entity("e", train, test, labels)


def test_wrong_width_names_rule_and_entity():
    train, test, labels = valid_inputs()
    with pytest.raises(ContractError, match=r"^e: .*38 columns"):
        check_entity("e", train[:, :37], test, labels)


def test_too_few_rows():
    train, test, labels = valid_inputs()
    with pytest.raises(ContractError, match="1,000 rows") as excinfo:
        check_entity("e", train, test[:-1], labels[:-1])
    assert str(excinfo.value).startswith("e: ")


def test_non_finite():
    train, test, labels = valid_inputs()
    test[500, 3] = np.nan
    with pytest.raises(ContractError, match="finite") as excinfo:
        check_entity("e", train, test, labels)
    assert str(excinfo.value).startswith("e: ")


def test_out_of_range():
    train, test, labels = valid_inputs()
    train[7, 2] = 1.0000001
    with pytest.raises(ContractError, match=r"\[0.0, 1.0\]") as excinfo:
        check_entity("e", train, test, labels)
    assert str(excinfo.value).startswith("e: ")


def test_non_binary_labels():
    train, test, labels = valid_inputs()
    labels[3] = 2
    with pytest.raises(ContractError, match="0, 1") as excinfo:
        check_entity("e", train, test, labels)
    assert str(excinfo.value).startswith("e: ")


def test_label_length_mismatch():
    train, test, labels = valid_inputs()
    with pytest.raises(ContractError, match="labels") as excinfo:
        check_entity("e", train, test, labels[:-1])
    assert str(excinfo.value).startswith("e: ")


@pytest.mark.parametrize("split", ["train", "test"])
def test_non_finite_names_the_split_row_and_column(split):
    train, test, labels = valid_inputs()
    {"train": train, "test": test}[split][500, 3] = np.inf
    with pytest.raises(ContractError) as excinfo:
        check_entity("e", train, test, labels)
    message = str(excinfo.value)
    assert split in message
    assert "row 500" in message
    assert "column 3" in message


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("bad", [-1e-9, 1.0000001])
def test_out_of_range_on_either_side_names_the_split_row_and_column(split, bad):
    train, test, labels = valid_inputs()
    {"train": train, "test": test}[split][12, 37] = bad
    with pytest.raises(ContractError) as excinfo:
        check_entity("e", train, test, labels)
    message = str(excinfo.value)
    assert split in message
    assert "row 12" in message
    assert "column 37" in message


@pytest.mark.parametrize("split", ["train", "test"])
def test_wrong_width_and_too_few_rows_name_the_split(split):
    train, test, labels = valid_inputs()
    arrays = {"train": train, "test": test}
    arrays[split] = arrays[split][:, :1]
    with pytest.raises(ContractError, match=split):
        check_entity("e", arrays["train"], arrays["test"], labels)
    arrays = {"train": train, "test": test}
    arrays[split] = arrays[split][:-1]
    labels = labels[: len(arrays["test"])]
    with pytest.raises(ContractError, match=split):
        check_entity("e", arrays["train"], arrays["test"], labels)


def test_integer_features_are_rejected():
    train, test, labels = valid_inputs()
    with pytest.raises(ContractError, match=r"^e: .*float"):
        check_entity("e", np.zeros(train.shape, dtype=np.int64), test, labels)


def test_one_dimensional_features_are_rejected():
    train, test, labels = valid_inputs()
    with pytest.raises(ContractError, match=r"^e: .*2-D"):
        check_entity("e", train[:, 0], test, labels)


def test_float32_features_are_accepted():
    train, test, labels = valid_inputs()
    check_entity("e", train.astype(np.float32), test.astype(np.float32), labels)


def test_the_closed_bounds_zero_and_one_are_accepted():
    train, test, labels = valid_inputs()
    train[0, 0], train[1, 0] = 0.0, 1.0
    check_entity("e", train, test, labels)


def test_two_dimensional_labels_are_rejected():
    train, test, labels = valid_inputs()
    with pytest.raises(ContractError, match=r"^e: .*1-D"):
        check_entity("e", train, test, labels.reshape(-1, 1))


def test_nan_labels_are_rejected():
    train, test, labels = valid_inputs()
    nan_labels = labels.astype(np.float64)
    nan_labels[10] = np.nan
    with pytest.raises(ContractError, match="0, 1"):
        check_entity("e", train, test, nan_labels)


def break_rule(rule: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train, test, labels = valid_inputs()
    if rule == 1:
        train = train[:, :37]
    elif rule == 2:
        train = train[:5]
    elif rule == 3:
        test[0, 0] = np.nan
    elif rule == 4:
        train = train + 1.0
    elif rule == 5:
        labels = labels + 2
    elif rule == 6:
        labels = labels[:-1]
    return train, test, labels


@pytest.mark.parametrize("rule", [1, 2, 3, 4, 5, 6])
def test_each_message_names_its_rule(rule):
    train, test, labels = break_rule(rule)
    with pytest.raises(ContractError, match=rf"^e: rule {rule} \("):
        check_entity("e", train, test, labels)


def test_contract_errors_are_tsguard_errors():
    train, test, labels = valid_inputs()
    with pytest.raises(TsguardError):
        check_entity("e", train, test, labels[:-1])
