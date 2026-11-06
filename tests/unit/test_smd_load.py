from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import make_entity
from tsguard.data import smd
from tsguard.data.labels import LabelAudit, Phase
from tsguard.data.smd import EntityData, load_entity
from tsguard.errors import ChecksumError, ContractError, LabelAccessError

NAME = "machine-1-1"


def write_files(root: Path, name: str, train, test, labels) -> None:
    for split, array, fmt in (
        ("train", train, "%.6f"),
        ("test", test, "%.6f"),
        ("test_label", labels, "%d"),
    ):
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        np.savetxt(folder / f"{name}.txt", array, fmt=fmt, delimiter=",")


@pytest.fixture
def written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, EntityData]:
    """A synthetic entity written in SMD's file format; checksum verification is bypassed."""
    source = make_entity(name=NAME)
    with source.labels.audit.phase(Phase.FINAL_EVALUATION):
        labels = source.labels.read("fixture")
    write_files(tmp_path, NAME, source.train, source.test, labels)
    monkeypatch.setattr(smd, "verify", lambda entities, *, dest=None: None)
    return tmp_path, source


def read_labels(entity: EntityData) -> np.ndarray:
    with entity.labels.audit.phase(Phase.FINAL_EVALUATION):
        return entity.labels.read("test_smd_load")


def test_verification_runs_before_anything_is_parsed(tmp_path):
    with pytest.raises(ChecksumError, match=rf"train/{NAME}.txt: missing"):
        load_entity(NAME, audit=LabelAudit(), dest=tmp_path)


def test_an_unknown_entity_is_refused(tmp_path):
    with pytest.raises(ValueError, match="machine-9-9"):
        load_entity("machine-9-9", audit=LabelAudit(), dest=tmp_path)


def test_verify_is_asked_about_exactly_this_entity_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[tuple[list[str], Path | None]] = []

    def record(entities, *, dest=None):
        calls.append((list(entities), dest))
        raise ChecksumError("stop here")

    monkeypatch.setattr(smd, "verify", record)
    with pytest.raises(ChecksumError, match="stop here"):
        load_entity(NAME, audit=LabelAudit(), dest=tmp_path)
    assert calls == [([NAME], tmp_path)]


def test_loads_the_three_files_into_an_entity(written):
    root, source = written
    audit = LabelAudit()
    entity = load_entity(NAME, audit=audit, dest=root)
    assert entity.name == NAME
    assert entity.train.dtype == np.float64
    assert entity.test.dtype == np.float64
    assert entity.train.shape == source.train.shape
    assert np.allclose(entity.train, source.train, atol=1e-6)
    assert np.allclose(entity.test, source.test, atol=1e-6)
    assert entity.train_end == len(source.train)
    assert entity.n_test == len(source.test)
    assert np.array_equal(entity.full, np.vstack([entity.train, entity.test]))
    labels = read_labels(entity)
    assert labels.dtype == np.int8
    assert np.array_equal(labels, read_labels(source))


def test_labels_report_to_the_audit_that_was_passed_in(written):
    root, _ = written
    audit = LabelAudit()
    entity = load_entity(NAME, audit=audit, dest=root)
    assert entity.labels.audit is audit
    with pytest.raises(LabelAccessError):
        entity.labels.read("too_early")
    with audit.phase(Phase.FINAL_EVALUATION):
        entity.labels.read("final_metrics")
    assert audit.accesses == ((NAME, "final_metrics", Phase.FINAL_EVALUATION),)


def test_loaded_arrays_are_read_only(written):
    root, _ = written
    entity = load_entity(NAME, audit=LabelAudit(), dest=root)
    for array in (entity.train, entity.test):
        assert array.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = 0.5


def test_the_default_directory_comes_from_the_environment(written, monkeypatch):
    root, _ = written
    monkeypatch.setenv("TSGUARD_DATA_DIR", str(root))
    entity = load_entity(NAME, audit=LabelAudit())
    assert entity.n_test == 2000


def test_the_contract_is_enforced_on_what_was_read(written):
    root, source = written
    bad = np.array(source.train)
    bad[5, 5] = 1.5
    write_files(root, NAME, bad, source.test, read_labels(source))
    with pytest.raises(ContractError, match=rf"^{NAME}: rule 4"):
        load_entity(NAME, audit=LabelAudit(), dest=root)


def test_a_file_with_a_single_row_fails_the_contract_instead_of_crashing(written):
    root, source = written
    write_files(root, NAME, source.train[:1], source.test, read_labels(source))
    with pytest.raises(ContractError, match=rf"^{NAME}: rule 2"):
        load_entity(NAME, audit=LabelAudit(), dest=root)


def test_labels_of_the_wrong_length_fail_the_contract(written):
    root, source = written
    write_files(root, NAME, source.train, source.test, read_labels(source)[:-1])
    with pytest.raises(ContractError, match=rf"^{NAME}: rule 6"):
        load_entity(NAME, audit=LabelAudit(), dest=root)
