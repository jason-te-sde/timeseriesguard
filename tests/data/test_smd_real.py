from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsguard.data.contract import N_CHANNELS
from tsguard.data.labels import LabelAudit, Phase
from tsguard.data.smd import ENTITIES, EntityData, load_entity

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def entities(smd_dir: Path) -> list[EntityData]:
    """All 28 machines, loaded once for the module; ``load_entity`` runs the contract."""
    audit = LabelAudit()
    return [load_entity(name, audit=audit, dest=smd_dir) for name in ENTITIES]


def count_segments(labels: np.ndarray) -> int:
    """Count maximal runs of ones, i.e. 0 to 1 transitions (Task 10's segments() replaces it)."""
    padded = np.concatenate([[0], labels.astype(np.int8)])
    return int(np.sum((padded[1:] == 1) & (padded[:-1] == 0)))


def test_every_entity_loads_and_passes_contract(entities):
    assert [entity.name for entity in entities] == list(ENTITIES)
    assert len(entities) == 28


def test_dataset_totals_match_the_spec(entities):
    assert sum(entity.train_end for entity in entities) == 708_405
    assert sum(entity.n_test for entity in entities) == 708_420
    assert {entity.train.shape[1] for entity in entities} == {N_CHANNELS}
    assert {entity.test.shape[1] for entity in entities} == {N_CHANNELS}


def test_anomaly_segment_count_is_327(entities):
    total = 0
    for entity in entities:
        with entity.labels.audit.phase(Phase.FINAL_EVALUATION):
            total += count_segments(entity.labels.read("test_smd_real"))
    assert total == 327


def test_every_entity_has_a_constant_train_channel(entities):
    for entity in entities:
        constant = int(np.sum(entity.train.std(axis=0) == 0.0))
        assert 2 <= constant <= 10, f"{entity.name}: {constant} constant train channels"
