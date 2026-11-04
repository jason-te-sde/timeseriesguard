"""Random scores beat the point-adjust protocol, which is why this project reports another one.

Point adjust credits a whole anomaly segment as soon as one of its points is flagged, so a
detector that knows nothing about the data scores as if it were state of the art. These tests
pin that fact on synthetic labels and on the real SMD labels. If the synthetic test fails,
point adjust is implemented wrongly: the protocol did not get better.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsguard.data.labels import LabelAudit, Phase
from tsguard.data.smd import ENTITIES, load_entity
from tsguard.eval.protocols import (
    ProtocolParams,
    adjust_point_adjust,
    best_f1,
    event_level_auc_pr,
)

SEEDS = (0, 1, 2, 3, 4)
CALLER = "test_point_adjust_is_gameable"
PA_SOTA_F1 = 0.9
MACRO_PA_F1 = 0.7
MACRO_POINT_WISE_F1 = 0.1
MACRO_EVENT_AUC_PR = 0.1
TRIVIAL_BASELINE_SLACK = 0.02


def _synthetic_labels() -> np.ndarray:
    """40,000 rows with four 1,000-row segments, anomaly rate 0.1 (shaped like machine-1-1)."""
    y = np.zeros(40_000, dtype=int)
    for start in (5_000, 15_000, 25_000, 35_000):
        y[start : start + 1_000] = 1
    return y


def _trivial_point_wise_f1(rate: float) -> float:
    """Best point-wise F1 of a detector that knows nothing: flag everything."""
    return 2 * rate / (1 + rate)


@pytest.fixture(scope="module")
def smd_labels(smd_dir: Path) -> dict[str, np.ndarray]:
    """The test labels of all 28 machines, read once inside the final-evaluation phase."""
    audit = LabelAudit()
    entities = [load_entity(name, audit=audit, dest=smd_dir) for name in ENTITIES]
    with audit.phase(Phase.FINAL_EVALUATION):
        return {entity.name: entity.labels.read(CALLER) for entity in entities}


@pytest.mark.parametrize("seed", SEEDS)
def test_random_scores_reach_sota_under_point_adjust(seed):
    """Random scores reach F1 > 0.9 under PA. If this test fails, PA is implemented wrong."""
    y = _synthetic_labels()
    r = np.random.default_rng(seed).random(len(y))
    a = y.mean()
    assert best_f1(adjust_point_adjust(r, y), y)[0] > PA_SOTA_F1
    # Point-wise stays at the trivial all-positive baseline.
    assert best_f1(r, y)[0] <= _trivial_point_wise_f1(a) + TRIVIAL_BASELINE_SLACK


@pytest.mark.data
@pytest.mark.parametrize("seed", SEEDS)
def test_random_scores_on_machine_1_1(seed, smd_labels):
    # Labels of machine-1-1 (a = 0.0946): PA F1* > 0.9 and point-wise F1* <= 2a/(1+a) + 0.02.
    y = smd_labels["machine-1-1"]
    r = np.random.default_rng(seed).random(len(y))
    a = y.mean()
    assert best_f1(adjust_point_adjust(r, y), y)[0] > PA_SOTA_F1
    assert best_f1(r, y)[0] <= _trivial_point_wise_f1(a) + TRIVIAL_BASELINE_SLACK


@pytest.mark.data
@pytest.mark.parametrize("seed", SEEDS)
def test_random_scores_across_smd(seed, smd_labels):
    # Over all 28 entities: macro PA F1* > 0.7, macro point-wise F1* < 0.1, and macro
    # event-level AUC-PR (L=60, C=60) < 0.1: the protocol this project reports cannot be
    # gamed this way.
    params = ProtocolParams()
    pa_f1, point_wise_f1, event_auc_pr = [], [], []
    for index, name in enumerate(ENTITIES):
        y = smd_labels[name]
        r = np.random.default_rng([seed, index]).random(len(y))
        pa_f1.append(best_f1(adjust_point_adjust(r, y), y)[0])
        point_wise_f1.append(best_f1(r, y)[0])
        event_auc_pr.append(event_level_auc_pr(r, y, params))
    assert np.mean(pa_f1) > MACRO_PA_F1, f"macro PA F1* = {np.mean(pa_f1):.4f}"
    assert np.mean(point_wise_f1) < MACRO_POINT_WISE_F1, (
        f"macro point-wise F1* = {np.mean(point_wise_f1):.4f}"
    )
    assert np.mean(event_auc_pr) < MACRO_EVENT_AUC_PR, (
        f"macro event-level AUC-PR = {np.mean(event_auc_pr):.4f}"
    )
