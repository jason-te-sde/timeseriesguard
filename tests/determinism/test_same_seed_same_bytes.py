"""One seed, one set of bytes: a run repeated in one process reproduces every digest.

The comparison is on `deterministic_view`, the metrics without their ``environment`` (versions
and a timestamp), hashed with `json_digest`, and on the digests of every fitted pipeline,
fitted detector and score array. A model that seeded anything from the clock, an unordered set
or a shared random stream would fail here. The seed test is the control: a different seed must
change the digests, or the equality above would hold for any code that ignored the seed.

The tiny profile runs on synthetic entities in seconds; the smoke test repeats the claim on a
real machine with the settings CI ships.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest
import torch

from tests.conftest import TINY, load_synthetic
from tsguard.data.labels import LabelAudit
from tsguard.data.smd import EntityData, load_entity
from tsguard.experiment.ablation import EntityLoader
from tsguard.experiment.reproduce import SMOKE, Profile, reproduce
from tsguard.experiment.tracking import MemoryTracker
from tsguard.models import MODEL_NAMES
from tsguard.repro.hashing import json_digest
from tsguard.repro.report import Metrics, deterministic_view, read_metrics

DIGEST_NAMES = {"pipeline", "detector", "test_scores"}
STOCHASTIC_MODELS = {"iforest", "cnn_ae", "lstm_ae"}  # zscore draws nothing from the seed
METRICS_FILE = "metrics.json"


@pytest.fixture(scope="module", autouse=True)
def restore_runtime() -> Iterator[None]:
    """Give back the torch settings that `reproduce` changes for the whole process."""
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


def run(profile: Profile, loader: EntityLoader, out_dir: Path) -> Metrics:
    """Run ``reproduce`` with a tracker of its own, so nothing is shared between two runs."""
    return reproduce(profile, loader=loader, tracker=MemoryTracker(), out_dir=out_dir)


def view_digest(metrics: Metrics) -> str:
    return json_digest(deterministic_view(metrics))


def without_seed(metrics: Metrics) -> dict[str, object]:
    """Return the deterministic view minus the root seed, which the view itself records."""
    return {key: value for key, value in deterministic_view(metrics).items() if key != "seed"}


class Difference(NamedTuple):
    """One per-entity digest that two runs disagree on."""

    model: str
    entity: str
    name: str
    left: str | None
    right: str | None

    def __str__(self) -> str:
        return f"{self.model}/{self.entity}/{self.name}: {self.left} != {self.right}"


def digest_differences(a: Metrics, b: Metrics) -> list[Difference]:
    """List every per-entity digest that differs between two runs of the same profile."""
    assert a.models.keys() == b.models.keys()
    differences = []
    for model in a.models:
        first, second = a.models[model].per_entity, b.models[model].per_entity
        assert [e.entity for e in first] == [e.entity for e in second]
        for left, right in zip(first, second, strict=True):
            for name in sorted(left.digests.keys() | right.digests.keys()):
                if left.digests.get(name) != right.digests.get(name):
                    differences.append(
                        Difference(
                            model,
                            left.entity,
                            name,
                            left.digests.get(name),
                            right.digests.get(name),
                        )
                    )
    return differences


def assert_digests_are_there(metrics: Metrics) -> None:
    """Refuse an empty comparison: every model and entity must carry all three digests."""
    assert set(metrics.models) == set(MODEL_NAMES)
    for summary in metrics.models.values():
        assert summary.per_entity
        for entity in summary.per_entity:
            assert set(entity.digests) == DIGEST_NAMES, entity.entity


def assert_same_bytes(a: Metrics, b: Metrics, *, seed: int) -> None:
    assert_digests_are_there(a)
    differences = digest_differences(a, b)
    assert not differences, "\n".join(map(str, differences)) + f"\nseed={seed}"
    assert view_digest(a) == view_digest(b), f"seed={seed}"


@pytest.fixture(scope="module")
def tiny_seed_0(tmp_path_factory: pytest.TempPathFactory) -> tuple[Metrics, Path]:
    """Run the tiny profile once at its own seed; both slow tests compare against this run."""
    out_dir = tmp_path_factory.mktemp("tiny-seed-0")
    return run(TINY, load_synthetic, out_dir), out_dir


@pytest.mark.slow
def test_tiny_reproduce_twice_gives_identical_bytes(tiny_seed_0, tmp_path):
    a, a_dir = tiny_seed_0
    b = run(TINY, load_synthetic, tmp_path)
    assert_same_bytes(a, b, seed=TINY.run.seed)
    # what reached the disk is what was returned, so the files agree as well
    assert view_digest(read_metrics(a_dir / METRICS_FILE)) == view_digest(a)
    assert view_digest(read_metrics(tmp_path / METRICS_FILE)) == view_digest(b)


@pytest.mark.slow
def test_a_different_seed_changes_the_bytes(tiny_seed_0, tmp_path):
    a, _ = tiny_seed_0
    other = dataclasses.replace(TINY, run=dataclasses.replace(TINY.run, seed=1))
    c = run(other, load_synthetic, tmp_path)
    assert (a.seed, c.seed) == (0, 1)
    # the view records the seed itself, so that field alone would make the digests differ
    assert json_digest(without_seed(a)) != json_digest(without_seed(c))
    changed = {difference.model for difference in digest_differences(a, c)}
    assert changed >= STOCHASTIC_MODELS, f"digests changed only for {sorted(changed)}"


@pytest.mark.data
def test_smoke_profile_twice_on_real_data(tmp_path, smd_dir):
    profile = dataclasses.replace(
        SMOKE,
        name="smoke-determinism",
        entities=("machine-1-1",),
        n_trials=1,
        ablation_seeds=(0,),
    )

    def loader(name: str, audit: LabelAudit) -> EntityData:
        return load_entity(name, audit=audit, dest=smd_dir)

    a = run(profile, loader, tmp_path / "a")
    b = run(profile, loader, tmp_path / "b")
    assert_same_bytes(a, b, seed=profile.run.seed)
