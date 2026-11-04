import itertools
import math

import numpy as np
import pytest

from tsguard.eval.inject import (
    KINDS,
    InjectionConfig,
    inject,
    reference_sigma,
)

CHANNELS = 38
START, END = 300, 4300


def _series(rng: np.random.Generator, rows: int = 5000) -> np.ndarray:
    return rng.random((rows, CHANNELS))


def _sigma(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.01, 0.2, CHANNELS)


def _labelled_rows(n_rows: int, start: int, labels: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_rows, dtype=bool)
    mask[start : start + len(labels)] = labels == 1
    return mask


def test_same_seed_same_output():
    data_rng = np.random.default_rng(3)
    X = data_rng.random((5000, CHANNELS))
    sigma = data_rng.uniform(0.01, 0.2, CHANNELS)
    first = inject(X, start=START, end=END, sigma=sigma, rng=np.random.default_rng(7))
    second = inject(X, start=START, end=END, sigma=sigma, rng=np.random.default_rng(7))
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_rows_outside_segments_are_untouched(rng):
    X = _series(rng)
    out, labels, segments = inject(X, start=START, end=END, sigma=_sigma(rng), rng=rng)
    labelled = _labelled_rows(len(X), START, labels)
    assert out.shape == X.shape
    assert out.dtype == np.float64
    assert out[~labelled].tobytes() == X[~labelled].tobytes()
    assert not np.array_equal(out[labelled], X[labelled])
    assert len(segments) > 0


def test_labels_mark_exactly_the_segments(rng):
    X = _series(rng)
    _, labels, segments = inject(X, start=START, end=END, sigma=_sigma(rng), rng=rng)
    expected = np.zeros(END - START, dtype=np.int8)
    for seg in segments:
        expected[seg.start - START : seg.start - START + seg.length] = 1
    assert labels.dtype == np.int8
    assert labels.shape == (END - START,)
    assert np.array_equal(labels, expected)
    assert labels.sum() == sum(seg.length for seg in segments)


def test_segments_keep_the_gap(rng):
    X = _series(rng)
    _, _, segments = inject(X, start=START, end=END, sigma=_sigma(rng), rng=rng)
    assert segments[0].start >= START + 60
    assert segments[-1].start + segments[-1].length <= END - 60
    for prev, nxt in itertools.pairwise(segments):
        assert nxt.start - (prev.start + prev.length) >= 120


def test_input_is_not_mutated(rng):
    X = _series(rng)
    before = X.copy()
    X.flags.writeable = False
    out, _, _ = inject(X, start=START, end=END, sigma=_sigma(rng), rng=rng)
    assert np.array_equal(X, before)
    assert out is not X
    assert not np.shares_memory(out, X)


def test_block_of_4000_rows_gets_4_segments(rng):
    X = _series(rng)
    _, _, segments = inject(X, start=START, end=END, sigma=_sigma(rng), rng=rng)
    assert len(segments) == 4
    for i, seg in enumerate(segments):
        assert START + i * 1000 <= seg.start
        assert seg.start + seg.length <= START + (i + 1) * 1000


def test_all_kinds_appear_in_a_long_block():
    X = np.zeros((40_000, CHANNELS))
    _, _, segments = inject(
        X, start=0, end=40_000, sigma=np.ones(CHANNELS), rng=np.random.default_rng(0)
    )
    assert {seg.kind for seg in segments} == set(KINDS)


def test_constant_channel_uses_fallback_sigma():
    X = np.zeros((100, 3))
    X[:, 1] = np.linspace(0, 1, 100)
    s = reference_sigma(X)
    assert s[0] == 0.1
    assert s[2] == 0.1
    assert s[1] == pytest.approx(np.std(X[:, 1]))
    assert reference_sigma(X, fallback=0.25)[0] == 0.25


def test_too_small_block_is_value_error(rng):
    X = _series(rng, rows=400)
    with pytest.raises(ValueError, match="200 rows"):
        inject(X, start=100, end=300, sigma=_sigma(rng), rng=rng)


def test_segment_fields_follow_the_config(rng):
    X = _series(rng, rows=40_000)
    config = InjectionConfig()
    _, _, segments = inject(X, start=0, end=40_000, sigma=_sigma(rng), rng=rng, config=config)
    assert len(segments) == 40
    for seg in segments:
        assert type(seg.start) is int
        assert type(seg.length) is int
        assert all(type(c) is int for c in seg.channels)
        assert seg.channels == tuple(sorted(set(seg.channels)))
        assert 1 <= len(seg.channels) <= config.max_channels
        assert seg.channels[0] >= 0
        assert seg.channels[-1] < CHANNELS
        assert config.min_sigma <= seg.magnitude <= config.max_sigma
        if seg.kind == "spike":
            assert 1 <= seg.length <= 3
        else:
            assert config.min_len <= seg.length <= config.max_len


def test_spike_and_level_shift_add_signed_magnitude_times_sigma(rng):
    X = _series(rng, rows=40_000)
    sigma = _sigma(rng)
    out, _, segments = inject(X, start=0, end=40_000, sigma=sigma, rng=rng)
    checked = set()
    for seg in segments:
        if seg.kind not in ("spike", "level_shift"):
            continue
        checked.add(seg.kind)
        rows = slice(seg.start, seg.start + seg.length)
        ch = list(seg.channels)
        delta = out[rows][:, ch] - X[rows][:, ch]
        assert np.allclose(np.abs(delta), seg.magnitude * sigma[ch])
        assert len(set(np.sign(delta).ravel())) == 1
        other = np.setdiff1d(np.arange(CHANNELS), ch)
        assert np.array_equal(out[rows][:, other], X[rows][:, other])
    assert checked == {"spike", "level_shift"}


def test_flatline_holds_the_value_of_its_first_row(rng):
    X = _series(rng, rows=40_000)
    out, _, segments = inject(X, start=0, end=40_000, sigma=_sigma(rng), rng=rng)
    flat = [seg for seg in segments if seg.kind == "flatline"]
    assert flat
    for seg in flat:
        rows = slice(seg.start, seg.start + seg.length)
        ch = list(seg.channels)
        assert np.array_equal(out[rows][:, ch], np.tile(X[seg.start, ch], (seg.length, 1)))
        other = np.setdiff1d(np.arange(CHANNELS), ch)
        assert np.array_equal(out[rows][:, other], X[rows][:, other])


def test_variance_burst_noise_scales_with_sigma(rng):
    X = _series(rng, rows=40_000)
    sigma = _sigma(rng)
    config = InjectionConfig(rows_per_segment=300)
    out, _, segments = inject(X, start=0, end=40_000, sigma=sigma, rng=rng, config=config)
    parts = []
    for seg in segments:
        if seg.kind != "variance_burst":
            continue
        rows = slice(seg.start, seg.start + seg.length)
        ch = list(seg.channels)
        delta = out[rows][:, ch] - X[rows][:, ch]
        parts.append((delta / (seg.magnitude * sigma[ch])).ravel())
    z = np.concatenate(parts)
    assert z.size > 1000
    assert abs(z.mean()) < 0.1
    assert 0.9 < z.std() < 1.1


def test_draw_order_is_the_documented_one():
    data_rng = np.random.default_rng(5)
    X = data_rng.random((6000, CHANNELS))
    sigma = data_rng.uniform(0.01, 0.2, CHANNELS)
    config = InjectionConfig()
    start, end = 500, 5500

    out, _, segments = inject(
        X, start=start, end=end, sigma=sigma, rng=np.random.default_rng(11), config=config
    )

    draws = np.random.default_rng(11)
    expected = X.copy()
    n = max(2, (end - start) // config.rows_per_segment)
    slot = (end - start) // n
    assert len(segments) == n
    for i, seg in enumerate(segments):
        kind = KINDS[int(draws.integers(4))]
        if kind == "spike":
            length = int(draws.integers(1, 4))
        else:
            length = round(
                math.exp(draws.uniform(math.log(config.min_len), math.log(config.max_len)))
            )
        length = min(length, slot - config.gap)
        offset = int(draws.integers(0, slot - config.gap - length + 1))
        s = start + i * slot + config.gap // 2 + offset
        m = int(draws.integers(1, config.max_channels + 1))
        channels = sorted(int(c) for c in draws.choice(CHANNELS, size=m, replace=False))
        magnitude = float(draws.uniform(config.min_sigma, config.max_sigma))
        sign = 1.0 if draws.random() < 0.5 else -1.0
        assert (seg.kind, seg.start, seg.length) == (kind, s, length)
        assert seg.channels == tuple(channels)
        assert seg.magnitude == magnitude
        rows = slice(s, s + length)
        if kind in ("spike", "level_shift"):
            expected[rows, channels] += sign * magnitude * sigma[channels]
        elif kind == "variance_burst":
            expected[rows, channels] += (
                draws.standard_normal((length, m)) * magnitude * sigma[channels]
            )
        else:
            expected[rows, channels] = X[s, channels]
    assert np.array_equal(out, expected)


@pytest.mark.parametrize(("start", "end"), [(-1, 4000), (1000, 99_999), (500, 500)])
def test_block_outside_the_series_is_value_error(rng, start, end):
    X = _series(rng)
    with pytest.raises(ValueError, match="does not lie inside"):
        inject(X, start=start, end=end, sigma=_sigma(rng), rng=rng)


def test_sigma_of_the_wrong_length_is_value_error(rng):
    X = _series(rng)
    with pytest.raises(ValueError, match="sigma"):
        inject(X, start=START, end=END, sigma=np.ones(CHANNELS - 1), rng=rng)
