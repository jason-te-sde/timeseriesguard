from pathlib import Path

import numpy as np
import pytest

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"


def test_default_seed_is_20261103(request, monkeypatch):
    monkeypatch.delenv("TSGUARD_TEST_SEED", raising=False)
    rng = request.getfixturevalue("rng")
    assert rng.integers(0, 2**32) == np.random.default_rng(20261103).integers(0, 2**32)


def test_seed_comes_from_the_environment_and_is_recorded(request, monkeypatch):
    monkeypatch.setenv("TSGUARD_TEST_SEED", "7")
    rng = request.getfixturevalue("rng")
    assert rng.integers(0, 2**32) == np.random.default_rng(7).integers(0, 2**32)
    assert ("seed", 7) in request.node.user_properties


def test_a_failing_test_reports_its_seed(pytester, monkeypatch):
    monkeypatch.setenv("TSGUARD_TEST_SEED", "424242")
    pytester.makeconftest(CONFTEST.read_text())
    pytester.makepyfile(
        """
        def test_boom(rng):
            assert False
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*seed=424242*"])


def test_a_passing_test_does_not_mention_a_seed(pytester, monkeypatch):
    monkeypatch.setenv("TSGUARD_TEST_SEED", "424242")
    pytester.makeconftest(CONFTEST.read_text())
    pytester.makepyfile(
        """
        def test_fine(rng):
            assert rng is not None
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)
    assert "seed=" not in result.stdout.str()


def test_a_non_integer_seed_names_the_variable(request, monkeypatch):
    monkeypatch.setenv("TSGUARD_TEST_SEED", "twelve")
    with pytest.raises(ValueError, match="TSGUARD_TEST_SEED") as excinfo:
        request.getfixturevalue("rng")
    assert "twelve" in str(excinfo.value)
