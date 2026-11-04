from __future__ import annotations

from pathlib import Path

import pytest

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"


def run_data_test(pytester: pytest.Pytester) -> pytest.RunResult:
    pytester.makeconftest(CONFTEST.read_text())
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.data
        def test_needs_the_dataset(smd_dir):
            raise AssertionError("the dataset should have been reported missing")
        """
    )
    return pytester.runpytest_subprocess("-m", "data", "-rs")


def test_missing_data_fails_instead_of_skipping(pytester, monkeypatch, tmp_path):
    monkeypatch.setenv("TSGUARD_DATA_DIR", str(tmp_path / "no-such-cache"))
    result = run_data_test(pytester)
    result.assert_outcomes(errors=1, skipped=0, passed=0)
    result.stdout.fnmatch_lines(["*SMD data missing: run `uv run tsguard data fetch`*"])


def test_the_failure_names_the_first_missing_file(pytester, monkeypatch, tmp_path):
    monkeypatch.setenv("TSGUARD_DATA_DIR", str(tmp_path / "no-such-cache"))
    result = run_data_test(pytester)
    result.stdout.fnmatch_lines(["*train/machine-1-1.txt: missing*"])
