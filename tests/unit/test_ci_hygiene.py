"""The ``hygiene`` job of ci.yml rejects a Co-Authored-By trailer in every commit a push brings.

The script lives in the workflow, so the test lifts it out of the YAML and runs it with ``bash -e``
(the shell GitHub gives a ``run`` step) against throwaway repositories whose history it can name.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

CI = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
STEP = "- name: Reject co-author trailers"
ZEROS = "0" * 40
TRAILER = "Co-Authored-By: Someone <someone@example.com>"


def hygiene_script() -> str:
    """Return the ``run`` block of the hygiene step, de-indented."""
    lines = CI.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == STEP)
    run = next(i for i in range(start, len(lines)) if lines[i].strip() == "run: |")
    indent = len(lines[run + 1]) - len(lines[run + 1].lstrip())
    body: list[str] = []
    for line in lines[run + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) < indent:
            break
        body.append(line[indent:])
    return "\n".join(body)


def git(cwd: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


@dataclass
class History:
    """Three commits: ``clean`` (the base), ``tainted`` (it has a trailer), ``tip`` (clean)."""

    path: Path
    clean: str
    tainted: str
    tip: str


def commit(path: Path, name: str, message: str) -> str:
    (path / name).write_text(name)
    git(path, "add", name)
    git(path, "-c", "commit.gpgsign=false", "commit", "-q", "-m", message)
    return git(path, "rev-parse", "HEAD")


@pytest.fixture
def history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> History:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    clean = commit(repo, "a", "first")
    tainted = commit(repo, "b", f"second\n\nwhy\n\n{TRAILER.lower()}")
    tip = commit(repo, "c", "third")
    return History(repo, clean, tainted, tip)


def run_hygiene(
    repo: History, *, event: str = "push", before: str = "", base_ref: str = "main"
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "EVENT_NAME": event,
        "BASE_REF": base_ref,
        "BEFORE": before,
        "SHA": repo.tip,
    }
    return subprocess.run(
        ["bash", "-e", "-c", hygiene_script()],
        cwd=repo.path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_script_is_found_in_the_workflow():
    script = hygiene_script()
    assert "co-authored-by" in script.lower()
    assert "exit 1" in script


def test_a_push_is_checked_from_the_commit_before_it(history):
    # The trailer is in the middle of the push, not in its last commit.
    done = run_hygiene(history, before=history.clean)
    assert done.returncode == 1
    assert "Co-Authored-By trailer" in done.stdout


def test_a_push_that_starts_after_the_tainted_commit_is_clean(history):
    assert run_hygiene(history, before=history.tainted).returncode == 0


def test_a_push_with_no_commit_before_it_checks_the_last_commit_only(history):
    # The first push of a branch has an all-zero `before`; there is no range to take.
    assert run_hygiene(history, before=ZEROS).returncode == 0
    (history.path / "d").write_text("d")
    git(history.path, "add", "d")
    git(history.path, "-c", "commit.gpgsign=false", "commit", "-q", "-m", f"last\n\n{TRAILER}")
    history.tip = git(history.path, "rev-parse", "HEAD")
    assert run_hygiene(history, before=ZEROS).returncode == 1


def test_a_before_commit_that_is_gone_falls_back_to_the_last_commit(history):
    # After a force push the old tip may not be in the clone.
    done = run_hygiene(history, before="1" * 40)
    assert done.returncode == 0, done.stderr


def test_a_pull_request_is_still_checked_against_its_base(history):
    git(history.path, "update-ref", "refs/remotes/origin/main", history.clean)
    assert run_hygiene(history, event="pull_request").returncode == 1
    git(history.path, "update-ref", "refs/remotes/origin/main", history.tainted)
    assert run_hygiene(history, event="pull_request").returncode == 0
