"""The documents state facts the code also states; these tests keep the two from drifting.

A list written into a document goes stale the first time someone extends the code and forgets the
prose. Each test here reads a document and compares it with the code it describes, so the stale
copy fails the build instead of misleading a reader.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from tests.leakage.defects import DEFECTS
from tsguard.repro.claims import BEGIN, END

ROOT = Path(__file__).resolve().parents[2]
TESTING_DOC = ROOT / "docs" / "testing.md"
README = ROOT / "README.md"
ADR_DIR = ROOT / "docs" / "adr"
ADRS = (
    "0001-event-level-protocol.md",
    "0002-label-free-model-selection.md",
    "0003-guards-live-in-the-classes.md",
    "0004-windowed-features.md",
)
ADR_SECTIONS = ("## Context", "## Decision", "## Consequences", "## Rejected alternatives")

DEFECT_TABLE_HEADER = ("id", "defect", "guard")
SCANNED_SUFFIXES = frozenset({".md", ".py", ".toml", ".yml", ".yaml"})
SKIPPED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".superpowers",
        "__pycache__",
        "artifacts",
        "htmlcov",
        "mlruns",
    }
)
SKIPPED_FILES = frozenset({"uv.lock"})
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})")
# Domains reserved for documentation and testing (RFC 2606, RFC 6761) reach no mailbox, so a
# placeholder such as the git identity a test configures is not an address that can leak.
RESERVED_DOMAINS = frozenset({"example.com", "example.net", "example.org"})
RESERVED_TLDS = frozenset({"example", "invalid", "localhost", "test"})


def real_addresses(line: str) -> list[str]:
    """Return the e-mail addresses in ``line`` whose domain is not reserved for examples."""
    found = []
    for match in EMAIL.finditer(line):
        domain = match.group(1).lower()
        reserved = domain in RESERVED_DOMAINS or any(
            domain.endswith(f".{reserved}") for reserved in RESERVED_DOMAINS
        )
        if not reserved and domain.rsplit(".", 1)[-1] not in RESERVED_TLDS:
            found.append(match.group(0))
    return found


def _cells(line: str) -> list[str]:
    return [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]


def defect_table(text: str) -> dict[str, str]:
    """Return ``{id: guard}`` from the table whose header is ``| id | defect | guard |``."""
    rows: dict[str, str] = {}
    inside = False
    for line in text.splitlines():
        if not line.startswith("|"):
            if inside:
                break
            continue
        cells = _cells(line)
        if tuple(cell.lower() for cell in cells) == DEFECT_TABLE_HEADER:
            inside = True
            continue
        if not inside or set("".join(cells)) <= {"-", ":", " "}:
            continue
        assert len(cells) == len(DEFECT_TABLE_HEADER), f"malformed defect row: {line!r}"
        assert cells[0] not in rows, f"defect {cells[0]!r} is listed twice"
        rows[cells[0]] = cells[2]
    return rows


def repository_text_files(root: Path) -> list[Path]:
    """Return every file under ``root`` whose suffix is scanned, outside caches and the venv."""
    found: list[Path] = []
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(
            d for d in subdirs if d not in SKIPPED_DIRS and not d.endswith(".egg-info")
        )
        for name in sorted(files):
            path = Path(directory) / name
            if name not in SKIPPED_FILES and path.suffix in SCANNED_SUFFIXES:
                found.append(path)
    return found


def test_testing_doc_defect_table_matches_defects():
    table = defect_table(TESTING_DOC.read_text(encoding="utf-8"))
    assert table == {ident: guard for ident, (_, guard) in DEFECTS.items()}


def test_defect_table_parser_reads_ids_and_guards():
    text = (
        "Intro.\n\n| id | defect | guard |\n| --- | --- | --- |\n"
        "| `1` | a centered window | `assert_causal` |\n"
        "| `5b` | late freeze | `label_access_audit` |\n"
        "\nAfter.\n\n| id | other | table |\n| --- | --- | --- |\n| `9` | x | y |\n"
    )
    assert defect_table(text) == {"1": "assert_causal", "5b": "label_access_audit"}


def test_readme_has_claims_markers():
    lines = README.read_text(encoding="utf-8").splitlines()
    assert lines.count(BEGIN) == 1, f"README.md needs exactly one line {BEGIN}"
    assert lines.count(END) == 1, f"README.md needs exactly one line {END}"
    assert lines.index(BEGIN) < lines.index(END)


def test_every_adr_has_the_four_sections():
    present = sorted(path.name for path in ADR_DIR.glob("*.md"))
    assert present == sorted(ADRS)
    for name in ADRS:
        headings = (ADR_DIR / name).read_text(encoding="utf-8").splitlines()
        missing = [section for section in ADR_SECTIONS if section not in headings]
        assert not missing, f"{name} lacks {missing}"


def test_no_email_address_in_repository_text():
    files = repository_text_files(ROOT)
    scanned = {path.relative_to(ROOT).as_posix() for path in files}
    # The scan must reach the documents and the workflows, or finding nothing proves nothing.
    assert {"README.md", "pyproject.toml", ".github/workflows/ci.yml"} <= scanned
    assert not {path for path in scanned if path.startswith(".venv/")}
    hits = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in files
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if real_addresses(line)
    ]
    assert hits == []


def test_email_scan_finds_a_real_address_and_ignores_reserved_ones():
    address = "someone" + "@" + "university.edu"
    assert real_addresses(f"write to {address} for help") == [address]
    assert real_addresses(f"<{address.upper()}>") == [address.upper()]
    for placeholder in ("t@example.com", "a@mail.example.org", "b@host.test", "c@x.invalid"):
        assert real_addresses(placeholder) == [], placeholder
    assert real_addresses("@pytest.mark.data and user@localhost") == []
