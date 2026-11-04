"""The Server Machine Dataset (SMD): pinned source, checksum manifest, verified fetch.

SMD is read from one fixed commit of the OmniAnomaly repository. A manifest packaged with
tsguard records the size and sha256 of every file at that commit, so a download is accepted
only if it is byte-for-byte what the experiments were run on. Raw data is never committed;
it lives in a cache directory (``$TSGUARD_DATA_DIR`` or ``~/.cache/tsguard/smd/<commit>``).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Final

from tsguard.errors import ChecksumError, ContractError, FetchError

SMD_COMMIT: Final = "7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4"
"""OmniAnomaly commit that SMD is read from."""

ENTITIES: Final[tuple[str, ...]] = tuple(
    f"machine-{group}-{index}"
    for group, count in ((1, 8), (2, 9), (3, 11))
    for index in range(1, count + 1)
)
"""The 28 machines in natural order: ``machine-1-1`` ... ``machine-3-11``."""

SPLITS: Final = ("train", "test", "test_label")
"""The three files that exist for every machine."""

DATA_DIR_ENV: Final = "TSGUARD_DATA_DIR"
MANIFEST_RESOURCE: Final = "smd_manifest.json"

_URL_TEMPLATE: Final = (
    "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/{commit}/ServerMachineDataset/{path}"
)
_TIMEOUT_SECONDS: Final = 60
_CHUNK_BYTES: Final = 1 << 20


@dataclass(frozen=True)
class FileEntry:
    """A file of the dataset and the checksum it must have.

    Attributes:
        path: Location relative to the data directory, such as ``"train/machine-1-1.txt"``.
        sha256: Lowercase hex sha256 digest of the file.
        size: Size of the file in bytes.
    """

    path: str
    sha256: str
    size: int


Opener = Callable[[str], bytes]
"""Fetches a URL and returns the response body; injected so tests never touch the network."""


def urllib_opener(url: str) -> bytes:
    """Fetch ``url`` with urllib.

    Args:
        url: The address to read.

    Returns:
        The response body.
    """
    with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
        body: bytes = response.read()
    return body


def load_manifest() -> dict[str, FileEntry]:
    """Read the packaged manifest.

    Returns:
        The entries keyed by relative path.

    Raises:
        ContractError: If the manifest was generated for a commit other than ``SMD_COMMIT``.
    """
    text = resources.files("tsguard.data").joinpath(MANIFEST_RESOURCE).read_text(encoding="utf-8")
    data = json.loads(text)
    if data["commit"] != SMD_COMMIT:
        raise ContractError(
            f"packaged SMD manifest is for commit {data['commit']}, but tsguard pins "
            f"{SMD_COMMIT}; regenerate it with scripts/build_manifest.py"
        )
    return {
        path: FileEntry(path=path, sha256=item["sha256"], size=item["size"])
        for path, item in data["files"].items()
    }


def default_data_dir() -> Path:
    """Return the directory SMD is cached in.

    Returns:
        ``$TSGUARD_DATA_DIR`` if it is set and non-empty, else
        ``~/.cache/tsguard/smd/<SMD_COMMIT>``.
    """
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "tsguard" / "smd" / SMD_COMMIT


def source_url(path: str) -> str:
    """Return the URL of one dataset file at the pinned commit.

    Args:
        path: Relative path such as ``"train/machine-1-1.txt"``.

    Returns:
        A raw.githubusercontent.com URL that names ``SMD_COMMIT``.
    """
    return _URL_TEMPLATE.format(commit=SMD_COMMIT, path=path)


def fetch(
    entities: Sequence[str] = ENTITIES,
    *,
    dest: Path | None = None,
    opener: Opener = urllib_opener,
) -> list[Path]:
    """Make the files of ``entities`` present and verified under ``dest``.

    A file that is already there and matches its manifest entry is kept. Any other file is
    downloaded to ``<file>.part``, verified, and moved into place, so an interrupted run
    never leaves a partial file under the final name.

    Args:
        entities: Machine names to fetch; all 28 by default.
        dest: Data directory; ``default_data_dir()`` if omitted.
        opener: Fetches a URL and returns its bytes.

    Returns:
        The verified paths, entity by entity and split by split.

    Raises:
        ValueError: If a name in ``entities`` is not an SMD machine.
        FetchError: If the opener raises an ``OSError`` (``URLError``, ``HTTPError`` and
            timeouts are all ``OSError``); the message names the file and the URL.
        ChecksumError: If a download does not match the manifest. The ``.part`` file is
            deleted first.
    """
    _check_entities(entities)
    root = dest if dest is not None else default_data_dir()
    manifest = load_manifest()
    paths: list[Path] = []
    for entry in _entries(manifest, entities):
        target = root / entry.path
        if not _matches(target, entry):
            _download(entry, target, opener)
        paths.append(target)
    return paths


def verify(entities: Sequence[str] = ENTITIES, *, dest: Path | None = None) -> None:
    """Check that the files of ``entities`` under ``dest`` match the manifest.

    Args:
        entities: Machine names to check; all 28 by default.
        dest: Data directory; ``default_data_dir()`` if omitted.

    Raises:
        ValueError: If a name in ``entities`` is not an SMD machine.
        ChecksumError: For the first file that is missing or does not match. A file that
            does not match is deleted, so a later ``fetch`` replaces it.
    """
    _check_entities(entities)
    root = dest if dest is not None else default_data_dir()
    manifest = load_manifest()
    for entry in _entries(manifest, entities):
        target = root / entry.path
        if not target.is_file():
            raise ChecksumError(f"{entry.path}: missing from {root}")
        problem = _mismatch(target, entry)
        if problem is not None:
            target.unlink()
            raise ChecksumError(f"{entry.path} in {root}: {problem}; deleted")


def _check_entities(entities: Sequence[str]) -> None:
    unknown = [name for name in entities if name not in ENTITIES]
    if unknown:
        raise ValueError(
            f"unknown SMD entity {', '.join(repr(name) for name in unknown)}; "
            f"valid names: {', '.join(ENTITIES)}"
        )


def _entries(manifest: dict[str, FileEntry], entities: Sequence[str]) -> list[FileEntry]:
    return [manifest[f"{split}/{entity}.txt"] for entity in entities for split in SPLITS]


def _measure(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _mismatch(path: Path, entry: FileEntry) -> str | None:
    """Describe how ``path`` differs from ``entry``, or return None if it matches."""
    size, digest = _measure(path)
    if size == entry.size and digest == entry.sha256:
        return None
    return (
        f"expected {entry.size} bytes with sha256 {entry.sha256}, "
        f"got {size} bytes with sha256 {digest}"
    )


def _matches(path: Path, entry: FileEntry) -> bool:
    return path.is_file() and _mismatch(path, entry) is None


def _download(entry: FileEntry, target: Path, opener: Opener) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(f"{target.name}.part")
    url = source_url(entry.path)
    try:
        body = opener(url)
    except OSError as error:
        raise FetchError(f"{entry.path}: could not download {url}: {error}") from error
    part.write_bytes(body)
    problem = _mismatch(part, entry)
    if problem is not None:
        part.unlink()
        raise ChecksumError(f"{entry.path}: {problem}")
    os.replace(part, target)
