from __future__ import annotations

import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from tsguard.data import smd
from tsguard.data.smd import (
    ENTITIES,
    SMD_COMMIT,
    SPLITS,
    FileEntry,
    default_data_dir,
    fetch,
    load_manifest,
    source_url,
    urllib_opener,
    verify,
)
from tsguard.errors import ChecksumError, ContractError, FetchError

ENTITY = "machine-1-1"
TRAIN_PATH = f"train/{ENTITY}.txt"
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_manifest.py"


class FakeOpener:
    """Serves known bytes by URL and records every request."""

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.urls: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.urls)

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        return self.bodies[url]


@pytest.fixture
def served() -> dict[str, bytes]:
    """Return the bytes the fake origin serves for each file of one entity."""
    return {f"{split}/{ENTITY}.txt": f"0.5,0.25,0.125 {split}\n".encode() * 200 for split in SPLITS}


@pytest.fixture
def fake_manifest(
    monkeypatch: pytest.MonkeyPatch, served: dict[str, bytes]
) -> dict[str, FileEntry]:
    """Replace the packaged manifest with one whose digests match ``served``."""
    manifest = {
        path: FileEntry(path=path, sha256=hashlib.sha256(body).hexdigest(), size=len(body))
        for path, body in served.items()
    }
    monkeypatch.setattr(smd, "load_manifest", lambda: manifest)
    return manifest


@pytest.fixture
def fake_opener(served: dict[str, bytes]) -> FakeOpener:
    return FakeOpener({source_url(path): body for path, body in served.items()})


def _populate(root: Path, served: dict[str, bytes]) -> None:
    for path, body in served.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(body)


def test_manifest_lists_84_files_for_28_entities():
    m = load_manifest()
    assert len(m) == 84
    assert len(ENTITIES) == 28
    assert {p.split("/")[0] for p in m} == {"train", "test", "test_label"}


def test_entities_are_in_natural_order():
    assert ENTITIES[:2] == ("machine-1-1", "machine-1-2")
    assert ENTITIES[-1] == "machine-3-11"
    assert ENTITIES.index("machine-3-10") == 26


def test_manifest_for_another_commit_is_refused(monkeypatch):
    monkeypatch.setattr(smd, "SMD_COMMIT", "0" * 40)
    with pytest.raises(ContractError, match="regenerate"):
        load_manifest()


def test_fetch_downloads_and_verifies(tmp_path, served, fake_manifest, fake_opener):
    paths = fetch([ENTITY], dest=tmp_path, opener=fake_opener)
    assert paths == [tmp_path / split / f"{ENTITY}.txt" for split in SPLITS]
    for path in served:
        assert (tmp_path / path).read_bytes() == served[path]
    assert fake_opener.calls == 3
    assert not list(tmp_path.rglob("*.part"))


def test_fetch_skips_files_that_already_verify(tmp_path, fake_manifest, fake_opener):
    fetch([ENTITY], dest=tmp_path, opener=fake_opener)
    paths = fetch([ENTITY], dest=tmp_path, opener=fake_opener)
    assert fake_opener.calls == 3
    assert len(paths) == 3


def test_fetch_without_dest_uses_the_data_dir(
    monkeypatch, tmp_path, served, fake_manifest, fake_opener
):
    monkeypatch.setenv("TSGUARD_DATA_DIR", str(tmp_path))
    fetch([ENTITY], opener=fake_opener)
    assert (tmp_path / TRAIN_PATH).read_bytes() == served[TRAIN_PATH]


def test_checksum_mismatch_raises_and_leaves_no_file(tmp_path, served, fake_manifest):
    wrong = b"not the served bytes"
    with pytest.raises(ChecksumError) as excinfo:
        fetch([ENTITY], dest=tmp_path, opener=lambda url: wrong)
    message = str(excinfo.value)
    assert TRAIN_PATH in message
    assert fake_manifest[TRAIN_PATH].sha256 in message
    assert hashlib.sha256(wrong).hexdigest() in message
    assert not (tmp_path / TRAIN_PATH).exists()
    assert not (tmp_path / f"{TRAIN_PATH}.part").exists()


def test_size_mismatch_is_reported_even_when_the_digest_matches(
    monkeypatch, tmp_path, served, fake_manifest, fake_opener
):
    entry = fake_manifest[TRAIN_PATH]
    wrong_size = {**fake_manifest, TRAIN_PATH: FileEntry(TRAIN_PATH, entry.sha256, entry.size + 1)}
    monkeypatch.setattr(smd, "load_manifest", lambda: wrong_size)
    with pytest.raises(ChecksumError, match=f"{entry.size + 1} bytes"):
        fetch([ENTITY], dest=tmp_path, opener=fake_opener)
    assert not (tmp_path / TRAIN_PATH).exists()


def test_truncated_file_is_removed_and_refetched(tmp_path, served, fake_manifest, fake_opener):
    _populate(tmp_path, served)
    (tmp_path / TRAIN_PATH).write_bytes(served[TRAIN_PATH][: len(served[TRAIN_PATH]) // 2])

    with pytest.raises(ChecksumError, match=TRAIN_PATH):
        verify([ENTITY], dest=tmp_path)
    assert not (tmp_path / TRAIN_PATH).exists()

    fetch([ENTITY], dest=tmp_path, opener=fake_opener)
    assert fake_opener.urls == [source_url(TRAIN_PATH)]
    assert (tmp_path / TRAIN_PATH).read_bytes() == served[TRAIN_PATH]


def test_verify_deletes_only_the_mismatching_file(tmp_path, served, fake_manifest):
    test_path = f"test/{ENTITY}.txt"
    label_path = f"test_label/{ENTITY}.txt"
    _populate(tmp_path, served)
    (tmp_path / test_path).write_bytes(b"corrupt")

    with pytest.raises(ChecksumError, match=test_path):
        verify([ENTITY], dest=tmp_path)

    assert (tmp_path / TRAIN_PATH).read_bytes() == served[TRAIN_PATH]
    assert not (tmp_path / test_path).exists()
    assert (tmp_path / label_path).read_bytes() == served[label_path]


def test_verify_mismatch_names_the_cache_root_and_says_the_file_was_deleted(
    tmp_path, served, fake_manifest
):
    _populate(tmp_path, served)
    (tmp_path / TRAIN_PATH).write_bytes(b"corrupt")

    with pytest.raises(ChecksumError) as excinfo:
        verify([ENTITY], dest=tmp_path)

    message = str(excinfo.value)
    assert f"{TRAIN_PATH} in {tmp_path}: expected" in message
    assert message.endswith("; deleted")
    assert fake_manifest[TRAIN_PATH].sha256 in message
    assert hashlib.sha256(b"corrupt").hexdigest() in message


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("connection refused"),
        urllib.error.HTTPError("http://example.test/x", 404, "Not Found", Message(), None),
        TimeoutError("timed out"),
    ],
    ids=lambda failure: type(failure).__name__,
)
def test_an_opener_failure_becomes_a_fetch_error_naming_the_file_and_url(
    tmp_path, fake_manifest, failure
):
    def failing_opener(url: str) -> bytes:
        raise failure

    with pytest.raises(FetchError) as excinfo:
        fetch([ENTITY], dest=tmp_path, opener=failing_opener)

    message = str(excinfo.value)
    assert TRAIN_PATH in message
    assert source_url(TRAIN_PATH) in message
    assert excinfo.value.__cause__ is failure
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_an_opener_bug_is_not_disguised_as_a_fetch_error(tmp_path, fake_manifest):
    def buggy_opener(url: str) -> bytes:
        raise RuntimeError("bug in the opener")

    with pytest.raises(RuntimeError, match="bug in the opener"):
        fetch([ENTITY], dest=tmp_path, opener=buggy_opener)


def test_a_crash_before_the_move_leaves_nothing_under_the_final_name(
    monkeypatch, tmp_path, served, fake_manifest, fake_opener
):
    def crash(src: object, dst: object) -> None:
        raise OSError("simulated crash")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", crash)
        with pytest.raises(OSError, match="simulated crash"):
            fetch([ENTITY], dest=tmp_path, opener=fake_opener)
    assert not (tmp_path / TRAIN_PATH).exists()

    fetch([ENTITY], dest=tmp_path, opener=fake_opener)
    assert (tmp_path / TRAIN_PATH).read_bytes() == served[TRAIN_PATH]
    assert not list(tmp_path.rglob("*.part"))


def test_fetch_replaces_a_corrupt_file_without_verify(tmp_path, served, fake_manifest, fake_opener):
    (tmp_path / "train").mkdir()
    (tmp_path / TRAIN_PATH).write_bytes(b"corrupt")
    fetch([ENTITY], dest=tmp_path, opener=fake_opener)
    assert (tmp_path / TRAIN_PATH).read_bytes() == served[TRAIN_PATH]


def test_verify_accepts_a_complete_download_and_leaves_it_untouched(
    tmp_path, served, fake_manifest, fake_opener
):
    fetch([ENTITY], dest=tmp_path, opener=fake_opener)
    verify([ENTITY], dest=tmp_path)
    for path, body in served.items():
        assert (tmp_path / path).read_bytes() == body


def test_verify_reports_a_missing_file(tmp_path, fake_manifest):
    with pytest.raises(ChecksumError, match=f"{TRAIN_PATH}: missing"):
        verify([ENTITY], dest=tmp_path)


def test_unknown_entity_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="machine-9-9"):
        fetch(["machine-9-9"], dest=tmp_path, opener=lambda url: b"")


def test_unknown_entity_message_lists_the_valid_names(tmp_path):
    with pytest.raises(ValueError, match=r"machine-1-1.*machine-3-11"):
        verify(["machine-9-9"], dest=tmp_path)


def test_source_url_pins_the_commit():
    assert source_url("train/machine-1-1.txt") == (
        "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/"
        "7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4/ServerMachineDataset/train/machine-1-1.txt"
    )


def test_default_data_dir_honours_the_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TSGUARD_DATA_DIR", str(tmp_path))
    assert default_data_dir() == tmp_path


def test_default_data_dir_falls_back_to_the_pinned_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("TSGUARD_DATA_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_data_dir() == tmp_path / ".cache" / "tsguard" / "smd" / SMD_COMMIT


def test_urllib_opener_returns_the_body_and_sets_a_60_second_timeout(monkeypatch):
    body = b"0.1,0.2,0.3\n"

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    timeouts: list[object] = []
    real_urlopen = urllib.request.urlopen

    def spy(url: str, *args: Any, **kwargs: Any) -> Any:
        timeouts.append(kwargs.get("timeout"))
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setattr(urllib.request, "urlopen", spy)
    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert urllib_opener(f"http://127.0.0.1:{server.server_port}/x") == body
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert timeouts == [60]


def _write_source(root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for entity in ENTITIES:
        for split in SPLITS:
            path = f"{split}/{entity}.txt"
            body = f"{path}\n".encode() * 3
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_bytes(body)
            files[path] = body
    return files


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


def test_build_manifest_writes_sorted_json_with_digests(tmp_path):
    source, output = tmp_path / "source", tmp_path / "manifest.json"
    files = _write_source(source)

    result = _run_script("--source", str(source), "--output", str(output))

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    data = json.loads(text)
    assert text == json.dumps(data, indent=2, sort_keys=True) + "\n"
    assert data["commit"] == SMD_COMMIT
    assert len(data["files"]) == 84
    for path, body in files.items():
        expected = {"sha256": hashlib.sha256(body).hexdigest(), "size": len(body)}
        assert data["files"][path] == expected


def test_build_manifest_names_a_missing_source_file_and_writes_nothing(tmp_path):
    source, output = tmp_path / "source", tmp_path / "manifest.json"
    _write_source(source)
    (source / "test_label" / "machine-2-5.txt").unlink()

    result = _run_script("--source", str(source), "--output", str(output))

    assert result.returncode != 0
    assert "test_label/machine-2-5.txt" in result.stderr
    assert not output.exists()


@pytest.mark.data
def test_real_cache_matches_the_packaged_manifest():
    verify()
