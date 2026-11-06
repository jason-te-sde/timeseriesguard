"""Regenerate the packaged SMD checksum manifest from a local copy of the dataset.

Run it after changing ``SMD_COMMIT``, against a directory downloaded from that commit::

    uv run python scripts/build_manifest.py --source ~/.cache/tsguard/smd/<commit>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from tsguard.data.smd import ENTITIES, SMD_COMMIT, SPLITS

DEFAULT_OUTPUT: Final = (
    Path(__file__).resolve().parents[1] / "src" / "tsguard" / "data" / "smd_manifest.json"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Hash every SMD file under ``--source`` and write the manifest.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` if omitted.

    Returns:
        0 on success, 1 if a file is missing (nothing is written in that case).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="directory holding train/, test/ and test_label/ downloaded from SMD_COMMIT",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="manifest to write (default: the one packaged with tsguard)",
    )
    args = parser.parse_args(argv)

    files: dict[str, dict[str, object]] = {}
    for entity in ENTITIES:
        for split in SPLITS:
            path = f"{split}/{entity}.txt"
            file = args.source / path
            if not file.is_file():
                print(f"error: {path} is missing from {args.source}", file=sys.stderr)
                return 1
            data = file.read_bytes()
            files[path] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}

    manifest = {"commit": SMD_COMMIT, "files": files}
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(files)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
