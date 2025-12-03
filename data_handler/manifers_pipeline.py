from __future__ import annotations

import shutil
from pathlib import Path

from nbia_downloader import create_manifest, recover_dataset


def main() -> None:
    source_root = Path("data")
    target_root = Path("data_2")

    create_manifest(source_root)

    target_root.mkdir(parents=True, exist_ok=True)

    source_manifest = source_root / "manifest.json"
    target_manifest = target_root / "manifest.json"

    shutil.move(source_manifest, target_manifest)

    recover_dataset(target_root)

if __name__ == "__main__":
    main()

