from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from nbia_downloader import DataDownloader, MetaDataCollector, save_json


def load_metadata(path: Path) -> list[dict[str, Any]]:
    """Load metadata JSON from a file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    dataset_path = Path("data")
    # collection_name = "Pediatric-CT-SEG"
    collection_name = "CC-Tumor-Heterogeneity"

    metadata_dir = dataset_path / "metadata"
    metadata_path = metadata_dir / f"{collection_name}.json"

    if metadata_path.exists():
        metadata: list[dict[str, Any]] = load_metadata(metadata_path)
    else:
        collector = MetaDataCollector(max_workers=12)
        metadata = collector.get_collection_metadata(collection_name)
        if not metadata:
            return
        save_json(metadata, metadata_path)

    downloader = DataDownloader(metadata)

    print(f"Found {len(downloader.get_ids())} unique studies.\n")
    for study in downloader.studies:
        print(
            f"[{study.index}] "
            f"Patient: {study.patient_id}, "
            f"StudyUID: {study.study_uid}, "
            f"Date: {study.series_date}, "
            f"Modalities: {study.modalities}"
        )

    filter_mods = ["MR", "RTSTRUCT"]
    downloader.filter_by_modalities(filter_mods)

    print(f"\nAfter filtering to {filter_mods}:")
    print(f"Found {len(downloader.get_ids())} studies.\n")
    for study in downloader.studies:
        print(
            f"[{study.index}] "
            f"Patient: {study.patient_id}, "
            f"StudyUID: {study.study_uid}, "
            f"Date: {study.series_date}, "
            f"Modalities: {study.modalities}"
        )

    ids: List[int] = downloader.get_ids()
    selected_ids = ids[:1]

    output_dir = dataset_path / collection_name
    downloader.download(selected_ids, path=output_dir)

    print(f"\nDownloaded studies with IDs {selected_ids} to {output_dir}")


if __name__ == "__main__":
    main()
