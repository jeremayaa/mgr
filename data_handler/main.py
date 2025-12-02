from __future__ import annotations

from pathlib import Path
from typing import List

from nbia_downloader import DataDownloader, MetaDataCollector, save_json


def main() -> None:
    collection_name = "CC-Tumor-Heterogeneity"

    collector = MetaDataCollector(max_workers=12)
    metadata = collector.get_collection_metadata(collection_name)

    if not metadata:
        return

    save_json(metadata, Path("metadata") / f"{collection_name}_metadata.json")

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

    filter_mods = ["MR", "RTSTRUCT", "REG"]
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
    selected_ids = ids[:3]

    output_dir = Path("DownloadedStudy") / collection_name
    downloader.download(selected_ids, path=output_dir)

    print(f"\nDownloaded studies with IDs {selected_ids} to {output_dir}")


if __name__ == "__main__":
    main()
