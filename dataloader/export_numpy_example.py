from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from dataloader import Dataloader
from series_to_numpy import paths_for_np_pairing
from dicom_visualizer import export_slice_from_paths


def _sorted_study_keys(np_pairing: dict) -> List[Tuple[int, str]]:
    """Return a list of (numeric_index, study_key) sorted by numeric_index."""
    items: List[Tuple[int, str]] = []
    for key in np_pairing.keys():
        try:
            idx = int(key.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        items.append((idx, key))
    items.sort(key=lambda x: x[0])
    return items


def main() -> None:
    data_root = Path("data_2")

    target_study_idx = 4  # change this to choose N-th study

    loader = Dataloader(data_root)
    collections = loader.available_collections()
    print("Available collections:", collections)

    if not collections:
        return

    collection_name = "Pediatric-CT-SEG"
    if collection_name not in collections:
        collection_name = collections[0]

    print(f"Using collection: {collection_name}")

    x_y_pairing = loader.x_y_pairing(collection_name)
    np_pairing = paths_for_np_pairing(x_y_pairing)

    if not np_pairing:
        print("No NumPy pairings found.")
        return

    studies_sorted = _sorted_study_keys(np_pairing)
    if not studies_sorted:
        print("No valid study keys found.")
        return

    if target_study_idx < 0 or target_study_idx >= len(studies_sorted):
        print(
            f"Requested study index {target_study_idx} out of range "
            f"[0, {len(studies_sorted) - 1}]"
        )
        return

    numeric_idx, study_key = studies_sorted[target_study_idx]
    series_map = np_pairing[study_key]
    if not series_map:
        print(f"No series map for {study_key}.")
        return

    ct_npy_path = next(iter(series_map))
    seg_npy_path = series_map[ct_npy_path][0] if series_map[ct_npy_path] else None

    print(f"\nExporting study: {study_key} (numeric index {numeric_idx})")
    print(f"CT volume: {ct_npy_path}")
    print(f"Segmentation volume: {seg_npy_path}")

    slice_idx = 100  # choose any slice you like
    output_dir = Path("exports") / collection_name / study_key

    meta = export_slice_from_paths(ct_npy_path, seg_npy_path, slice_idx, output_dir)

    print("\nExported:")
    print(f"  CT PNG:  {meta['ct_png']}")
    print(f"  Seg PNG: {meta['seg_png']}")
    print("  Labels:")
    for lab_id, info in meta["labels"].items():
        print(f"    {lab_id}: {info['name']} color={info['color_rgb']}")


if __name__ == "__main__":
    main()
