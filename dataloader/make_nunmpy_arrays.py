from __future__ import annotations

from pathlib import Path

from dataloader import Dataloader
from series_to_numpy import pairs_to_numpy


def main() -> None:
    data_root = Path("data_2")

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
    print("\nOriginal x_y_pairing (folders):")
    for study_key, series_map in x_y_pairing.items():
        print(f"{study_key}:")
        for ct_series, seg_series_list in series_map.items():
            print(f"  CT: {ct_series}")
            for seg in seg_series_list:
                print(f"  RTSTRUCT: {seg}")

    np_pairing = pairs_to_numpy(x_y_pairing)

    print("\nNumPy-based x_y_pairing (.npy paths):")
    for study_key, series_map in np_pairing.items():
        print(f"{study_key}:")
        for ct_npy, seg_npy_list in series_map.items():
            print(f"  CT volume: {ct_npy}")
            for seg_npy in seg_npy_list:
                print(f"  Segmentation volume: {seg_npy}")


if __name__ == "__main__":
    main()
