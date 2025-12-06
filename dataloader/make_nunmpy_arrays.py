from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from dataloader import Dataloader
from series_to_numpy import pairs_to_numpy


def main() -> None:
    """
    Entry-point function that loads a dataset from a given root folder and prints CT/RTSTRUCT pairings.
    It constructs a Dataloader, queries available collections, and picks the hard-coded "Pediatric-CT-SEG" collection if present.
    For that collection it prints the original per-study mapping from CT series folders to RTSTRUCT series folders.
    It then calls pairs_to_numpy to convert folder-based paths into .npy volume paths (and trigger volume computation) and prints the resulting mapping.
    """
    data_root = Path("data_2")

    loader = Dataloader(data_root)
    collections = loader.available_collections()
    print("Available collections:", collections)

    if not collections:
        return

    collection_name = "Pediatric-CT-SEG"
    if collection_name not in collections:
        return

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
