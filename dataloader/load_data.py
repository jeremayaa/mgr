from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pydicom

from dataloader import Dataloader
from dicom_visualizer import visualize_rtstruct_overlay


ImageRtstructPair = Tuple[Path, Path]


def _find_rtstruct_dicom(rtstruct_series_dir: Path) -> Path:
    """Return the first RTSTRUCT DICOM file found in a series directory."""
    for path in sorted(rtstruct_series_dir.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            modality = getattr(ds, "Modality", None)
            if isinstance(modality, str) and modality.upper() == "RTSTRUCT":
                return path
        except Exception:
            continue
    raise FileNotFoundError(f"No RTSTRUCT DICOM found in {rtstruct_series_dir}")


def _print_rtstruct_info(seg_dcm_path: Path) -> None:
    """Print basic info about ROIs in an RTSTRUCT file."""
    ds = pydicom.dcmread(str(seg_dcm_path))

    print("\nRTSTRUCT summary")
    print("Type:", "RTSTRUCT")

    names = {
        int(r.ROINumber): getattr(r, "ROIName", f"ROI {int(r.ROINumber)}")
        for r in ds.get("StructureSetROISequence", [])
    }

    for rc in ds.get("ROIContourSequence", []):
        n = int(rc.ReferencedROINumber)
        name = names.get(n, f"ROI {n}")
        color = tuple(rc.ROIDisplayColor) if "ROIDisplayColor" in rc else None
        print(f"- #{n}: {name}, color RGB={color}")


def build_image_rtstruct_pairs(
    data_root: Path,
    collection_name: str,
) -> List[ImageRtstructPair]:
    """Return a flat list of (image_series_dir, rtstruct_series_dir) pairs for a collection."""
    loader = Dataloader(data_root)
    pairing = loader.x_y_pairing(collection_name)

    pairs: List[ImageRtstructPair] = []
    for study_key, series_map in pairing.items():
        for img_series, seg_series_list in series_map.items():
            if not seg_series_list:
                continue
            img_dir = Path(img_series)
            seg_dir = Path(seg_series_list[0])
            pairs.append((img_dir, seg_dir))

    return pairs


def show_image_slice(
    pair_list: List[ImageRtstructPair],
    image_id: int,
    slice_idx: int,
) -> None:
    """Visualize one image/RTSTRUCT pair by index and slice number."""
    if not pair_list:
        raise ValueError("Pair list is empty")

    if image_id < 0 or image_id >= len(pair_list):
        raise IndexError(f"Image id {image_id} out of range [0, {len(pair_list) - 1}]")

    img_series_dir, seg_series_dir = pair_list[image_id]

    seg_dcm_path = _find_rtstruct_dicom(seg_series_dir)
    _print_rtstruct_info(seg_dcm_path)

    fig = visualize_rtstruct_overlay(img_series_dir, seg_series_dir, slice_idx)
    plt.show()


def main() -> None:
    data_root = Path("data_2")

    loader = Dataloader(data_root)
    collections = loader.available_collections()
    print("Available collections:", collections)

    if not collections:
        return

    collection_name = collections[1]
    pairs = build_image_rtstruct_pairs(data_root, collection_name)

    print(f"\nFound {len(pairs)} image–RTSTRUCT pairs in collection {collection_name}:")
    for idx, (img_dir, seg_dir) in enumerate(pairs):
        print(f"[{idx}]")
        print(f"  Image series: {img_dir}")
        print(f"  RTSTRUCT series: {seg_dir}")

    if not pairs:
        return

    image_id = 1
    slice_idx = 160
    show_image_slice(pairs, image_id=image_id, slice_idx=slice_idx)


if __name__ == "__main__":
    main()
