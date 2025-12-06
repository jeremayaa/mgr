from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pydicom

from dataloader import Dataloader
from series_to_numpy import paths_for_np_pairing


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


def _extract_roi_colors(seg_dcm_path: Path) -> Dict[int, Dict[str, object]]:
    """Extract ROI names and RGB colors from an RTSTRUCT DICOM file."""
    ds = pydicom.dcmread(str(seg_dcm_path), stop_before_pixels=True)

    names = {
        int(r.ROINumber): getattr(r, "ROIName", f"ROI_{int(r.ROINumber)}")
        for r in ds.get("StructureSetROISequence", [])
    }

    labels: Dict[int, Dict[str, object]] = {}
    for rc in ds.get("ROIContourSequence", []):
        n = int(rc.ReferencedROINumber)
        name = names.get(n, f"ROI_{n}")
        if "ROIDisplayColor" in rc and rc.ROIDisplayColor is not None:
            rgb = list(map(int, rc.ROIDisplayColor))
        else:
            rgb = [255, 0, 0]
        labels[n] = {"name": name, "color_rgb": rgb}

    return labels


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
    np_pairing = paths_for_np_pairing(x_y_pairing)

    if not np_pairing:
        print("No NumPy pairings found.")
        return

    for study_key, series_map in np_pairing.items():
        print(f"\nProcessing {study_key}:")
        for ct_npy_str, seg_npy_list in series_map.items():
            if not seg_npy_list:
                continue

            seg_npy_path = Path(seg_npy_list[0])
            seg_dir = seg_npy_path.parent

            try:
                rtstruct_path = _find_rtstruct_dicom(seg_dir)
            except FileNotFoundError as e:
                print(f"  Skipping {seg_dir}: {e}")
                continue

            labels = _extract_roi_colors(rtstruct_path)

            if not labels:
                print(f"  No labels found in {rtstruct_path}")
                continue

            out_json = seg_dir / "segmentation_labels.json"
            with out_json.open("w", encoding="utf-8") as f:
                json.dump({"labels": labels}, f, indent=2)

            print(f"  Saved label colors to {out_json}")


if __name__ == "__main__":
    main()
