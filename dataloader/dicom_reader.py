from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pydicom
from dataloader import Dataloader


def pick_first_segmentation(
    pairing: Dict[str, Dict[str, List[str]]]
) -> Optional[Tuple[str, str]]:
    """Return (study_key, segmentation_series_dir) for the first available segmentation."""
    for study_key, series_map in pairing.items():
        for _, seg_series_list in series_map.items():
            if seg_series_list:
                return study_key, seg_series_list[0]
    return None


def first_dicom_in_series(series_dir: Path) -> Optional[Path]:
    """Return the first DICOM file in a series directory."""
    for path in sorted(series_dir.rglob("*")):
        if path.is_file():
            return path
    return None


def print_rtstruct_info(dcm_path: Path) -> None:
    """Print basic information about an RTSTRUCT DICOM file."""
    ds = pydicom.dcmread(str(dcm_path), force=True)

    patient_id = getattr(ds, "PatientID", None)
    modality = getattr(ds, "Modality", None)
    study_uid = getattr(ds, "StudyInstanceUID", None)
    series_uid = getattr(ds, "SeriesInstanceUID", None)

    structure_seq = getattr(ds, "StructureSetROISequence", [])
    obs_seq = getattr(ds, "RTROIObservationsSequence", [])

    roi_names: Dict[int, str] = {}
    for roi in structure_seq:
        number = int(getattr(roi, "ROINumber", -1))
        name = str(getattr(roi, "ROIName", "")).strip()
        roi_names[number] = name

    roi_types: Dict[int, str] = {}
    for obs in obs_seq:
        number = int(getattr(obs, "ReferencedROINumber", -1))
        roi_type = str(getattr(obs, "RTROIInterpretedType", "")).strip()
        roi_types[number] = roi_type

    print(f"DICOM path: {dcm_path}")
    print(f"PatientID: {patient_id}")
    print(f"Modality: {modality}")
    print(f"StudyInstanceUID: {study_uid}")
    print(f"SeriesInstanceUID: {series_uid}")
    print(f"Number of ROIs: {len(structure_seq)}")

    print("\nROIs:")
    for number, name in roi_names.items():
        roi_type = roi_types.get(number, "")
        if roi_type:
            print(f"  ROINumber: {number}, Name: {name}, Type: {roi_type}")
        else:
            print(f"  ROINumber: {number}, Name: {name}")


def main() -> None:
    data_root = Path("data_2")

    loader = Dataloader(data_root)
    collections = loader.available_collections()

    if not collections:
        print("No collections found in dataset.")
        return

    collection_name = collections[1]
    print(f"Using collection: {collection_name}")

    pairing = loader.x_y_pairing(collection_name)

    picked = pick_first_segmentation(pairing)
    if picked is None:
        print("No segmentation series found in dataset.")
        return

    study_key, seg_series_dir_str = picked
    seg_series_dir = Path(seg_series_dir_str)

    print(f"Using study: {study_key}")
    print(f"Segmentation series directory: {seg_series_dir}")

    dcm_path = first_dicom_in_series(seg_series_dir)
    if dcm_path is None:
        print("No DICOM files found in segmentation series.")
        return

    print_rtstruct_info(dcm_path)


if __name__ == "__main__":
    main()
