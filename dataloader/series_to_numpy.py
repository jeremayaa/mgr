from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pydicom
from matplotlib.path import Path as MplPath


PairingDict = Dict[str, Dict[str, List[str]]]


def _zpos(ds: pydicom.dataset.Dataset) -> float:
    """
    Helper function that extracts the slice z-position from a DICOM dataset to allow sorting slices along the patient axis.
    It tries to use ImagePositionPatient[2] and falls back to InstanceNumber if that is not available.
    This function is used in _load_ct_volume as the key function for ordering DICOM slices.
    """
    try:
        return float(ds.ImagePositionPatient[2])
    except Exception:
        return float(getattr(ds, "InstanceNumber", 0))


def _world_to_rc(
    points_xyz: np.ndarray,
    origin_xyz: np.ndarray,
    row_cos: np.ndarray,
    col_cos: np.ndarray,
    pix_spacing: np.ndarray,
) -> np.ndarray:
    """
    Convert 3D world (patient) coordinates into 2D image row/column coordinates using CT geometry.
    It subtracts the slice origin, projects onto the row and column direction cosine vectors, and scales by pixel spacing.
    The function returns an array of (row, col) indices that are later used to rasterize RTSTRUCT contours.
    It is called from _rtstruct_to_volume for each contour in an RTSTRUCT structure.
    """

    v = points_xyz - origin_xyz[None, :]
    r = (v @ row_cos) / pix_spacing[0]
    c = (v @ col_cos) / pix_spacing[1]

    return np.stack([r, c], axis=1)


def _fill_polygon_mask_bbox(
    rows: int,
    cols: int,
    poly_rc: np.ndarray,
) -> np.ndarray:
    """
    Rasterize a polygon defined in (row, col) coordinates into a 2D boolean mask of shape (rows, cols).
    It computes a bounding box around the polygon, samples points on a pixel-centered grid within that box, and uses matplotlib.path.Path to test point inclusion.
    The mask is initially all False and is set to True where points fall inside the polygon.
    This function is called from _rtstruct_to_volume to generate slice-wise segmentation masks from RTSTRUCT contours.
    """
    mask = np.zeros((rows, cols), dtype=bool)

    r_min = max(int(np.floor(poly_rc[:, 0].min())), 0)
    r_max = min(int(np.ceil(poly_rc[:, 0].max())) + 1, rows)
    c_min = max(int(np.floor(poly_rc[:, 1].min())), 0)
    c_max = min(int(np.ceil(poly_rc[:, 1].max())) + 1, cols)

    if r_min >= r_max or c_min >= c_max:
        return mask

    rr = np.arange(r_min, r_max) + 0.5
    cc = np.arange(c_min, c_max) + 0.5
    CC, RR = np.meshgrid(cc, rr)
    pts = np.stack([CC.ravel(), RR.ravel()], axis=1)

    path = MplPath(poly_rc[:, ::-1])
    inside = path.contains_points(pts)
    inside = inside.reshape(r_max - r_min, c_max - c_min)

    mask[r_min:r_max, c_min:c_max] = inside
    return mask


def _load_ct_volume(ct_series_dir: Path) -> tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Load a CT series from a directory of DICOM files into a 3D volume in Hounsfield units, along with per-slice geometry metadata.
    It finds all .dcm files, reads them with pydicom, sorts them by z-position using _zpos, stacks pixel_array into a volume, and applies RescaleSlope/RescaleIntercept.
    For each slice it also extracts direction cosines, image position, pixel spacing, and image size, storing these as geometry dictionaries.
    The function is called by pairs_to_numpy when a CT volume needs to be computed and saved as .npy.
    """
    ct_files = sorted(
        [p for p in ct_series_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"]
    )
    if not ct_files:
        raise FileNotFoundError(f"No DICOM files found in CT series directory: {ct_series_dir}")

    dsets = [pydicom.dcmread(str(f)) for f in ct_files]
    dsets.sort(key=_zpos)

    imgs = np.stack([ds.pixel_array for ds in dsets]).astype(np.int16)

    slope = float(getattr(dsets[0], "RescaleSlope", 1.0))
    intercept = float(getattr(dsets[0], "RescaleIntercept", 0.0))
    ct_hu = imgs * slope + intercept

    sop_uids = [str(ds.SOPInstanceUID) for ds in dsets]

    geometries: List[Dict[str, Any]] = []
    for ds in dsets:
        iop = np.array(ds.ImageOrientationPatient, dtype=float)
        row_cos = iop[:3]
        col_cos = iop[3:]
        ipp = np.array(ds.ImagePositionPatient, dtype=float)
        ps = np.array(ds.PixelSpacing, dtype=float)
        geometries.append(
            {
                "row_cos": row_cos,
                "col_cos": col_cos,
                "ipp": ipp,
                "ps": ps,
                "rows": int(ds.Rows),
                "cols": int(ds.Columns),
            }
        )

    return ct_hu.astype(np.float32), sop_uids, geometries


def _find_rtstruct_dicom(rtstruct_series_dir: Path) -> Path:
    """
    Search a directory tree for a DICOM file whose Modality is RTSTRUCT and return its path.
    It recursively scans for .dcm files, reads them with pydicom (without pixel data), and checks the Modality tag.
    The function returns the first found RTSTRUCT file and raises FileNotFoundError if none is found.
    It is used by pairs_to_numpy to locate the RTSTRUCT file corresponding to a segmentation series directory.
    """

    for path in sorted(rtstruct_series_dir.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            modality = getattr(ds, "Modality", None)
            if isinstance(modality, str) and modality.upper() == "RTSTRUCT":
                return path
        except Exception:
            continue
    raise FileNotFoundError(f"No RTSTRUCT DICOM found in {rtstruct_series_dir}")


def _rtstruct_to_volume(
    rtstruct_path: Path,
    sop_uids: List[str],
    geometries: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Convert an RTSTRUCT DICOM file into a 3D label volume aligned with a given CT volume.
    It reads the RTSTRUCT, iterates over ROIContourSequence and ContourSequence, 
    maps contour points from world space to image (row, col) using _world_to_rc, 
    and rasterizes them with _fill_polygon_mask_bbox.
    For each contour, it assigns the ROI number as a label in the corresponding slice of seg_vol, 
    applying a fixed rotation and flip so that the mask matches the CT orientation.

    The function is called by pairs_to_numpy after loading the CT volume via _load_ct_volume.
    """
    ds = pydicom.dcmread(str(rtstruct_path), stop_before_pixels=True)

    depth = len(sop_uids)
    rows = int(geometries[0]["rows"])
    cols = int(geometries[0]["cols"])
    seg_vol = np.zeros((depth, rows, cols), dtype=np.int16)

    sop_to_index = {sop: idx for idx, sop in enumerate(sop_uids)}

    for rc in getattr(ds, "ROIContourSequence", []):
        roi_num = int(rc.ReferencedROINumber)

        for cnt in getattr(rc, "ContourSequence", []):
            cis = getattr(cnt, "ContourImageSequence", [])
            if not cis:
                continue

            ref_sop = str(cis[0].ReferencedSOPInstanceUID)
            if ref_sop not in sop_to_index:
                continue

            slice_idx = sop_to_index[ref_sop]
            geom = geometries[slice_idx]
            row_cos = geom["row_cos"]
            col_cos = geom["col_cos"]
            ipp = geom["ipp"]
            ps = geom["ps"]

            coords = np.array(cnt.ContourData, dtype=float).reshape(-1, 3)
            poly_rc = _world_to_rc(coords, ipp, row_cos, col_cos, ps)

            mask2d = _fill_polygon_mask_bbox(rows, cols, poly_rc)

            mask2d = np.rot90(mask2d, k=1)
            mask2d = np.flipud(mask2d)

            seg_vol[slice_idx][mask2d] = roi_num

    return seg_vol


def pairs_to_numpy(x_y_pairing: PairingDict) -> PairingDict:
    """
    Ensure that .npy volumes exist for each CT/RTSTRUCT pair described by the x_y_pairing mapping and return a new mapping with .npy paths.
    It first calls paths_for_np_pairing to convert series directories into target .npy file paths, then iterates over these and checks whether the CT and segmentation .npy files already exist.
    For missing volumes, it calls _load_ct_volume to build the CT volume, _find_rtstruct_dicom to locate the RTSTRUCT file, and _rtstruct_to_volume to derive the segmentation volume, then saves both arrays to disk.
    The function returns a new pairing dict that includes only those pairs for which .npy volumes exist or have been generated.
    """
    from series_to_numpy import paths_for_np_pairing  # or import at top if same module

    np_pairing = paths_for_np_pairing(x_y_pairing)
    new_pairing: PairingDict = {}

    for study_key, series_map in np_pairing.items():
        new_series_map: Dict[str, List[str]] = {}

        for ct_npy_str, seg_npy_list in series_map.items():
            if not seg_npy_list:
                continue

            ct_npy_path = Path(ct_npy_str)
            seg_npy_path = Path(seg_npy_list[0])

            ct_exists = ct_npy_path.exists()
            seg_exists = seg_npy_path.exists()

            if ct_exists and seg_exists:
                print(f"Skipping existing volumes: {ct_npy_path}, {seg_npy_path}")
                new_series_map[str(ct_npy_path)] = [str(seg_npy_path)]
                continue

            ct_dir = ct_npy_path.parent
            seg_dir = seg_npy_path.parent

            print(f"Computing volumes for: {ct_dir} and {seg_dir}")
            ct_vol, sop_uids, geometries = _load_ct_volume(ct_dir)
            rtstruct_path = _find_rtstruct_dicom(seg_dir)
            seg_vol = _rtstruct_to_volume(rtstruct_path, sop_uids, geometries)

            ct_npy_path.parent.mkdir(parents=True, exist_ok=True)
            seg_npy_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(ct_npy_path, ct_vol)
            np.save(seg_npy_path, seg_vol)

            new_series_map[str(ct_npy_path)] = [str(seg_npy_path)]

        if new_series_map:
            new_pairing[study_key] = new_series_map

    return new_pairing


def paths_for_np_pairing(x_y_pairing: PairingDict) -> PairingDict:
    """
    Convert a mapping from CT/RTSTRUCT series directories into a mapping from CT/RTSTRUCT .npy file paths.
    For each CT series directory, it constructs a fixed output file name 'ct_volume.npy' in that directory and 'rtstruct_labels.npy' in the corresponding RTSTRUCT series directory.
    It returns a new pairing dict with the same per-study structure but using string paths to the future .npy files instead of the original DICOM series folders.
    This function is called by pairs_to_numpy to determine where the NumPy volumes should be stored.
    """
    new_pairing: PairingDict = {}

    for study_key, series_map in x_y_pairing.items():
        new_series_map: Dict[str, List[str]] = {}

        for ct_series_str, seg_series_list in series_map.items():
            if not seg_series_list:
                continue

            ct_dir = Path(ct_series_str)
            seg_dir = Path(seg_series_list[0])

            ct_npy_path = ct_dir / "ct_volume.npy"
            seg_npy_path = seg_dir / "rtstruct_labels.npy"

            new_series_map[str(ct_npy_path)] = [str(seg_npy_path)]

        if new_series_map:
            new_pairing[study_key] = new_series_map

    return new_pairing
