from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pydicom
from matplotlib.path import Path as MplPath


PairingDict = Dict[str, Dict[str, List[str]]]


class CTRTStructVolumeGenerator:
    """
    Helper class that converts a CT series directory and a corresponding RTSTRUCT
    series directory into aligned NumPy volumes.

    It wraps the logic for loading the CT DICOM stack, extracting geometric
    metadata, locating the RTSTRUCT file, and rasterizing RTSTRUCT contours
    into a 3D label volume. This keeps pairs_to_numpy simple and makes it easier
    to extend or customize the conversion behavior later.
    """

    def __init__(self) -> None:
        # You can add configuration options later (e.g., label mapping, ROI filters, etc.)
        pass

    # ---- Public API -----------------------------------------------------
    def compute_volumes(
        self, ct_dir: Path, seg_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given a CT series directory and an RTSTRUCT series directory, return
        aligned (ct_volume, seg_volume) NumPy arrays.

        This method orchestrates the full pipeline: it loads the CT volume and
        slice geometry, finds the RTSTRUCT DICOM file under seg_dir, converts
        RTSTRUCT contours to a label volume, and returns both arrays using the
        original CT orientation.
        """
        ct_vol, sop_uids, geometries = self._load_ct_volume(ct_dir)
        rtstruct_path = self._find_rtstruct_dicom(seg_dir)
        seg_vol = self._rtstruct_to_volume(rtstruct_path, sop_uids, geometries)
        return ct_vol, seg_vol

    # ---- Internal helpers (moved from your free functions) --------------

    @staticmethod
    def _zpos(ds: pydicom.dataset.Dataset) -> float:
        """Same logic as your old _zpos helper."""
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            return float(getattr(ds, "InstanceNumber", 0))

    def _load_ct_volume(
        self, ct_series_dir: Path
    ) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """
        Load CT DICOM series into a 3D Hounsfield volume and collect per-slice geometry.

        It reads all .dcm files, sorts them by z-position using _zpos, stacks
        pixel_array into a volume, applies RescaleSlope/RescaleIntercept, and
        extracts direction cosines, image position, pixel spacing, and image size
        for each slice. Returns (volume, sop_uids, geometries).
        """
        ct_files = sorted(
            [p for p in ct_series_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"]
        )
        if not ct_files:
            raise FileNotFoundError(
                f"No DICOM files found in CT series directory: {ct_series_dir}"
            )

        dsets = [pydicom.dcmread(str(f)) for f in ct_files]
        dsets.sort(key=self._zpos)

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

    @staticmethod
    def _find_rtstruct_dicom(rtstruct_series_dir: Path) -> Path:
        """
        Find and return the first RTSTRUCT DICOM file under a segmentation series directory.

        It recursively scans for files with .dcm extension, reads them without
        pixel data, and checks their Modality attribute. The first DICOM with
        Modality 'RTSTRUCT' is returned, otherwise FileNotFoundError is raised.
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

    @staticmethod
    def _world_to_rc(
        points_xyz: np.ndarray,
        origin_xyz: np.ndarray,
        row_cos: np.ndarray,
        col_cos: np.ndarray,
        pix_spacing: np.ndarray,
    ) -> np.ndarray:
        """
        Map 3D world (patient) coordinates into 2D image (row, col) indices
        using the CT slice geometry.

        It subtracts the slice origin, projects onto row and column direction
        cosines, and divides by pixel spacing to obtain continuous indices.
        The result is a (N, 2) array used to rasterize RTSTRUCT contours.
        """
        v = points_xyz - origin_xyz[None, :]
        r = (v @ row_cos) / pix_spacing[0]
        c = (v @ col_cos) / pix_spacing[1]
        return np.stack([r, c], axis=1)

    @staticmethod
    def _fill_polygon_mask_bbox(
        rows: int,
        cols: int,
        poly_rc: np.ndarray,
    ) -> np.ndarray:
        """
        Rasterize a polygon expressed in (row, col) coordinates into a 2D
        boolean mask.

        It computes a tight bounding box around the polygon, samples a
        pixel-centered grid within that box, and uses matplotlib's Path
        to test which points lie inside the polygon. The function returns
        a mask of shape (rows, cols) with True where the polygon is filled.
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

    def _rtstruct_to_volume(
        self,
        rtstruct_path: Path,
        sop_uids: List[str],
        geometries: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Convert an RTSTRUCT DICOM file to a 3D label volume aligned with the CT slices.

        It reads the RTSTRUCT, iterates over ROIs and contours, maps contour
        points into image index space with _world_to_rc, rasterizes each
        polygon using _fill_polygon_mask_bbox, optionally applies a rotation
        and flip for orientation, and writes the ROI number into the 3D label
        volume at the appropriate slice indices.
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
                poly_rc = self._world_to_rc(coords, ipp, row_cos, col_cos, ps)

                mask2d = self._fill_polygon_mask_bbox(rows, cols, poly_rc)

                # Keep your current orientation fix (can be removed once math is fixed)
                mask2d = np.rot90(mask2d, k=1)
                mask2d = np.flipud(mask2d)

                seg_vol[slice_idx][mask2d] = roi_num

        return seg_vol


def pairs_to_numpy(x_y_pairing: PairingDict) -> PairingDict:
    """
    Ensure NumPy volumes exist for CT/RTSTRUCT pairs and return updated pairing.

    This implementation uses CTRTStructVolumeGenerator to convert each
    (ct_series_dir, seg_series_dir) pair into aligned volumes, saving them
    as .npy files. Existing volumes are reused and not recomputed.
    """
    from series_to_numpy import paths_for_np_pairing  # if in same module, move to top

    np_pairing = paths_for_np_pairing(x_y_pairing)
    new_pairing: PairingDict = {}

    generator = CTRTStructVolumeGenerator()

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

            ct_vol, seg_vol = generator.compute_volumes(ct_dir, seg_dir)

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
