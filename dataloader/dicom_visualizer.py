# from __future__ import annotations

# from pathlib import Path
# from typing import Any, Dict, List, Optional

# import matplotlib.pyplot as plt
# import numpy as np
# import pydicom
# from matplotlib.path import Path as MplPath


# def _zpos(ds: pydicom.dataset.Dataset) -> float:
#     try:
#         return float(ds.ImagePositionPatient[2])
#     except Exception:
#         return float(getattr(ds, "InstanceNumber", 0))


# def _world_to_rc(
#     points_xyz: np.ndarray,
#     origin_xyz: np.ndarray,
#     row_cos: np.ndarray,
#     col_cos: np.ndarray,
#     pix_spacing: np.ndarray,
# ) -> np.ndarray:
#     v = points_xyz - origin_xyz[None, :]
#     r = (v @ row_cos) / pix_spacing[0]
#     c = (v @ col_cos) / pix_spacing[1]
#     return np.stack([r, c], axis=1)


# def _fill_polygon_mask(rows: int, cols: int, poly_rc: np.ndarray) -> np.ndarray:
#     rr = np.arange(rows) + 0.5
#     cc = np.arange(cols) + 0.5
#     CC, RR = np.meshgrid(cc, rr)
#     pts = np.stack([CC.ravel(), RR.ravel()], axis=1)
#     path = MplPath(poly_rc[:, ::-1])
#     inside = path.contains_points(pts)
#     return inside.reshape(rows, cols)


# def _detect_rtstruct_file(rtstruct_dir: Path) -> Path:
#     for path in sorted(rtstruct_dir.rglob("*.dcm")):
#         try:
#             ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
#             modality = getattr(ds, "Modality", None)
#             if isinstance(modality, str) and modality.upper() == "RTSTRUCT":
#                 return path
#         except Exception:
#             continue
#     raise FileNotFoundError(f"No RTSTRUCT DICOM found in {rtstruct_dir}")


# def visualize_rtstruct_overlay(
#     ct_series_dir: str | Path,
#     rtstruct_series_dir: str | Path,
#     slice_idx: int,
# ) -> plt.Figure:
#     """Return a figure with CT slice and RTSTRUCT overlay for a given series and slice index."""
#     ct_dir = Path(ct_series_dir)
#     rt_dir = Path(rtstruct_series_dir)

#     ct_files = sorted(
#         [p for p in ct_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"]
#     )
#     if not ct_files:
#         raise FileNotFoundError(f"No DICOM files found in CT series directory: {ct_dir}")

#     ct_dsets = [pydicom.dcmread(str(f)) for f in ct_files]
#     ct_dsets.sort(key=_zpos)

#     imgs = np.stack([ds.pixel_array for ds in ct_dsets]).astype(np.int16)

#     slope = float(getattr(ct_dsets[0], "RescaleSlope", 1.0))
#     intercept = float(getattr(ct_dsets[0], "RescaleIntercept", 0.0))
#     ct_hu = imgs * slope + intercept

#     slice_idx = max(0, min(slice_idx, len(ct_dsets) - 1))
#     ds = ct_dsets[slice_idx]

#     rows = int(ds.Rows)
#     cols = int(ds.Columns)
#     iop = np.array(ds.ImageOrientationPatient, dtype=float)
#     row_cos, col_cos = iop[:3], iop[3:]
#     ipp = np.array(ds.ImagePositionPatient, dtype=float)
#     ps = np.array(ds.PixelSpacing, dtype=float)
#     chosen_sop = ds.SOPInstanceUID

#     seg_path = _detect_rtstruct_file(rt_dir)
#     seg = pydicom.dcmread(str(seg_path))

#     roi_colors: Dict[int, Any] = {}
#     for rc in getattr(seg, "ROIContourSequence", []):
#         roi_num = int(rc.ReferencedROINumber)
#         rgb = getattr(rc, "ROIDisplayColor", None)
#         if rgb is not None and len(rgb) == 3:
#             roi_colors[roi_num] = tuple(c / 255.0 for c in rgb)

#     roi_masks: Dict[int, np.ndarray] = {}
#     for rc in getattr(seg, "ROIContourSequence", []):
#         roi_num = int(rc.ReferencedROINumber)
#         for cnt in getattr(rc, "ContourSequence", []):
#             cis = getattr(cnt, "ContourImageSequence", [])
#             if not cis or cis[0].ReferencedSOPInstanceUID != chosen_sop:
#                 continue
#             coords = np.array(cnt.ContourData, dtype=float).reshape(-1, 3)
#             poly_rc = _world_to_rc(coords, ipp, row_cos, col_cos, ps)
#             m = _fill_polygon_mask(rows, cols, poly_rc)

#             m = np.rot90(m, k=1)
#             m = np.flipud(m)

#             if roi_num not in roi_masks:
#                 roi_masks[roi_num] = np.zeros((rows, cols), dtype=bool)
#             roi_masks[roi_num] |= m

#     overlay = np.zeros((rows, cols, 4), dtype=float)
#     alpha = 0.35
#     for roi_num, m in roi_masks.items():
#         color = roi_colors.get(roi_num, (1.0, 0.0, 0.0))
#         overlay[m, :3] = color
#         overlay[m, 3] = alpha

#     fig, ax = plt.subplots()
#     ax.imshow(ct_hu[slice_idx], cmap="gray")
#     ax.imshow(overlay)
#     ax.set_title(f"Slice {slice_idx + 1} — RTSTRUCT overlay")
#     ax.axis("off")

#     return fig



from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _label_to_colors(unique_labels: Sequence[int]) -> dict[int, tuple[float, float, float]]:
    """Assign simple RGB colors to integer labels > 0."""
    palette = [
        (1.0, 0.0, 0.0),  # red
        (0.0, 1.0, 0.0),  # green
        (0.0, 0.0, 1.0),  # blue
        (1.0, 1.0, 0.0),  # yellow
        (1.0, 0.0, 1.0),  # magenta
        (0.0, 1.0, 1.0),  # cyan
    ]
    colors: dict[int, tuple[float, float, float]] = {}
    idx = 0
    for lab in unique_labels:
        if lab == 0:
            continue
        colors[lab] = palette[idx % len(palette)]
        idx += 1
    return colors


def plot_volume_slice(
    ct_volume: np.ndarray,
    seg_volume: np.ndarray | None,
    slice_idx: int,
    alpha: float = 0.35,
) -> plt.Figure:
    """Plot a single slice from a 3D CT volume with segmentation overlay."""
    if ct_volume.ndim != 3:
        raise ValueError(f"ct_volume must be 3D, got shape {ct_volume.shape}")

    depth = ct_volume.shape[0]
    slice_idx = max(0, min(slice_idx, depth - 1))

    ct_slice = ct_volume[slice_idx]

    fig, ax = plt.subplots()
    ax.imshow(ct_slice, cmap="gray")

    if seg_volume is not None:
        if seg_volume.shape != ct_volume.shape:
            raise ValueError(
                f"segmentation volume shape {seg_volume.shape} "
                f"does not match CT volume shape {ct_volume.shape}"
            )

        seg_slice = seg_volume[slice_idx]
        unique_labels = np.unique(seg_slice)
        colors = _label_to_colors(unique_labels)

        rows, cols = seg_slice.shape
        overlay = np.zeros((rows, cols, 4), dtype=float)

        for lab, color in colors.items():
            mask = seg_slice == lab
            overlay[mask, :3] = color
            overlay[mask, 3] = alpha

        ax.imshow(overlay)

    ax.set_title(f"Volume slice {slice_idx + 1}")
    ax.axis("off")
    return fig


def plot_volume_slice_from_paths(
    ct_npy_path: str | Path,
    seg_npy_path: str | Path | None,
    slice_idx: int,
    alpha: float = 0.35,
) -> plt.Figure:
    """Load CT and segmentation volumes from .npy files and plot a slice."""
    ct_path = Path(ct_npy_path)
    if not ct_path.exists():
        raise FileNotFoundError(f"CT volume not found: {ct_path}")
    ct_volume = np.load(ct_path)

    seg_volume = None
    if seg_npy_path is not None:
        seg_path = Path(seg_npy_path)
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation volume not found: {seg_path}")
        seg_volume = np.load(seg_path)

    return plot_volume_slice(ct_volume, seg_volume, slice_idx, alpha=alpha)
