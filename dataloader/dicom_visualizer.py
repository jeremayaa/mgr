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







import json
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


def export_slice_from_paths(
    ct_npy_path: str | Path,
    seg_npy_path: str | Path | None,
    slice_idx: int,
    output_dir: str | Path,
) -> dict:
    """Export CT and segmentation slice as PNGs plus JSON with label colors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    if ct_volume.ndim != 3:
        raise ValueError(f"ct_volume must be 3D, got shape {ct_volume.shape}")

    depth = ct_volume.shape[0]
    slice_idx = max(0, min(slice_idx, depth - 1))

    ct_slice = ct_volume[slice_idx]

    slice_tag = f"{slice_idx:03d}"
    ct_png_path = output_dir / f"ct_slice_{slice_tag}.png"
    seg_png_path = output_dir / f"seg_slice_{slice_tag}.png"

    plt.imsave(ct_png_path, ct_slice, cmap="gray")

    labels_dict: dict[int, dict[str, list[int] | str]] = {}

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
        seg_rgb = np.zeros((rows, cols, 3), dtype=np.float32)

        for lab, color in colors.items():
            mask = seg_slice == lab
            seg_rgb[mask] = color

            labels_dict[int(lab)] = {
                "name": f"ROI_{int(lab)}",
                "color_rgb": [int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)],
            }

        seg_rgb_u8 = (seg_rgb * 255).astype(np.uint8)
        plt.imsave(seg_png_path, seg_rgb_u8)

    else:
        seg_png_path = None

    meta = {
        "slice_index": int(slice_idx),
        "ct_png": str(ct_png_path),
        "seg_png": str(seg_png_path) if seg_png_path is not None else None,
        "labels": labels_dict,
    }

    json_path = output_dir / f"slice_{slice_tag}_meta.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta
