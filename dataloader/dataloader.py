from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pydicom


class Dataloader:
    """Load a dataset from a root folder containing a manifest and downloaded series."""

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        manifest_path = self.data_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        return data

    def available_collections(self) -> List[str]:
        """Return a list of collection names available in the dataset."""
        collections = self.manifest.get("collections", [])
        return [c.get("name") for c in collections if c.get("name")]

    def x_y_pairing(self, collection_name: str) -> Dict[str, Dict[str, List[str]]]:
        """Return per-study mappings from image series paths to segmentation series paths."""
        collection = self._get_collection_entry(collection_name)
        if collection is None:
            raise ValueError(f"Collection {collection_name} not found in manifest")

        result: Dict[str, Dict[str, List[str]]] = {}

        for study in collection.get("studies", []):
            index = int(study.get("index"))
            series_uids: List[str] = study.get("series_uids", [])
            study_key = f"study_{index}"

            image_series_dirs: List[Path] = []
            seg_series_dirs: List[Path] = []

            for series_uid in series_uids:
                series_dir = (
                    self.data_root
                    / collection_name
                    / f"study_{index}"
                    / series_uid
                )
                if not series_dir.exists() or not series_dir.is_dir():
                    continue

                modality = self._detect_modality(series_dir)

                if modality == "RTSTRUCT":
                    seg_series_dirs.append(series_dir)
                else:
                    image_series_dirs.append(series_dir)

            study_mapping: Dict[str, List[str]] = {}
            seg_series_strs = [str(p) for p in seg_series_dirs]

            for img_dir in image_series_dirs:
                study_mapping[str(img_dir)] = seg_series_strs

            result[study_key] = study_mapping

        return result

    def _get_collection_entry(self, collection_name: str) -> Optional[Mapping[str, Any]]:
        collections = self.manifest.get("collections", [])
        for collection in collections:
            if collection.get("name") == collection_name:
                return collection
        return None

    @staticmethod
    def _detect_modality(series_dir: Path) -> Optional[str]:
        """Infer modality from the first readable DICOM file in a series directory."""
        for path in series_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
                modality = getattr(ds, "Modality", None)
                if isinstance(modality, str):
                    return modality
            except Exception:
                continue
        return None
