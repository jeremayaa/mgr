from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
from tcia_utils import nbia


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON-serializable data to a file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class MetaDataCollector:
    """Fetch collection lists and per-collection metadata from NBIA."""

    def __init__(self, max_workers: int = 12) -> None:
        self.max_workers = max_workers

    def get_collections_list(self) -> List[Dict[str, Any]]:
        """Return a sorted list of available collections."""
        try:
            collections = nbia.getCollections()
        except AttributeError:
            collections = nbia.getCollectionValues()

        normalized: List[Dict[str, Any]] = []
        for item in collections:
            collection_name = item.get("Collection") or item.get("collection")
            description = item.get("Description") or item.get("description")
            normalized.append(
                {
                    "Collection": collection_name,
                    "Description": description,
                }
            )

        normalized.sort(key=lambda x: x.get("Collection") or "")
        return normalized

    def get_collection_metadata(self, collection_name: str) -> List[Dict[str, Any]]:
        """Return series-level metadata for a given collection."""
        try:
            patients = nbia.getPatients(collection=collection_name)
        except AttributeError:
            patients = nbia.getPatient(collection=collection_name)

        if not patients:
            return []

        first = patients[0]
        if "PatientID" in first:
            pid_key = "PatientID"
        elif "PatientId" in first:
            pid_key = "PatientId"
        else:
            raise KeyError("Unable to determine patient ID key in NBIA response.")

        return asyncio.run(
            self._gather_metadata(
                collection_name=collection_name,
                patients=patients,
                pid_key=pid_key,
            )
        )

    async def _gather_metadata(
        self,
        collection_name: str,
        patients: Sequence[Mapping[str, Any]],
        pid_key: str,
    ) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        rows: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            tasks = [
                loop.run_in_executor(
                    pool,
                    self._fetch_series_for_patient,
                    collection_name,
                    str(patient[pid_key]),
                )
                for patient in patients
            ]
            for result in await asyncio.gather(*tasks):
                rows.extend(result)

        return rows

    @staticmethod
    def _fetch_series_for_patient(collection_name: str, patient_id: str) -> List[Dict[str, Any]]:
        series_list = nbia.getSeries(collection=collection_name, patientId=patient_id)
        if not series_list:
            return []

        rows: List[Dict[str, Any]] = []
        for s in series_list:
            rows.append(
                {
                    "PatientID": patient_id,
                    "StudyInstanceUID": s.get("StudyInstanceUID"),
                    "SeriesInstanceUID": s.get("SeriesInstanceUID"),
                    "Modality": s.get("Modality"),
                    "BodyPartExamined": s.get("BodyPartExamined"),
                    "SeriesDescription": s.get("SeriesDescription"),
                    "SeriesDate": s.get("SeriesDate"),
                    "ImagesInSeries": s.get("ImageCount"),
                }
            )
        return rows


@dataclass
class Study:
    """Container for metadata about a single study."""

    index: int
    study_uid: str
    patient_id: str
    series_date: Optional[str]
    modalities: List[str]
    series_rows: pd.DataFrame


class DataDownloader:
    """Prepare and download studies from NBIA metadata."""

    def __init__(self, metadata: Iterable[Mapping[str, Any]]) -> None:
        self._df_full = pd.DataFrame(list(metadata))
        self._validate_columns()
        self._df = self._df_full.copy()
        self._studies = self._build_studies(self._df)
        self.max_workers = 6

    @property
    def studies(self) -> Sequence[Study]:
        """Return the list of available studies for the current filter."""
        return tuple(self._studies)

    def _validate_columns(self) -> None:
        required = {"StudyInstanceUID", "PatientID", "SeriesInstanceUID", "Modality"}
        missing = required.difference(self._df_full.columns)
        if missing:
            raise ValueError(f"Missing required metadata columns: {missing}")

    def _build_studies(self, df: pd.DataFrame) -> List[Study]:
        studies: List[Study] = []
        grouped = df.groupby("StudyInstanceUID", sort=False)
        for idx, (study_uid, group) in enumerate(grouped):
            patient_id = str(group["PatientID"].iloc[0])
            series_date_value = group["SeriesDate"].iloc[0] if "SeriesDate" in group.columns else None
            series_date = str(series_date_value) if series_date_value is not None else None
            modalities = sorted(set(map(str, group["Modality"].dropna().tolist())))
            studies.append(
                Study(
                    index=idx,
                    study_uid=str(study_uid),
                    patient_id=patient_id,
                    series_date=series_date,
                    modalities=modalities,
                    series_rows=group.copy(),
                )
            )
        return studies

    def get_ids(self) -> List[int]:
        """Return numeric study IDs for the current filter."""
        return [study.index for study in self._studies]

    def get_study(self, study_id: int) -> Optional[Study]:
        """Return a single study by numeric ID, or None if it does not exist."""
        for study in self._studies:
            if study.index == study_id:
                return study
        return None

    def filter_by_modalities(self, modalities: Optional[Sequence[str]] = None) -> None:
        """Filter by modalities if provided, otherwise reset to all studies."""
        if not modalities:
            self._df = self._df_full.copy()
            self._studies = self._build_studies(self._df)
            return

        mask = self._df_full["Modality"].isin(modalities)
        self._df = self._df_full[mask].copy()
        self._studies = self._build_studies(self._df)

    def reset_filters(self) -> None:
        """Reset all filters and rebuild the study list."""
        self._df = self._df_full.copy()
        self._studies = self._build_studies(self._df)

    def download(self, ids: Sequence[int], path: str | Path) -> None:
        """Download all series belonging to the selected study IDs into the given path."""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)

        for study_id in ids:
            study = self.get_study(study_id)
            if study is None:
                continue

            series_df = study.series_rows
            if series_df.empty:
                continue

            series_uids = series_df["SeriesInstanceUID"].dropna().astype(str).tolist()
            if not series_uids:
                continue

            study_path = output_path / f"study_{study.index}"
            study_path.mkdir(parents=True, exist_ok=True)

            try:
                nbia.downloadSeries(
                    series_uids,
                    input_type="uids",
                    path=str(study_path),
                    as_zip=False,
                    max_workers=self.max_workers,
                )
            except TypeError:
                series_rows = [{"SeriesInstanceUID": uid} for uid in series_uids]
                nbia.downloadSeries(
                    series_rows,
                    path=str(study_path),
                    as_zip=False,
                    max_workers=self.max_workers,
                )


def _downloaded_study_indices(collection_dir: Path) -> List[int]:
    """Return study indices based on existing study_<index> folders."""
    indices: List[int] = []
    if not collection_dir.exists():
        return indices
    for child in collection_dir.iterdir():
        if child.is_dir() and child.name.startswith("study_"):
            try:
                idx = int(child.name.split("_", 1)[1])
            except ValueError:
                continue
            indices.append(idx)
    return sorted(set(indices))

def create_manifest(data_root: str | Path, manifest_name: str = "manifest.json") -> None:
    """Create a manifest describing downloaded studies and series under a data root."""
    root = Path(data_root)

    collections: List[Dict[str, Any]] = []

    for collection_dir in sorted(root.iterdir()):
        if not collection_dir.is_dir():
            continue
        if collection_dir.name == "metadata":
            continue

        studies_entries: List[Dict[str, Any]] = []

        for study_dir in sorted(collection_dir.glob("study_*")):
            if not study_dir.is_dir():
                continue
            try:
                index = int(study_dir.name.split("_", 1)[1])
            except ValueError:
                continue

            series_uids = [
                child.name
                for child in sorted(study_dir.iterdir())
                if child.is_dir()
            ]

            if not series_uids:
                continue

            studies_entries.append(
                {
                    "index": index,
                    "series_uids": series_uids,
                }
            )

        if studies_entries:
            collections.append(
                {
                    "name": collection_dir.name,
                    "studies": studies_entries,
                }
            )

    manifest: Dict[str, Any] = {
        "version": 1,
        "collections": collections,
    }

    save_json(manifest, root / manifest_name)


def recover_dataset(
    data_root: str | Path,
    manifest_name: str = "manifest.json",
    max_workers: int = 6,
) -> None:
    """Reconstruct a dataset from a manifest file by re-downloading the saved series."""
    root = Path(data_root)
    manifest_path = root / manifest_name

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest: Dict[str, Any] = json.load(f)

    collections = manifest.get("collections", [])
    for collection in collections:
        collection_name = collection.get("name")
        if not collection_name:
            continue

        collection_root = root / collection_name
        collection_root.mkdir(parents=True, exist_ok=True)

        for study in collection.get("studies", []):
            study_index = int(study.get("index"))
            series_uids = study.get("series_uids", [])
            if not series_uids:
                continue

            study_path = collection_root / f"study_{study_index}"
            study_path.mkdir(parents=True, exist_ok=True)

            try:
                nbia.downloadSeries(
                    series_uids,
                    input_type="uids",
                    path=str(study_path),
                    as_zip=False,
                    max_workers=max_workers,
                )
            except TypeError:
                series_rows = [{"SeriesInstanceUID": uid} for uid in series_uids]
                nbia.downloadSeries(
                    series_rows,
                    path=str(study_path),
                    as_zip=False,
                    max_workers=max_workers,
                )