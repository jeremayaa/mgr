# collection_to_metadata_csv.py
# pip install -U tcia_utils pandas
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tcia_utils import nbia

def _get_patients(collection):
    try:
        pts = nbia.getPatients(collection=collection)
    except AttributeError:
        pts = nbia.getPatient(collection=collection)
    if not pts:
        raise SystemExit("No patients found.")
    pid_key = "PatientID" if "PatientID" in pts[0] else "PatientId"
    return pts, pid_key

def _fetch_series(collection, pid):
    rows = []
    for s in nbia.getSeries(collection=collection, patientId=pid):
        rows.append({
            "PatientID": pid,
            "StudyInstanceUID": s.get("StudyInstanceUID"),
            "SeriesInstanceUID": s.get("SeriesInstanceUID"),
            "Modality": s.get("Modality"),
            "BodyPartExamined": s.get("BodyPartExamined"),
            "SeriesDescription": s.get("SeriesDescription"),
            "SeriesDate": s.get("SeriesDate"),
            "ImagesInSeries": s.get("ImageCount"),
        })
    return rows

def get_collection_metadata(collection_name, out_path=None, max_workers=12):
    patients, pid_key = _get_patients(collection_name)
    all_rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch_series, collection_name, p[pid_key]) for p in patients]
        for fut in as_completed(futures):
            all_rows.extend(fut.result())
    if not all_rows:
        raise SystemExit("No series found.")
    df = pd.DataFrame(all_rows)
    out = out_path or os.path.abspath(f"{collection_name}_metadata.csv")
    df.to_csv(out, index=False)
    print(f"Patients: {len(patients)} | Series rows: {len(df)}")
    print(f"Saved: {out}")
    return out

# simple call:
# get_collection_metadata("Pediatric-CT-SEG")
