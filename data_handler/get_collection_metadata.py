# pip install -U tcia_utils
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os
from tcia_utils import nbia

# collection_name = 'Pediatric-CT-SEG'
collection_name = 'Pancreas-CT'

# 1) Get all patients
try:
    patients = nbia.getPatients(collection=collection_name)
except AttributeError:
    patients = nbia.getPatient(collection=collection_name)

pid_key = "PatientID" if "PatientID" in patients[0] else "PatientId"
print(f"Found {len(patients)} patients.")


# 2) Define a worker
def fetch_series(pid):
    """Fetch all series metadata for one patient ID"""
    series_list = nbia.getSeries(collection=collection_name, patientId=pid)
    rows = []
    for s in series_list:
        rows.append({
            "PatientID": pid,
            "StudyInstanceUID": s.get("StudyInstanceUID"),
            "SeriesInstanceUID": s.get("SeriesInstanceUID"),
            "Modality": s.get("Modality"),
            "BodyPartExamined": s.get("BodyPartExamined"),
            "SeriesDescription": s.get("SeriesDescription"),
            "SeriesDate": s.get("SeriesDate"),
            "ImagesInSeries": s.get("ImageCount")
        })
    return rows


# 3) Run with asyncio
async def gather_metadata(patients, max_workers=12):
    loop = asyncio.get_event_loop()
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        tasks = [
            loop.run_in_executor(pool, fetch_series, p[pid_key])
            for p in patients
        ]
        for result in await asyncio.gather(*tasks):
            rows.extend(result)
    return rows


# 4) Execute
all_series_metadata = asyncio.run(gather_metadata(patients, max_workers=12))

df = pd.DataFrame(all_series_metadata)
print(df.head())
print(f"\nTotal series metadata rows: {len(df)}")

# 5) Save to the same folder as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
out_file = os.path.join(script_dir, f"{collection_name}_metadata.csv")
df.to_csv(out_file, index=False)
print(f"Saved metadata to {out_file}")
