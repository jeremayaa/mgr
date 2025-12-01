# pip install -U tcia_utils pandas
import pandas as pd
from tcia_utils import nbia

# 1) Load metadata CSV
# df = pd.read_csv("code/data_handler/Pediatric-CT-SEG_metadata.csv")  # change filename as needed
df = pd.read_csv("code/data_handler/CC-Tumor-Heterogeneity_metadata.csv")  # change filename as needed

# Group by StudyInstanceUID so we can see how many studies are inside
studies = df.groupby("StudyInstanceUID")
print(f"Found {len(studies)} unique studies.\n")

# 2) Present studies to the user
for i, (study_uid, group) in enumerate(studies):
    patient_id = group["PatientID"].iloc[0]
    date = group["SeriesDate"].iloc[0]
    mods = group["Modality"].unique()
    print(f"[{i}] Patient: {patient_id}, StudyUID: {study_uid}, Date: {date}, Modalities: {mods}")

# 3) Pick one study (default = first one)
choice = input("\nEnter study index to download (default 0): ").strip()
if choice == "":
    choice = 0
else:
    choice = int(choice)

study_uid, study_rows = list(studies)[choice]
print(f"\nSelected Study: {study_uid}, with {len(study_rows)} series.")

# 4) optionally, filter the modalities you want
filter_mods = ["MR", "RTSTRUCT", 'REG']
filtered = study_rows[study_rows["Modality"].isin(filter_mods)]
print(f"Will download {len(filtered)} series ({filter_mods}).")


# 5) Prepare list of series for download
series_uids = filtered["SeriesInstanceUID"].tolist()

# Some nbia versions want "uids", others want dicts
try:
    nbia.downloadSeries(series_uids,
                        input_type="uids",
                        path="./DownloadedStudy",
                        as_zip=False,
                        max_workers=6)
except TypeError:
    series_rows = [{"SeriesInstanceUID": uid} for uid in series_uids]
    nbia.downloadSeries(series_rows,
                        path="./DownloadedStudy",
                        as_zip=False,
                        max_workers=6)

print(f"\nDone. {filter_mods} series downloaded to ./DownloadedStudy/")
