# pip install -U tcia_utils pandas
from tcia_utils import nbia
import pandas as pd

# -------------------------------
# 1) Get all NBIA radiology collections
# -------------------------------
try:
    # Newer tcia_utils (NBIA v4-style)
    collections = nbia.getCollections()
except AttributeError:
    # Fallback for older versions
    collections = nbia.getCollectionValues()

# Turn into a DataFrame for easy viewing
df = pd.DataFrame(collections)

# Try to normalize column names a bit
if "Collection" not in df.columns and "collection" in df.columns:
    df.rename(columns={"collection": "Collection"}, inplace=True)
if "Description" not in df.columns and "description" in df.columns:
    df.rename(columns={"description": "Description"}, inplace=True)

# Sort alphabetically
df = df.sort_values("Collection").reset_index(drop=True)

print(f"\nFound {len(df)} collections available via NBIA / tcia_utils:\n")
for _, row in df.iterrows():
    name = row["Collection"]
    desc = row.get("Description", "")
    print(f"- {name}" + (f"  |  {desc}" if isinstance(desc, str) else ""))

# Optional: save to CSV so you can browse it comfortably
df.to_csv("nbia_collections_list.csv", index=False)
print("\nSaved full list to nbia_collections_list.csv")
