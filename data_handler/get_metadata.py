import asyncio
import aiohttp
import csv
import json
import time
import os
from collections import OrderedDict

BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

async def fetch_json(session: aiohttp.ClientSession, path: str, params=None):
    """Robust JSON fetch: tolerates 200 + empty body and missing Content-Type."""
    url = f"{BASE}/{path}"
    for attempt in range(3):
        try:
            async with session.get(
                url, params=params, headers={"Accept": "application/json"}, timeout=120
            ) as r:
                if r.status == 204 or r.headers.get("Content-Length") == "0":
                    return []
                text = await r.text()  # don't use r.json() – header may be missing
                if not text.strip():
                    return []
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    # Sometimes servers return junk/HTML when under load; retry a bit.
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (2**attempt))
                        continue
                    return []
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt < 2:
                await asyncio.sleep(0.5 * (2**attempt))
                continue
            return []


async def get_collections(session):
    data = await fetch_json(session, "getCollectionValues")
    return [d.get("Collection") for d in (data or []) if d.get("Collection")]


async def summarize_collection(session, collection: str, sem: asyncio.Semaphore):
    async with sem:
        # Fire requests concurrently for this collection
        patients_task = asyncio.create_task(
            fetch_json(session, "getPatient", {"Collection": collection})
        )
        series_ct_task = asyncio.create_task(
            fetch_json(
                session, "getSeries", {"Collection": collection, "Modality": "CT"}
            )
        )
        mods_task = asyncio.create_task(
            fetch_json(session, "getModalityValues", {"Collection": collection})
        )
        body_ct_task = asyncio.create_task(
            fetch_json(
                session,
                "getBodyPartValues",
                {"Collection": collection, "Modality": "CT"},
            )
        )
        vendors_task = asyncio.create_task(
            fetch_json(session, "getManufacturerValues", {"Collection": collection})
        )

        patients, series_ct, mods, body_ct, vendors = await asyncio.gather(
            patients_task,
            series_ct_task,
            mods_task,
            body_ct_task,
            vendors_task,
            return_exceptions=True,
        )

    # Normalize exceptions → treat as missing
    def ok(x):
        return [] if isinstance(x, Exception) or x is None else x

    patients = ok(patients)
    series_ct = ok(series_ct)
    mods = ok(mods)
    body_ct = ok(body_ct)
    vendors = ok(vendors)

    patients_count = len(patients)
    series_ct_count = len(series_ct)
    studies_ct_count = len(
        {s.get("StudyInstanceUID") for s in series_ct if s.get("StudyInstanceUID")}
    )

    modalities_all = ", ".join(
        sorted({m.get("Modality") for m in mods if m.get("Modality")})
    )
    bodyparts_ct = ", ".join(
        sorted(
            {b.get("BodyPartExamined") for b in body_ct if b.get("BodyPartExamined")}
        )
    )
    vendors_all = sorted(
        {v.get("Manufacturer") for v in vendors if v.get("Manufacturer")}
    )
    vendors_sample = ", ".join(vendors_all[:3])

    return OrderedDict(
        [
            ("Collection", collection),
            ("Patients", patients_count),
            ("Series(CT)", series_ct_count),
            ("Studies(CT)", studies_ct_count),
            ("Modalities(all)", modalities_all),
            ("BodyParts(CT)", bodyparts_ct),
            ("Vendors(sample)", vendors_sample),
        ]
    )


async def main():
    connector = aiohttp.TCPConnector(limit=12)
    sem = asyncio.Semaphore(8)  # per-collection burst cap
    async with aiohttp.ClientSession(connector=connector) as session:
        collections = await get_collections(session)
        # Optionally filter here while testing to avoid hammering the API:
        # collections = collections[:10]

        rows = await asyncio.gather(
            *(summarize_collection(session, c, sem) for c in collections)
        )

    # Save output into the same folder as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, "metadata.csv")

    fieldnames = [
        "Collection",
        "Patients",
        "Series(CT)",
        "Studies(CT)",
        "Modalities(all)",
        "BodyParts(CT)",
        "Vendors(sample)",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if r:
                w.writerow(r)
    print(f"Wrote {out} with {len([r for r in rows if r])} collections.")


if __name__ == "__main__":
    t0 = time.perf_counter()
    try:
        asyncio.run(main())
    finally:
        print(f"Total runtime: {time.perf_counter() - t0:.2f}s")
