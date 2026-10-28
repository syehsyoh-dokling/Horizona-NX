"""Fetch the public source series behind the Track 2 panels (FRED CSV, Ken French daily factors),
for LOCAL evaluation only: realized outcomes of the public cards after their as-of dates.
No API key is used: these are the public, anonymous download endpoints."""
import io
import pathlib
import time
import urllib.request
import zipfile

OUT = pathlib.Path(__file__).resolve().parent / "public_series"
OUT.mkdir(exist_ok=True)
FRED = ["DGS2", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30", "DEXUSEU", "DEXUSUK", "DEXJPUS", "DEXSZUS",
        "DEXUSAL", "DEXCAUS", "DEXUSNZ", "DEXSDUS", "DEXNOUS", "DEXDNUS", "DEXCHUS", "DEXINUS",
        "DEXBZUS"]
UA = {"User-Agent": "Mozilla/5.0 (research backtest)"}


def get(url: str) -> bytes:
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print("  retry", url, e)
            time.sleep(3)
    raise RuntimeError(url)


for s in FRED:
    p = OUT / f"{s}.csv"
    if p.exists() and p.stat().st_size > 1000:
        continue
    try:
        p.write_bytes(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"))
        print("fred", s, p.stat().st_size)
    except RuntimeError as e:
        print("FAILED", s, e)

KF = {"ff3_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
      "mom_daily": "F-F_Momentum_Factor_daily_CSV.zip"}
for name, z in KF.items():
    p = OUT / f"{name}.csv"
    if p.exists() and p.stat().st_size > 1000:
        continue
    try:
        raw = get(f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{z}")
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            p.write_bytes(zf.read(zf.namelist()[0]))
        print("kf", name, p.stat().st_size)
    except RuntimeError as e:
        print("FAILED", name, e)
print("done", sorted((x.name, x.stat().st_size) for x in OUT.iterdir()))
