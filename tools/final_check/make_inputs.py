"""Build the runner's /input layout for each -ps1 unit and tar it for the VPS.

Layout per unit (what the organizers' runner mounts, per forecast_card.md):
  <unit>/card.toml, forecast_spec.json, manifest.json, forecast_card.md
  <unit>/panels/<panel_id>.parquet
  <unit>/text/...
No practice-answers are copied (the mount must stay answer-free, like a Final input).
"""
import pathlib
import shutil
import sys
import tarfile

PUB = pathlib.Path(sys.argv[1])
DST = pathlib.Path(sys.argv[2])
TAR = pathlib.Path(sys.argv[3])

if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)
n = 0
for unit in sorted(PUB.glob("units/*-ps1")):
    d = DST / unit.name
    (d / "panels").mkdir(parents=True)
    for f in ("card.toml", "forecast_spec.json", "manifest.json", "forecast_card.md"):
        shutil.copy2(unit / f, d / f)
    for pq in unit.glob("*.parquet"):
        shutil.copy2(pq, d / "panels" / pq.name)
    shutil.copytree(unit / "text", d / "text")
    n += 1
with tarfile.open(TAR, "w:gz") as tf:
    tf.add(DST, arcname=DST.name)
print(f"{n} units -> {DST} ; tar {TAR} ({TAR.stat().st_size/1e6:.1f} MB)")
