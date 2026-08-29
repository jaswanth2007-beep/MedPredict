from pathlib import Path
import zipfile
import io
import requests

URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

r = requests.get(URL, timeout=60)
r.raise_for_status()
with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    z.extractall(OUT)

print("Downloaded and extracted C-MAPSS data to", OUT)
