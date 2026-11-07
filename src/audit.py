import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

DATA = Path("data")
CLASSES = ["ai", "real"]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


rows = []
for cls in CLASSES:
    for p in sorted((DATA / cls).rglob("*")):
        if not p.is_file():
            continue
        row = {
            "path": p.as_posix(), "class": cls,
            "file_size": p.stat().st_size, "sha256": sha256(p),
            "width": None, "height": None, "mode": None,
            "format": None, "corrupt": False, "error": "",
        }
        try:
            with Image.open(p) as im:
                row["format"], row["mode"] = im.format, im.mode
                row["width"], row["height"] = im.size
                im.verify()              # structural check
            with Image.open(p) as im:
                im.load()                # full decode catches truncated files
        except Exception as e:
            row["corrupt"] = True
            row["error"] = str(e)[:100]
        rows.append(row)

df = pd.DataFrame(rows)
df["is_duplicate"] = df.duplicated("sha256", keep="first")
df.to_csv("audit.csv", index=False)

ok = df[~df["corrupt"]]
print("\n=== Files per class ===")
print(df.groupby("class").size())
print("\n=== Min / median / max per class ===")
print(ok.groupby("class")[["width", "height", "file_size"]].agg(["min", "median", "max"]).T)
print("\n=== Colour modes ===")
print(ok.groupby(["class", "mode"]).size())
print("\n=== Formats ===")
print(ok.groupby(["class", "format"]).size())
print("\nCorrupt files:", int(df["corrupt"].sum()))
print("Exact duplicates (extra copies):", int(df["is_duplicate"].sum()))
groups = df[df.duplicated("sha256", keep=False)].groupby("sha256")["class"].nunique()
print("Duplicate groups spanning both classes:", int((groups > 1).sum()))