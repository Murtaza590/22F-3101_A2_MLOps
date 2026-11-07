from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
df = pd.read_csv("audit.csv")

# Drop corrupt files, then every file whose hash appears in BOTH classes
# (ambiguous label), then keep one copy of the remaining duplicates.
df = df[~df["corrupt"]]
n_classes = df.groupby("sha256")["class"].transform("nunique")
df = df[n_classes == 1]
df = df.drop_duplicates("sha256", keep="first").copy()

df["label"] = (df["class"] == "ai").astype(int)   # ai = 1 (positive), real = 0

train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

Path("splits").mkdir(exist_ok=True)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part[["path", "label"]].to_csv(f"splits/{name}.txt", index=False)
    print(name, len(part), part["label"].value_counts().to_dict())
    