#!/usr/bin/env python3
"""Download and preprocess Hippocorpus for probing experiments."""
from datasets import load_dataset
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "processed" / "hippocorpus.parquet"

ds = load_dataset("allenai/hippocorpus", split="train")
df = ds.to_pandas()
print(f"Hippocorpus: {len(df)} rows")

age_col = "annotatorAge"
gender_col = "annotatorGender"
print(f"Age: {df[age_col].value_counts().to_dict()}")
print(f"Gender: {df[gender_col].value_counts().to_dict()}")

age_map = {"18-24": 1, "25-29": 2, "30-34": 2, "35-39": 3, "40-44": 3, "45-49": 3, "50-54": 3, "55+": 3}
gender_map = {"Female": "female", "Male": "male"}

rows = []
for _, row in df.iterrows():
    ab = age_map.get(row[age_col])
    g = gender_map.get(row[gender_col])
    text = str(row["story"])
    if ab is None or g is None or len(text) < 20:
        continue
    rows.append({
        "text": text,
        "blogger_id": str(row.get("WorkerId", "unknown")),
        "age_bin": ab,
        "gender": g,
        "star_sign": "Unknown",
        "n_tokens": len(text.split()),
    })

hdf = pd.DataFrame(rows)
print(f"Valid: {len(hdf)}, Authors: {hdf['blogger_id'].nunique()}")
print(f"Age bins: {hdf['age_bin'].value_counts().sort_index().to_dict()}")
print(f"Gender: {hdf['gender'].value_counts().to_dict()}")

authors = hdf["blogger_id"].unique()
train_a, temp_a = train_test_split(authors, test_size=0.3, random_state=42)
val_a, test_a = train_test_split(temp_a, test_size=0.5, random_state=42)
sm = {}
for a in train_a: sm[a] = "train"
for a in val_a: sm[a] = "val"
for a in test_a: sm[a] = "test"
hdf["split"] = hdf["blogger_id"].map(sm)

for s in ["train", "val", "test"]:
    print(f"  {s}: {len(hdf[hdf['split']==s])}")

hdf.to_parquet(OUT_PATH, index=False)
print(f"Saved {OUT_PATH}")
