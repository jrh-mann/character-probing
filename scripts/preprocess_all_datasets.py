#!/usr/bin/env python3
"""Preprocess all non-blog datasets. Run after blog corpus is ready."""
import os, sys, urllib.request, json, io
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

os.environ.setdefault("HF_HOME", "/root/hf_cache")
BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def save_dataset(df, name):
    path = PROC / f"{name}.parquet"
    if path.exists() and path.stat().st_size > 1000:
        print(f"  {name}: already exists ({len(pd.read_parquet(path))} rows)")
        return
    df.to_parquet(str(path), index=False)
    print(f"  {name}: saved {len(df)} rows")


def prep_hippocorpus():
    print("\n=== Hippocorpus ===")
    csv_path = "/workspace/hippo_data/hippoCorpusV2.csv"
    if not os.path.exists(csv_path):
        print("  Downloading from Kaggle...")
        url = "https://www.kaggle.com/api/v1/datasets/download/saurabhshahane/hippocorpus"
        urllib.request.urlretrieve(url, "/workspace/hippo.zip")
        os.makedirs("/workspace/hippo_data", exist_ok=True)
        import subprocess
        subprocess.run(["unzip", "-qo", "/workspace/hippo.zip", "-d", "/workspace/hippo_data"])

    df = pd.read_csv(csv_path)
    gender_map = {"man": "male", "woman": "female"}
    rows = []
    for _, row in df.iterrows():
        age = row.get("annotatorAge")
        g = gender_map.get(str(row.get("annotatorGender", "")))
        text = str(row.get("story", ""))
        if pd.isna(age) or g is None or len(text) < 20:
            continue
        age = int(age)
        ab = 1 if age <= 29 else (2 if age <= 44 else 3)
        rows.append({"text": text, "blogger_id": str(row.get("WorkerId", "unknown")),
                     "age_bin": ab, "gender": g, "star_sign": "Unknown", "n_tokens": len(text.split())})
    hdf = pd.DataFrame(rows)
    n = len(hdf)
    perm = np.random.RandomState(42).permutation(n)
    hdf["split"] = "train"
    hdf.iloc[perm[int(0.7*n):int(0.85*n)], hdf.columns.get_loc("split")] = "val"
    hdf.iloc[perm[int(0.85*n):], hdf.columns.get_loc("split")] = "test"
    save_dataset(hdf, "hippocorpus")


def prep_ellipse():
    print("\n=== ELLIPSE ===")
    url = "https://raw.githubusercontent.com/scrosseye/ELLIPSE-Corpus/main/ELLIPSE_Final_github_train.csv"
    print("  Downloading from GitHub...")
    df = pd.read_csv(url)
    gender_map = {"Male": "male", "Female": "female"}
    rows = []
    for _, row in df.iterrows():
        g = gender_map.get(str(row.get("gender", "")))
        text = str(row.get("full_text", ""))
        if g is None or len(text) < 20:
            continue
        grade = row.get("grade", 0)
        ab = 1 if grade <= 9 else (2 if grade <= 10 else 3)
        rows.append({"text": text, "blogger_id": f"student_{_}",
                     "age_bin": ab, "gender": g, "star_sign": "Unknown", "n_tokens": len(text.split())})
    hdf = pd.DataFrame(rows)
    n = len(hdf)
    perm = np.random.RandomState(42).permutation(n)
    hdf["split"] = "train"
    hdf.iloc[perm[int(0.7*n):int(0.85*n)], hdf.columns.get_loc("split")] = "val"
    hdf.iloc[perm[int(0.85*n):], hdf.columns.get_loc("split")] = "test"
    save_dataset(hdf, "ellipse")


def prep_europarl():
    print("\n=== Europarl ===")
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("samzirbo/europarl.en-es.gendered",
                           "europarl.en-es.simple.json", repo_type="dataset")
    data = [json.loads(line) for line in open(path)]
    rows = []
    for d in data:
        text = d.get("en", "")
        gender = d.get("gender", "")
        if gender not in ("male", "female") or len(text) < 50:
            continue
        rows.append({"text": text, "blogger_id": "unknown", "age_bin": 1,
                     "gender": gender, "star_sign": "Unknown", "n_tokens": len(text.split())})
    df = pd.DataFrame(rows)
    min_count = min(df["gender"].value_counts())
    balanced = pd.concat([df[df["gender"]==g].sample(min(min_count, 50000), random_state=42)
                         for g in ["male", "female"]])
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(balanced)
    balanced["split"] = "train"
    balanced.iloc[int(0.7*n):int(0.85*n), balanced.columns.get_loc("split")] = "val"
    balanced.iloc[int(0.85*n):, balanced.columns.get_loc("split")] = "test"
    save_dataset(balanced, "europarl_gender")


def prep_prism():
    print("\n=== PRISM ===")
    from datasets import load_dataset
    survey = load_dataset("HannahRoseKirk/prism-alignment", "survey", split="train").to_pandas()
    convos = load_dataset("HannahRoseKirk/prism-alignment", "conversations", split="train").to_pandas()
    merged = convos.merge(survey[["user_id", "age", "gender"]], on="user_id", how="inner")
    age_map = {"18-24 years old": 1, "25-34 years old": 1, "35-44 years old": 2,
               "45-54 years old": 2, "55-64 years old": 3, "65+ years old": 3}
    gender_map = {"Male": "male", "Female": "female"}
    rows = []
    for _, row in merged.iterrows():
        text = str(row["opening_prompt"])
        ab = age_map.get(row["age"])
        g = gender_map.get(row["gender"])
        if ab is None or g is None or len(text) < 20:
            continue
        rows.append({"text": text, "blogger_id": str(row["user_id"]),
                     "age_bin": ab, "gender": g, "star_sign": "Unknown", "n_tokens": len(text.split())})
    hdf = pd.DataFrame(rows)
    n = len(hdf)
    perm = np.random.RandomState(42).permutation(n)
    hdf["split"] = "train"
    hdf.iloc[perm[int(0.7*n):int(0.85*n)], hdf.columns.get_loc("split")] = "val"
    hdf.iloc[perm[int(0.85*n):], hdf.columns.get_loc("split")] = "test"
    save_dataset(hdf, "prism")


if __name__ == "__main__":
    prep_hippocorpus()
    prep_ellipse()
    prep_europarl()
    prep_prism()
    print("\n=== ALL DATASETS READY ===")
    for f in sorted(PROC.glob("*.parquet")):
        df = pd.read_parquet(f)
        print(f"  {f.name}: {len(df)} rows")
