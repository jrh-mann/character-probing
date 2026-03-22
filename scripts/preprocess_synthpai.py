#!/usr/bin/env python3
"""Preprocess SynthPAI dataset for probing experiments.

Downloads from HuggingFace, creates train/val/test split by author,
saves as parquet in the same format as the blog corpus.
"""

import os
from pathlib import Path
from collections import Counter

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "processed" / "synthpai_corpus.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading SynthPAI from HuggingFace...")
    ds = load_dataset("RobinSta/SynthPAI", split="train")
    print(f"  {len(ds)} rows")

    rows = []
    for item in ds:
        profile = item["profile"]
        text = item["text"]
        if not text or len(text.strip()) < 20:
            continue

        age = profile.get("age")
        sex = profile.get("sex")
        if age is None or sex is None:
            continue

        # 3-class age bins
        if age <= 30:
            age_bin = 1  # young
        elif age <= 50:
            age_bin = 2  # middle
        else:
            age_bin = 3  # older

        rows.append({
            "text": text,
            "blogger_id": item.get("author", "unknown"),
            "age_bin": age_bin,
            "gender": sex,
            "star_sign": "Unknown",  # not available
            "income_level": profile.get("income_level", "unknown"),
            "n_tokens": len(text.split()),  # rough estimate
        })

    df = pd.DataFrame(rows)
    print(f"  {len(df)} valid rows, {df['blogger_id'].nunique()} authors")

    # Print distributions
    print(f"\n  Age bins: {dict(df['age_bin'].value_counts().sort_index())}")
    print(f"  Gender: {dict(df['gender'].value_counts())}")
    print(f"  Income: {dict(df['income_level'].value_counts())}")

    # Split by author (70/15/15)
    authors = df["blogger_id"].unique()
    train_auth, temp_auth = train_test_split(authors, test_size=0.30, random_state=42)
    val_auth, test_auth = train_test_split(temp_auth, test_size=0.50, random_state=42)

    split_map = {}
    for a in train_auth: split_map[a] = "train"
    for a in val_auth: split_map[a] = "val"
    for a in test_auth: split_map[a] = "test"

    df["split"] = df["blogger_id"].map(split_map)

    # Print split sizes
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        print(f"\n  {split}: {len(sub)} texts, {sub['blogger_id'].nunique()} authors")
        print(f"    age: {dict(sub['age_bin'].value_counts().sort_index())}")
        print(f"    gender: {dict(sub['gender'].value_counts())}")

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()
