#!/usr/bin/env python3
"""Run probing on blog corpus, SynthPAI, and Hippocorpus.
Also run transfer tests: blog probes evaluated on other datasets."""

import os, sys, subprocess, time, json
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/character-probing/hf_cache")
## HF_TOKEN must be set in environment before running

BASE = Path("/workspace/character-probing")
SCRIPT = str(BASE / "scripts" / "02_run_probes.py")

# Models that fit on A40 (46GB)
MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "google/gemma-3-1b-pt",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
]

DATASETS = {
    "blog": str(BASE / "data" / "processed" / "blog_corpus.parquet"),
    "synthpai": str(BASE / "data" / "processed" / "synthpai_corpus.parquet"),
    "hippocorpus": str(BASE / "data" / "processed" / "hippocorpus.parquet"),
}


def run_probe(model, dataset_name, data_path, max_train=0, max_test=0):
    short = model.split("/")[-1]
    out_dir = str(BASE / "results" / dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    result_file = Path(out_dir) / f"{short}_per_layer_results.csv"
    if result_file.exists():
        print(f"  SKIP {short} ({dataset_name})", flush=True)
        return True

    cmd = [sys.executable, SCRIPT, "--model_name", model,
           "--data_path", data_path, "--output_dir", out_dir,
           "--max_train_texts", str(max_train), "--max_test_texts", str(max_test)]

    log_file = BASE / "results" / "logs" / f"{dataset_name}_{short}.log"
    print(f"  ==> {time.strftime('%H:%M:%S')} {short} ({dataset_name})", flush=True)
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

    ok = proc.returncode == 0
    print(f"  ==> {time.strftime('%H:%M:%S')} {short} {'OK' if ok else 'FAIL'}", flush=True)
    return ok


def preprocess_blog():
    """Download and preprocess blog corpus if needed."""
    parquet = BASE / "data" / "processed" / "blog_corpus.parquet"
    if parquet.exists() and parquet.stat().st_size > 1000:
        print("Blog corpus already preprocessed", flush=True)
        return

    raw_dir = BASE / "data" / "raw" / "blogs"
    if not raw_dir.exists() or not list(raw_dir.glob("*.xml")):
        print("Downloading Blog Authorship Corpus...", flush=True)
        os.makedirs(BASE / "data" / "raw", exist_ok=True)
        subprocess.run("apt-get update -qq && apt-get install -y -qq unzip wget > /dev/null 2>&1",
                       shell=True, cwd=str(BASE / "data" / "raw"))
        subprocess.run(["wget", "-q", "https://u.cs.biu.ac.il/~koppel/blogs/blogs.zip", "-O", "blogs.zip"],
                       cwd=str(BASE / "data" / "raw"))
        subprocess.run(["unzip", "-qo", "blogs.zip", "-d", "blogs_tmp"], cwd=str(BASE / "data" / "raw"))
        os.makedirs(raw_dir, exist_ok=True)
        subprocess.run("find blogs_tmp -name '*.xml' -exec mv {} blogs/ \\;",
                       shell=True, cwd=str(BASE / "data" / "raw"))
        subprocess.run(["rm", "-rf", "blogs_tmp", "blogs.zip"], cwd=str(BASE / "data" / "raw"))

    print("Preprocessing blog corpus...", flush=True)
    subprocess.run([sys.executable, str(BASE / "scripts" / "01_preprocess_data.py")])


def preprocess_hippocorpus():
    """Download and preprocess Hippocorpus."""
    parquet = BASE / "data" / "processed" / "hippocorpus.parquet"
    if parquet.exists() and parquet.stat().st_size > 1000:
        print("Hippocorpus already preprocessed", flush=True)
        return

    print("Preprocessing Hippocorpus...", flush=True)
    from datasets import load_dataset
    import pandas as pd
    from sklearn.model_selection import train_test_split

    ds = load_dataset("allenai/hippocorpus", split="train")
    df = ds.to_pandas()

    age_map = {"18-24": 1, "25-29": 2, "30-34": 2, "35-39": 3,
               "40-44": 3, "45-49": 3, "50-54": 3, "55+": 3}
    gender_map = {"Female": "female", "Male": "male"}

    rows = []
    for _, row in df.iterrows():
        ab = age_map.get(row["annotatorAge"])
        g = gender_map.get(row["annotatorGender"])
        text = str(row["story"])
        if ab is None or g is None or len(text) < 20:
            continue
        rows.append({"text": text, "blogger_id": str(row.get("WorkerId", "unknown")),
                     "age_bin": ab, "gender": g, "star_sign": "Unknown",
                     "n_tokens": len(text.split())})

    hdf = pd.DataFrame(rows)
    authors = hdf["blogger_id"].unique()
    tr, tmp = train_test_split(authors, test_size=0.3, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, random_state=42)
    sm = {}
    for a in tr: sm[a] = "train"
    for a in va: sm[a] = "val"
    for a in te: sm[a] = "test"
    hdf["split"] = hdf["blogger_id"].map(sm)
    hdf.to_parquet(str(parquet), index=False)
    print(f"Saved {len(hdf)} rows ({hdf['blogger_id'].nunique()} authors)", flush=True)


if __name__ == "__main__":
    os.makedirs(BASE / "results" / "logs", exist_ok=True)

    # 1. Preprocess all datasets
    print("=" * 60, flush=True)
    print("PREPROCESSING", flush=True)
    print("=" * 60, flush=True)
    preprocess_blog()
    preprocess_hippocorpus()

    # 2. Run probes on all datasets
    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'=' * 60}", flush=True)
        print(f"PROBING: {ds_name}", flush=True)
        print(f"{'=' * 60}", flush=True)

        max_train = 100000 if ds_name == "blog" else 0
        max_test = 10000 if ds_name == "blog" else 0

        for model in MODELS:
            run_probe(model, ds_name, ds_path, max_train, max_test)

    print(f"\n{'=' * 60}", flush=True)
    print("ALL DONE", flush=True)
    print(f"{'=' * 60}", flush=True)
