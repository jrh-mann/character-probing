#!/usr/bin/env python3
"""
03_data_scaling.py — Quick data scaling experiment on one model.

Runs the probe pipeline at multiple training set sizes to determine
diminishing returns. Uses a single model (default 0.5B) for speed.
"""

import subprocess
import sys
import os
import time
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-0.5B"
SIZES = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000]
TEST_SIZE = 10000
SCRIPT = str(BASE_DIR / "scripts" / "02_run_probes.py")
OUTPUT_DIR = str(BASE_DIR / "results" / "data_scaling")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

results = []
for n_train in SIZES:
    print(f"\n{'='*60}")
    print(f"  Training with {n_train} texts")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--model_name", MODEL,
         "--max_train_texts", str(n_train),
         "--max_test_texts", str(TEST_SIZE),
         "--output_dir", OUTPUT_DIR,
         "--seed", "42"],
        capture_output=True, text=True,
        env={**os.environ, "HF_HOME": str(BASE_DIR / "hf_cache")}
    )
    elapsed = time.time() - t0

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"FAILED: {result.stderr[-500:]}")
        continue

    # Read results
    csv_path = Path(OUTPUT_DIR) / "Qwen2.5-0.5B_per_layer_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for task in ["age_bin", "gender", "star_sign"]:
            task_df = df[df["task"] == task]
            if len(task_df) == 0:
                continue
            best = task_df.loc[task_df["text_balanced_acc"].idxmax()]
            results.append({
                "n_train": n_train,
                "task": task,
                "best_layer": int(best["layer"]),
                "text_balanced_acc": best["text_balanced_acc"],
                "macro_f1": best["macro_f1"],
                "time_s": round(elapsed, 1),
            })
        # Rename to avoid overwrite
        csv_path.rename(csv_path.parent / f"Qwen2.5-0.5B_n{n_train}_results.csv")

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_DIR}/data_scaling_summary.csv", index=False)

print(f"\n{'='*60}")
print("DATA SCALING SUMMARY")
print(f"{'='*60}")
for task in ["age_bin", "gender", "star_sign"]:
    print(f"\n{task}:")
    t = results_df[results_df["task"] == task].sort_values("n_train")
    for _, r in t.iterrows():
        print(f"  {int(r['n_train']):>6} texts: bal_acc={r['text_balanced_acc']:.4f}  f1={r['macro_f1']:.4f}  ({r['time_s']:.0f}s)")
