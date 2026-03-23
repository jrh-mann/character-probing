#!/usr/bin/env python3
"""Run transfer eval for all models. One model at a time, clear cache between."""
import subprocess, sys, os, time, shutil
from pathlib import Path

os.environ["HF_HOME"] = "/root/hf_cache"
os.environ.setdefault("HF_TOKEN", "")

BASE = Path(__file__).resolve().parent.parent
SCRIPT = str(BASE / "scripts" / "08_transfer_eval.py")

MODELS = [
    "pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b",
    "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B", "Qwen2.5-14B",
    "Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base", "Qwen3-8B-Base",
    "gemma-3-270m", "gemma-3-1b-pt", "gemma-3-4b-pt", "gemma-3-12b-pt",
    "Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.1-8B",
]


def clear_cache():
    cache = Path("/root/hf_cache/hub")
    if cache.exists():
        for d in cache.glob("models--*"):
            shutil.rmtree(d, ignore_errors=True)


def main():
    results_file = BASE / "results" / "transfer_eval.csv"

    # Check which models already done
    done = set()
    if results_file.exists():
        import pandas as pd
        df = pd.read_csv(results_file)
        done = set(df["model"].unique())

    todo = [m for m in MODELS if m not in done]
    print(f"Transfer eval: {len(todo)} models to run ({len(done)} already done)")

    for ms in todo:
        probe_dir = BASE / "probes" / ms
        if not probe_dir.exists():
            print(f"  {ms}: no probes, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"=== {ms} ({time.strftime('%H:%M:%S')}) ===")

        log = BASE / "results" / "logs" / f"transfer_{ms}.log"
        os.makedirs(log.parent, exist_ok=True)
        with open(log, "w") as lf:
            proc = subprocess.run(
                [sys.executable, SCRIPT, "--model_name", ms],
                stdout=lf, stderr=subprocess.STDOUT, timeout=3600)

        if proc.returncode == 0:
            print(f"  OK")
        else:
            print(f"  FAIL (exit {proc.returncode})")
            content = log.read_text()
            for line in content.strip().split("\n")[-3:]:
                print(f"    {line}")

        clear_cache()

    print(f"\n{'='*50}")
    print("TRANSFER EVAL DONE")


if __name__ == "__main__":
    main()
