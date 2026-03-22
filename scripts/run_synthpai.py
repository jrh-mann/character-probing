#!/usr/bin/env python3
"""Run probing experiments on SynthPAI dataset across multiple models."""
import subprocess, sys, os, time, pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(BASE_DIR / "hf_cache"))

SCRIPT = str(BASE_DIR / "scripts" / "02_run_probes.py")
LOG_DIR = BASE_DIR / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = str(BASE_DIR / "data" / "processed" / "synthpai_corpus.parquet")
RESULTS_DIR = BASE_DIR / "results" / "synthpai"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Models that fit on 20GB VRAM
MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "google/gemma-3-1b-pt",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
]

for model in MODELS:
    short = model.split("/")[-1]
    result_file = RESULTS_DIR / f"{short}_per_layer_results.csv"
    if result_file.exists():
        print(f"SKIP {short}", flush=True)
        continue

    cmd = [sys.executable, SCRIPT,
           "--model_name", model,
           "--data_path", DATA_PATH,
           "--output_dir", str(RESULTS_DIR),
           "--max_train_texts", "0",
           "--max_test_texts", "0"]

    print(f"\n{'='*60}", flush=True)
    print(f"==> {time.strftime('%H:%M:%S')}: {short} (SynthPAI)", flush=True)
    print(f"{'='*60}", flush=True)

    log_file = LOG_DIR / f"synthpai_{short}.log"
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

    status = "OK" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
    print(f"==> {time.strftime('%H:%M:%S')}: {short} {status}", flush=True)

print(f"\n==> {time.strftime('%H:%M:%S')}: ALL SYNTHPAI DONE", flush=True)
