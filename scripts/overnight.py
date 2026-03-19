#!/usr/bin/env python3
"""Overnight runner. Python so it can't be killed by pkill -f bash."""
import subprocess, sys, os, time

os.environ["HF_HOME"] = "/workspace/hf_cache"
SCRIPT = "/workspace/characterprobing/scripts/02_run_probes.py"
LOG_DIR = "/workspace/characterprobing/results/logs"

RUNS = [
    # (model, extra_flags, log_suffix)
    ("Qwen/Qwen2.5-0.5B", [], ""),   # rerun with 50k × 3 epochs for consistency
    ("Qwen/Qwen2.5-1.5B", [], ""),   # rerun with 3 epochs + fixed ridge
    ("Qwen/Qwen2.5-3B", [], ""),     # rerun with 3 epochs + fixed ridge
    ("Qwen/Qwen2.5-7B", [], ""),     # rerun (crashed last time)
    ("Qwen/Qwen2.5-0.5B-Instruct", [], ""),
    ("Qwen/Qwen2.5-0.5B-Instruct", ["--chat_template"], "_chat"),
    ("Qwen/Qwen2.5-1.5B-Instruct", [], ""),
    ("Qwen/Qwen2.5-1.5B-Instruct", ["--chat_template"], "_chat"),
    ("Qwen/Qwen2.5-3B-Instruct", [], ""),
    ("Qwen/Qwen2.5-3B-Instruct", ["--chat_template"], "_chat"),
    ("Qwen/Qwen2.5-7B-Instruct", [], ""),
    ("Qwen/Qwen2.5-7B-Instruct", ["--chat_template"], "_chat"),
]

# Skip models that already have results
import pathlib
results_dir = pathlib.Path("/workspace/characterprobing/results")

for model, extra, suffix in RUNS:
    short = model.split("/")[-1] + suffix
    result_file = results_dir / f"{short}_per_layer_results.csv"
    if result_file.exists():
        print(f"SKIP {short} (results exist)", flush=True)
        continue

    log_file = f"{LOG_DIR}/{short}.log"
    cmd = [sys.executable, SCRIPT, "--model_name", model,
           "--max_train_texts", "100000", "--max_test_texts", "10000"] + extra

    print(f"\n{'='*60}", flush=True)
    print(f"==> {time.strftime('%H:%M:%S')}: Starting {short}", flush=True)
    print(f"==> CMD: {' '.join(cmd)}", flush=True)
    print(f"{'='*60}", flush=True)

    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

    print(f"==> {time.strftime('%H:%M:%S')}: {short} exit={proc.returncode}", flush=True)

print(f"\n==> {time.strftime('%H:%M:%S')}: ALL DONE", flush=True)
