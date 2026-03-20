#!/usr/bin/env python3
"""Overnight runner. Python so it can't be killed by pkill -f bash.

Runs all model experiments (02_run_probes.py + 05_attention_probe.py) in sequence.
Skips models that already have results. Logs everything.
"""
import subprocess, sys, os, time, pathlib, shutil

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(BASE_DIR / "hf_cache"))

PROBE_SCRIPT = str(BASE_DIR / "scripts" / "02_run_probes.py")
ATTN_SCRIPT = str(BASE_DIR / "scripts" / "05_attention_probe.py")
LOG_DIR = BASE_DIR / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    # (model, extra_flags, log_suffix)
    # Shuffled-label controls are trained inline (no separate runs needed).
    # Each run trains 5 EMA configs + 1 shuffled control per (layer, task).
    #
    # ── Qwen2.5 base (scaling curve) ──
    ("Qwen/Qwen2.5-0.5B", [], ""),
    ("Qwen/Qwen2.5-1.5B", [], ""),
    ("Qwen/Qwen2.5-3B", [], ""),
    ("Qwen/Qwen2.5-7B", [], ""),
    ("Qwen/Qwen2.5-14B", [], ""),
    # ── Qwen3 base (scaling curve) ──
    ("Qwen/Qwen3-0.6B-Base", [], ""),
    ("Qwen/Qwen3-1.7B-Base", [], ""),
    ("Qwen/Qwen3-4B-Base", [], ""),
    ("Qwen/Qwen3-8B-Base", [], ""),
    # ── Gemma 3 base (scaling curve) ──
    ("google/gemma-3-1b-pt", [], ""),
    ("google/gemma-3-4b-pt", [], ""),
    ("google/gemma-3-12b-pt", [], ""),
    # ── Llama 3.x base (scaling curve) ──
    ("meta-llama/Llama-3.2-1B", [], ""),
    ("meta-llama/Llama-3.2-3B", [], ""),
    ("meta-llama/Llama-3.1-8B", [], ""),
]

# Pre-flight: check data exists
data_path = BASE_DIR / "data" / "processed" / "blog_corpus.parquet"
if not data_path.exists():
    print(f"ERROR: {data_path} not found. Run 01_preprocess_data.py first.", flush=True)
    sys.exit(1)

results_dir = BASE_DIR / "results"
results_dir.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 2
successes, failures = [], []


def run_with_retry(cmd, log_file, label, result_file=None, probe_dir=None, max_retries=MAX_RETRIES):
    """Run a command with retries. Cleans partial output before retry."""
    for attempt in range(max_retries + 1):
        suffix = f" (retry {attempt})" if attempt > 0 else ""
        print(f"==> {time.strftime('%H:%M:%S')}: {label}{suffix}", flush=True)

        # Clean partial outputs from previous failed attempt
        if attempt > 0:
            if result_file and result_file.exists():
                print(f"    Removing partial result: {result_file.name}", flush=True)
                result_file.unlink()
            if probe_dir and probe_dir.exists() and not (probe_dir / "_COMPLETE").exists():
                print(f"    Removing incomplete probe dir: {probe_dir.name}", flush=True)
                shutil.rmtree(probe_dir)

        with open(log_file, "a" if attempt > 0 else "w") as lf:
            if attempt > 0:
                lf.write(f"\n{'='*40}\nRETRY {attempt}\n{'='*40}\n")
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

        if proc.returncode == 0:
            print(f"==> {time.strftime('%H:%M:%S')}: {label} OK", flush=True)
            return True
        else:
            print(f"==> {time.strftime('%H:%M:%S')}: {label} FAIL (exit {proc.returncode})", flush=True)
            if attempt < max_retries:
                print(f"    Retrying in 10s...", flush=True)
                time.sleep(10)
    return False


for model, extra, suffix in RUNS:
    short = model.split("/")[-1] + suffix

    # ── Linear probes ──────────────────────────────────────────────
    result_file = results_dir / f"{short}_per_layer_results.csv"
    if result_file.exists():
        print(f"SKIP {short} probes (results exist)", flush=True)
    else:
        cmd = [sys.executable, PROBE_SCRIPT, "--model_name", model,
               "--max_train_texts", "100000", "--max_test_texts", "10000"] + extra

        print(f"\n{'='*60}", flush=True)
        print(f"==> CMD: {' '.join(cmd)}", flush=True)
        print(f"{'='*60}", flush=True)

        ok = run_with_retry(cmd, LOG_DIR / f"{short}.log", f"{short} probes",
                           result_file=result_file,
                           probe_dir=BASE_DIR / "probes" / short)
        (successes if ok else failures).append(f"{short} probes")

    # ── Attention probes ───────────────────────────────────────────
    attn_result = results_dir / f"{short}_attn_results.csv"
    if attn_result.exists():
        print(f"SKIP {short} attention (results exist)", flush=True)
    else:
        attn_cmd = [sys.executable, ATTN_SCRIPT, "--model_name", model,
                    "--max_train_texts", "100000", "--max_test_texts", "10000"] + extra

        ok = run_with_retry(attn_cmd, LOG_DIR / f"{short}_attn.log", f"{short} attention",
                           result_file=attn_result)
        (successes if ok else failures).append(f"{short} attention")

# ── Summary ────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print(f"==> {time.strftime('%H:%M:%S')}: ALL DONE", flush=True)
print(f"  Successes: {len(successes)}", flush=True)
for s in successes:
    print(f"    OK  {s}", flush=True)
if failures:
    print(f"  Failures: {len(failures)}", flush=True)
    for f in failures:
        print(f"    FAIL  {f}", flush=True)
print(f"{'='*60}", flush=True)
