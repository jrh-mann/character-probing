#!/usr/bin/env python3
"""Re-evaluate all completed models with corrected per-token eval."""
import subprocess, sys, os, time, pathlib

os.environ["HF_HOME"] = "/workspace/hf_cache"
SCRIPT = "/workspace/characterprobing/scripts/04_reeval.py"
PROBE_DIR = pathlib.Path("/workspace/characterprobing/probes")

# Discover all saved probe directories
RUNS = []
for d in sorted(PROBE_DIR.iterdir()):
    if not d.is_dir(): continue
    name = d.name  # e.g. "Qwen2.5-0.5B-Instruct_chat"
    is_chat = name.endswith("_chat")
    base_name = name.replace("_chat", "")
    model_id = f"Qwen/{base_name}"
    RUNS.append((model_id, is_chat, name))

for model_id, is_chat, short_name in RUNS:
    result_file = pathlib.Path(f"/workspace/characterprobing/results/{short_name}_reeval_results.csv")
    if result_file.exists():
        print(f"SKIP {short_name} (reeval results exist)", flush=True)
        continue

    cmd = [sys.executable, SCRIPT, "--model_name", model_id,
           "--max_test_texts", "10000"]
    if is_chat:
        cmd.append("--chat_template")

    print(f"\n{'='*60}", flush=True)
    print(f"==> {time.strftime('%H:%M:%S')}: Re-evaluating {short_name}", flush=True)
    print(f"==> CMD: {' '.join(cmd)}", flush=True)
    print(f"{'='*60}", flush=True)

    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT)
    print(f"==> {time.strftime('%H:%M:%S')}: {short_name} exit={proc.returncode}", flush=True)

print(f"\n==> {time.strftime('%H:%M:%S')}: ALL RE-EVALS DONE", flush=True)
