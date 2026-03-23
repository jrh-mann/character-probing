#!/usr/bin/env python3
"""Fill missing data points and run 7B on all datasets.

Uses /root/hf_cache for model downloads.
Clears model cache between models to save space.
"""
import subprocess, sys, os, time
from pathlib import Path

os.environ["HF_HOME"] = "/root/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

BASE = Path(__file__).resolve().parent.parent
SCRIPT = str(BASE / "scripts" / "02_run_probes.py")

DATASETS = {
    "hippocorpus": (str(BASE / "data" / "processed" / "hippocorpus.parquet"), "0", "0"),
    "ellipse": (str(BASE / "data" / "processed" / "ellipse.parquet"), "0", "0"),
    "prism": (str(BASE / "data" / "processed" / "prism.parquet"), "0", "0"),
    "synthpai": (str(BASE / "data" / "processed" / "synthpai_corpus.parquet"), "0", "0"),
    "europarl": (str(BASE / "data" / "processed" / "europarl_gender.parquet"), "10000", "5000"),
}

# Models to run — load once, cycle all datasets, then discard
MODELS = [
    # Gap fills
    ("EleutherAI/pythia-160m", 16),  # (model_id, batch_size)
    # 7B
    ("Qwen/Qwen2.5-7B", 4),
]


def run_one(model, ds_name, data_path, max_train, max_test, batch_size):
    """Run one model on one dataset. Returns True if successful."""
    short = model.split("/")[-1]
    out_dir = str(BASE / "results" / ds_name)
    os.makedirs(out_dir, exist_ok=True)
    result_file = Path(out_dir) / f"{short}_per_layer_results.csv"

    if result_file.exists():
        return True  # already done

    if not Path(data_path).exists():
        print(f"    NO DATA: {data_path}", flush=True)
        return False

    cmd = [sys.executable, SCRIPT, "--model_name", model,
           "--data_path", data_path, "--output_dir", out_dir,
           "--batch_size", str(batch_size),
           "--max_train_texts", max_train, "--max_test_texts", max_test]

    log_file = BASE / "results" / "logs" / f"{ds_name}_{short}.log"
    os.makedirs(log_file.parent, exist_ok=True)

    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        # Check if it was OOM
        try:
            log_content = log_file.read_text()
            if "FATAL" in log_content:
                print(f"    OOM/FATAL", flush=True)
            else:
                print(f"    FAIL (exit {proc.returncode})", flush=True)
                # Print last few lines of error
                lines = log_content.strip().split("\n")
                for line in lines[-3:]:
                    print(f"      {line}", flush=True)
        except:
            print(f"    FAIL (exit {proc.returncode})", flush=True)
        return False

    # Verify result file was created
    if not result_file.exists():
        print(f"    No result file produced!", flush=True)
        return False

    return True


def main():
    # Test one small model first to verify everything works
    print("=== SMOKE TEST ===", flush=True)
    test_model = "EleutherAI/pythia-14m"
    test_ds = "prism"
    dp, mt, mte = DATASETS[test_ds]
    short = test_model.split("/")[-1]
    result = Path(BASE / "results" / test_ds / f"{short}_per_layer_results.csv")
    if result.exists():
        print(f"  Smoke test: {short}/{test_ds} already exists, skipping", flush=True)
    else:
        print(f"  Running {short} on {test_ds}...", flush=True)
        ok = run_one(test_model, test_ds, dp, mt, mte, 64)
        if not ok:
            print("  SMOKE TEST FAILED — aborting", flush=True)
            sys.exit(1)
        print(f"  Smoke test passed!", flush=True)
    os.system("rm -rf /root/hf_cache/hub/models--* 2>/dev/null")

    # Run all models
    for model, default_bs in MODELS:
        short = model.split("/")[-1]
        print(f"\n{'='*60}", flush=True)
        print(f"=== {short} (bs={default_bs}) ===", flush=True)
        print(f"{'='*60}", flush=True)

        for ds_name, (data_path, max_train, max_test) in DATASETS.items():
            result = Path(BASE / "results" / ds_name / f"{short}_per_layer_results.csv")
            if result.exists():
                print(f"  SKIP {ds_name}", flush=True)
                continue

            print(f"  ==> {time.strftime('%H:%M:%S')} {ds_name}", flush=True)
            ok = run_one(model, ds_name, data_path, max_train, max_test, default_bs)
            status = "OK" if ok else "FAIL"
            print(f"  ==> {time.strftime('%H:%M:%S')} {ds_name} {status}", flush=True)

        # Clear model cache
        os.system("rm -rf /root/hf_cache/hub/models--* 2>/dev/null")
        print(f"  [cache cleared]", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"ALL DONE", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
