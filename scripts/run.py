#!/usr/bin/env python3
"""Comprehensive runner: fills gaps, runs bigger models, classifier, position accuracy.

Usage: python scripts/run.py [--skip-gaps] [--skip-classifier] [--skip-position]

Designed for A40 (46GB). Loads one model at a time, runs all needed experiments,
then clears cache before loading the next.
"""
import subprocess, sys, os, time, shutil
from pathlib import Path

os.environ["HF_HOME"] = "/root/hf_cache"
os.environ.setdefault("HF_TOKEN", "")

BASE = Path(__file__).resolve().parent.parent
SCRIPT = str(BASE / "scripts" / "02_run_probes.py")
POS_SCRIPT = str(BASE / "scripts" / "05_position_accuracy.py")
CLS_SCRIPT = str(BASE / "scripts" / "06_train_classifier.py")
BLOG_DATA = str(BASE / "data" / "processed" / "blog_corpus.parquet")

DATASETS = {
    "blog":        (BLOG_DATA, "0", "10000"),
    "hippocorpus": (str(BASE / "data" / "processed" / "hippocorpus.parquet"), "0", "0"),
    "ellipse":     (str(BASE / "data" / "processed" / "ellipse.parquet"), "0", "0"),
    "prism":       (str(BASE / "data" / "processed" / "prism.parquet"), "0", "0"),
    "synthpai":    (str(BASE / "data" / "processed" / "synthpai_corpus.parquet"), "0", "0"),
    "europarl":    (str(BASE / "data" / "processed" / "europarl_gender.parquet"), "10000", "5000"),
}

# All models ordered small→large. Load once, run all datasets, discard.
ALL_MODELS = [
    # Pythia
    ("EleutherAI/pythia-14m",  64),
    ("EleutherAI/pythia-31m",  64),
    ("EleutherAI/pythia-70m",  64),
    ("EleutherAI/pythia-160m", 32),
    ("EleutherAI/pythia-410m", 32),
    ("EleutherAI/pythia-1b",   16),
    # Qwen2.5
    ("Qwen/Qwen2.5-0.5B", 64),
    ("Qwen/Qwen2.5-1.5B", 32),
    ("Qwen/Qwen2.5-3B",   16),
    ("Qwen/Qwen2.5-7B",    8),
    # Qwen3
    ("Qwen/Qwen3-0.6B-Base", 64),
    ("Qwen/Qwen3-1.7B-Base", 32),
    ("Qwen/Qwen3-4B-Base",   16),
    ("Qwen/Qwen3-8B-Base",    8),
    # Gemma 3
    ("google/gemma-3-270m",    64),
    ("google/gemma-3-1b-pt",   32),
    ("google/gemma-3-4b-pt",   16),
    ("google/gemma-3-12b-pt",  16),
    # Llama 3
    ("meta-llama/Llama-3.2-1B", 32),
    ("meta-llama/Llama-3.2-3B", 16),
    ("meta-llama/Llama-3.1-8B", 16),
]


def model_short(m):
    return m.rstrip("/").split("/")[-1]


def result_exists(model, ds_name):
    ms = model_short(model)
    if ds_name == "blog":
        return (BASE / "results" / f"{ms}_per_layer_results.csv").exists()
    return (BASE / "results" / ds_name / f"{ms}_per_layer_results.csv").exists()


def position_exists(model):
    ms = model_short(model)
    return (BASE / "results" / f"{ms}_position_accuracy.csv").exists()


def run_probes(model, ds_name, data_path, max_train, max_test, batch_size):
    ms = model_short(model)
    if ds_name == "blog":
        out_dir = str(BASE / "results")
    else:
        out_dir = str(BASE / "results" / ds_name)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [sys.executable, SCRIPT,
           "--model_name", model,
           "--data_path", data_path,
           "--output_dir", out_dir,
           "--batch_size", str(batch_size),
           "--max_train_texts", max_train,
           "--max_test_texts", max_test]

    log_dir = BASE / "results" / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"{ds_name}_{ms}.log"

    print(f"    Running {ms} on {ds_name} ...", flush=True)
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=7200)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        log_content = log_file.read_text()
        lines = log_content.strip().split("\n")
        print(f"    FAIL ({elapsed:.0f}s, exit {proc.returncode})", flush=True)
        for line in lines[-5:]:
            print(f"      {line}", flush=True)
        return False

    print(f"    OK ({elapsed:.0f}s)", flush=True)
    return True


def run_position(model, batch_size):
    ms = model_short(model)
    if position_exists(model):
        return True

    cmd = [sys.executable, POS_SCRIPT,
           "--model_name", model,
           "--data_path", BLOG_DATA,
           "--output_dir", str(BASE / "results"),
           "--batch_size", str(batch_size)]

    log_dir = BASE / "results" / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / f"position_{ms}.log"

    print(f"    Position accuracy for {ms} ...", flush=True)
    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"    FAIL ({elapsed:.0f}s)", flush=True)
        return False
    print(f"    OK ({elapsed:.0f}s)", flush=True)
    return True


def clear_cache():
    """Clear HF model cache to free disk."""
    cache_dir = Path("/root/hf_cache/hub")
    if cache_dir.exists():
        for d in cache_dir.glob("models--*"):
            shutil.rmtree(d, ignore_errors=True)
    print("  [cache cleared]", flush=True)


def main():
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--skip-gaps", action="store_true")
    pa.add_argument("--skip-classifier", action="store_true")
    pa.add_argument("--skip-position", action="store_true")
    pa.add_argument("--models", nargs="*", help="Only run these models (short names)")
    args = pa.parse_args()

    # Verify data exists
    print("=== CHECKING DATA ===", flush=True)
    missing_data = []
    for ds_name, (data_path, _, _) in DATASETS.items():
        if Path(data_path).exists():
            print(f"  {ds_name}: OK", flush=True)
        else:
            print(f"  {ds_name}: MISSING ({data_path})", flush=True)
            missing_data.append(ds_name)

    if "blog" in missing_data:
        print("\nERROR: Blog corpus not found. Run 01_preprocess_data.py first.", flush=True)
        sys.exit(1)

    # Build work plan
    work = []  # (model, tasks_list)
    models_to_run = ALL_MODELS
    if args.models:
        models_to_run = [(m, bs) for m, bs in ALL_MODELS if model_short(m) in args.models]

    for model, default_bs in models_to_run:
        ms = model_short(model)
        tasks = []

        if not args.skip_gaps:
            for ds_name, (data_path, mt, mte) in DATASETS.items():
                if ds_name in missing_data:
                    continue
                if not result_exists(model, ds_name):
                    tasks.append(("probe", ds_name, data_path, mt, mte))

        if not args.skip_position and "blog" not in missing_data:
            if not position_exists(model):
                tasks.append(("position",))

        if tasks:
            work.append((model, default_bs, tasks))

    # Summary
    total_probe_runs = sum(1 for _, _, tasks in work for t in tasks if t[0] == "probe")
    total_position = sum(1 for _, _, tasks in work for t in tasks if t[0] == "position")
    print(f"\n=== WORK PLAN ===", flush=True)
    print(f"  Probe runs needed: {total_probe_runs}", flush=True)
    print(f"  Position accuracy: {total_position}", flush=True)
    print(f"  Models: {len(work)}", flush=True)

    for model, bs, tasks in work:
        ms = model_short(model)
        probe_tasks = [t for t in tasks if t[0] == "probe"]
        pos_tasks = [t for t in tasks if t[0] == "position"]
        ds_list = ", ".join(t[1] for t in probe_tasks)
        extras = []
        if pos_tasks: extras.append("position")
        extra_str = f" + {', '.join(extras)}" if extras else ""
        print(f"  {ms} (bs={bs}): {ds_list}{extra_str}", flush=True)

    # Phase 1: Gap-filling probe runs (load model once, run all datasets, discard)
    successes, failures = 0, 0
    probe_work = [(m, bs, [t for t in tasks if t[0] == "probe"]) for m, bs, tasks in work]
    probe_work = [(m, bs, tasks) for m, bs, tasks in probe_work if tasks]

    if probe_work:
        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 1: GAP-FILLING PROBES ({sum(len(t) for _,_,t in probe_work)} runs)", flush=True)
        print(f"{'='*60}", flush=True)

    for model, default_bs, tasks in probe_work:
        ms = model_short(model)
        print(f"\n  === {ms} ({time.strftime('%H:%M:%S')}) ===", flush=True)
        for task in tasks:
            _, ds_name, data_path, mt, mte = task
            ok = run_probes(model, ds_name, data_path, mt, mte, default_bs)
            if ok: successes += 1
            else: failures += 1
        clear_cache()

    # Phase 2: Classifier
    if not args.skip_classifier and Path(BLOG_DATA).exists():
        cls_result = BASE / "results" / "classifier_comparison.csv"
        if not cls_result.exists():
            print(f"\n{'='*60}", flush=True)
            print(f"PHASE 2: TRANSFORMER CLASSIFIER ({time.strftime('%H:%M:%S')})", flush=True)
            print(f"{'='*60}", flush=True)
            log_dir = BASE / "results" / "logs" / "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = log_dir / "classifier.log"
            with open(log_file, "w") as lf:
                proc = subprocess.run([sys.executable, CLS_SCRIPT, "--max_train", "20000"],
                                      stdout=lf, stderr=subprocess.STDOUT, timeout=7200)
            if proc.returncode == 0:
                print("  Classifier: OK", flush=True)
                successes += 1
            else:
                print("  Classifier: FAIL", flush=True)
                failures += 1
            clear_cache()

    # Phase 3: Position accuracy (load model, train Ridge from scratch, eval)
    pos_work = [(m, bs) for m, bs, tasks in work if any(t[0] == "position" for t in tasks)]
    if pos_work:
        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 3: POSITION ACCURACY ({len(pos_work)} models)", flush=True)
        print(f"{'='*60}", flush=True)

    for model, default_bs in pos_work:
        ms = model_short(model)
        print(f"\n  === {ms} ({time.strftime('%H:%M:%S')}) ===", flush=True)
        ok = run_position(model, default_bs)
        if ok: successes += 1
        else: failures += 1
        clear_cache()

    print(f"\n{'='*60}", flush=True)
    print(f"ALL DONE: {successes} succeeded, {failures} failed", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
