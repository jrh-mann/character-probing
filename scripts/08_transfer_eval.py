#!/usr/bin/env python3
"""Cross-dataset transfer: evaluate blog-trained Ridge probes on other datasets.

Loads saved Ridge Gram matrices (trained on blog corpus), solves Ridge,
then runs a forward pass on each target dataset's test set and evaluates.

Gender only (binary, same definition everywhere). Age bins differ across
datasets so transfer is not meaningful.

Output: results/transfer_eval.csv
"""

import argparse, gc, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(BASE / "hf_cache"))
NUM_WORKERS = min(8, os.cpu_count() or 1)
MAX_SEQ_LEN = 1024

GENDER_MAP = {"female": 0, "male": 1}
TASK_N_CLASSES = {"gender": 2}

TARGET_DATASETS = {
    "hippocorpus": BASE / "data" / "processed" / "hippocorpus.parquet",
    "ellipse":     BASE / "data" / "processed" / "ellipse.parquet",
    "prism":       BASE / "data" / "processed" / "prism.parquet",
    "europarl":    BASE / "data" / "processed" / "europarl_gender.parquet",
    "synthpai":    BASE / "data" / "processed" / "synthpai_corpus.parquet",
}

MODEL_PARAMS = {
    "pythia-14m": 0.014, "pythia-31m": 0.031, "pythia-70m": 0.070,
    "pythia-160m": 0.160, "pythia-410m": 0.410, "pythia-1b": 1.0,
    "Qwen2.5-0.5B": 0.5, "Qwen2.5-1.5B": 1.5, "Qwen2.5-3B": 3.0,
    "Qwen2.5-7B": 7.0, "Qwen2.5-14B": 14.0,
    "Qwen3-0.6B-Base": 0.6, "Qwen3-1.7B-Base": 1.7,
    "Qwen3-4B-Base": 4.0, "Qwen3-8B-Base": 8.0,
    "gemma-3-270m": 0.27, "gemma-3-1b-pt": 1.0,
    "gemma-3-4b-pt": 4.0, "gemma-3-12b-pt": 12.0,
    "Llama-3.2-1B": 1.2, "Llama-3.2-3B": 3.2, "Llama-3.1-8B": 8.0,
}

# Map full HF names to short names (for probe dir lookup)
MODEL_FULL_NAMES = {
    "pythia-14m": "EleutherAI/pythia-14m",
    "pythia-31m": "EleutherAI/pythia-31m",
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "Qwen2.5-14B": "Qwen/Qwen2.5-14B",
    "Qwen3-0.6B-Base": "Qwen/Qwen3-0.6B-Base",
    "Qwen3-1.7B-Base": "Qwen/Qwen3-1.7B-Base",
    "Qwen3-4B-Base": "Qwen/Qwen3-4B-Base",
    "Qwen3-8B-Base": "Qwen/Qwen3-8B-Base",
    "gemma-3-270m": "google/gemma-3-270m",
    "gemma-3-1b-pt": "google/gemma-3-1b-pt",
    "gemma-3-4b-pt": "google/gemma-3-4b-pt",
    "gemma-3-12b-pt": "google/gemma-3-12b-pt",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
}


class GenderDS(Dataset):
    def __init__(self, path, split="test"):
        df = pd.read_parquet(path)
        df = df[df["split"] == split].reset_index(drop=True)
        df = df[df["gender"].isin(["male", "female"])].reset_index(drop=True)
        self.texts = df["text"].tolist()
        self.y = torch.tensor(df["gender"].map(GENDER_MAP).values, dtype=torch.long)
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], self.y[i]


def make_collate(tok):
    def fn(batch):
        texts, labels = zip(*batch)
        enc = tok(list(texts), padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        return enc, torch.stack(labels)
    return fn


def short(s): return s.rstrip("/").split("/")[-1]


def _get_transformer_backbone(model):
    if hasattr(model, 'gpt_neox'): return model.gpt_neox
    inner = model.model
    if hasattr(inner, 'language_model'): return inner.language_model
    return inner


def get_inner_model(model):
    return _get_transformer_backbone(model)


def eval_layers_list(nl, stride=4):
    ls = list(range(0, nl+1, stride))
    if nl not in ls: ls.append(nl)
    return sorted(ls)


def auto_bs(model):
    n = sum(p.numel() for p in model.parameters()) / 1e9
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 24
    if vram_gb >= 38:
        return 128 if n < 2 else (64 if n < 5 else (32 if n < 10 else 16))
    return 64 if n < 2 else (32 if n < 5 else (16 if n < 10 else 8))


class ActivationCapture:
    def __init__(self, model, eval_layers, n_layers):
        self.captured = {}
        self.hooks = []
        inner = _get_transformer_backbone(model)
        embed = getattr(inner, 'embed_tokens', None) or getattr(inner, 'embed_in', None)
        norm = getattr(inner, 'norm', None) or getattr(inner, 'final_layer_norm', None)
        layers = getattr(inner, 'layers', None)
        for l in eval_layers:
            if l == 0 and embed is not None:
                self.hooks.append(embed.register_forward_hook(self._make_hook(l)))
            elif l == n_layers and norm is not None:
                self.hooks.append(norm.register_forward_hook(self._make_hook(l)))
            elif layers is not None and 0 < l <= n_layers:
                self.hooks.append(layers[l-1].register_forward_hook(self._make_hook(l)))

    def _make_hook(self, layer_idx):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self.captured[layer_idx] = t.detach()
        return hook

    def clear(self): self.captured.clear()
    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks.clear()


def solve_ridge(state_dict, lam=1.0):
    """Solve Ridge for gender from saved Gram matrix."""
    A = state_dict["A"].double()
    sx = state_dict["sx"].double()
    sx2 = state_dict["sx2"].double()
    n = state_dict["n"]
    B = state_dict["B_gender"].double()
    cc = state_dict["cc_gender"].double()
    D = A.shape[0]

    mean = sx / n
    var = (sx2 / n - mean**2).clamp(min=1e-8)
    std = var.sqrt()
    inv = 1.0 / std

    A_c = A - n * mean.unsqueeze(1) @ mean.unsqueeze(0)
    A_z = A_c * inv.unsqueeze(1) * inv.unsqueeze(0)
    mean_y = cc / n
    B_c = B - n * mean.unsqueeze(1) @ mean_y.unsqueeze(0)
    B_z = B_c * inv.unsqueeze(1)

    W = torch.linalg.solve(A_z / n + lam * torch.eye(D, dtype=torch.float64), B_z / n)
    return W.float(), mean_y.float(), mean.float(), std.float()


def find_best_lambda(model_short_name):
    """Find best ridge lambda for gender from blog val results."""
    val_path = BASE / "results" / f"{model_short_name}_val_results.csv"
    if not val_path.exists():
        return 1.0
    vdf = pd.read_csv(val_path)
    ridge = vdf[(vdf["task"] == "gender") & (vdf["strategy"].str.startswith("ridge_"))]
    if ridge.empty:
        return 1.0
    best = ridge.loc[ridge["text_balanced_acc"].idxmax()]
    return float(best["strategy"].replace("ridge_", ""))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_name", required=True, help="Short name e.g. pythia-14m")
    pa.add_argument("--batch_size", type=int, default=0)
    pa.add_argument("--max_test_texts", type=int, default=5000)
    args = pa.parse_args()

    ms = args.model_name
    full_name = MODEL_FULL_NAMES.get(ms)
    if full_name is None:
        sys.exit(f"Unknown model: {ms}")

    probe_dir = BASE / "probes" / ms
    if not probe_dir.exists():
        sys.exit(f"No probes for {ms}")

    # Find best lambda from blog val
    best_lam = find_best_lambda(ms)
    print(f"Model: {ms}, best lambda: {best_lam}")

    # Load all ridge probes and solve
    ridge_files = sorted(probe_dir.glob("L*_ridge.pt"))
    if not ridge_files:
        sys.exit(f"No ridge probes in {probe_dir}")

    ridge_solutions = {}  # layer -> (W, bias, mean, std)
    for rf in ridge_files:
        layer = int(rf.stem.split("_")[0][1:])
        sd = torch.load(rf, map_location="cpu", weights_only=True)
        if "B_gender" not in sd:
            continue
        W, bias, mean, std = solve_ridge(sd, lam=best_lam)
        ridge_solutions[layer] = (W, bias, mean, std)
    print(f"  Solved {len(ridge_solutions)} layers")

    # Load model
    print(f"Loading {full_name} ...")
    tok = AutoTokenizer.from_pretrained(full_name, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(full_name, dtype=torch.bfloat16,
                                                  device_map="auto", trust_remote_code=True)
    model.eval()
    cfg = model.config
    if hasattr(cfg, 'text_config'): cfg = cfg.text_config
    nl = cfg.num_hidden_layers
    elvs = sorted(ridge_solutions.keys())
    bs = args.batch_size if args.batch_size > 0 else auto_bs(model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Move solutions to device
    for l in ridge_solutions:
        W, bias, mean, std = ridge_solutions[l]
        ridge_solutions[l] = (W.to(dev), bias.to(dev), mean.to(dev), std.to(dev))

    rows = []

    # Also evaluate on blog test (as baseline)
    all_datasets = {"blog": BASE / "data" / "processed" / "blog_corpus.parquet"}
    all_datasets.update(TARGET_DATASETS)

    for ds_name, ds_path in all_datasets.items():
        if not ds_path.exists():
            print(f"  {ds_name}: MISSING, skipping")
            continue

        tst = GenderDS(ds_path, "test")
        if len(tst) == 0:
            print(f"  {ds_name}: no test data")
            continue
        if args.max_test_texts > 0 and len(tst) > args.max_test_texts:
            torch.manual_seed(42)
            tst = Subset(tst, torch.randperm(len(tst))[:args.max_test_texts].tolist())

        loader = DataLoader(tst, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS,
                           pin_memory=True, collate_fn=make_collate(tok), persistent_workers=True)

        capture = ActivationCapture(model, elvs, nl)

        # Accumulate predictions per layer
        layer_preds = {l: [] for l in elvs}
        all_labels = []

        for enc, labels in tqdm(loader, desc=f"  {ds_name}", leave=False):
            ids = enc["input_ids"].to(dev)
            amask = enc["attention_mask"].to(dev)
            try:
                capture.clear()
                with torch.no_grad():
                    get_inner_model(model)(input_ids=ids, attention_mask=amask)
            except torch.cuda.OutOfMemoryError:
                capture.clear(); torch.cuda.empty_cache(); gc.collect(); continue

            B, S = ids.shape
            bi, ti = torch.where(amask)
            if bi.shape[0] == 0:
                capture.clear(); continue

            text_counts = torch.zeros(B, 1, device=dev)
            text_counts.scatter_add_(0, bi.unsqueeze(1), torch.ones(bi.shape[0], 1, device=dev))

            for l in elvs:
                hs = capture.captured.get(l)
                if hs is None: continue
                acts = hs[bi, ti].float()

                W, bias, mean, std = ridge_solutions[l]
                acts_z = (acts - mean) / (std + 1e-8)
                tok_logits = acts_z @ W + bias

                text_logits = torch.zeros(B, 2, device=dev)
                text_logits.scatter_add_(0, bi.unsqueeze(1).expand_as(tok_logits), tok_logits)
                text_logits = text_logits / text_counts.clamp(min=1)

                layer_preds[l].extend(text_logits.argmax(-1).cpu().tolist())

            all_labels.extend(labels.tolist())
            capture.clear()

        capture.remove()

        for l in elvs:
            if not layer_preds[l]: continue
            acc = balanced_accuracy_score(all_labels, layer_preds[l])
            rows.append({
                "model": ms, "n_params": MODEL_PARAMS[ms],
                "source": "blog", "target": ds_name,
                "layer": l, "task": "gender",
                "balanced_acc": round(acc, 5),
                "n_test": len(all_labels),
            })

        best_acc = max(rows[-len(elvs):], key=lambda r: r["balanced_acc"])["balanced_acc"] if rows else 0
        print(f"  {ds_name}: best={best_acc:.3f} (n={len(all_labels)})")

    del model; torch.cuda.empty_cache(); gc.collect()

    # Save
    df = pd.DataFrame(rows)
    out = BASE / "results" / "transfer_eval.csv"
    if out.exists():
        existing = pd.read_csv(out)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["model", "target", "layer"], keep="last")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
