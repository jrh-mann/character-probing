#!/usr/bin/env python3
"""Evaluate probe accuracy as a function of position in text.

Loads saved Ridge probes (trained on blog corpus), runs ONE forward pass
on the test set, and computes cumulative accuracy at position thresholds.

No training — just loads probes and evaluates.

Output: {model_short}_position_accuracy.csv
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

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(BASE_DIR / "hf_cache"))
NUM_WORKERS = min(8, os.cpu_count() or 1)
MAX_SEQ_LEN = 1024

AGE_BIN_MAP = {1: 0, 2: 1, 3: 2}
GENDER_MAP = {"female": 0, "male": 1}
TASK_N_CLASSES = {"age_bin": 3, "gender": 2}
TASKS = ["age_bin", "gender"]

POSITION_BINS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class BlogDS(Dataset):
    def __init__(self, path, split):
        df = pd.read_parquet(path)
        df = df[df["split"] == split].reset_index(drop=True)
        self.texts = df["text"].tolist()
        self.y_age = torch.tensor(df["age_bin"].map(AGE_BIN_MAP).fillna(0).astype(int).values, dtype=torch.long)
        self.y_gen = torch.tensor(df["gender"].map(GENDER_MAP).fillna(0).astype(int).values, dtype=torch.long)
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], self.y_age[i], self.y_gen[i]


def make_collate(tok):
    def fn(batch):
        texts, a, g = zip(*batch)
        enc = tok(list(texts), padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        return enc, {"age_bin": torch.stack(a), "gender": torch.stack(g)}
    return fn


def short(s): return s.rstrip("/").split("/")[-1]


def _get_transformer_backbone(model):
    if hasattr(model, 'gpt_neox'): return model.gpt_neox
    inner = model.model
    if hasattr(inner, 'language_model'): return inner.language_model
    return inner


def get_inner_model(model): return _get_transformer_backbone(model)


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


def solve_ridge(state_dict, lam, task):
    """Solve Ridge from saved Gram matrix."""
    A = state_dict["A"].double()
    sx = state_dict["sx"].double()
    sx2 = state_dict["sx2"].double()
    n = state_dict["n"]
    B = state_dict[f"B_{task}"].double()
    cc = state_dict[f"cc_{task}"].double()
    D = A.shape[0]

    mean = sx / n
    var = (sx2 / n - mean**2).clamp(min=1e-8); std = var.sqrt(); inv = 1.0 / std
    A_c = A - n * mean.unsqueeze(1) @ mean.unsqueeze(0)
    A_z = A_c * inv.unsqueeze(1) * inv.unsqueeze(0)
    mean_y = cc / n
    B_c = B - n * mean.unsqueeze(1) @ mean_y.unsqueeze(0)
    B_z = B_c * inv.unsqueeze(1)
    W = torch.linalg.solve(A_z / n + lam * torch.eye(D, dtype=torch.float64), B_z / n)
    return W.float(), mean_y.float(), mean.float(), std.float()


def find_best_lambda(ms, task):
    """Find best ridge lambda from blog val results."""
    val_path = BASE_DIR / "results" / f"{ms}_val_results.csv"
    if not val_path.exists():
        return 1.0
    vdf = pd.read_csv(val_path)
    ridge = vdf[(vdf["task"] == task) & (vdf["strategy"].str.startswith("ridge_"))]
    if ridge.empty:
        return 1.0
    best = ridge.loc[ridge["text_balanced_acc"].idxmax()]
    return float(best["strategy"].replace("ridge_", ""))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_name", required=True)
    pa.add_argument("--data_path", default=str(BASE_DIR / "data" / "processed" / "blog_corpus.parquet"))
    pa.add_argument("--output_dir", default=str(BASE_DIR / "results"))
    pa.add_argument("--max_test_texts", type=int, default=10000)
    pa.add_argument("--batch_size", type=int, default=0)
    args = pa.parse_args()

    ms = short(args.model_name)
    od = Path(args.output_dir)
    probe_dir = BASE_DIR / "probes" / ms
    out_file = od / f"{ms}_position_accuracy.csv"

    if out_file.exists():
        print(f"Already exists: {out_file}")
        return

    if not probe_dir.exists():
        sys.exit(f"No probes for {ms} at {probe_dir}")

    # Load and solve Ridge probes from saved Gram matrices
    ridge_files = sorted(probe_dir.glob("L*_ridge.pt"))
    if not ridge_files:
        sys.exit(f"No ridge probes in {probe_dir}")

    print(f"Loading {args.model_name} ...")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16,
                                                  device_map="auto", trust_remote_code=True)
    model.eval()
    cfg = model.config
    if hasattr(cfg, 'text_config'): cfg = cfg.text_config
    hdim = cfg.hidden_size; nl = cfg.num_hidden_layers
    bs = args.batch_size if args.batch_size > 0 else auto_bs(model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Solve Ridge for each layer × task
    ridge_sols = {}  # (layer, task) -> (W, bias, mean, std)
    elvs = []
    for rf in ridge_files:
        layer = int(rf.stem.split("_")[0][1:])
        elvs.append(layer)
        sd = torch.load(rf, map_location="cpu", weights_only=True)
        for t in TASKS:
            if f"B_{t}" not in sd:
                continue
            lam = find_best_lambda(ms, t)
            W, bias, mean, std = solve_ridge(sd, lam, t)
            ridge_sols[(layer, t)] = (W.to(dev), bias.to(dev), mean.to(dev), std.to(dev))
    elvs = sorted(set(elvs))
    print(f"  {len(ridge_sols)} probe solutions across {len(elvs)} layers, bs={bs}")

    # Load test data
    tst = BlogDS(args.data_path, "test")
    if args.max_test_texts > 0 and len(tst) > args.max_test_texts:
        torch.manual_seed(42)
        tst = Subset(tst, torch.randperm(len(tst))[:args.max_test_texts].tolist())
    print(f"  Test texts: {len(tst)}")

    col = make_collate(tok)
    loader = DataLoader(tst, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True, collate_fn=col, persistent_workers=True)

    # Accumulate predictions per (layer, task, position_bin)
    results = {l: {t: {p: {"preds": [], "labels": []} for p in POSITION_BINS}
                   for t in TASKS}
               for l in elvs}

    capture = ActivationCapture(model, elvs, nl)

    t0 = time.time()
    for bn, (enc, labs) in enumerate(tqdm(loader, desc="Position eval")):
        ids = enc["input_ids"].to(dev)
        amask = enc["attention_mask"].to(dev)
        try:
            capture.clear()
            with torch.no_grad():
                get_inner_model(model)(input_ids=ids, attention_mask=amask)
        except torch.cuda.OutOfMemoryError:
            capture.clear(); torch.cuda.empty_cache(); gc.collect(); continue

        B, S = ids.shape
        lengths = amask.sum(1)
        batch_labels = {t: labs[t].to(dev) for t in TASKS}

        for l in elvs:
            hs = capture.captured.get(l)
            if hs is None: continue
            hs = hs.float()

            for t in TASKS:
                if (l, t) not in ridge_sols: continue
                W, bias, mean, std = ridge_sols[(l, t)]

                # Per-token logits
                hs_z = (hs - mean) / (std + 1e-8)
                tok_logits = hs_z @ W + bias  # (B, S, C)

                for pct in POSITION_BINS:
                    cutoff = (lengths.float() * pct).long().clamp(min=1)
                    pos_range = torch.arange(S, device=dev).unsqueeze(0)
                    pos_mask = (pos_range < cutoff.unsqueeze(1)) & amask.bool()

                    mask_f = pos_mask.unsqueeze(-1).float()
                    sum_logits = (tok_logits * mask_f).sum(1)
                    count = mask_f.sum(1).clamp(min=1)
                    mean_logits = sum_logits / count

                    preds = mean_logits.argmax(-1)
                    results[l][t][pct]["preds"].extend(preds.cpu().tolist())
                    results[l][t][pct]["labels"].extend(batch_labels[t].cpu().tolist())

        capture.clear()
        if bn % 200 == 0: gc.collect()

    capture.remove()
    print(f"  Eval: {time.time()-t0:.0f}s")

    # Compute balanced accuracy
    rows = []
    for l in results:
        for t in results[l]:
            for pct in POSITION_BINS:
                d = results[l][t][pct]
                if not d["labels"]: continue
                acc = balanced_accuracy_score(d["labels"], d["preds"])
                rows.append({
                    "model_name": args.model_name, "task": t, "layer": l,
                    "position_pct": pct, "accuracy": round(acc, 5),
                    "n_texts": len(d["labels"]),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print(f"Saved {out_file} ({len(df)} rows)")

    # Summary
    for t in TASKS:
        tdf = df[df["task"] == t]
        if tdf.empty: continue
        full = tdf[tdf["position_pct"] == 1.0]
        if len(full) > 0:
            best = full.loc[full["accuracy"].idxmax()]
            early = tdf[(tdf["layer"] == best["layer"]) & (tdf["position_pct"] == 0.1)]
            early_acc = early["accuracy"].values[0] if len(early) > 0 else 0
            print(f"  {t}: best L{int(best['layer'])} — 10%={early_acc:.3f}, 100%={best['accuracy']:.3f}")

    del model; torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    main()
