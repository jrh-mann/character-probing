#!/usr/bin/env python3
"""Evaluate probe accuracy as a function of position in text.

For each model with trained probes, loads the model, runs forward pass on test
texts, and computes cumulative Ridge probe accuracy at different position
thresholds. Shows how much of the text a model needs to "see" to make
an accurate prediction.

Output: {model_short}_position_accuracy.csv with columns:
  model_name, task, layer, position_pct, accuracy, n_texts
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
STAR_SIGNS_SORTED = sorted(["Aquarius","Aries","Cancer","Capricorn","Gemini","Leo",
                             "Libra","Pisces","Sagittarius","Scorpio","Taurus","Virgo"])
STAR_SIGN_MAP = {n: i for i, n in enumerate(STAR_SIGNS_SORTED)}
TASK_N_CLASSES = {"age_bin": 3, "gender": 2, "star_sign": 12}
TASKS = ["age_bin", "gender"]  # skip star_sign for position analysis (chance-level)

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
    if hasattr(model, 'gpt_neox'):
        return model.gpt_neox
    inner = model.model
    if hasattr(inner, 'language_model'):
        return inner.language_model
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
    if vram_gb >= 75:
        bs = 256 if n < 5 else (128 if n < 15 else 32)
    elif vram_gb >= 38:
        bs = 128 if n < 2 else (64 if n < 5 else (32 if n < 10 else 16))
    else:
        bs = 64 if n < 2 else (32 if n < 5 else (16 if n < 10 else 8))
    return bs


class ActivationCapture:
    def __init__(self, model, eval_layers, n_layers):
        self.captured = {}
        self.hooks = []
        self.eval_layers = eval_layers
        inner = _get_transformer_backbone(model)
        embed = getattr(inner, 'embed_tokens', None) or getattr(inner, 'embed_in', None)
        norm = getattr(inner, 'norm', None) or getattr(inner, 'final_layer_norm', None)
        layers = getattr(inner, 'layers', None) or getattr(inner, 'layers', None)
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

    def clear(self):
        self.captured.clear()

    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks.clear()


class Ridge:
    """Minimal Ridge class — just enough to load state_dict and solve."""
    def __init__(self, D, dev="cuda"):
        self.D, self.dev = D, dev
        self.A = torch.zeros(D, D, device=dev)
        self.B = {t: torch.zeros(D, TASK_N_CLASSES[t], device=dev) for t in TASKS}
        self.class_counts = {t: torch.zeros(TASK_N_CLASSES[t], device=dev) for t in TASKS}
        self.sx = torch.zeros(D, device=dev)
        self.sx2 = torch.zeros(D, device=dev)
        self.n = 0

    def load_state_dict(self, d):
        self.A = d["A"].to(self.dev)
        self.sx = d["sx"].to(self.dev)
        self.sx2 = d["sx2"].to(self.dev)
        self.n = d["n"]
        for t in TASKS:
            if f"B_{t}" in d:
                self.B[t] = d[f"B_{t}"].to(self.dev)
            if f"cc_{t}" in d:
                self.class_counts[t] = d[f"cc_{t}"].to(self.dev)

    def solve(self, lam=1.0, task="age_bin"):
        n = max(self.n, 1)
        sx = self.sx.double(); sx2 = self.sx2.double()
        A = self.A.double(); Bt = self.B[task].double()
        cc = self.class_counts[task].double()
        mean = sx / n
        var = (sx2 / n - mean**2).clamp(min=1e-8); std = var.sqrt(); inv = 1.0/std
        A_c = A - n * mean.unsqueeze(1) @ mean.unsqueeze(0)
        A_z = A_c * inv.unsqueeze(1) * inv.unsqueeze(0)
        mean_y = cc / n
        B_c = Bt - n * mean.unsqueeze(1) @ mean_y.unsqueeze(0)
        B_z = B_c * inv.unsqueeze(1)
        W_z = torch.linalg.solve(A_z / n + lam * torch.eye(self.D, device=self.dev, dtype=torch.float64), B_z / n)
        bias = mean_y
        return W_z.float(), bias.float(), mean.float(), std.float()


def find_best_layer_and_lambda(model_short, output_dir):
    """Find best layer and ridge lambda from val results."""
    val_path = Path(output_dir) / f"{model_short}_val_results.csv"
    if not val_path.exists():
        return None
    vdf = pd.read_csv(val_path)
    best = {}
    for t in TASKS:
        ridge_rows = vdf[(vdf["task"] == t) & (vdf["strategy"].str.startswith("ridge_"))]
        if len(ridge_rows) == 0:
            continue
        best_row = ridge_rows.loc[ridge_rows["text_balanced_acc"].idxmax()]
        lam_str = best_row["strategy"].replace("ridge_", "")
        best[t] = {"layer": int(best_row["layer"]), "lambda": float(lam_str)}
    return best


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_name", required=True)
    pa.add_argument("--data_path", default=str(BASE_DIR / "data" / "processed" / "blog_corpus.parquet"))
    pa.add_argument("--output_dir", default=str(BASE_DIR / "results"))
    pa.add_argument("--max_test_texts", type=int, default=10000)
    pa.add_argument("--batch_size", type=int, default=0)
    pa.add_argument("--eval_layer_stride", type=int, default=4)
    args = pa.parse_args()

    ms = short(args.model_name)
    od = Path(args.output_dir)
    probe_dir = BASE_DIR / "probes" / ms

    out_file = od / f"{ms}_position_accuracy.csv"
    if out_file.exists():
        print(f"Already exists: {out_file}")
        return

    if not Path(args.data_path).exists():
        sys.exit(f"Data not found: {args.data_path}")

    # Find best ridge lambdas from val results
    best_config = find_best_layer_and_lambda(ms, od)
    if best_config is None:
        print(f"No val results for {ms}, will use lambda=1.0 for all layers")

    # Load ridge probe state dicts
    ridge_files = sorted(probe_dir.glob("L*_ridge.pt"))
    if not ridge_files:
        sys.exit(f"No ridge probes found in {probe_dir}")

    print(f"Loading {args.model_name} ...")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16,
                                                  device_map="auto", trust_remote_code=True)
    model.eval()
    cfg = model.config
    if hasattr(cfg, 'text_config'):
        cfg = cfg.text_config
    hdim = cfg.hidden_size
    nl = cfg.num_hidden_layers
    elvs = eval_layers_list(nl, args.eval_layer_stride)
    bs = args.batch_size if args.batch_size > 0 else auto_bs(model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  hdim={hdim}, layers={nl}, bs={bs}, eval_layers={elvs}")

    # Load ridge probes for all eval layers
    ridges = {}
    for rf in ridge_files:
        layer_idx = int(rf.stem.split("_")[0][1:])  # L4_ridge.pt -> 4
        if layer_idx not in elvs:
            continue
        r = Ridge(hdim, dev)
        r.load_state_dict(torch.load(rf, map_location="cpu", weights_only=True))
        ridges[layer_idx] = r
    print(f"  Loaded {len(ridges)} ridge probes")

    # Solve ridge for each layer × task
    ridge_sols = {}
    for l in ridges:
        for t in TASKS:
            if best_config and t in best_config:
                lam = best_config[t]["lambda"]
            else:
                lam = 1.0
            W, bias, mean, std = ridges[l].solve(lam=lam, task=t)
            ridge_sols[(l, t)] = (W.to(dev), bias.to(dev), mean.to(dev), std.to(dev))
    print(f"  Solved {len(ridge_sols)} ridge solutions")
    del ridges; gc.collect()

    # Load test data
    tst = BlogDS(args.data_path, "test")
    if args.max_test_texts > 0 and len(tst) > args.max_test_texts:
        torch.manual_seed(42)
        tst = Subset(tst, torch.randperm(len(tst))[:args.max_test_texts].tolist())
    print(f"  Test texts: {len(tst)}")

    col = make_collate(tok)
    loader = DataLoader(tst, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=True, collate_fn=col, persistent_workers=True)

    # For each position bin, accumulate predictions
    # Structure: results[layer][task][pct_bin] = {"preds": [], "labels": []}
    results = {l: {t: {p: {"preds": [], "labels": []} for p in POSITION_BINS}
                   for t in TASKS}
               for l in elvs if any((l, t) in ridge_sols for t in TASKS)}

    capture = ActivationCapture(model, elvs, nl)

    print(f"\nEvaluating position accuracy ...")
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
        lengths = amask.sum(1)  # (B,) — number of real tokens per text
        batch_labels = {t: labs[t].to(dev) for t in TASKS}

        for l in results:
            hs = capture.captured.get(l)
            if hs is None:
                continue
            hs = hs.float()  # (B, S, D)

            for t in TASKS:
                if (l, t) not in ridge_sols:
                    continue
                W, bias, mean, std = ridge_sols[(l, t)]

                # Per-token logits: (B, S, C)
                hs_z = (hs - mean.unsqueeze(0).unsqueeze(0)) / (std.unsqueeze(0).unsqueeze(0) + 1e-8)
                tok_logits = hs_z @ W + bias.unsqueeze(0).unsqueeze(0)  # (B, S, C)

                # For each position bin, compute cumulative mean logit and predict
                for pct in POSITION_BINS:
                    # For each text, take tokens up to position pct * length
                    cutoff = (lengths.float() * pct).long().clamp(min=1)  # (B,)

                    # Create mask: (B, S) where position < cutoff
                    pos_range = torch.arange(S, device=dev).unsqueeze(0).expand(B, S)  # (B, S)
                    pos_mask = (pos_range < cutoff.unsqueeze(1)) & amask.bool()  # (B, S)

                    # Cumulative mean logits per text
                    mask_expanded = pos_mask.unsqueeze(-1).float()  # (B, S, 1)
                    sum_logits = (tok_logits * mask_expanded).sum(1)  # (B, C)
                    count = mask_expanded.sum(1).clamp(min=1)  # (B, 1)
                    mean_logits = sum_logits / count  # (B, C)

                    preds = mean_logits.argmax(-1)  # (B,)
                    results[l][t][pct]["preds"].extend(preds.cpu().tolist())
                    results[l][t][pct]["labels"].extend(batch_labels[t].cpu().tolist())

        capture.clear()
        if bn % 200 == 0:
            gc.collect()

    capture.remove()

    # Compute balanced accuracy for each bin
    rows = []
    for l in results:
        for t in results[l]:
            for pct in POSITION_BINS:
                d = results[l][t][pct]
                if not d["labels"]:
                    continue
                acc = balanced_accuracy_score(d["labels"], d["preds"])
                rows.append({
                    "model_name": args.model_name,
                    "task": t,
                    "layer": l,
                    "position_pct": pct,
                    "accuracy": round(acc, 5),
                    "n_texts": len(d["labels"]),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print(f"\nSaved {out_file} ({len(df)} rows)")

    # Print summary: best layer per task
    for t in TASKS:
        tdf = df[df["task"] == t]
        if len(tdf) == 0:
            continue
        full = tdf[tdf["position_pct"] == 1.0]
        if len(full) > 0:
            best = full.loc[full["accuracy"].idxmax()]
            early = tdf[(tdf["layer"] == best["layer"]) & (tdf["position_pct"] == 0.1)]
            early_acc = early["accuracy"].values[0] if len(early) > 0 else 0
            print(f"  {t}: best layer L{int(best['layer'])} — "
                  f"10%={early_acc:.3f}, 100%={best['accuracy']:.3f}")

    del model; torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    main()
