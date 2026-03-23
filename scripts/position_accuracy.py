#!/usr/bin/env python3
"""Evaluate probe accuracy as a function of position in text.

Self-contained: trains Ridge probe from scratch, then evaluates position accuracy.
No need to upload saved probes — just needs model + data.

Pass 1: Forward pass on training data → accumulate Ridge Gram matrices
Pass 2: Forward pass on test data → compute cumulative accuracy at position bins

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


class Ridge:
    """Incremental Gram matrix on GPU. Solve post-hoc in float64."""
    def __init__(self, D, dev="cuda"):
        self.D, self.dev = D, dev
        self.A = torch.zeros(D, D, device=dev)
        self.B = {t: torch.zeros(D, TASK_N_CLASSES[t], device=dev) for t in TASKS}
        self.sx = torch.zeros(D, device=dev)
        self.sx2 = torch.zeros(D, device=dev)
        self.n = 0

    @torch.no_grad()
    def update(self, x, labels):
        self.A.addmm_(x.T, x)
        self.sx += x.sum(0)
        self.sx2 += (x**2).sum(0)
        self.n += x.shape[0]
        for t in TASKS:
            nc = TASK_N_CLASSES[t]
            oh = torch.zeros(x.shape[0], nc, device=self.dev)
            oh.scatter_(1, labels[t].unsqueeze(1), 1.0)
            self.B[t].addmm_(x.T, oh)

    def solve(self, lam=1.0, task="age_bin"):
        n = max(self.n, 1)
        sx = self.sx.double(); sx2 = self.sx2.double()
        A = self.A.double(); Bt = self.B[task].double()
        mean = sx / n
        var = (sx2 / n - mean**2).clamp(min=1e-8); std = var.sqrt(); inv = 1.0/std
        A_c = A - n * mean.unsqueeze(1) @ mean.unsqueeze(0)
        A_z = A_c * inv.unsqueeze(1) * inv.unsqueeze(0)
        # class priors as bias
        n_per_class = torch.zeros(TASK_N_CLASSES[task], device=self.dev, dtype=torch.float64)
        for t_idx in range(TASK_N_CLASSES[task]):
            n_per_class[t_idx] = (Bt[:, t_idx].sum()) if False else 0  # handled below
        # B centering
        cc = Bt.sum(0)  # class counts
        mean_y = cc / n
        B_c = Bt - n * mean.unsqueeze(1) @ mean_y.unsqueeze(0)
        B_z = B_c * inv.unsqueeze(1)
        W_z = torch.linalg.solve(A_z / n + lam * torch.eye(self.D, device=self.dev, dtype=torch.float64), B_z / n)
        return W_z.float(), mean_y.float(), mean.float(), std.float()


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_name", required=True)
    pa.add_argument("--data_path", default=str(BASE_DIR / "data" / "processed" / "blog_corpus.parquet"))
    pa.add_argument("--output_dir", default=str(BASE_DIR / "results"))
    pa.add_argument("--max_train_texts", type=int, default=0)
    pa.add_argument("--max_test_texts", type=int, default=10000)
    pa.add_argument("--batch_size", type=int, default=0)
    pa.add_argument("--eval_layer_stride", type=int, default=4)
    pa.add_argument("--ridge_lambda", type=float, default=1.0)
    args = pa.parse_args()

    ms = short(args.model_name)
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    out_file = od / f"{ms}_position_accuracy.csv"

    if out_file.exists():
        print(f"Already exists: {out_file}")
        return

    if not Path(args.data_path).exists():
        sys.exit(f"Data not found: {args.data_path}")

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

    col = make_collate(tok)

    # ── PASS 1: Train Ridge probes ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PASS 1: Training Ridge probes")
    print(f"{'='*60}")

    trn = BlogDS(args.data_path, "train")
    if args.max_train_texts > 0 and len(trn) > args.max_train_texts:
        torch.manual_seed(42)
        trn = Subset(trn, torch.randperm(len(trn))[:args.max_train_texts].tolist())
    print(f"  Training on {len(trn)} texts")

    train_loader = DataLoader(trn, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, collate_fn=col, persistent_workers=True)

    ridges = {l: Ridge(hdim, dev) for l in elvs}
    capture = ActivationCapture(model, elvs, nl)

    t0 = time.time()
    for bn, (enc, labs) in enumerate(tqdm(train_loader, desc="Train Ridge")):
        ids = enc["input_ids"].to(dev)
        amask = enc["attention_mask"].to(dev)
        try:
            capture.clear()
            with torch.no_grad():
                get_inner_model(model)(input_ids=ids, attention_mask=amask)

            bi, ti = torch.where(amask)
            if bi.shape[0] == 0:
                capture.clear(); continue

            batch_labels = {t: labs[t].to(dev) for t in TASKS}
            tlabs = {t: batch_labels[t][bi] for t in TASKS}

            for l in elvs:
                hs = capture.captured.get(l)
                if hs is None: continue
                acts = hs[bi, ti].float()
                ridges[l].update(acts, tlabs)
            capture.clear()
        except torch.cuda.OutOfMemoryError:
            capture.clear(); torch.cuda.empty_cache(); gc.collect()
            continue
        if bn % 500 == 0: gc.collect()

    capture.remove()
    print(f"  Training: {time.time()-t0:.0f}s")

    # Solve
    print("  Solving Ridge ...")
    lam = args.ridge_lambda
    ridge_sols = {}
    for l in elvs:
        for t in TASKS:
            W, bias, mean, std = ridges[l].solve(lam=lam, task=t)
            ridge_sols[(l, t)] = (W.to(dev), bias.to(dev), mean.to(dev), std.to(dev))
    del ridges; gc.collect()
    print(f"  Solved {len(ridge_sols)} (layer, task) pairs")

    # ── PASS 2: Position accuracy on test set ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"PASS 2: Position accuracy evaluation")
    print(f"{'='*60}")

    tst = BlogDS(args.data_path, "test")
    if args.max_test_texts > 0 and len(tst) > args.max_test_texts:
        torch.manual_seed(42)
        tst = Subset(tst, torch.randperm(len(tst))[:args.max_test_texts].tolist())
    print(f"  Evaluating on {len(tst)} texts")

    test_loader = DataLoader(tst, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True, collate_fn=col, persistent_workers=True)

    # Accumulate predictions per (layer, task, position_bin)
    results = {l: {t: {p: {"preds": [], "labels": []} for p in POSITION_BINS}
                   for t in TASKS}
               for l in elvs}

    capture = ActivationCapture(model, elvs, nl)

    t0 = time.time()
    for bn, (enc, labs) in enumerate(tqdm(test_loader, desc="Position eval")):
        ids = enc["input_ids"].to(dev)
        amask = enc["attention_mask"].to(dev)
        try:
            capture.clear()
            with torch.no_grad():
                get_inner_model(model)(input_ids=ids, attention_mask=amask)
        except torch.cuda.OutOfMemoryError:
            capture.clear(); torch.cuda.empty_cache(); gc.collect(); continue

        B, S = ids.shape
        lengths = amask.sum(1)  # (B,)
        batch_labels = {t: labs[t].to(dev) for t in TASKS}

        for l in elvs:
            hs = capture.captured.get(l)
            if hs is None: continue
            hs = hs.float()  # (B, S, D)

            for t in TASKS:
                if (l, t) not in ridge_sols: continue
                W, bias, mean, std = ridge_sols[(l, t)]

                # Per-token logits: (B, S, C)
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
    print(f"  Evaluation: {time.time()-t0:.0f}s")

    # Compute balanced accuracy
    rows = []
    for l in results:
        for t in results[l]:
            for pct in POSITION_BINS:
                d = results[l][t][pct]
                if not d["labels"]: continue
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

    # Summary
    for t in TASKS:
        tdf = df[df["task"] == t]
        if len(tdf) == 0: continue
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
