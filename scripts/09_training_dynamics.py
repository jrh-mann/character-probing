#!/usr/bin/env python3
"""Training dynamics: probe accuracy across Pythia training checkpoints.

For each Pythia model size × checkpoint, trains Ridge probes via Gram matrix
accumulation (single forward pass) and evaluates on the test set.

Produces a 2D surface: training steps × parameter count → probe accuracy.

Output: results/training_dynamics.csv
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

# Log-spaced checkpoints covering full Pythia training (143K steps)
CHECKPOINTS = [0, 1, 2, 4, 8, 16, 64, 256, 1000, 2000, 4000,
               8000, 16000, 32000, 64000, 128000, 143000]

PYTHIA_MODELS = [
    ("EleutherAI/pythia-14m", 0.014),
    ("EleutherAI/pythia-31m", 0.031),
    ("EleutherAI/pythia-70m", 0.070),
    ("EleutherAI/pythia-160m", 0.160),
    ("EleutherAI/pythia-410m", 0.410),
    ("EleutherAI/pythia-1b", 1.0),
]

LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]


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
    def __init__(self, D, dev="cuda"):
        self.D, self.dev = D, dev
        self.A = torch.zeros(D, D, device=dev)
        self.B = {t: torch.zeros(D, TASK_N_CLASSES[t], device=dev) for t in TASKS}
        self.cc = {t: torch.zeros(TASK_N_CLASSES[t], device=dev) for t in TASKS}
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
            self.cc[t] += oh.sum(0)

    def solve(self, lam=1.0, task="age_bin"):
        n = max(self.n, 1)
        A = self.A.double(); sx = self.sx.double(); sx2 = self.sx2.double()
        Bt = self.B[task].double(); cc = self.cc[task].double()
        mean = sx / n
        var = (sx2 / n - mean**2).clamp(min=1e-8); std = var.sqrt(); inv = 1.0/std
        A_c = A - n * mean.unsqueeze(1) @ mean.unsqueeze(0)
        A_z = A_c * inv.unsqueeze(1) * inv.unsqueeze(0)
        mean_y = cc / n
        B_c = Bt - n * mean.unsqueeze(1) @ mean_y.unsqueeze(0)
        B_z = B_c * inv.unsqueeze(1)
        W = torch.linalg.solve(A_z / n + lam * torch.eye(self.D, device=self.dev, dtype=torch.float64), B_z / n)
        return W.float(), mean_y.float(), mean.float(), std.float()


def run_checkpoint(model_name, revision, n_params, train_loader, test_loader, dev,
                   max_train_batches=0):
    """Train and evaluate Ridge probes for one checkpoint."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()

    cfg = model.config
    if hasattr(cfg, 'text_config'): cfg = cfg.text_config
    hdim = cfg.hidden_size; nl = cfg.num_hidden_layers
    elvs = eval_layers_list(nl, stride=4)

    # Train: accumulate Ridge Gram matrices
    ridges = {l: Ridge(hdim, dev) for l in elvs}
    capture = ActivationCapture(model, elvs, nl)

    for bn, (enc, labs) in enumerate(train_loader):
        if max_train_batches > 0 and bn >= max_train_batches:
            break
        ids = enc["input_ids"].to(dev)
        amask = enc["attention_mask"].to(dev)
        try:
            capture.clear()
            with torch.no_grad():
                get_inner_model(model)(input_ids=ids, attention_mask=amask)
            bi, ti = torch.where(amask)
            if bi.shape[0] == 0: capture.clear(); continue
            tlabs = {t: labs[t].to(dev)[bi] for t in TASKS}
            for l in elvs:
                hs = capture.captured.get(l)
                if hs is None: continue
                ridges[l].update(hs[bi, ti].float(), tlabs)
            capture.clear()
        except torch.cuda.OutOfMemoryError:
            capture.clear(); torch.cuda.empty_cache(); gc.collect()

    # Solve for all lambdas
    solutions = {}  # (layer, task, lam) -> (W, bias, mean, std)
    for l in elvs:
        for t in TASKS:
            for lam in LAMBDAS:
                W, bias, mean, std = ridges[l].solve(lam=lam, task=t)
                solutions[(l, t, lam)] = (W.to(dev), bias.to(dev), mean.to(dev), std.to(dev))
    del ridges; gc.collect()

    # Eval: per-text predictions
    preds = {(l, t, lam): {"p": [], "y": []} for l in elvs for t in TASKS for lam in LAMBDAS}

    for enc, labs in test_loader:
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
        if bi.shape[0] == 0: capture.clear(); continue
        batch_labels = {t: labs[t].to(dev) for t in TASKS}
        text_counts = torch.zeros(B, 1, device=dev)
        text_counts.scatter_add_(0, bi.unsqueeze(1), torch.ones(bi.shape[0], 1, device=dev))

        for l in elvs:
            hs = capture.captured.get(l)
            if hs is None: continue
            acts = hs[bi, ti].float()
            for t in TASKS:
                for lam in LAMBDAS:
                    W, bias, mean, std = solutions[(l, t, lam)]
                    logits = (acts - mean) / (std + 1e-8) @ W + bias
                    text_logits = torch.zeros(B, logits.shape[1], device=dev)
                    text_logits.scatter_add_(0, bi.unsqueeze(1).expand_as(logits), logits)
                    text_logits = text_logits / text_counts.clamp(min=1)
                    preds[(l, t, lam)]["p"].extend(text_logits.argmax(-1).cpu().tolist())
                    preds[(l, t, lam)]["y"].extend(batch_labels[t].cpu().tolist())
        capture.clear()

    capture.remove()
    del model; torch.cuda.empty_cache(); gc.collect()

    # Compute accuracy, pick best lambda per (layer, task)
    rows = []
    for t in TASKS:
        best_acc, best_layer, best_lam = 0, 0, 1.0
        for l in elvs:
            for lam in LAMBDAS:
                d = preds[(l, t, lam)]
                if not d["y"]: continue
                acc = balanced_accuracy_score(d["y"], d["p"])
                if acc > best_acc:
                    best_acc, best_layer, best_lam = acc, l, lam
        rows.append({
            "model": short(model_name), "n_params": n_params,
            "revision": revision, "task": t,
            "best_layer": best_layer, "best_lambda": best_lam,
            "balanced_acc": round(best_acc, 5),
        })

    return rows


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_path", default=str(BASE_DIR / "data" / "processed" / "blog_corpus.parquet"))
    pa.add_argument("--max_train_texts", type=int, default=20000)
    pa.add_argument("--max_test_texts", type=int, default=10000)
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--max_train_batches", type=int, default=0)
    pa.add_argument("--models", nargs="*", help="Only run these model short names")
    pa.add_argument("--checkpoints", nargs="*", type=int, help="Only these steps")
    args = pa.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_file = BASE_DIR / "results" / "training_dynamics.csv"

    # Load existing results to skip completed
    done = set()
    if out_file.exists():
        edf = pd.read_csv(out_file)
        for _, row in edf.iterrows():
            done.add((row["model"], str(row["revision"])))

    # Load data
    trn = BlogDS(args.data_path, "train")
    tst = BlogDS(args.data_path, "test")
    if args.max_train_texts > 0 and len(trn) > args.max_train_texts:
        torch.manual_seed(42)
        trn = Subset(trn, torch.randperm(len(trn))[:args.max_train_texts].tolist())
    if args.max_test_texts > 0 and len(tst) > args.max_test_texts:
        torch.manual_seed(42)
        tst = Subset(tst, torch.randperm(len(tst))[:args.max_test_texts].tolist())

    checkpoints = args.checkpoints or CHECKPOINTS
    models_to_run = PYTHIA_MODELS
    if args.models:
        models_to_run = [(m, p) for m, p in PYTHIA_MODELS if short(m) in args.models]

    total = sum(1 for m, _ in models_to_run for s in checkpoints
                if (short(m), f"step{s}") not in done)
    print(f"Training dynamics: {total} checkpoint runs needed")
    print(f"  Train: {len(trn)} texts, Test: {len(tst)} texts, bs={args.batch_size}")

    all_rows = []
    completed = 0

    for model_name, n_params in models_to_run:
        ms = short(model_name)
        # Load tokenizer once per model (shared across checkpoints)
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tok.padding_side = "right"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        col = make_collate(tok)

        train_loader = DataLoader(trn, batch_size=args.batch_size, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True,
                                  collate_fn=col, persistent_workers=True)
        test_loader = DataLoader(tst, batch_size=args.batch_size, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True,
                                 collate_fn=col, persistent_workers=True)

        for step in checkpoints:
            revision = f"step{step}"
            if (ms, revision) in done:
                continue

            t0 = time.time()
            print(f"\n  {ms} {revision} ({completed}/{total}) ...", end="", flush=True)

            try:
                rows = run_checkpoint(model_name, revision, n_params,
                                     train_loader, test_loader, dev,
                                     max_train_batches=args.max_train_batches)
                all_rows.extend(rows)
                elapsed = time.time() - t0
                accs = {r["task"]: r["balanced_acc"] for r in rows}
                print(f" age={accs.get('age_bin',0):.3f} gen={accs.get('gender',0):.3f} ({elapsed:.0f}s)")
                completed += 1

                # Save incrementally
                if all_rows:
                    new_df = pd.DataFrame(all_rows)
                    if out_file.exists():
                        existing = pd.read_csv(out_file)
                        new_df = pd.concat([existing, new_df], ignore_index=True)
                        new_df = new_df.drop_duplicates(subset=["model", "revision", "task"], keep="last")
                    new_df.to_csv(out_file, index=False)
                    all_rows = []

            except Exception as e:
                print(f" FAILED: {e}")
                torch.cuda.empty_cache(); gc.collect()

        # Clear model cache between sizes
        import shutil
        cache = Path(os.environ.get("HF_HOME", "")) / "hub"
        if cache.exists():
            for d in cache.glob(f"models--*{ms.replace('-', '*')}*"):
                shutil.rmtree(d, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"Done: {completed} checkpoints completed")
    if out_file.exists():
        df = pd.read_csv(out_file)
        print(f"Total rows: {len(df)}")


if __name__ == "__main__":
    main()
