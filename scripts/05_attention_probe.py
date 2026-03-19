#!/usr/bin/env python3
"""
05_attention_probe.py — Train and evaluate attention probes (Dower et al. 2024).

Architecture:
  q = softmax(A @ w_q + b_q)      # (S,) attention weights over tokens
  z = sigmoid(q^T @ (A @ w_v) + b_v)  # scalar logit per class

Extended to multi-class via C value heads:
  q = softmax(A @ w_q + b_q)          # (S,) shared attention weights
  logits = q^T @ (A @ W_v) + b_v      # (C,) one logit per class

Trained with AdamW, lr=1e-4, wd=1e-5, cross-entropy loss.
Padding tokens are masked out of the softmax via -inf.

Usage:
  python 05_attention_probe.py --model_name Qwen/Qwen2.5-0.5B
"""

import argparse, gc, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_SEQ_LEN = 1024
SEED = 42

AGE_BIN_MAP = {1: 0, 2: 1, 3: 2}
GENDER_MAP = {"female": 0, "male": 1}
STAR_SIGNS_SORTED = sorted(["Aquarius","Aries","Cancer","Capricorn","Gemini","Leo",
                             "Libra","Pisces","Sagittarius","Scorpio","Taurus","Virgo"])
STAR_SIGN_MAP = {n: i for i, n in enumerate(STAR_SIGNS_SORTED)}
TASK_N_CLASSES = {"age_bin": 3, "gender": 2, "star_sign": 12}
TASKS = list(TASK_N_CLASSES.keys())

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)

# ── Dataset ──────────────────────────────────────────────────────────────

class BlogDS(Dataset):
    def __init__(self, path, split):
        df = pd.read_parquet(path)
        df = df[df["split"] == split].reset_index(drop=True)
        self.texts = df["text"].tolist()
        self.y_age = torch.tensor(df["age_bin"].map(AGE_BIN_MAP).values, dtype=torch.long)
        self.y_gen = torch.tensor(df["gender"].map(GENDER_MAP).values, dtype=torch.long)
        self.y_star = torch.tensor(df["star_sign"].map(STAR_SIGN_MAP).values, dtype=torch.long)
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i], self.y_age[i], self.y_gen[i], self.y_star[i]

def make_collate(tok, chat_template=False):
    def fn(batch):
        texts, a, g, s = zip(*batch)
        if chat_template and hasattr(tok, "apply_chat_template"):
            templated = [tok.apply_chat_template([{"role": "user", "content": t}],
                         tokenize=False, add_generation_prompt=False) for t in texts]
            enc = tok(templated, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        else:
            enc = tok(list(texts), padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        return enc, {"age_bin": torch.stack(a), "gender": torch.stack(g), "star_sign": torch.stack(s)}
    return fn

# ── Attention Probe ──────────────────────────────────────────────────────

class AttentionProbe(torch.nn.Module):
    """Attention probe following Dower et al. (2024).

    Shared query vector produces attention weights over tokens.
    Per-class value projection produces logits.
    Padding is masked via -inf before softmax.
    """
    def __init__(self, D, C, dev="cuda"):
        super().__init__()
        self.D, self.C = D, C
        sc = 1.0 / D**0.5
        self.w_q = torch.nn.Parameter(torch.randn(D, device=dev) * sc)
        self.b_q = torch.nn.Parameter(torch.zeros(1, device=dev))
        self.W_v = torch.nn.Parameter(torch.randn(D, C, device=dev) * sc)
        self.b_v = torch.nn.Parameter(torch.zeros(C, device=dev))

    def forward(self, A, mask):
        """
        A: (B, S, D) activations
        mask: (B, S) bool, True for real tokens, False for padding
        Returns: (B, C) logits
        """
        # Query scores: (B, S)
        scores = A @ self.w_q + self.b_q
        # Mask padding with -inf before softmax
        scores = scores.masked_fill(~mask, float('-inf'))
        q = torch.softmax(scores, dim=1)  # (B, S)
        # Value projection: (B, S, C)
        values = A @ self.W_v  # (B, S, C)
        # Attention-weighted sum: (B, C)
        logits = torch.einsum('bs,bsc->bc', q, values) + self.b_v
        return logits

# ── Hook-based activation capture ───────────────────────────────────────

class ActivationCapture:
    def __init__(self, model, eval_layers, n_layers):
        self.captured = {}
        self.hooks = []
        inner = model.model
        for l in eval_layers:
            if l == 0:
                h = inner.embed_tokens.register_forward_hook(self._make_hook(0))
            elif l == n_layers:
                h = inner.norm.register_forward_hook(self._make_hook(l))
            else:
                h = inner.layers[l - 1].register_forward_hook(self._make_hook(l))
            self.hooks.append(h)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.captured[layer_idx] = output[0]
            else:
                self.captured[layer_idx] = output
        return hook

    def clear(self): self.captured.clear()
    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks.clear()

# ── Helpers ──────────────────────────────────────────────────────────────

def eval_layers_list(nl, stride=4):
    ls = list(range(0, nl+1, stride))
    if nl not in ls: ls.append(nl)
    return sorted(ls)

def auto_bs(model):
    n = sum(p.numel() for p in model.parameters()) / 1e9
    if n < 1:    return 128
    elif n < 2:  return 128
    elif n < 5:  return 64
    elif n < 10: return 32
    else:        return 16

def short(s): return s.rstrip("/").split("/")[-1]

# ── Main ────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_name", required=True)
    pa.add_argument("--max_train_texts", type=int, default=100000)
    pa.add_argument("--max_test_texts", type=int, default=10000)
    pa.add_argument("--eval_layer_stride", type=int, default=4)
    pa.add_argument("--epochs", type=int, default=3)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--wd", type=float, default=1e-5)
    pa.add_argument("--output_dir", default="/workspace/characterprobing/results/")
    pa.add_argument("--data_path", default="/workspace/characterprobing/data/processed/blog_corpus.parquet")
    pa.add_argument("--seed", type=int, default=SEED)
    pa.add_argument("--chat_template", action="store_true")
    args = pa.parse_args()

    set_seed(args.seed)
    ms = short(args.model_name)
    if args.chat_template: ms += "_chat"
    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)
    probe_dir = Path("/workspace/characterprobing/probes") / ms
    probe_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print(f"Loading {args.model_name} fp16 ...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16,
                                                  device_map="auto", trust_remote_code=True)
    model.eval()
    hdim = model.config.hidden_size
    nl = model.config.num_hidden_layers
    elvs = eval_layers_list(nl, args.eval_layer_stride)
    bs = auto_bs(model)
    print(f"  hdim={hdim}, layers={nl}, bs={bs}, eval_layers={elvs}")

    col = make_collate(tok, chat_template=args.chat_template)
    trn = BlogDS(args.data_path, "train"); tst = BlogDS(args.data_path, "test")
    if args.max_train_texts > 0 and len(trn) > args.max_train_texts:
        trn = Subset(trn, torch.randperm(len(trn))[:args.max_train_texts].tolist())
    if args.max_test_texts > 0 and len(tst) > args.max_test_texts:
        tst = Subset(tst, torch.randperm(len(tst))[:args.max_test_texts].tolist())
    print(f"  train={len(trn)}, test={len(tst)}")

    tl = DataLoader(trn, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True,
                    collate_fn=col, persistent_workers=True)
    el = DataLoader(tst, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True,
                    collate_fn=col, persistent_workers=True)

    dev = "cuda"
    capture = ActivationCapture(model, elvs, nl)

    # ── Create probes and optimizers ─────────────────────────────────────
    probes = {}  # (layer, task) -> AttentionProbe
    for l in elvs:
        for t in TASKS:
            probes[(l, t)] = AttentionProbe(hdim, TASK_N_CLASSES[t], dev=dev)

    optimizers = {}
    for key, probe in probes.items():
        optimizers[key] = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.wd)

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TRAINING ATTENTION PROBES — {len(trn)} texts, {args.epochs} epoch(s)")
    print(f"  lr={args.lr}, wd={args.wd}, AdamW")
    print(f"{'='*60}")

    loss_logs = []
    t0 = time.time()
    global_step = 0

    for epoch in range(args.epochs):
        print(f"\n  Epoch {epoch+1}/{args.epochs}")
        for probe in probes.values(): probe.train()

        for bn, (enc, labs) in enumerate(tqdm(tl, desc=f"Train e{epoch+1}", dynamic_ncols=True)):
            ids = enc["input_ids"].to(dev)
            amask = enc["attention_mask"].to(dev)

            try:
                capture.clear()
                with torch.no_grad():
                    model.model(input_ids=ids, attention_mask=amask)
            except torch.cuda.OutOfMemoryError:
                capture.clear(); torch.cuda.empty_cache(); gc.collect(); continue

            mask = amask.bool()  # (B, S)
            batch_labels = {t: labs[t].to(dev) for t in TASKS}

            log = {"batch": global_step}

            for l in elvs:
                A = capture.captured[l].float().detach()  # (B, S, D) — detach from model graph
                for t in TASKS:
                    key = (l, t)
                    optimizers[key].zero_grad()
                    logits = probes[key](A, mask)  # (B, C)
                    loss = F.cross_entropy(logits, batch_labels[t])
                    loss.backward()
                    optimizers[key].step()
                    log[f"loss_L{l}_{t}"] = round(loss.item(), 5)

            loss_logs.append(log)
            capture.clear()
            global_step += 1

            if global_step % 500 == 0: gc.collect()

    train_time = time.time() - t0
    print(f"\nTraining: {train_time:.1f}s")

    # Save loss logs
    ldf = pd.DataFrame(loss_logs)
    ldf.to_csv(od / f"{ms}_attn_training_losses.csv", index=False)

    # Save probe checkpoints
    for (l, t), probe in probes.items():
        torch.save(probe.state_dict(), probe_dir / f"L{l}_{t}_attn.pt")
    print(f"  Saved probes and training losses")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"EVALUATING ATTENTION PROBES — {len(tst)} texts")
    print(f"{'='*60}")

    for probe in probes.values(): probe.eval()

    text_preds = {(l, t): {"pred": [], "true": []} for l in elvs for t in TASKS}

    t0 = time.time()
    for bn, (enc, labs) in enumerate(tqdm(el, desc="Eval", dynamic_ncols=True)):
        ids = enc["input_ids"].to(dev)
        amask = enc["attention_mask"].to(dev)

        try:
            capture.clear()
            with torch.no_grad():
                model.model(input_ids=ids, attention_mask=amask)
        except torch.cuda.OutOfMemoryError:
            capture.clear(); torch.cuda.empty_cache(); gc.collect(); continue

        mask = amask.bool()
        batch_labels = {t: labs[t].to(dev) for t in TASKS}

        with torch.no_grad():
            for l in elvs:
                A = capture.captured[l].float()
                for t in TASKS:
                    logits = probes[(l, t)](A, mask)  # (B, C)
                    preds = logits.argmax(-1)
                    text_preds[(l, t)]["pred"].extend(preds.cpu().tolist())
                    text_preds[(l, t)]["true"].extend(batch_labels[t].cpu().tolist())

        capture.clear()
        if bn % 200 == 0: gc.collect()

    capture.remove()
    eval_time = time.time() - t0
    print(f"Eval time: {eval_time:.1f}s")

    # ── Results ──────────────────────────────────────────────────────────
    result_rows = []
    print(f"\n{'='*70}")
    print(f"ATTENTION PROBE RESULTS: {ms}")
    print(f"{'='*70}")
    for (l, t), data in sorted(text_preds.items()):
        if not data["true"]: continue
        true = np.array(data["true"])
        pred = np.array(data["pred"])
        ba = balanced_accuracy_score(true, pred)
        f1 = f1_score(true, pred, average="macro", zero_division=0)
        result_rows.append({
            "model_name": args.model_name,
            "strategy": "attention",
            "layer": l,
            "task": t,
            "text_balanced_acc": round(ba, 5),
            "macro_f1": round(f1, 5),
        })
        print(f"  L{l:>3} {t:>10}: bal_acc={ba:.4f}  f1={f1:.4f}")

    rdf = pd.DataFrame(result_rows)
    rdf.to_csv(od / f"{ms}_attn_results.csv", index=False)
    print(f"\nSaved {od / f'{ms}_attn_results.csv'}")

    del model; torch.cuda.empty_cache(); gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
