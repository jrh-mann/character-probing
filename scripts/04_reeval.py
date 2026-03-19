#!/usr/bin/env python3
"""
04_reeval.py — Re-evaluate saved probes with per-token predictions.

Fixes the mean-pooling eval bug: probes were trained on individual token
activations but evaluated on mean-pooled text representations.

This script:
  1. Loads a model and its saved EMA probes
  2. Runs the test set through the model
  3. Predicts per-token, aggregates per-text via mean logits
  4. Reports corrected balanced accuracy per layer per task
  5. Collects position-accuracy data: how does probe accuracy vary
     with relative position in the text (causal context window)

Usage:
  python 04_reeval.py --model_name Qwen/Qwen2.5-0.5B
  python 04_reeval.py --model_name Qwen/Qwen2.5-0.5B-Instruct --chat_template
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

# ── Dataset (same as 02_run_probes.py) ──────────────────────────────────

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
            templated = []
            for text in texts:
                msgs = [{"role": "user", "content": text}]
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                templated.append(t)
            enc = tok(templated, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        else:
            enc = tok(list(texts), padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
        return enc, {"age_bin": torch.stack(a), "gender": torch.stack(g), "star_sign": torch.stack(s)}
    return fn

# ── EMA Probe (prediction only — loaded from checkpoint) ────────────────

class EMAProbe:
    def __init__(self, state, dev="cuda"):
        self.rmean = state["rmean"].to(dev)
        self.rvar = state["rvar"].to(dev)
        self.W_ema = state["W_ema"].to(dev)
        self.b_ema = state["b_ema"].to(dev)
        self.decay = state.get("decay", 0.999)
        self.step = state["step"]

    @torch.no_grad()
    def predict(self, x):
        xn = (x - self.rmean) / (self.rvar.sqrt() + 1e-8)
        bc = max(1.0 - self.decay ** self.step, 1e-8)
        return xn @ (self.W_ema / bc).T + self.b_ema / bc

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

# ── Position buckets ────────────────────────────────────────────────────
N_POS_BUCKETS = 20  # 5% increments

# ── Main ────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_name", required=True)
    pa.add_argument("--max_test_texts", type=int, default=10000)
    pa.add_argument("--eval_layer_stride", type=int, default=4)
    pa.add_argument("--output_dir", default="/workspace/characterprobing/results/")
    pa.add_argument("--data_path", default="/workspace/characterprobing/data/processed/blog_corpus.parquet")
    pa.add_argument("--probe_dir", default="/workspace/characterprobing/probes")
    pa.add_argument("--seed", type=int, default=SEED)
    pa.add_argument("--chat_template", action="store_true")
    args = pa.parse_args()

    set_seed(args.seed)
    ms = short(args.model_name)
    if args.chat_template:
        ms += "_chat"
    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)
    probe_path = Path(args.probe_dir) / ms

    if not probe_path.exists():
        print(f"ERROR: No saved probes at {probe_path}")
        return

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

    # Load saved EMA probes
    ema = {}
    for l in elvs:
        for t in TASKS:
            fp = probe_path / f"L{l}_{t}_ema.pt"
            if not fp.exists():
                print(f"  WARNING: missing {fp}")
                continue
            state = torch.load(fp, map_location="cpu", weights_only=False)
            ema[(l, t)] = EMAProbe(state, dev="cuda")
    print(f"  Loaded {len(ema)} EMA probes")

    # Load test data
    col = make_collate(tok, chat_template=args.chat_template)
    tst = BlogDS(args.data_path, "test")
    if args.max_test_texts > 0 and len(tst) > args.max_test_texts:
        tst = Subset(tst, torch.randperm(len(tst))[:args.max_test_texts].tolist())
    print(f"  test={len(tst)}")
    el = DataLoader(tst, batch_size=bs, shuffle=False, num_workers=8,
                    pin_memory=True, collate_fn=col, persistent_workers=True)

    # ── Eval ────────────────────────────────────────────────────────────
    dev = "cuda"
    capture = ActivationCapture(model, elvs, nl)

    # Per-text aggregated predictions (mean logits → argmax)
    text_preds = {(l, t): {"pred": [], "true": []} for l in elvs for t in TASKS if (l, t) in ema}

    # Per-token position accuracy: bucket by relative position in text
    # pos_data[layer][task][bucket] = {"correct": int, "total": int}
    pos_data = {}
    for l in elvs:
        pos_data[l] = {}
        for t in TASKS:
            if (l, t) not in ema: continue
            pos_data[l][t] = [{"correct": 0, "total": 0} for _ in range(N_POS_BUCKETS)]

    print(f"\nEvaluating {len(tst)} texts ...")
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

        B, S = ids.shape
        batch_labels = {t: labs[t].to(dev) for t in TASKS}

        # Get lengths (number of non-pad tokens per text)
        lengths = amask.sum(1)  # (B,)

        for l in elvs:
            hs = capture.captured[l]  # (B, S, D)

            for t in TASKS:
                if (l, t) not in ema: continue
                probe = ema[(l, t)]

                # ── Per-token predictions ────────────────────────────
                # Reshape to (B*S, D), predict, reshape back to (B, S, C)
                flat = hs.reshape(B * S, -1).float()
                logits_flat = probe.predict(flat)  # (B*S, C)
                logits = logits_flat.reshape(B, S, -1)  # (B, S, C)

                # ── Text-level: mean logits over non-pad tokens ──────
                # Zero out padding positions
                mask_3d = amask.unsqueeze(-1)  # (B, S, 1)
                masked_logits = logits * mask_3d  # (B, S, C)
                sum_logits = masked_logits.sum(1)  # (B, C)
                mean_logits = sum_logits / lengths.unsqueeze(1).clamp(min=1)  # (B, C)
                text_pred = mean_logits.argmax(-1)  # (B,)

                text_preds[(l, t)]["pred"].extend(text_pred.cpu().tolist())
                text_preds[(l, t)]["true"].extend(batch_labels[t].cpu().tolist())

                # ── Position-accuracy data (fully vectorized) ─────────
                tok_preds = logits.argmax(-1)  # (B, S)

                # Build position indices: for each (b, s), relative position = s / length[b]
                pos_idx = torch.arange(S, device=dev).unsqueeze(0).expand(B, S)  # (B, S)
                rel_pos = pos_idx.float() / lengths.unsqueeze(1).clamp(min=1).float()  # (B, S)
                buckets = (rel_pos * N_POS_BUCKETS).long().clamp(max=N_POS_BUCKETS - 1)  # (B, S)

                # Expand labels to match token shape
                labels_exp = batch_labels[t].unsqueeze(1).expand(B, S)  # (B, S)
                correct = (tok_preds == labels_exp)  # (B, S) bool

                # Only count non-pad positions
                valid = amask.bool()  # (B, S)

                for bucket_idx in range(N_POS_BUCKETS):
                    bucket_mask = (buckets == bucket_idx) & valid
                    pos_data[l][t][bucket_idx]["total"] += bucket_mask.sum().item()
                    pos_data[l][t][bucket_idx]["correct"] += (correct & bucket_mask).sum().item()

        capture.clear()
        if bn % 200 == 0: gc.collect()

    capture.remove()
    eval_time = time.time() - t0
    print(f"Eval time: {eval_time:.1f}s")

    # ── Compute and save text-level metrics ──────────────────────────────
    result_rows = []
    print(f"\n{'='*70}")
    print(f"CORRECTED RESULTS (per-token eval, mean-logit aggregation): {ms}")
    print(f"{'='*70}")
    for (l, t), data in sorted(text_preds.items()):
        if not data["true"]: continue
        true = np.array(data["true"])
        pred = np.array(data["pred"])
        ba = balanced_accuracy_score(true, pred)
        f1 = f1_score(true, pred, average="macro", zero_division=0)
        result_rows.append({
            "model_name": args.model_name,
            "strategy": "ema",
            "layer": l,
            "task": t,
            "text_balanced_acc": round(ba, 5),
            "macro_f1": round(f1, 5),
        })
        print(f"  L{l:>3} {t:>10}: bal_acc={ba:.4f}  f1={f1:.4f}")

    rdf = pd.DataFrame(result_rows)
    rdf.to_csv(od / f"{ms}_reeval_results.csv", index=False)
    print(f"\nSaved {od / f'{ms}_reeval_results.csv'}")

    # ── Save position-accuracy data ──────────────────────────────────────
    pos_rows = []
    for l in elvs:
        for t in TASKS:
            if t not in pos_data.get(l, {}): continue
            for bucket_idx, d in enumerate(pos_data[l][t]):
                if d["total"] == 0: continue
                pos_rows.append({
                    "model_name": args.model_name + ("_chat" if args.chat_template else ""),
                    "layer": l,
                    "task": t,
                    "position_pct": round((bucket_idx + 0.5) / N_POS_BUCKETS * 100, 1),
                    "correct": d["correct"],
                    "total": d["total"],
                    "accuracy": round(d["correct"] / d["total"], 5),
                })

    pdf = pd.DataFrame(pos_rows)
    pdf.to_csv(od / f"{ms}_position_accuracy.csv", index=False)
    print(f"Saved {od / f'{ms}_position_accuracy.csv'}")

    del model; torch.cuda.empty_cache(); gc.collect()
    print("Done.")


if __name__ == "__main__":
    main()
