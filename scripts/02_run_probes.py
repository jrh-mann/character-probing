#!/usr/bin/env python3
"""
02_run_probes.py — v5: All-GPU, no threading, maximum throughput.

All probe strategies run on GPU after each forward pass. No CPU bottleneck.

Strategies:
  1. EMA Linear Probe — per-layer, online SGD + EMA
  2. Ridge Regression — per-layer, incremental Gram matrix (float32 GPU, solve in float64 CPU)
  3. Multi-layer Ridge — reservoir of concat features, solved post-hoc
  4. Mass-Mean Probe — per-layer, zero-parameter nearest-centroid
"""

import argparse, gc, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import save_file as safetensors_save
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_SEQ_LEN = 1024
SEED = 42
RESERVOIR_SIZE = 50000
MAX_MULTI_LAYERS = 6  # cap concat layers to limit reservoir/solve memory

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
            # Wrap each text as a user message
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

# ── EMA Linear Probe (GPU) ──────────────────────────────────────────────

class EMAProbe:
    def __init__(self, D, C, dev="cuda", lr=0.01, decay=0.999, wd=1e-4):
        self.D, self.C, self.lr, self.decay, self.wd, self.dev = D, C, lr, decay, wd, dev
        sc = (2.0/(D+C))**0.5
        self.W = torch.randn(C, D, device=dev) * sc
        self.b = torch.zeros(C, device=dev)
        self.W_ema = torch.zeros_like(self.W)
        self.b_ema = torch.zeros_like(self.b)
        self.step = 0
        self.rmean = torch.zeros(D, device=dev)
        self.rvar = torch.ones(D, device=dev)
        self.ns = 0
        self.tot_c, self.tot_n = 0, 0

    @torch.no_grad()
    def _ustats(self, x):
        B = x.shape[0]
        bm, bv = x.mean(0), (x.var(0, unbiased=False) if B > 1 else torch.zeros(x.shape[1], device=x.device))
        nn = self.ns + B; d = bm - self.rmean
        self.rmean += d * (B / nn)
        self.rvar = (self.rvar * self.ns + bv * B + d**2 * self.ns * B / nn) / nn
        self.ns = nn

    @torch.no_grad()
    def update(self, x, y):
        self._ustats(x)
        xn = (x - self.rmean) / (self.rvar.sqrt() + 1e-8)
        lo = xn @ self.W.T + self.b
        loss = F.cross_entropy(lo, y).item()
        self.tot_c += (lo.argmax(-1) == y).sum().item(); self.tot_n += y.shape[0]
        p = torch.softmax(lo, -1)
        oh = torch.zeros_like(p).scatter_(1, y.unsqueeze(1), 1.0)
        gl = (p - oh) / xn.shape[0]
        self.W -= self.lr * (gl.T @ xn + self.wd * self.W)
        self.b -= self.lr * gl.sum(0)
        self.W_ema.mul_(self.decay).add_(self.W, alpha=1-self.decay)
        self.b_ema.mul_(self.decay).add_(self.b, alpha=1-self.decay)
        self.step += 1
        return loss

    @torch.no_grad()
    def predict(self, x):
        xn = (x - self.rmean) / (self.rvar.sqrt() + 1e-8)
        bc = max(1.0 - self.decay ** self.step, 1e-8)
        return xn @ (self.W_ema / bc).T + self.b_ema / bc

    def state_dict(self):
        return {k: getattr(self, k).cpu() if isinstance(getattr(self, k), torch.Tensor) else getattr(self, k)
                for k in ("W","b","W_ema","b_ema","rmean","rvar","ns","step")}

# ── Incremental Ridge (GPU, float32, solve in float64) ───────────────────

class Ridge:
    """Incremental Gram matrix on GPU. Solve post-hoc on CPU in float64."""
    def __init__(self, D, dev="cuda"):
        self.D, self.dev = D, dev
        self.A = torch.zeros(D, D, device=dev)          # float32
        self.B = {t: torch.zeros(D, TASK_N_CLASSES[t], device=dev) for t in TASKS}
        self.class_counts = {t: torch.zeros(TASK_N_CLASSES[t], device=dev) for t in TASKS}
        self.sx = torch.zeros(D, device=dev)
        self.sx2 = torch.zeros(D, device=dev)
        self.n = 0

    @torch.no_grad()
    def update(self, x, labels):
        """x: (N, D) float32 GPU. labels: {task: (N,) long GPU}."""
        self.A.addmm_(x.T, x)
        self.sx += x.sum(0)
        self.sx2 += (x**2).sum(0)
        self.n += x.shape[0]
        for t in TASKS:
            nc = TASK_N_CLASSES[t]
            oh = torch.zeros(x.shape[0], nc, device=self.dev)
            oh.scatter_(1, labels[t].unsqueeze(1), 1.0)
            self.B[t].addmm_(x.T, oh)
            self.class_counts[t] += oh.sum(0)

    def solve(self, lam=1.0, task="age_bin"):
        """Returns (W_z, bias, mean, std) — weights in z-scored space. All on CPU."""
        n = max(self.n, 1)
        sx = self.sx.cpu().double(); sx2 = self.sx2.cpu().double()
        A = self.A.cpu().double(); Bt = self.B[task].cpu().double()
        cc = self.class_counts[task].cpu().double()  # actual class counts

        mean = sx / n
        var = (sx2 / n - mean**2).clamp(min=1e-8); std = var.sqrt(); inv = 1.0/std

        # Center and z-score the Gram matrix
        A_c = A - n * mean.unsqueeze(1) @ mean.unsqueeze(0)
        A_z = A_c * inv.unsqueeze(1) * inv.unsqueeze(0)

        # Center the cross-covariance: B_c = X^T Y - n * mean_x @ mean_y^T
        mean_y = cc / n  # class proportions (sums to 1)
        B_c = Bt - n * mean.unsqueeze(1) @ mean_y.unsqueeze(0)
        B_z = B_c * inv.unsqueeze(1)

        # Normalize by n so lambda is on a sensible scale
        # Solve: (A_z/n + lam * I) W = B_z/n
        W_z = torch.linalg.solve(A_z / n + lam * torch.eye(self.D, dtype=torch.float64), B_z / n)
        bias = mean_y
        return W_z.float(), bias.float(), mean.float(), std.float()

    def state_dict(self):
        d = {"A": self.A.cpu(), "sx": self.sx.cpu(), "sx2": self.sx2.cpu(), "n": self.n}
        for t, b in self.B.items(): d[f"B_{t}"] = b.cpu()
        for t, c in self.class_counts.items(): d[f"cc_{t}"] = c.cpu()
        return d

# ── Mass-Mean Probe (GPU) ───────────────────────────────────────────────

class MassMean:
    def __init__(self, D, C, dev="cuda"):
        self.C = C
        self.sums = torch.zeros(C, D, device=dev)
        self.counts = torch.zeros(C, device=dev)
    @torch.no_grad()
    def update(self, x, y):
        self.sums.index_add_(0, y, x)
        self.counts.scatter_add_(0, y, torch.ones(y.shape[0], device=y.device))
    @torch.no_grad()
    def predict(self, x):
        ctr = self.sums / self.counts.unsqueeze(1).clamp(min=1)
        return -torch.cdist(x.unsqueeze(0), ctr.unsqueeze(0)).squeeze(0)

# ── Cross-Layer Gram (GPU, for multi-layer ridge) ────────────────────────

class CrossLayerGram:
    """Incremental cross-layer Gram blocks on GPU for exact multi-layer ridge.

    Maintains A_ij = X_i^T X_j for all pairs (i,j) of selected layers.
    B_i = X_i^T Y comes from per-layer Ridge objects (reused, not duplicated).
    At solve time, assembles the full block matrix and solves on GPU float32.
    """
    def __init__(self, multi_layers, D, dev="cuda"):
        self.multi_layers = multi_layers
        self.K = len(multi_layers)
        self.D = D
        self.dev = dev
        # Store A_ij blocks. Only upper triangle needed (symmetric).
        self.blocks = {}
        for i, li in enumerate(multi_layers):
            for j, lj in enumerate(multi_layers):
                if j >= i:  # upper triangle only
                    self.blocks[(li, lj)] = torch.zeros(D, D, device=dev)

    @torch.no_grad()
    def update(self, layer_acts):
        """layer_acts: dict[layer_idx -> (N, D) float32 GPU]."""
        for i, li in enumerate(self.multi_layers):
            xi = layer_acts[li]
            for j, lj in enumerate(self.multi_layers):
                if j >= i:
                    xj = layer_acts[lj]
                    self.blocks[(li, lj)].addmm_(xi.T, xj)

    def solve(self, ridges, lam=1.0, task="age_bin"):
        """Assemble full block Gram and solve on GPU float32.

        Uses per-layer Ridge objects for mean/std/B (avoids duplication).
        Returns (W, bias, mean_concat, std_concat) all float32 GPU.
        """
        K, D = self.K, self.D
        KD = K * D
        ml = self.multi_layers

        # Get per-layer stats from Ridge objects
        means, stds, invs = [], [], []
        for li in ml:
            r = ridges[li]
            n = max(r.n, 1)
            mean = (r.sx / n).float()
            var = (r.sx2 / n - mean**2).clamp(min=1e-8)
            std = var.sqrt()
            means.append(mean)
            stds.append(std)
            invs.append(1.0 / std)

        # Assemble and z-score the full block Gram on GPU (float32)
        A_full = torch.zeros(KD, KD, device=self.dev)
        n = ridges[ml[0]].n

        for i, li in enumerate(ml):
            for j, lj in enumerate(ml):
                if j >= i:
                    A_ij = self.blocks[(li, lj)].float()
                else:
                    A_ij = self.blocks[(lj, li)].float().T  # symmetric

                # Center: A_c_ij = A_ij - n * mean_i @ mean_j^T
                A_c = A_ij - n * means[i].unsqueeze(1) @ means[j].unsqueeze(0)
                # Z-score: A_z_ij = diag(inv_i) @ A_c @ diag(inv_j)
                A_z = A_c * invs[i].unsqueeze(1) * invs[j].unsqueeze(0)
                # Normalize by n
                A_full[i*D:(i+1)*D, j*D:(j+1)*D] = A_z / n

        # Assemble B_full from per-layer Ridge B matrices
        nc = TASK_N_CLASSES[task]
        B_full = torch.zeros(KD, nc, device=self.dev)
        cc_total = ridges[ml[0]].class_counts[task].float()
        mean_y = cc_total / n

        for i, li in enumerate(ml):
            Bi = ridges[li].B[task].float()
            Bi_c = Bi - n * means[i].unsqueeze(1) @ mean_y.unsqueeze(0)
            Bi_z = Bi_c * invs[i].unsqueeze(1)
            B_full[i*D:(i+1)*D] = Bi_z / n

        # Solve on GPU float32: (A_full + lam*I) W = B_full
        reg = A_full + lam * torch.eye(KD, device=self.dev)
        W = torch.linalg.solve(reg, B_full)
        bias = mean_y

        # Concat mean and std for prediction
        mean_concat = torch.cat(means)
        std_concat = torch.cat(stds)

        return W, bias, mean_concat, std_concat

# ── Helpers ──────────────────────────────────────────────────────────────

def all_real_positions(amask):
    """Return (batch_indices, token_indices, text_ids) for ALL non-pad tokens."""
    B, S = amask.shape; dev = amask.device; ab, at = [], []
    for b in range(B):
        r = amask[b].nonzero(as_tuple=False).squeeze(-1)
        if r.shape[0] == 0: continue
        ab.append(torch.full((r.shape[0],), b, device=dev, dtype=torch.long))
        at.append(r)
    if not ab:
        e = torch.zeros(0, dtype=torch.long, device=dev); return e, e, e
    return torch.cat(ab), torch.cat(at), torch.cat(ab)

def eval_layers_list(nl, stride=4):
    ls = list(range(0, nl+1, stride))
    if nl not in ls: ls.append(nl)
    return sorted(ls)

def auto_bs(model):
    """Fixed batch sizes based on empirical VRAM profiling on A40 (46GB).
    Conservative enough to never OOM; OOM handler catches rare outlier batches."""
    n = sum(p.numel() for p in model.parameters()) / 1e9
    # A40 (46GB) with hooks. Measured: 0.5B@128=29GB, 1.5B@64=19GB, 3B@32=17GB.
    # Previous sizes were too conservative. Targeting ~35GB peak.
    if n < 1:    bs = 128
    elif n < 2:  bs = 128
    elif n < 5:  bs = 64
    elif n < 10: bs = 32
    else:        bs = 16
    print(f"  Batch size: {bs} (model {n:.1f}B params)")
    return bs

def short(s): return s.rstrip("/").split("/")[-1]


# ── Hook-based activation capture ───────────────────────────────────────

class ActivationCapture:
    """Register forward hooks on specific layers to capture hidden states.

    For Qwen2 models:
      - Layer 0 (embedding output) = output of model.model.embed_tokens
      - Layer i (transformer layer i-1 output) = output of model.model.layers[i-1]
      - Layer n_layers (final layer output) = output of model.model.norm

    Hook captures the output tensor (the residual stream at that point).
    """

    def __init__(self, model, eval_layers, n_layers):
        self.captured = {}
        self.hooks = []
        self.eval_layers = eval_layers

        inner = model.model  # Qwen2Model (no lm_head)

        for l in eval_layers:
            if l == 0:
                # Embedding output
                h = inner.embed_tokens.register_forward_hook(self._make_hook(0))
            elif l == n_layers:
                # After final norm
                h = inner.norm.register_forward_hook(self._make_hook(l))
            else:
                # Transformer layer l-1 output (layers are 0-indexed, but our layer l
                # corresponds to the output AFTER transformer block l-1)
                h = inner.layers[l - 1].register_forward_hook(self._make_hook(l))
            self.hooks.append(h)

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            # Transformer layers return a tuple (hidden_states, ...), embedding returns tensor
            if isinstance(output, tuple):
                self.captured[layer_idx] = output[0]
            else:
                self.captured[layer_idx] = output
        return hook

    def clear(self):
        self.captured.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ── Train ────────────────────────────────────────────────────────────────

def _process_batch(capture, elvs, bi, ti, tlabs, ema, mm, ridges, cross_gram, bn, dev):
    """Process one batch of activations through all probe strategies. Returns log dict."""
    log = {"batch": bn}
    layer_acts = {}
    for l in elvs:
        a = capture.captured[l][bi, ti].float()
        layer_acts[l] = a
        for t in TASKS:
            loss = ema[(l,t)].update(a, tlabs[t])
            log[f"loss_L{l}_{t}"] = round(loss, 5)
            p = ema[(l,t)]
            if p.tot_n > 0:
                log[f"acc_L{l}_{t}"] = round(p.tot_c / p.tot_n, 5)
            mm[(l,t)].update(a, tlabs[t])
        ridges[l].update(a, tlabs)
    # Cross-layer Gram update (only for multi_layers subset)
    cross_gram.update(layer_acts)
    return log


def train(model, loader, nl, hdim, tpt, elvs, epochs=1, dev="cuda"):
    ema = {(l,t): EMAProbe(hdim, TASK_N_CLASSES[t], dev) for l in elvs for t in TASKS}
    mm = {(l,t): MassMean(hdim, TASK_N_CLASSES[t], dev) for l in elvs for t in TASKS}
    ridges = {l: Ridge(hdim, dev) for l in elvs}

    # Multi-layer: use at most MAX_MULTI_LAYERS evenly spaced from eval layers
    if len(elvs) > MAX_MULTI_LAYERS:
        step = max(1, len(elvs) // MAX_MULTI_LAYERS)
        multi_layers = elvs[::step][:MAX_MULTI_LAYERS]
    else:
        multi_layers = elvs
    cross_gram = CrossLayerGram(multi_layers, hdim, dev)
    print(f"  Multi-layer concat: {multi_layers} -> dim={hdim * len(multi_layers)}")

    logs = []

    capture = ActivationCapture(model, elvs, nl)
    oom_count = 0

    model.eval()
    global_bn = 0
    for epoch in range(epochs):
        if epochs > 1:
            print(f"  Epoch {epoch+1}/{epochs}")
        for bn, (enc, labs) in enumerate(tqdm(loader, desc=f"Train e{epoch+1}", dynamic_ncols=True)):
            ids = enc["input_ids"].to(dev)
            amask = enc["attention_mask"].to(dev)

            try:
                capture.clear()
                with torch.no_grad():
                    model.model(input_ids=ids, attention_mask=amask)

                bi, ti, txid = all_real_positions(amask)
                if bi.shape[0] == 0:
                    capture.clear(); continue

                tlabs = {t: labs[t].to(dev)[txid] for t in TASKS}
                log = _process_batch(capture, elvs, bi, ti, tlabs, ema, mm, ridges, cross_gram, global_bn, dev)
                logs.append(log)
                capture.clear()

            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                capture.clear(); torch.cuda.empty_cache(); gc.collect()
                if oom_count <= 3:
                    print(f"\n  [OOM] Skipped batch {global_bn}. OOM count: {oom_count}")
                elif oom_count == 4:
                    print(f"\n  [OOM] Suppressing further OOM warnings.")
                continue

            global_bn += 1
            if global_bn % 500 == 0: gc.collect()

    capture.remove()
    if oom_count > 0:
        print(f"  Total OOM skips: {oom_count}/{bn+1} batches ({oom_count/(bn+1)*100:.1f}%)")
    return ema, mm, ridges, cross_gram, multi_layers, pd.DataFrame(logs)

# ── Evaluate ─────────────────────────────────────────────────────────────

def evaluate(model, loader, ema, mm, ridge_sols, multi_sols, elvs, multi_layers, nl, dev="cuda"):
    keys = []
    for l in elvs:
        for t in TASKS:
            keys += [("ema",l,t), ("ema_pool",l,t), ("mm",l,t)]
            for lam in ridge_sols.get((l,t), {}):
                keys.append((f"ridge_{lam}",l,t))
    for t in TASKS:
        for lam in multi_sols.get(t, {}):
            keys.append((f"multi_{lam}","all",t))

    acc = {k: {"tp":[], "tl":[], "pr":[]} for k in keys}

    capture = ActivationCapture(model, elvs, nl)

    model.eval()
    for bn, (enc, labs) in enumerate(tqdm(loader, desc="Eval", dynamic_ncols=True)):
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
        lengths = amask.sum(1)  # (B,)

        # ── Per-token predictions, mean-logit aggregation per text ────
        bi, ti, txid = all_real_positions(amask)
        if bi.shape[0] == 0:
            capture.clear(); continue

        # Also compute mean-pooled activations for legacy comparison
        per_text_acts = {}
        per_tok_acts = {}
        for l in elvs:
            hs = capture.captured[l]  # (B, S, D)
            per_tok_acts[l] = hs[bi, ti].float()
            masked = hs * amask.unsqueeze(-1)
            per_text_acts[l] = (masked.sum(1).float() / lengths.unsqueeze(1).clamp(min=1).float())

        text_counts = torch.zeros(B, 1, device=dev)
        text_counts.scatter_add_(0, txid.unsqueeze(1), torch.ones(txid.shape[0], 1, device=dev))

        for l in elvs:
            a_tok = per_tok_acts[l]   # (N_tokens, D)
            a_pool = per_text_acts[l]  # (B, D)
            for t in TASKS:
                # ── EMA: per-token mean-logit aggregation ──
                logits_tok = ema[(l,t)].predict(a_tok)
                text_logits = torch.zeros(B, logits_tok.shape[1], device=dev)
                text_logits.scatter_add_(0, txid.unsqueeze(1).expand_as(logits_tok), logits_tok)
                text_logits = text_logits / text_counts.clamp(min=1)
                preds_ema = text_logits.argmax(-1)
                acc[("ema",l,t)]["tp"].extend(preds_ema.cpu().tolist())
                acc[("ema",l,t)]["tl"].extend(batch_labels[t].cpu().tolist())
                if t == "gender":
                    probs = torch.softmax(text_logits, -1).cpu().numpy()
                    acc[("ema",l,t)]["pr"].extend(probs.tolist())

                # ── EMA: mean-pooled activations (legacy) ──
                preds_pool = ema[(l,t)].predict(a_pool).argmax(-1)
                acc[("ema_pool",l,t)]["tp"].extend(preds_pool.cpu().tolist())
                acc[("ema_pool",l,t)]["tl"].extend(batch_labels[t].cpu().tolist())

                # ── Mass-mean: per-token ──
                logits_mm = mm[(l,t)].predict(a_tok)
                text_logits_mm = torch.zeros(B, logits_mm.shape[1], device=dev)
                text_logits_mm.scatter_add_(0, txid.unsqueeze(1).expand_as(logits_mm), logits_mm)
                text_logits_mm = text_logits_mm / text_counts.clamp(min=1)
                acc[("mm",l,t)]["tp"].extend(text_logits_mm.argmax(-1).cpu().tolist())
                acc[("mm",l,t)]["tl"].extend(batch_labels[t].cpu().tolist())

                # ── Ridge: per-token ──
                for lam, (Wz, bi2, mn, sd) in ridge_sols.get((l,t), {}).items():
                    lo = ((a_tok - mn) / sd) @ Wz + bi2
                    text_lo = torch.zeros(B, lo.shape[1], device=dev)
                    text_lo.scatter_add_(0, txid.unsqueeze(1).expand_as(lo), lo)
                    text_lo = text_lo / text_counts.clamp(min=1)
                    acc[(f"ridge_{lam}",l,t)]["tp"].extend(text_lo.argmax(-1).cpu().tolist())
                    acc[(f"ridge_{lam}",l,t)]["tl"].extend(batch_labels[t].cpu().tolist())

        # ── Multi-layer: per-token ──
        cat_acts = torch.cat([per_tok_acts[l] for l in multi_layers], dim=1)
        for t in TASKS:
            for lam, (Wz, bi2, mn, sd) in multi_sols.get(t, {}).items():
                lo = ((cat_acts - mn) / sd) @ Wz + bi2
                text_lo = torch.zeros(B, lo.shape[1], device=dev)
                text_lo.scatter_add_(0, txid.unsqueeze(1).expand_as(lo), lo)
                text_lo = text_lo / text_counts.clamp(min=1)
                acc[(f"multi_{lam}","all",t)]["tp"].extend(text_lo.argmax(-1).cpu().tolist())
                acc[(f"multi_{lam}","all",t)]["tl"].extend(batch_labels[t].cpu().tolist())

        capture.clear()
        if bn % 500 == 0: gc.collect()

    capture.remove()
    # Metrics
    rows = []
    for k, a in acc.items():
        if not a["tl"]: continue
        s, l, t = k
        tp, tl = np.array(a["tp"]), np.array(a["tl"])
        row = {"strategy": s, "layer": str(l), "task": t,
               "text_balanced_acc": round(balanced_accuracy_score(tl, tp), 5),
               "macro_f1": round(f1_score(tl, tp, average="macro", zero_division=0), 5)}
        if t == "gender" and s == "ema" and a["pr"]:
            try: row["auc_roc"] = round(roc_auc_score(tl, np.stack(a["pr"])[:,1]), 5)
            except: pass
        if t == "star_sign" and s == "ema":
            rng = np.random.default_rng(42); n = len(tl)
            boots = [balanced_accuracy_score(tl[ii], tp[ii]) for ii in (rng.integers(0,n,size=n) for _ in range(1000))]
            row["bootstrap_ci_lo"] = round(float(np.percentile(boots, 2.5)), 5)
            row["bootstrap_ci_hi"] = round(float(np.percentile(boots, 97.5)), 5)
        rows.append(row)
    return pd.DataFrame(rows)

# ── Main ─────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model_name", required=True)
    pa.add_argument("--batch_size", type=int, default=0)
    pa.add_argument("--tokens_per_text", type=int, default=50)
    pa.add_argument("--max_train_texts", type=int, default=0)
    pa.add_argument("--max_test_texts", type=int, default=10000)
    pa.add_argument("--epochs", type=int, default=1)
    pa.add_argument("--eval_layer_stride", type=int, default=4)
    pa.add_argument("--output_dir", default="/workspace/characterprobing/results/")
    pa.add_argument("--data_path", default="/workspace/characterprobing/data/processed/blog_corpus.parquet")
    pa.add_argument("--seed", type=int, default=SEED)
    pa.add_argument("--chat_template", action="store_true",
                    help="Wrap texts in chat template for instruct models")
    args = pa.parse_args()

    set_seed(args.seed)
    ms = short(args.model_name)
    if args.chat_template:
        ms += "_chat"
    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)
    pd_dir = Path("/workspace/characterprobing/probes") / ms

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    print(f"Loading {args.model_name} fp16 ...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16,
                                                  device_map="auto", trust_remote_code=True)
    model.eval()
    # torch.compile disabled — causes recompilation with variable-length inputs + hooks
    hdim = model.config.hidden_size; nl = model.config.num_hidden_layers
    elvs = eval_layers_list(nl, args.eval_layer_stride)
    bs = args.batch_size if args.batch_size > 0 else auto_bs(model)
    print(f"  hdim={hdim}, layers={nl}, bs={bs}, eval_layers={elvs}")

    col = make_collate(tok, chat_template=args.chat_template)
    if args.chat_template:
        print("  Using chat template for tokenization")
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

    # TRAIN
    print(f"\n{'='*60}\nTRAINING — {len(trn)} texts, {args.epochs} epoch(s)\n{'='*60}")
    t0 = time.time()
    ema, mm, ridges, cross_gram, multi_layers, ldf = train(model, tl, nl, hdim, args.tokens_per_text, elvs, epochs=args.epochs)
    tt = time.time() - t0; print(f"Training: {tt:.1f}s")

    # Save everything
    pd_dir.mkdir(parents=True, exist_ok=True)
    for (l,t), p in ema.items(): torch.save(p.state_dict(), pd_dir / f"L{l}_{t}_ema.pt")
    for (l,t), p in mm.items(): torch.save({"sums": p.sums.cpu(), "counts": p.counts.cpu()}, pd_dir / f"L{l}_{t}_mm.pt")
    for l, r in ridges.items(): torch.save(r.state_dict(), pd_dir / f"L{l}_ridge.pt")
    # Save cross-layer Gram blocks (enables re-solving multi-layer without rerunning)
    cross_state = {"multi_layers": cross_gram.multi_layers, "K": cross_gram.K, "D": cross_gram.D}
    for (li, lj), block in cross_gram.blocks.items():
        cross_state[f"block_{li}_{lj}"] = block.cpu()
    torch.save(cross_state, pd_dir / "cross_gram.pt")
    ldf.to_csv(od / f"{ms}_training_losses.csv", index=False)
    print(f"  Saved: probes, ridge grams, cross-layer grams, mass-mean, losses")

    # Solve ridges
    print("Solving ridge regressions ...")
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_sols = {}  # (layer, task) -> {lam: (Wz, bias, mean, std)}
    for l in elvs:
        for t in TASKS:
            ridge_sols[(l,t)] = {}
            for lam in lambdas:
                Wz, bias, mean, std = ridges[l].solve(lam=lam, task=t)
                ridge_sols[(l,t)][lam] = (Wz.cuda(), bias.cuda(), mean.cuda(), std.cuda())

    # Multi-layer ridge from cross-layer Gram blocks (exact, GPU solve)
    multi_sols = {}  # task -> {lam: (W, bias, mean, std)}
    print(f"Solving multi-layer ridge (GPU, {len(multi_layers)} layers, dim={hdim * len(multi_layers)}) ...")
    for t in TASKS:
        multi_sols[t] = {}
        for lam in lambdas:
            W, bias, mean_c, std_c = cross_gram.solve(ridges, lam=lam, task=t)
            multi_sols[t][lam] = (W, bias, mean_c, std_c)
        print(f"  {t}: done")
    del cross_gram

    # EVALUATE
    print(f"\n{'='*60}\nEVALUATION — {len(tst)} texts\n{'='*60}")
    t0 = time.time()
    rdf = evaluate(model, el, ema, mm, ridge_sols, multi_sols, elvs, multi_layers, nl)
    et = time.time() - t0; print(f"Evaluation: {et:.1f}s")

    rdf.insert(0, "model_name", args.model_name)
    rdf.to_csv(od / f"{ms}_per_layer_results.csv", index=False)

    # Summary
    print(f"\n{'='*60}\nBEST PER STRATEGY\n{'='*60}")
    for s in rdf["strategy"].unique():
        sdf = rdf[rdf["strategy"] == s]
        for t in TASKS:
            tdf = sdf[sdf["task"] == t]
            if len(tdf) == 0: continue
            b = tdf.loc[tdf["text_balanced_acc"].idxmax()]
            print(f"  {s:>15} | {t:>10} | L{b['layer']:>3} | {b['text_balanced_acc']:.4f}")

    print(f"\nTotal: {tt+et:.0f}s (train={tt:.0f}s, eval={et:.0f}s)")

if __name__ == "__main__":
    main()
