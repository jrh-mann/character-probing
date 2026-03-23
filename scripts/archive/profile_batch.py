#!/usr/bin/env python3
"""Profile every phase of a single training batch."""
import torch, time
import torch.nn.functional as F
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tok.pad_token = tok.eos_token
tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16, device_map="auto")
model.eval()

# Load real data
df = pd.read_parquet("/workspace/character-probing/data/processed/blog_corpus.parquet")
texts = df[df["split"] == "train"]["text"].tolist()[:256]
enc = tok(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
ids = enc["input_ids"].cuda()
amask = enc["attention_mask"].cuda()
B, S = ids.shape
n_real = amask.sum().item()
print(f"Batch: {B} texts x {S} max tokens, {n_real} non-pad tokens")

D = model.config.hidden_size
nl = model.config.num_hidden_layers
eval_layers = [0, 4, 8, 12, 16, 20, 24]

# Hook setup
captured = {}
hooks = []
inner = model.model
for l in eval_layers:
    def make_hook(li):
        def hook(mod, inp, out):
            captured[li] = out[0] if isinstance(out, tuple) else out
        return hook
    if l == 0:
        h = inner.embed_tokens.register_forward_hook(make_hook(l))
    elif l == nl:
        h = inner.norm.register_forward_hook(make_hook(l))
    else:
        h = inner.layers[l - 1].register_forward_hook(make_hook(l))
    hooks.append(h)

# Warmup
with torch.no_grad():
    model.model(input_ids=ids, attention_mask=amask)
captured.clear()
torch.cuda.synchronize()

timings = {}

# 1. Forward pass
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    model.model(input_ids=ids, attention_mask=amask)
torch.cuda.synchronize()
timings["1_forward_pass"] = time.perf_counter() - t0

# 2. all_real_positions (Python loop with GPU ops)
torch.cuda.synchronize()
t0 = time.perf_counter()
ab, at = [], []
for b in range(B):
    r = amask[b].nonzero(as_tuple=False).squeeze(-1)
    ab.append(torch.full((r.shape[0],), b, device="cuda", dtype=torch.long))
    at.append(r)
bi = torch.cat(ab)
ti = torch.cat(at)
torch.cuda.synchronize()
timings["2_all_real_positions"] = time.perf_counter() - t0
N = bi.shape[0]

# 3. Activation extraction (fancy index + float cast, per layer)
torch.cuda.synchronize()
t0 = time.perf_counter()
acts = {}
for l in eval_layers:
    acts[l] = captured[l][bi, ti].float()
torch.cuda.synchronize()
timings["3_extract_7_layers"] = time.perf_counter() - t0

# 4. EMA-style matmul updates WITH .item()
dummy_y = torch.randint(0, 3, (N,), device="cuda")
W = torch.randn(3, D, device="cuda") * 0.01
b_param = torch.zeros(3, device="cuda")

torch.cuda.synchronize()
t0 = time.perf_counter()
for l in eval_layers:
    a = acts[l]
    for task_i in range(3):
        for cfg_i in range(7):  # 7 EMA configs
            lo = a @ W.T + b_param
            loss_val = F.cross_entropy(lo, dummy_y).item()
            acc_val = (lo.argmax(-1) == dummy_y).sum().item() / N
torch.cuda.synchronize()
timings["4a_147_ema_WITH_item"] = time.perf_counter() - t0

torch.cuda.synchronize()
t0 = time.perf_counter()
for l in eval_layers:
    a = acts[l]
    for task_i in range(3):
        for cfg_i in range(7):
            lo = a @ W.T + b_param
            loss_val = F.cross_entropy(lo, dummy_y)
            acc_val = (lo.argmax(-1) == dummy_y).float().mean()
torch.cuda.synchronize()
timings["4b_147_ema_NO_item"] = time.perf_counter() - t0

# 5. MLP backward updates WITH .item()
mlp_net = torch.nn.Sequential(torch.nn.Linear(D, 128), torch.nn.ReLU(), torch.nn.Linear(128, 3)).cuda()
mlp_opt = torch.optim.Adam(mlp_net.parameters(), lr=1e-3)

torch.cuda.synchronize()
t0 = time.perf_counter()
for l in eval_layers:
    a = acts[l].detach()
    for task_i in range(3):
        for h_i in range(4):  # 4 hidden sizes
            mlp_opt.zero_grad()
            lo = mlp_net(a)
            loss = F.cross_entropy(lo, dummy_y)
            loss.backward()
            mlp_opt.step()
            _ = loss.item()
            _ = (lo.argmax(-1) == dummy_y).float().mean().item()
torch.cuda.synchronize()
timings["5a_84_mlp_WITH_item"] = time.perf_counter() - t0

torch.cuda.synchronize()
t0 = time.perf_counter()
for l in eval_layers:
    a = acts[l].detach()
    for task_i in range(3):
        for h_i in range(4):
            mlp_opt.zero_grad()
            lo = mlp_net(a)
            loss = F.cross_entropy(lo, dummy_y)
            loss.backward()
            mlp_opt.step()
torch.cuda.synchronize()
timings["5b_84_mlp_NO_item"] = time.perf_counter() - t0

# 6. Ridge addmm
torch.cuda.synchronize()
t0 = time.perf_counter()
for l in eval_layers:
    A_gram = torch.zeros(D, D, device="cuda")
    A_gram.addmm_(acts[l].T, acts[l])
torch.cuda.synchronize()
timings["6_ridge_7_addmm"] = time.perf_counter() - t0

# 7. Shuffled control (21 probes with .item)
torch.cuda.synchronize()
t0 = time.perf_counter()
for l in eval_layers:
    a = acts[l]
    for task_i in range(3):
        perm = torch.randperm(N, device="cuda")
        lo = a @ W.T + b_param
        _ = F.cross_entropy(lo, dummy_y[perm]).item()
        _ = (lo.argmax(-1) == dummy_y[perm]).sum().item() / N
torch.cuda.synchronize()
timings["7_21_shuffled_WITH_item"] = time.perf_counter() - t0

# 8. Logging
t0 = time.perf_counter()
rows = []
for i in range(252):
    rows.append({"batch": 0, "layer": 0, "task": "x", "config": "y",
                 "lr": 0.01, "wd": 1e-4, "loss": 0.5, "batch_acc": 0.3, "n_tokens": N})
timings["8_logging_252_dicts"] = time.perf_counter() - t0

# Print
sep = "=" * 60
print(f"\n{sep}")
print(f"PROFILE RESULTS")
print(sep)
total = 0
for k, v in sorted(timings.items()):
    print(f"  {k:40s} {v * 1000:8.1f} ms")
    total += v
print(f"  {'---':40s} {'---':>8s}")
print(f"  {'TOTAL':40s} {total * 1000:8.1f} ms")

print(f"\n  .item() cost:")
ema_diff = timings["4a_147_ema_WITH_item"] - timings["4b_147_ema_NO_item"]
mlp_diff = timings["5a_84_mlp_WITH_item"] - timings["5b_84_mlp_NO_item"]
print(f"    EMA 147 probes: {ema_diff * 1000:.1f} ms ({ema_diff / timings['4a_147_ema_WITH_item'] * 100:.0f}% of EMA time)")
print(f"    MLP 84 probes:  {mlp_diff * 1000:.1f} ms ({mlp_diff / timings['5a_84_mlp_WITH_item'] * 100:.0f}% of MLP time)")
print(f"    Total .item():  {(ema_diff + mlp_diff) * 1000:.1f} ms")

no_item_total = total - ema_diff - mlp_diff
print(f"\n  Projected batch time without .item(): {no_item_total * 1000:.0f} ms")
print(f"  Projected batch time current:         {total * 1000:.0f} ms")
print(f"  Forward pass alone:                   {timings['1_forward_pass'] * 1000:.0f} ms")

for h in hooks:
    h.remove()
