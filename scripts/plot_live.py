#!/usr/bin/env python3
"""Plot training losses and accuracies from all available loss CSVs. Run anytime."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

plt.rcParams.update({
    'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#444', 'axes.labelcolor': '#ddd',
    'text.color': '#ddd', 'xtick.color': '#aaa', 'ytick.color': '#aaa',
    'grid.color': '#333', 'grid.alpha': 0.5,
})

TASK_COLORS = {"age_bin": "#FF6B6B", "gender": "#4ECDC4", "star_sign": "#FFE66D"}
RESULTS = Path("/workspace/characterprobing/results")
FIGS = Path("/root/characterprobing/figures")
FIGS.mkdir(parents=True, exist_ok=True)

csvs = sorted(RESULTS.glob("*_training_losses.csv"))
if not csvs:
    print("No training loss CSVs found."); sys.exit(0)

print(f"Found {len(csvs)} loss files")

# Find which layers are logged (varies by model)
def get_logged_layers(df):
    cols = [c for c in df.columns if c.startswith("loss_L")]
    return sorted(set(int(c.split("_")[1][1:]) for c in cols))

# Plot losses
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Training Loss (best logged layer per model)", fontsize=16, fontweight='bold')

for ax, task in zip(axes, ["age_bin", "gender", "star_sign"]):
    ax.set_title(task.replace("_", " ").title(), fontsize=14, color=TASK_COLORS[task])
    ax.set_xlabel("Batch (log)"); ax.set_ylabel("Loss"); ax.set_xscale("log"); ax.grid(True, alpha=0.3)

    for csv in csvs:
        name = csv.stem.replace("_training_losses", "")
        df = pd.read_csv(csv)
        layers = get_logged_layers(df)
        best_l, best_loss = layers[0], 999
        for l in layers:
            col = f"loss_L{l}_{task}"
            if col in df.columns:
                fl = df[col].iloc[-max(1, len(df)//10):].mean()
                if fl < best_loss:
                    best_l, best_loss = l, fl

        col = f"loss_L{best_l}_{task}"
        if col not in df.columns: continue
        raw = df[col].values
        w = max(1, len(raw) // 50)
        smooth = pd.Series(raw).rolling(w, min_periods=1, center=True).mean().values

        is_it = "Instruct" in name or "chat" in name
        size = name.replace("Qwen2.5-", "").replace("-Instruct", "").replace("_chat", "")
        ax.plot(range(len(smooth)), smooth, linewidth=1.5,
                linestyle="--" if is_it else "-",
                label=f"{size}{'(IT)' if 'Instruct' in name else ''}{' chat' if 'chat' in name else ''}")
    ax.legend(fontsize=8, loc="upper right", facecolor='#1a1a2e', edgecolor='#444')

plt.tight_layout()
fig.savefig(FIGS / "live_training_losses.png", dpi=150, bbox_inches="tight")
print(f"Saved {FIGS / 'live_training_losses.png'}")

# Plot accuracies (if available)
has_acc = False
for csv in csvs:
    df = pd.read_csv(csv)
    if any("acc_L" in c for c in df.columns):
        has_acc = True; break

if has_acc:
    # Batch sizes used per model size (from auto_bs)
    MODEL_BS = {"0.5B": 128, "1.5B": 128, "3B": 64, "7B": 32, "14B": 16}
    MAX_SEQ = 1024

    def get_bs(name):
        for k in MODEL_BS:
            if k in name: return MODEL_BS[k]
        return 64

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle("Running Train Accuracy vs Tokens Seen (best logged layer)", fontsize=16, fontweight='bold')

    for ax, task in zip(axes2, ["age_bin", "gender", "star_sign"]):
        ax.set_title(task.replace("_", " ").title(), fontsize=14, color=TASK_COLORS[task])
        ax.set_xlabel("Tokens (log)"); ax.set_ylabel("Accuracy"); ax.set_xscale("log"); ax.grid(True, alpha=0.3)
        chance = {"age_bin": 1/3, "gender": 0.5, "star_sign": 1/12}
        ax.axhline(chance[task], color='white', linestyle=':', alpha=0.4)

        for csv in csvs:
            name = csv.stem.replace("_training_losses", "")
            df = pd.read_csv(csv)
            layers = get_logged_layers(df)
            best_l = layers[-1]  # use deepest layer for acc
            col = f"acc_L{best_l}_{task}"
            if col not in df.columns: continue

            bs = get_bs(name)
            tokens = df["batch"].values * bs * MAX_SEQ

            is_it = "Instruct" in name or "chat" in name
            size = name.replace("Qwen2.5-", "").replace("-Instruct", "").replace("_chat", "")
            ax.plot(tokens, df[col].values, linewidth=1.5,
                    linestyle="--" if is_it else "-",
                    label=f"{size}{'(IT)' if 'Instruct' in name else ''}{'chat' if 'chat' in name else ''}")
        ax.legend(fontsize=8, loc="lower right", facecolor='#1a1a2e', edgecolor='#444')

    plt.tight_layout()
    fig2.savefig(FIGS / "live_training_acc.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIGS / 'live_training_acc.png'}")

    # ── Per-model-size comparison: base vs instruct vs chat ────────────
    import re
    SIZES = ["0.5B", "1.5B", "3B", "7B"]
    TASK_LINES = {"age_bin": "-", "gender": "--", "star_sign": ":"}
    VARIANT_COLORS = {"base": "#4ECDC4", "instruct": "#FF6B6B", "chat": "#FFE66D"}

    # Group CSVs by model size
    size_groups = {s: [] for s in SIZES}
    for csv in csvs:
        name = csv.stem.replace("_training_losses", "")
        for s in SIZES:
            if s in name:
                # Determine variant
                if "_chat" in name:
                    variant = "chat"
                elif "Instruct" in name:
                    variant = "instruct"
                else:
                    variant = "base"
                size_groups[s].append((csv, name, variant))
                break

    active_sizes = [s for s in SIZES if size_groups[s]]
    n_sizes = len(active_sizes)
    if n_sizes > 0:
        fig3, axes3 = plt.subplots(1, n_sizes, figsize=(6 * n_sizes, 6))
        if n_sizes == 1: axes3 = [axes3]
        fig3.suptitle("Accuracy vs Tokens — Base vs Instruct vs Chat (by model size)",
                       fontsize=16, fontweight='bold')

        for ax, size in zip(axes3, active_sizes):
            ax.set_title(f"Qwen2.5-{size}", fontsize=14)
            ax.set_xlabel("Tokens"); ax.set_ylabel("Accuracy")
            ax.set_xscale("log"); ax.grid(True, alpha=0.3)
            # Chance lines
            for task, ch in [("age_bin", 1/3), ("gender", 0.5), ("star_sign", 1/12)]:
                ax.axhline(ch, color='grey', linestyle=':', alpha=0.3, linewidth=0.8)

            for csv_f, name, variant in size_groups[size]:
                df = pd.read_csv(csv_f)
                layers = get_logged_layers(df)
                best_l = layers[-1]
                bs = get_bs(name)
                tokens = df["batch"].values * bs * MAX_SEQ

                for task, ls in TASK_LINES.items():
                    col = f"acc_L{best_l}_{task}"
                    if col not in df.columns: continue
                    # Smooth a bit
                    raw = df[col].values
                    w = max(1, len(raw) // 80)
                    smooth = pd.Series(raw).rolling(w, min_periods=1, center=True).mean().values
                    ax.plot(tokens, smooth, linewidth=1.8,
                            color=VARIANT_COLORS[variant], linestyle=ls,
                            label=f"{variant} · {task.replace('_',' ')}")

            # De-duplicate legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=7,
                      loc="upper left", facecolor='#1a1a2e', edgecolor='#444')

        plt.tight_layout()
        fig3.savefig(FIGS / "acc_by_model_size.png", dpi=150, bbox_inches="tight")
        print(f"Saved {FIGS / 'acc_by_model_size.png'}")

# Also print a quick summary table of completed results
result_csvs = sorted(RESULTS.glob("*_per_layer_results.csv"))
if result_csvs:
    print(f"\n{'='*60}")
    print(f"COMPLETED RESULTS ({len(result_csvs)} models)")
    print(f"{'='*60}")
    for csv in result_csvs:
        df = pd.read_csv(csv)
        model = csv.stem.replace("_per_layer_results", "")
        for strat in ["ema", "ridge_1.0", "multi_1.0"]:
            sdf = df[df["strategy"] == strat]
            if len(sdf) == 0: continue
            line = f"  {model:>30} [{strat:>10}]"
            for task in ["age_bin", "gender", "star_sign"]:
                tdf = sdf[sdf["task"] == task]
                if len(tdf) == 0: line += "     N/A"; continue
                best = tdf["text_balanced_acc"].max()
                line += f"  {task}={best:.3f}"
            print(line)

# ── Scaling curve ────────────────────────────────────────────────────────
if len(result_csvs) >= 2:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#1a1a2e',
        'axes.edgecolor': '#444', 'axes.labelcolor': '#ddd',
        'text.color': '#ddd', 'xtick.color': '#aaa', 'ytick.color': '#aaa',
        'grid.color': '#333', 'grid.alpha': 0.5,
    })
    all_df = pd.concat([pd.read_csv(c) for c in result_csvs], ignore_index=True)
    def get_size(n):
        s = n.split("/")[-1].replace("Qwen2.5-","").replace("-Instruct","").replace("_chat","")
        return float(s.replace("B",""))
    all_df["size_B"] = all_df["model_name"].apply(get_size)
    all_df["is_instruct"] = all_df["model_name"].str.contains("Instruct")

    strats = {"ema": ("EMA Linear","o"), "mm": ("Mass-Mean","s"), "multi_1.0": ("Multi-layer Ridge","D")}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Scaling Curve: Probe Accuracy vs Model Size", fontsize=16, fontweight='bold')
    for ax, task in zip(axes, ["age_bin", "gender", "star_sign"]):
        ax.set_title(task.replace("_"," ").title(), fontsize=14, color=TASK_COLORS[task])
        ax.set_xlabel("Model Size (B)"); ax.set_ylabel("Balanced Accuracy")
        ax.set_xscale("log"); ax.grid(True, alpha=0.3)
        chance = {"age_bin":1/3,"gender":0.5,"star_sign":1/12}
        ax.axhline(chance[task], color='white', linestyle=':', alpha=0.4)
        base = all_df[~all_df["is_instruct"]]
        for st, (label, marker) in strats.items():
            tdf = base[(base["strategy"]==st) & (base["task"]==task)]
            if len(tdf)==0: continue
            best = tdf.groupby("size_B")["text_balanced_acc"].max().reset_index().sort_values("size_B")
            ax.plot(best["size_B"], best["text_balanced_acc"], marker=marker, markersize=8, linewidth=2, label=label)
        ax.legend(fontsize=9, loc="lower right", facecolor='#1a1a2e', edgecolor='#444')
    plt.tight_layout()
    fig.savefig(FIGS / "scaling_curve.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "scaling_curve.pdf", bbox_inches="tight")
    print(f"Saved {FIGS / 'scaling_curve.png'}")
