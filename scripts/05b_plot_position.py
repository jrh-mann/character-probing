#!/usr/bin/env python3
"""Plot probe accuracy vs token position (% of text seen)."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#444', 'axes.labelcolor': '#ddd',
    'text.color': '#ddd', 'xtick.color': '#aaa', 'ytick.color': '#aaa',
    'grid.color': '#333', 'grid.alpha': 0.5,
})

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS = BASE_DIR / "results"
FIGS = BASE_DIR / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

TASK_COLORS = {"age_bin": "#FF6B6B", "gender": "#4ECDC4", "star_sign": "#FFE66D"}
SIZE_COLORS = {"0.5B": "#66c2a5", "1.5B": "#fc8d62", "3B": "#8da0cb", "7B": "#e78ac3", "14B": "#a6d854"}

csvs = sorted(RESULTS.glob("*_position_accuracy.csv"))
if not csvs:
    print("No position accuracy CSVs found."); exit()

all_df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
print(f"Loaded {len(csvs)} position accuracy files, {len(all_df)} rows")

# Clean up model names
def short_name(m):
    return m.split("/")[-1].replace("Qwen2.5-", "")

def get_size(m):
    for k in ["0.5B", "1.5B", "3B", "7B", "14B"]:
        if k in m: return k
    return "?"

all_df["short"] = all_df["model_name"].apply(short_name)
all_df["size"] = all_df["model_name"].apply(get_size)

# For each model, pick the best layer (highest mean accuracy for age_bin)
best_layers = {}
for model in all_df["short"].unique():
    mdf = all_df[(all_df["short"] == model) & (all_df["task"] == "age_bin")]
    if len(mdf) == 0: continue
    layer_acc = mdf.groupby("layer")["accuracy"].mean()
    best_layers[model] = layer_acc.idxmax()

print("Best layers:", best_layers)

# ── Plot 1: Best layer per model, all tasks ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Probe Accuracy vs Position in Text (best layer per model)", fontsize=16, fontweight='bold')

for ax, task in zip(axes, ["age_bin", "gender", "star_sign"]):
    ax.set_title(task.replace("_", " ").title(), fontsize=14, color=TASK_COLORS[task])
    ax.set_xlabel("Position in Text (%)"); ax.set_ylabel("Token-Level Accuracy")
    ax.grid(True, alpha=0.3)
    chance = {"age_bin": 1/3, "gender": 0.5, "star_sign": 1/12}
    ax.axhline(chance[task], color='white', linestyle=':', alpha=0.4)

    for model in sorted(all_df["short"].unique()):
        if model not in best_layers: continue
        bl = best_layers[model]
        mdf = all_df[(all_df["short"] == model) & (all_df["task"] == task) & (all_df["layer"] == bl)]
        if len(mdf) == 0: continue
        mdf = mdf.sort_values("position_pct")
        size = get_size(model)
        is_it = "Instruct" in model
        is_chat = "chat" in model
        ls = ":" if is_chat else ("--" if is_it else "-")
        ax.plot(mdf["position_pct"], mdf["accuracy"], linewidth=1.8,
                color=SIZE_COLORS.get(size, "#999"), linestyle=ls,
                marker="o", markersize=3, label=model)
    ax.legend(fontsize=7, loc="best", facecolor='#1a1a2e', edgecolor='#444')

plt.tight_layout()
fig.savefig(FIGS / "position_accuracy.png", dpi=150, bbox_inches="tight")
fig.savefig(FIGS / "position_accuracy.pdf", bbox_inches="tight")
print(f"Saved {FIGS / 'position_accuracy.png'}")

# ── Plot 2: Per model size, all layers, age_bin only ─────────────────────
# Show how different layers build up signal across position
base_models = [m for m in all_df["short"].unique() if "Instruct" not in m and "chat" not in m]
if base_models:
    n = len(base_models)
    fig2, axes2 = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1: axes2 = [axes2]
    fig2.suptitle("Age Probe Accuracy vs Position — By Layer (base models)", fontsize=16, fontweight='bold')

    layer_cmap = plt.cm.viridis

    for ax, model in zip(axes2, sorted(base_models)):
        ax.set_title(f"Qwen2.5-{model}", fontsize=14)
        ax.set_xlabel("Position in Text (%)"); ax.set_ylabel("Token-Level Accuracy")
        ax.grid(True, alpha=0.3)
        ax.axhline(1/3, color='white', linestyle=':', alpha=0.4)

        mdf = all_df[(all_df["short"] == model) & (all_df["task"] == "age_bin")]
        layers = sorted(mdf["layer"].unique())
        for i, l in enumerate(layers):
            ldf = mdf[mdf["layer"] == l].sort_values("position_pct")
            color = layer_cmap(i / max(len(layers) - 1, 1))
            ax.plot(ldf["position_pct"], ldf["accuracy"], linewidth=1.5,
                    color=color, marker="o", markersize=2, label=f"L{l}")
        ax.legend(fontsize=8, loc="best", facecolor='#1a1a2e', edgecolor='#444')

    plt.tight_layout()
    fig2.savefig(FIGS / "position_accuracy_by_layer.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIGS / 'position_accuracy_by_layer.png'}")
