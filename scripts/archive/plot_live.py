#!/usr/bin/env python3
"""Plot training losses, accuracies, and scaling curves from all available CSVs. Run anytime."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import sys

plt.rcParams.update({
    'figure.facecolor': '#1a1a2e', 'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#444', 'axes.labelcolor': '#ddd',
    'text.color': '#ddd', 'xtick.color': '#aaa', 'ytick.color': '#aaa',
    'grid.color': '#333', 'grid.alpha': 0.5,
})

TASK_COLORS = {"age_bin": "#FF6B6B", "gender": "#4ECDC4", "star_sign": "#FFE66D"}
CHANCE = {"age_bin": 1/3, "gender": 0.5, "star_sign": 1/12}
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS = BASE_DIR / "results"
FIGS = BASE_DIR / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────

def model_short_name(name):
    """Extract short display name from model_name or filename."""
    s = name.split("/")[-1]
    for prefix in ["Qwen2.5-", "Qwen3-", "gemma-3-", "Llama-3.2-", "Llama-3.1-", "Meta-Llama-3-"]:
        s = s.replace(prefix, "")
    return s.replace("-Base", "").replace("-pt", "")

def model_size_B(name):
    """Extract approximate size in billions from model name."""
    s = name.split("/")[-1]
    m = re.search(r'(\d+\.?\d*)[Bb]', s)
    if m:
        return float(m.group(1))
    # Handle Gemma naming like "gemma-3-1b-pt"
    m = re.search(r'-(\d+)b', s, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Handle "270m" etc
    m = re.search(r'(\d+)[Mm]', s)
    if m:
        return float(m.group(1)) / 1000
    return 0

def model_family(name):
    """Extract model family from name."""
    s = name.split("/")[-1] if "/" in name else name
    if "Qwen2.5" in s: return "Qwen2.5"
    if "Qwen3" in s: return "Qwen3"
    if "gemma" in s.lower(): return "Gemma3"
    if "llama" in s.lower() or "Llama" in s: return "Llama"
    return "Other"

FAMILY_COLORS = {
    "Qwen2.5": "#4ECDC4",
    "Qwen3": "#FF6B6B",
    "Gemma3": "#FFE66D",
    "Llama": "#B39DDB",
    "Other": "#999999",
}

# ── Training loss/accuracy plots (long-format CSVs) ──────────────────────

csvs = sorted(RESULTS.glob("*_training_log.csv"))
if csvs:
    print(f"Found {len(csvs)} training log files")

    # Use the default EMA config for plotting
    DEFAULT_CONFIG = "ema_lr1e-2"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Training Loss (default config, best layer per model)", fontsize=16, fontweight='bold')

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle("Training Accuracy (default config, best layer)", fontsize=16, fontweight='bold')

    for task_i, task in enumerate(["age_bin", "gender", "star_sign"]):
        ax_loss = axes[task_i]
        ax_acc = axes2[task_i]
        for ax in (ax_loss, ax_acc):
            ax.set_title(task.replace("_", " ").title(), fontsize=14, color=TASK_COLORS[task])
            ax.set_xlabel("Batch"); ax.grid(True, alpha=0.3)
        ax_loss.set_ylabel("Loss")
        ax_acc.set_ylabel("Batch Accuracy")
        ax_acc.axhline(CHANCE[task], color='white', linestyle=':', alpha=0.4)

        for csv in csvs:
            name = csv.stem.replace("_training_log", "")
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue

            # Filter to default config and this task
            sub = df[(df["config"] == DEFAULT_CONFIG) & (df["task"] == task)]
            if len(sub) == 0:
                continue

            # Pick best layer (lowest final loss)
            final_loss = sub.groupby("layer")["loss"].apply(lambda x: x.iloc[-max(1, len(x)//10):].mean())
            best_layer = final_loss.idxmin()
            layer_data = sub[sub["layer"] == best_layer].sort_values("batch")

            # Smooth
            w = max(1, len(layer_data) // 50)
            loss_smooth = layer_data["loss"].rolling(w, min_periods=1, center=True).mean().values
            acc_smooth = layer_data["batch_acc"].rolling(w, min_periods=1, center=True).mean().values
            batches = layer_data["batch"].values

            family = model_family(name)
            color = FAMILY_COLORS.get(family, "#999")
            ax_loss.plot(batches, loss_smooth, linewidth=1.5, color=color, label=model_short_name(name))
            ax_acc.plot(batches, acc_smooth, linewidth=1.5, color=color, label=model_short_name(name))

        for ax in (ax_loss, ax_acc):
            ax.legend(fontsize=7, loc="best", facecolor='#1a1a2e', edgecolor='#444')

    plt.figure(fig.number)
    plt.tight_layout()
    fig.savefig(FIGS / "live_training_losses.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIGS / 'live_training_losses.png'}")

    plt.figure(fig2.number)
    plt.tight_layout()
    fig2.savefig(FIGS / "live_training_acc.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIGS / 'live_training_acc.png'}")

    plt.close('all')
else:
    print("No training log CSVs found.")

# ── Results summary + scaling curve ──────────────────────────────────────

result_csvs = sorted(RESULTS.glob("*_per_layer_results.csv"))
if result_csvs:
    print(f"\n{'='*60}")
    print(f"COMPLETED RESULTS ({len(result_csvs)} models)")
    print(f"{'='*60}")

    all_df = pd.concat([pd.read_csv(c) for c in result_csvs], ignore_index=True)

    # Print summary for default EMA config
    default_strat = "ema_lr1e-2"
    for csv in result_csvs:
        df = pd.read_csv(csv)
        model = csv.stem.replace("_per_layer_results", "")
        sdf = df[df["strategy"] == default_strat]
        if len(sdf) == 0:
            # Fallback to old "ema" strategy name
            sdf = df[df["strategy"] == "ema"]
        if len(sdf) == 0:
            continue
        line = f"  {model:>30}"
        for task in ["age_bin", "gender", "star_sign"]:
            tdf = sdf[sdf["task"] == task]
            if len(tdf) == 0:
                line += "     N/A"
            else:
                best = tdf["text_balanced_acc"].max()
                line += f"  {task}={best:.3f}"
        print(line)

    # Print shuffled control comparison
    shuf = all_df[all_df["strategy"] == "shuffled"]
    if len(shuf) > 0:
        print(f"\n  SHUFFLED CONTROL (should be near chance):")
        for task in ["age_bin", "gender", "star_sign"]:
            tdf = shuf[shuf["task"] == task]
            if len(tdf) > 0:
                mean_acc = tdf.groupby("layer")["text_balanced_acc"].max().mean()
                print(f"    {task}: {mean_acc:.3f} (chance={CHANCE[task]:.3f})")

    # ── Scaling curve ────────────────────────────────────────────────
    if len(result_csvs) >= 2:
        all_df["size_B"] = all_df["model_name"].apply(model_size_B)
        all_df["family"] = all_df["model_name"].apply(model_family)

        # Pick best EMA strategy (any config starting with "ema_")
        ema_df = all_df[all_df["strategy"].str.startswith("ema_")].copy()
        # Exclude shuffled
        ema_df = ema_df[ema_df["strategy"] != "shuffled"]

        if len(ema_df) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle("Scaling Curve: Best EMA Probe Accuracy vs Model Size", fontsize=16, fontweight='bold')

            for ax, task in zip(axes, ["age_bin", "gender", "star_sign"]):
                ax.set_title(task.replace("_", " ").title(), fontsize=14, color=TASK_COLORS[task])
                ax.set_xlabel("Model Size (B)"); ax.set_ylabel("Balanced Accuracy")
                ax.set_xscale("log"); ax.grid(True, alpha=0.3)
                ax.axhline(CHANCE[task], color='white', linestyle=':', alpha=0.4)

                tdf = ema_df[ema_df["task"] == task]
                for family in sorted(tdf["family"].unique()):
                    fdf = tdf[tdf["family"] == family]
                    best = fdf.groupby("size_B")["text_balanced_acc"].max().reset_index().sort_values("size_B")
                    if len(best) == 0:
                        continue
                    ax.plot(best["size_B"], best["text_balanced_acc"],
                            marker="o", markersize=8, linewidth=2,
                            color=FAMILY_COLORS.get(family, "#999"),
                            label=family)

                ax.legend(fontsize=9, loc="lower right", facecolor='#1a1a2e', edgecolor='#444')

            plt.tight_layout()
            fig.savefig(FIGS / "scaling_curve.png", dpi=150, bbox_inches="tight")
            fig.savefig(FIGS / "scaling_curve.pdf", bbox_inches="tight")
            print(f"Saved {FIGS / 'scaling_curve.png'}")
            plt.close('all')

    # ── Layer-by-layer accuracy ──────────────────────────────────────
    ema_results = all_df[all_df["strategy"].str.startswith("ema_") & (all_df["strategy"] != "shuffled")].copy()
    if len(ema_results) > 0:
        ema_results["layer"] = pd.to_numeric(ema_results["layer"], errors="coerce")
        # Compute max layer per model to normalize depth
        max_layers = ema_results.groupby("model_name")["layer"].max()
        ema_results["total_layers"] = ema_results["model_name"].map(max_layers)
        ema_results["layer_frac"] = ema_results["layer"] / ema_results["total_layers"]
        ema_results["family"] = ema_results["model_name"].apply(model_family)
        ema_results["short"] = ema_results["model_name"].apply(model_short_name)
        ema_results["size_B"] = ema_results["model_name"].apply(model_size_B)

        # Best accuracy across all EMA configs per (model, layer, task)
        best_per_layer = ema_results.groupby(
            ["model_name", "short", "family", "size_B", "layer", "layer_frac", "total_layers", "task"]
        )["text_balanced_acc"].max().reset_index()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Probe Accuracy by Relative Layer Depth (best EMA config)", fontsize=16, fontweight='bold')

        for ax, task in zip(axes, ["age_bin", "gender", "star_sign"]):
            ax.set_title(task.replace("_", " ").title(), fontsize=14, color=TASK_COLORS[task])
            ax.set_xlabel("Relative Layer Depth"); ax.set_ylabel("Balanced Accuracy")
            ax.grid(True, alpha=0.3)
            ax.axhline(CHANCE[task], color='white', linestyle=':', alpha=0.4)

            tdf = best_per_layer[best_per_layer["task"] == task]
            for _, grp in tdf.groupby("model_name"):
                row = grp.iloc[0]
                color = FAMILY_COLORS.get(row["family"], "#999")
                alpha = 0.4 + 0.6 * min(row["size_B"] / 14, 1)  # bigger = more opaque
                grp_sorted = grp.sort_values("layer_frac")
                ax.plot(grp_sorted["layer_frac"], grp_sorted["text_balanced_acc"],
                        marker="o", markersize=3, linewidth=1.5,
                        color=color, alpha=alpha,
                        label=f"{row['family']} {row['short']}")

            ax.legend(fontsize=6, loc="best", facecolor='#1a1a2e', edgecolor='#444', ncol=2)

        plt.tight_layout()
        fig.savefig(FIGS / "layer_by_layer_acc.png", dpi=150, bbox_inches="tight")
        fig.savefig(FIGS / "layer_by_layer_acc.pdf", bbox_inches="tight")
        print(f"Saved {FIGS / 'layer_by_layer_acc.png'}")
        plt.close('all')

print("\nDone.")
