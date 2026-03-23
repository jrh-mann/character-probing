#!/usr/bin/env python3
"""Generate publication-quality paper figures.

Handles all 5 model families (Pythia, Qwen2.5, Qwen3, Gemma 3, Llama 3)
across 6 datasets (blog, hippocorpus, ellipse, prism, synthpai, europarl).
"""
import re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
FIGS = BASE / "figures" / "paper_figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Dataset config: (subdir, tasks_with_real_labels)
# Europarl has age_bin=1 for all texts → age is meaningless there
DATASETS = {
    "Blog Corpus":  ("",          ["age_bin", "gender"]),
    "Hippocorpus":  ("hippocorpus", ["age_bin", "gender"]),
    "ELLIPSE":      ("ellipse",   ["gender"]),          # "age_bin" is school grade, not age
    "PRISM":        ("prism",     ["age_bin", "gender"]),
    "Europarl":     ("europarl",  ["gender"]),           # age_bin is constant=1
    "SynthPAI":     ("synthpai",  ["age_bin", "gender"]),
}

DATASET_COLORS = {
    "Blog Corpus": "#2196F3",
    "Hippocorpus": "#4CAF50",
    "ELLIPSE": "#FF9800",
    "PRISM": "#9C27B0",
    "Europarl": "#F44336",
    "SynthPAI": "#607D8B",
}
DATASET_MARKERS = {
    "Blog Corpus": "o",
    "Hippocorpus": "s",
    "ELLIPSE": "D",
    "PRISM": "^",
    "Europarl": "v",
    "SynthPAI": "x",
}

# Model param counts (billions)
MODEL_PARAMS = {
    "pythia-14m": 0.014, "pythia-31m": 0.031, "pythia-70m": 0.070,
    "pythia-160m": 0.160, "pythia-410m": 0.410, "pythia-1b": 1.0,
    "Qwen2.5-0.5B": 0.5, "Qwen2.5-1.5B": 1.5, "Qwen2.5-3B": 3.0,
    "Qwen2.5-7B": 7.0, "Qwen2.5-14B": 14.0,
    "Qwen3-0.6B-Base": 0.6, "Qwen3-1.7B-Base": 1.7,
    "Qwen3-4B-Base": 4.0, "Qwen3-8B-Base": 8.0,
    "gemma-3-270m": 0.27, "gemma-3-1b-pt": 1.0,
    "gemma-3-4b-pt": 4.0, "gemma-3-12b-pt": 12.0,
    "Llama-3.2-1B": 1.2, "Llama-3.2-3B": 3.2, "Llama-3.1-8B": 8.0,
}

FAMILY_MAP = {}
for m in MODEL_PARAMS:
    if "pythia" in m: FAMILY_MAP[m] = "Pythia"
    elif "Qwen2.5" in m: FAMILY_MAP[m] = "Qwen 2.5"
    elif "Qwen3" in m: FAMILY_MAP[m] = "Qwen 3"
    elif "gemma" in m: FAMILY_MAP[m] = "Gemma 3"
    elif "Llama" in m: FAMILY_MAP[m] = "Llama 3"

FAMILY_COLORS = {
    "Pythia": "#e41a1c", "Qwen 2.5": "#377eb8", "Qwen 3": "#4daf4a",
    "Gemma 3": "#984ea3", "Llama 3": "#ff7f00",
}
FAMILY_MARKERS = {
    "Pythia": "o", "Qwen 2.5": "s", "Qwen 3": "D",
    "Gemma 3": "^", "Llama 3": "v",
}

CHANCE = {"age_bin": 1/3, "gender": 0.5}
TASK_LABELS = {"age_bin": "Age (3-class)", "gender": "Gender (2-class)"}


def model_short(name):
    return name.rstrip("/").split("/")[-1]


def load_dataset_results(subdir):
    """Load per_layer_results CSVs from a dataset directory."""
    d = RESULTS / subdir if subdir else RESULTS
    csvs = sorted(d.glob("*_per_layer_results.csv"))
    if not csvs:
        return pd.DataFrame()
    frames = []
    for f in csvs:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def best_ridge_acc(df, task):
    """Get best ridge accuracy per model for a task (val-selected preferred)."""
    # Filter to ridge strategies
    ridge = df[(df["task"] == task) & (df["strategy"].str.startswith("ridge_"))].copy()
    if ridge.empty:
        return pd.DataFrame()

    # Prefer val_selected if available
    if "val_selected" in ridge.columns:
        val_sel = ridge[ridge["val_selected"] == True]
        if not val_sel.empty:
            ridge = val_sel

    # Best layer per model
    ridge["short"] = ridge["model_name"].apply(model_short)
    best = ridge.groupby("short")["text_balanced_acc"].max().reset_index()
    best["n_params"] = best["short"].map(MODEL_PARAMS)
    best["family"] = best["short"].map(FAMILY_MAP)
    best = best.dropna(subset=["n_params"])
    return best


def fig_scaling_law():
    """Main scaling law figure: all models on blog dataset, both tasks."""
    df = load_dataset_results("")
    if df.empty:
        print("No blog results"); return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Scaling Law: Probe Accuracy vs Model Parameters", fontsize=14, fontweight="bold")

    for ax, task in zip(axes, ["age_bin", "gender"]):
        best = best_ridge_acc(df, task)
        if best.empty:
            continue

        # Plot by family
        for family, color in FAMILY_COLORS.items():
            fdf = best[best["family"] == family]
            if fdf.empty: continue
            ax.scatter(fdf["n_params"], fdf["text_balanced_acc"],
                      color=color, marker=FAMILY_MARKERS[family], s=60,
                      label=family, zorder=3, edgecolors="white", linewidths=0.5)

        # Fit log-linear
        x = np.log10(best["n_params"].values)
        y = best["text_balanced_acc"].values
        slope, intercept, r, p, se = stats.linregress(x, y)
        x_fit = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
        ax.plot(10**x_fit, slope * x_fit + intercept, "k--", linewidth=1.5, alpha=0.7)
        ax.text(0.05, 0.95, f"R²={r**2:.3f}\nslope={slope:.3f}/decade\nn={len(best)}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        ax.axhline(CHANCE[task], color="gray", linestyle=":", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (billions)")
        ax.set_ylabel("Balanced Accuracy (best layer, ridge)")
        ax.set_title(TASK_LABELS[task])
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGS / "paper_scaling_law.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "paper_scaling_law.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved paper_scaling_law")


def fig_6datasets(task):
    """6-dataset scaling figure for a specific task."""
    fig, ax = plt.subplots(figsize=(8, 6))
    title_task = TASK_LABELS.get(task, task)
    ax.set_title(f"Cross-Dataset Scaling: {title_task}", fontsize=13, fontweight="bold")

    for ds_name, (subdir, valid_tasks) in DATASETS.items():
        if task not in valid_tasks:
            continue
        df = load_dataset_results(subdir)
        if df.empty:
            continue
        best = best_ridge_acc(df, task)
        if len(best) < 3:
            continue

        color = DATASET_COLORS[ds_name]
        marker = DATASET_MARKERS[ds_name]
        ax.scatter(best["n_params"], best["text_balanced_acc"],
                  color=color, marker=marker, s=50, zorder=3,
                  edgecolors="white", linewidths=0.3)

        # Fit
        x = np.log10(best["n_params"].values)
        y = best["text_balanced_acc"].values
        slope, intercept, r, p, se = stats.linregress(x, y)
        x_fit = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
        ax.plot(10**x_fit, slope * x_fit + intercept, color=color,
                linestyle="--", linewidth=1.2, alpha=0.7)
        label = f"{ds_name} (R²={r**2:.2f}, n={len(best)})"
        # Re-scatter with label for legend
        ax.scatter([], [], color=color, marker=marker, s=50, label=label)

    ax.axhline(CHANCE[task], color="gray", linestyle=":", alpha=0.5,
               label=f"Chance ({CHANCE[task]:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (billions)")
    ax.set_ylabel("Balanced Accuracy (best layer, ridge)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    suffix = task.replace("_", "")
    fig.savefig(FIGS / f"paper_6datasets_{suffix}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / f"paper_6datasets_{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved paper_6datasets_{suffix}")


def fig_layer_curves():
    """All models' layer curves on blog dataset, colored by model size."""
    df = load_dataset_results("")
    if df.empty:
        return

    # Filter to best ridge strategy per model
    ridge = df[df["strategy"].str.startswith("ridge_")].copy()
    if "val_selected" in ridge.columns:
        vs = ridge[ridge["val_selected"] == True]
        if not vs.empty: ridge = vs

    ridge["short"] = ridge["model_name"].apply(model_short)
    ridge["n_params"] = ridge["short"].map(MODEL_PARAMS)
    ridge["family"] = ridge["short"].map(FAMILY_MAP)
    ridge = ridge.dropna(subset=["n_params"])

    # Normalized layer
    # Layer column might be string with "all" entries — filter to numeric
    ridge = ridge[ridge["layer"] != "all"].copy()
    ridge["layer"] = ridge["layer"].astype(int)

    for model in ridge["short"].unique():
        mask = ridge["short"] == model
        max_l = ridge.loc[mask, "layer"].max()
        if max_l > 0:
            ridge.loc[mask, "norm_layer"] = ridge.loc[mask, "layer"] / max_l
        else:
            ridge.loc[mask, "norm_layer"] = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Layer-wise Probe Accuracy (all models)", fontsize=14, fontweight="bold")

    cmap = plt.cm.viridis
    log_params = np.log10(ridge["n_params"])
    norm = plt.Normalize(log_params.min(), log_params.max())

    for ax, task in zip(axes, ["age_bin", "gender"]):
        td = ridge[ridge["task"] == task]
        # Best strategy per (model, layer)
        td = td.groupby(["short", "layer", "norm_layer", "n_params", "family"])["text_balanced_acc"].max().reset_index()

        for model in sorted(td["short"].unique(), key=lambda m: MODEL_PARAMS.get(m, 0)):
            mdf = td[td["short"] == model].sort_values("norm_layer")
            if mdf.empty: continue
            color = cmap(norm(np.log10(mdf["n_params"].iloc[0])))
            ax.plot(mdf["norm_layer"], mdf["text_balanced_acc"],
                    color=color, linewidth=1.2, alpha=0.8, marker=".", markersize=3)

        ax.axhline(CHANCE[task], color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Normalized Layer Depth")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(TASK_LABELS[task])
        ax.grid(True, alpha=0.3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("log₁₀(params in B)")

    fig.tight_layout()
    fig.savefig(FIGS / "paper_layer_all_models.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "paper_layer_all_models.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved paper_layer_all_models")


def fig_signal_decomposition():
    """Signal decomposition: chance → BoW → embedding → best layer."""
    df = load_dataset_results("")
    if df.empty: return

    # BoW baselines
    bow_path = RESULTS / "bow_baseline_results.csv"
    bow = {}
    if bow_path.exists():
        bdf = pd.read_csv(bow_path)
        # Best BoW per task
        for task_name in ["age_bin", "gender"]:
            tdf = bdf[bdf["task"] == task_name]
            if not tdf.empty:
                bow[task_name] = tdf["test_balanced_acc"].max()

    ridge = df[df["strategy"].str.startswith("ridge_")].copy()
    if "val_selected" in ridge.columns:
        vs = ridge[ridge["val_selected"] == True]
        if not vs.empty: ridge = vs

    ridge["short"] = ridge["model_name"].apply(model_short)
    ridge["n_params"] = ridge["short"].map(MODEL_PARAMS)
    ridge = ridge.dropna(subset=["n_params"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Signal Decomposition", fontsize=14, fontweight="bold")

    for ax, task in zip(axes, ["age_bin", "gender"]):
        models_sorted = sorted(ridge["short"].unique(), key=lambda m: MODEL_PARAMS.get(m, 0))
        l0_accs, best_accs, names = [], [], []

        for model in models_sorted:
            mdf = ridge[(ridge["short"] == model) & (ridge["task"] == task)]
            if mdf.empty: continue
            l0 = mdf[mdf["layer"] == 0]["text_balanced_acc"].max()
            best = mdf["text_balanced_acc"].max()
            if pd.isna(l0) or pd.isna(best): continue
            l0_accs.append(l0)
            best_accs.append(best)
            names.append(model)

        if not names: continue
        y = np.arange(len(names))
        l0_arr = np.array(l0_accs)
        best_arr = np.array(best_accs)
        chance = CHANCE[task]

        ax.barh(y, l0_arr - chance, left=chance, color="#4FC3F7", label="Embedding (L0)")
        ax.barh(y, best_arr - l0_arr, left=l0_arr, color="#81C784", label="Contextual gain")
        ax.axvline(chance, color="gray", linestyle=":", label=f"Chance ({chance:.2f})")
        if task in bow:
            ax.axvline(bow[task], color="#FF7043", linestyle="--", linewidth=2,
                       label=f"BoW baseline ({bow[task]:.3f})")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Balanced Accuracy")
        ax.set_title(TASK_LABELS[task])
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(FIGS / "paper_signal_decomposition.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "paper_signal_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved paper_signal_decomposition")


def fig_bow_text_truncation():
    """BoW accuracy vs text truncation length."""
    path = RESULTS / "bow_text_truncation.csv"
    if not path.exists(): return

    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("BoW Accuracy vs Text Truncation", fontsize=13, fontweight="bold")

    for task, color in [("age_bin", "#FF6B6B"), ("gender", "#4ECDC4")]:
        tdf = df[df["task"] == task].sort_values("n_words")
        ax.plot(tdf["n_words"], tdf["accuracy"], marker="o", color=color,
                linewidth=2, label=TASK_LABELS[task])
        ax.axhline(CHANCE[task], color=color, linestyle=":", alpha=0.4)

    ax.set_xlabel("Words per Document")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGS / "paper_bow_text_truncation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved paper_bow_text_truncation")


def fig_training_curves():
    """Training loss/accuracy curves from training logs."""
    logs = sorted(RESULTS.glob("*_training_log.csv"))
    if not logs: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves (EMA probe, default config)", fontsize=14, fontweight="bold")

    cmap = plt.cm.viridis
    # Collect all models
    models = {}
    for f in logs:
        try:
            df = pd.read_csv(f)
            ms = f.stem.replace("_training_log", "")
            if ms in MODEL_PARAMS:
                models[ms] = df
        except:
            pass

    if not models: return
    params_list = sorted(models.keys(), key=lambda m: MODEL_PARAMS.get(m, 0))
    log_params = [np.log10(MODEL_PARAMS[m]) for m in params_list]
    norm = plt.Normalize(min(log_params), max(log_params))

    for ax, task in zip(axes, ["age_bin", "gender"]):
        for ms in params_list:
            df = models[ms]
            tdf = df[(df["task"] == task) & (df["config"] == "ema_lr1e-2")]
            if tdf.empty: continue
            # Average across layers
            avg = tdf.groupby("batch")["batch_acc"].mean().reset_index()
            if len(avg) < 2: continue
            # Smooth
            window = max(1, len(avg) // 50)
            smoothed = avg["batch_acc"].rolling(window, min_periods=1).mean()
            color = cmap(norm(np.log10(MODEL_PARAMS[ms])))
            ax.plot(avg["batch"], smoothed, color=color, linewidth=1, alpha=0.8)

        ax.axhline(CHANCE[task], color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Batch Accuracy (smoothed)")
        ax.set_title(TASK_LABELS[task])
        ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8)
    cbar.set_label("log₁₀(params in B)")

    fig.tight_layout()
    fig.savefig(FIGS / "paper_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved paper_training_curves")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_scaling_law()
    fig_6datasets("gender")
    fig_6datasets("age_bin")
    fig_layer_curves()
    fig_signal_decomposition()
    fig_bow_text_truncation()
    fig_training_curves()
    print("Done.")
