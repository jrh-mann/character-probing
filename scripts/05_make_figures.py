#!/usr/bin/env python3
"""
05_make_figures.py — Generate all research figures from probe results.

Reads per-layer CSVs from /workspace/characterprobing/results/*.csv and produces
five publication-quality figures saved as both PNG and PDF in
/workspace/characterprobing/figures/.

Usage:
    python 05_make_figures.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("/workspace/characterprobing/results")
FIGURES_DIR = Path("/root/characterprobing/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a2a4a"
TASK_COLORS = {
    "age_bin": "#FF6B6B",
    "gender": "#4ECDC4",
    "star_sign": "#FFE66D",
}
TASK_LABELS = {
    "age_bin": "Age (3-class)",
    "gender": "Gender (2-class)",
    "star_sign": "Star Sign (12-class)",
}
CHANCE_LEVELS = {
    "age_bin": 1.0 / 3,
    "gender": 1.0 / 2,
    "star_sign": 1.0 / 12,
}
TASKS = ["age_bin", "gender", "star_sign"]

# Model size ordering (parameter count in billions for x-axis)
SIZE_ORDER = ["0.5B", "1.5B", "3B", "7B", "14B"]
SIZE_PARAMS = {"0.5B": 0.5, "1.5B": 1.5, "3B": 3.0, "7B": 7.0, "14B": 14.0}

# Marker shapes for scatter plot, keyed by size string
SIZE_MARKERS = {"0.5B": "o", "1.5B": "s", "3B": "D", "7B": "^", "14B": "P"}

# ---------------------------------------------------------------------------
# Helper: apply dark style to axes / figure
# ---------------------------------------------------------------------------

def _apply_dark_style(fig, axes):
    """Set dark background and light text on figure and all axes."""
    fig.patch.set_facecolor(BG_COLOR)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR, which="both")
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)


def _save(fig, name):
    """Save figure as PNG and PDF."""
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {png_path.name} and {pdf_path.name}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results() -> pd.DataFrame:
    """Load and concatenate all result CSVs."""
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {RESULTS_DIR}")
        sys.exit(1)
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception as exc:
            print(f"  Warning: could not read {f.name}: {exc}")
    if not frames:
        print("ERROR: No valid CSV files loaded.")
        sys.exit(1)
    data = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(data)} rows from {len(frames)} CSV file(s).")
    return data


def parse_model_info(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns: size, is_instruct, n_params."""
    df = df.copy()
    sizes = []
    instructs = []
    for name in df["model_name"]:
        is_inst = name.endswith("-Instruct")
        # Extract size token: after "Qwen2.5-" or similar pattern
        base = name.split("/")[-1]  # e.g. "Qwen2.5-7B-Instruct"
        parts = base.split("-")
        # Find the part that looks like a size (ends with B)
        size = None
        for p in parts:
            if p.endswith("B") and p[:-1].replace(".", "").isdigit():
                size = p
                break
        sizes.append(size)
        instructs.append(is_inst)
    df["size"] = sizes
    df["is_instruct"] = instructs
    df["n_params"] = df["size"].map(SIZE_PARAMS)
    return df


def compute_normalized_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized_layer column in [0, 1]."""
    df = df.copy()
    norm_layers = []
    for _, group in df.groupby("model_name"):
        max_layer = group["layer"].max()
        if max_layer == 0:
            norm = group["layer"].astype(float)
        else:
            norm = group["layer"] / max_layer
        norm_layers.append(norm)
    df["normalized_layer"] = pd.concat(norm_layers).reindex(df.index)
    return df


def best_layer_acc(df: pd.DataFrame) -> pd.DataFrame:
    """Return df of (model_name, task) -> best text_balanced_acc across layers."""
    return (
        df.groupby(["model_name", "task", "size", "is_instruct", "n_params"])
        ["text_balanced_acc"]
        .max()
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Figure 1: Hero figure — layer sweep for one model
# ---------------------------------------------------------------------------

def figure1_hero(df: pd.DataFrame):
    """3-panel layer sweep for the largest available instruct model."""
    # Pick the largest instruct model available
    instruct_models = df[df["is_instruct"]]["model_name"].unique()
    if len(instruct_models) == 0:
        # Fall back to any model
        instruct_models = df["model_name"].unique()
    if len(instruct_models) == 0:
        print("  SKIP figure1: no models found.")
        return

    # Sort by n_params descending and pick the first
    model_params = (
        df[df["model_name"].isin(instruct_models)]
        .drop_duplicates("model_name")[["model_name", "n_params"]]
        .sort_values("n_params", ascending=False)
    )
    hero_model = model_params.iloc[0]["model_name"]
    sub = df[df["model_name"] == hero_model].copy()
    short_name = hero_model.split("/")[-1]
    print(f"  Hero model: {short_name}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _apply_dark_style(fig, axes)
    fig.suptitle(
        f"Layer-wise Probe Accuracy — {short_name}",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, task in zip(axes, TASKS):
        td = sub[sub["task"] == task].sort_values("normalized_layer")
        if td.empty:
            ax.set_title(TASK_LABELS.get(task, task))
            continue
        ax.plot(
            td["normalized_layer"], td["text_balanced_acc"],
            color=TASK_COLORS[task], linewidth=2, marker="o", markersize=3,
        )
        chance = CHANCE_LEVELS[task]
        ax.axhline(chance, color="white", linestyle="--", linewidth=1, alpha=0.5,
                    label=f"Chance ({chance:.2f})")
        ax.set_xlabel("Normalized Layer Position")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xlim(0, 1)
        ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
                  fontsize=8)

    fig.tight_layout()
    _save(fig, "fig1_hero_layer_sweep")


# ---------------------------------------------------------------------------
# Figure 2: Scaling laws
# ---------------------------------------------------------------------------

def figure2_scaling(df: pd.DataFrame):
    """Best-layer accuracy vs model params (log scale), base vs instruct."""
    best = best_layer_acc(df)
    if best.empty:
        print("  SKIP figure2: no data.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _apply_dark_style(fig, axes)
    fig.suptitle(
        "Scaling Laws — Best-Layer Balanced Accuracy vs Model Size",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, task in zip(axes, TASKS):
        td = best[best["task"] == task]
        for is_inst, label, ls in [(False, "Base", "-"), (True, "Instruct", "--")]:
            sub = td[td["is_instruct"] == is_inst].sort_values("n_params")
            if sub.empty:
                continue
            ax.plot(
                sub["n_params"], sub["text_balanced_acc"],
                marker="o", linewidth=2, linestyle=ls,
                label=label, color=TASK_COLORS[task] if not is_inst
                else _lighten(TASK_COLORS[task], 0.4),
            )
        chance = CHANCE_LEVELS[task]
        ax.axhline(chance, color="white", linestyle=":", linewidth=1, alpha=0.5,
                    label=f"Chance ({chance:.2f})")
        ax.set_xscale("log")
        ax.set_xlabel("Model Parameters (B)")
        ax.set_ylabel("Best-Layer Balanced Accuracy")
        ax.set_title(TASK_LABELS.get(task, task))
        # Nice x ticks
        ax.set_xticks([0.5, 1.5, 3, 7, 14])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
                  fontsize=8)

    fig.tight_layout()
    _save(fig, "fig2_scaling_laws")


def _lighten(hex_color: str, amount: float = 0.3) -> str:
    """Lighten a hex color toward white by *amount* (0-1)."""
    c = matplotlib.colors.to_rgb(hex_color)
    lightened = [min(1.0, ch + (1.0 - ch) * amount) for ch in c]
    return matplotlib.colors.to_hex(lightened)


# ---------------------------------------------------------------------------
# Figure 3: Base vs Instruct scatter
# ---------------------------------------------------------------------------

def figure3_scatter(df: pd.DataFrame):
    """Scatter: base best-layer acc vs instruct best-layer acc."""
    best = best_layer_acc(df)
    # Pivot to get base and instruct side by side
    base = best[~best["is_instruct"]][["size", "task", "text_balanced_acc"]].rename(
        columns={"text_balanced_acc": "base_acc"}
    )
    inst = best[best["is_instruct"]][["size", "task", "text_balanced_acc"]].rename(
        columns={"text_balanced_acc": "instruct_acc"}
    )
    merged = pd.merge(base, inst, on=["size", "task"], how="inner")
    if merged.empty:
        print("  SKIP figure3: need both base and instruct models.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    _apply_dark_style(fig, ax)
    ax.set_title("Base vs Instruct — Best-Layer Balanced Accuracy",
                 fontsize=13, fontweight="bold")

    # Plot diagonal
    lims = [0, 1]
    ax.plot(lims, lims, color="white", linestyle="--", linewidth=1, alpha=0.4)

    # Plot points
    for _, row in merged.iterrows():
        ax.scatter(
            row["base_acc"], row["instruct_acc"],
            color=TASK_COLORS[row["task"]],
            marker=SIZE_MARKERS.get(row["size"], "o"),
            s=100, edgecolors="white", linewidths=0.5, zorder=3,
        )

    # Legends
    task_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TASK_COLORS[t],
               markersize=8, label=TASK_LABELS[t])
        for t in TASKS if t in merged["task"].values
    ]
    size_handles = [
        Line2D([0], [0], marker=SIZE_MARKERS[s], color="none",
               markerfacecolor=TEXT_COLOR, markersize=8, label=s)
        for s in SIZE_ORDER if s in merged["size"].values
    ]
    leg1 = ax.legend(handles=task_handles, title="Task", loc="upper left",
                     facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
                     title_fontsize=9, fontsize=8)
    leg1.get_title().set_color(TEXT_COLOR)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=size_handles, title="Size", loc="lower right",
                     facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
                     title_fontsize=9, fontsize=8)
    leg2.get_title().set_color(TEXT_COLOR)

    ax.set_xlabel("Base Model Best-Layer Balanced Accuracy")
    ax.set_ylabel("Instruct Model Best-Layer Balanced Accuracy")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    fig.tight_layout()
    _save(fig, "fig3_base_vs_instruct_scatter")


# ---------------------------------------------------------------------------
# Figure 4: Layer emergence heatmap
# ---------------------------------------------------------------------------

def figure4_heatmap(df: pd.DataFrame):
    """Heatmap: normalized layer position x model size, colored by accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _apply_dark_style(fig, axes)
    fig.suptitle(
        "Layer Emergence Heatmap — Balanced Accuracy by Layer and Model Size",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, task in zip(axes, TASKS):
        td = df[df["task"] == task].copy()
        if td.empty:
            ax.set_title(TASK_LABELS.get(task, task))
            continue

        # Build a grid: rows = model sizes (sorted), cols = normalized layer bins
        n_layer_bins = 50
        layer_bins = np.linspace(0, 1, n_layer_bins)

        available_sizes = sorted(
            td["size"].dropna().unique(),
            key=lambda s: SIZE_PARAMS.get(s, 0),
        )
        if len(available_sizes) < 2:
            # Not enough models for a meaningful heatmap
            ax.set_title(TASK_LABELS.get(task, task) + " (insufficient data)")
            continue

        grid = np.full((len(available_sizes), n_layer_bins), np.nan)
        for i, size in enumerate(available_sizes):
            sd = td[td["size"] == size].sort_values("normalized_layer")
            if sd.empty:
                continue
            # Interpolate onto the regular grid
            xp = sd["normalized_layer"].values
            yp = sd["text_balanced_acc"].values
            grid[i, :] = np.interp(layer_bins, xp, yp)

        vmin = np.nanmin(grid) if not np.all(np.isnan(grid)) else 0
        vmax = np.nanmax(grid) if not np.all(np.isnan(grid)) else 1

        im = ax.imshow(
            grid, aspect="auto", origin="lower",
            extent=[0, 1, 0, len(available_sizes) - 1],
            cmap="plasma", vmin=vmin, vmax=vmax,
            interpolation="bicubic",
        )
        ax.set_xlabel("Normalized Layer Position")
        ax.set_ylabel("Model Size")
        ax.set_yticks(range(len(available_sizes)))
        ax.set_yticklabels(available_sizes)
        ax.set_title(TASK_LABELS.get(task, task))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        cbar.ax.yaxis.set_tick_params(labelcolor=TEXT_COLOR)
        cbar.set_label("Balanced Accuracy", color=TEXT_COLOR)

    fig.tight_layout()
    _save(fig, "fig4_layer_emergence_heatmap")


# ---------------------------------------------------------------------------
# Figure 5: Cross-layer accuracy curves (all models overlaid)
# ---------------------------------------------------------------------------

def figure5_crosslayer(df: pd.DataFrame):
    """All models overlaid per task; colored by size, dashed for instruct."""
    cmap = plt.cm.viridis
    available_sizes = sorted(
        df["size"].dropna().unique(), key=lambda s: SIZE_PARAMS.get(s, 0)
    )
    if not available_sizes:
        print("  SKIP figure5: no data.")
        return

    norm = Normalize(
        vmin=np.log10(SIZE_PARAMS.get(available_sizes[0], 0.5)),
        vmax=np.log10(SIZE_PARAMS.get(available_sizes[-1], 14)),
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _apply_dark_style(fig, axes)
    fig.suptitle(
        "Cross-Layer Balanced Accuracy — All Models",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=1.02,
    )

    models_sorted = (
        df.drop_duplicates("model_name")[["model_name", "size", "is_instruct", "n_params"]]
        .sort_values(["n_params", "is_instruct"])
    )

    for ax, task in zip(axes, TASKS):
        for _, mrow in models_sorted.iterrows():
            mname = mrow["model_name"]
            size = mrow["size"]
            is_inst = mrow["is_instruct"]
            td = df[(df["model_name"] == mname) & (df["task"] == task)].sort_values(
                "normalized_layer"
            )
            if td.empty:
                continue
            color = cmap(norm(np.log10(SIZE_PARAMS.get(size, 1))))
            ls = "--" if is_inst else "-"
            short = mname.split("/")[-1]
            ax.plot(
                td["normalized_layer"], td["text_balanced_acc"],
                color=color, linestyle=ls, linewidth=1.5, alpha=0.85,
                label=short,
            )
        chance = CHANCE_LEVELS[task]
        ax.axhline(chance, color="white", linestyle=":", linewidth=1, alpha=0.4)
        ax.set_xlabel("Normalized Layer Position")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(TASK_LABELS.get(task, task))
        ax.set_xlim(0, 1)

    # Build a unified legend — size colors + linestyle key
    legend_handles = []
    for size in available_sizes:
        color = cmap(norm(np.log10(SIZE_PARAMS[size])))
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=2, label=size)
        )
    legend_handles.append(
        Line2D([0], [0], color=TEXT_COLOR, linewidth=1.5, linestyle="-", label="Base")
    )
    legend_handles.append(
        Line2D([0], [0], color=TEXT_COLOR, linewidth=1.5, linestyle="--", label="Instruct")
    )

    axes[-1].legend(
        handles=legend_handles, loc="upper right",
        facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
        fontsize=7, title="Model", title_fontsize=8,
    )
    if axes[-1].get_legend():
        axes[-1].get_legend().get_title().set_color(TEXT_COLOR)

    fig.tight_layout()
    _save(fig, "fig5_crosslayer_all_models")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("05_make_figures.py — Generating research figures")
    print("=" * 60)

    df = load_all_results()
    df = parse_model_info(df)
    df = compute_normalized_layer(df)

    # Ensure metric column exists
    if "text_balanced_acc" not in df.columns:
        print("ERROR: 'text_balanced_acc' column not found in data.")
        sys.exit(1)

    available_tasks = df["task"].unique()
    available_models = df["model_name"].unique()
    print(f"Tasks:  {sorted(available_tasks)}")
    print(f"Models: {sorted(available_models)}")
    print()

    figures_generated = []

    # Figure 1
    print("Figure 1: Hero layer sweep")
    try:
        figure1_hero(df)
        figures_generated.append("fig1_hero_layer_sweep")
    except Exception as exc:
        print(f"  FAILED: {exc}")

    # Figure 2
    print("Figure 2: Scaling laws")
    try:
        figure2_scaling(df)
        figures_generated.append("fig2_scaling_laws")
    except Exception as exc:
        print(f"  FAILED: {exc}")

    # Figure 3
    print("Figure 3: Base vs Instruct scatter")
    try:
        figure3_scatter(df)
        figures_generated.append("fig3_base_vs_instruct_scatter")
    except Exception as exc:
        print(f"  FAILED: {exc}")

    # Figure 4
    print("Figure 4: Layer emergence heatmap")
    try:
        figure4_heatmap(df)
        figures_generated.append("fig4_layer_emergence_heatmap")
    except Exception as exc:
        print(f"  FAILED: {exc}")

    # Figure 5
    print("Figure 5: Cross-layer accuracy curves")
    try:
        figure5_crosslayer(df)
        figures_generated.append("fig5_crosslayer_all_models")
    except Exception as exc:
        print(f"  FAILED: {exc}")

    print()
    print("=" * 60)
    print(f"Done. Generated {len(figures_generated)} figure(s):")
    for name in figures_generated:
        print(f"  - {FIGURES_DIR / name}.png / .pdf")
    print("=" * 60)


if __name__ == "__main__":
    main()
