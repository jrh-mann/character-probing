#!/usr/bin/env python3
"""PCA-controlled probes: test whether scaling law is a d_model artifact.

For each model, loads saved Ridge Gram matrices, projects into the top-k
PCA dimensions, re-solves Ridge in the reduced space. If the scaling law
persists at fixed k (e.g., 128 dims for all models), the finding reflects
genuine representation quality, not just probe capacity growing with d_model.

No GPU needed. No forward passes. Just linear algebra on saved Gram matrices.

Output: results/pca_controlled_probes.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent.parent
PROBES = BASE / "probes"
RESULTS = BASE / "results"

TASKS = ["age_bin", "gender"]
TASK_N_CLASSES = {"age_bin": 3, "gender": 2}
LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]
PCA_DIMS = [32, 64, 128, 256, 512]  # fixed capacities to test

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


def load_ridge(path):
    """Load saved Ridge Gram matrix state dict."""
    d = torch.load(path, map_location="cpu", weights_only=True)
    return d


def pca_basis(A, sx, sx2, n, k):
    """Compute top-k PCA eigenvectors from sufficient statistics.

    Returns V_k (D, k) — the projection matrix to k dims.
    """
    D = A.shape[0]
    if k >= D:
        return torch.eye(D, dtype=torch.float64)

    mean = sx / n
    # Centered covariance: C = A/n - mean @ mean^T
    C = A / n - mean.unsqueeze(1) @ mean.unsqueeze(0)

    # Eigendecompose (symmetric → use eigh, returns ascending order)
    eigenvalues, eigenvectors = torch.linalg.eigh(C)
    # Take top-k (last k, since eigh returns ascending)
    V_k = eigenvectors[:, -k:]  # (D, k)
    return V_k


def solve_ridge_pca(A, B_task, sx, sx2, cc, n, V_k, lam=1.0):
    """Solve Ridge regression in PCA-projected subspace.

    Project the Gram matrix and targets into V_k's column space,
    then solve the k-dimensional Ridge problem.

    Returns: training accuracy estimate (using the Ridge solution
    applied back to the training statistics).
    """
    D = A.shape[0]
    k = V_k.shape[1]
    C = TASK_N_CLASSES.get("age_bin", 3)  # will be overridden
    C = B_task.shape[1]

    # Project sufficient statistics into PCA space
    # A_k = V_k^T A V_k  (k×k)
    A_k = V_k.T @ A @ V_k

    # B_k = V_k^T B  (k×C)
    B_k = V_k.T @ B_task

    # sx_k = V_k^T sx  (k,)
    sx_k = V_k.T @ sx

    # sx2_k = diag(V_k^T diag(sx2) V_k) — but we need per-feature variance
    # Actually easier: just compute mean and center in projected space
    mean_k = sx_k / n

    # Centered Gram in projected space
    A_k_c = A_k - n * mean_k.unsqueeze(1) @ mean_k.unsqueeze(0)

    # Variance in projected space (from centered Gram diagonal)
    var_k = (torch.diag(A_k) / n - mean_k**2).clamp(min=1e-8)
    std_k = var_k.sqrt()
    inv_k = 1.0 / std_k

    # Z-score the Gram
    A_k_z = A_k_c * inv_k.unsqueeze(1) * inv_k.unsqueeze(0)

    # Target centering
    mean_y = cc / n
    B_k_c = B_k - n * mean_k.unsqueeze(1) @ mean_y.unsqueeze(0)
    B_k_z = B_k_c * inv_k.unsqueeze(1)

    # Solve: (A_k_z/n + lam*I) W = B_k_z/n
    W_k = torch.linalg.solve(
        A_k_z / n + lam * torch.eye(k, dtype=torch.float64),
        B_k_z / n
    )

    return W_k, mean_k, std_k, mean_y


def training_accuracy_estimate(A, B_task, sx, cc, n, V_k, W_k, mean_k, std_k, mean_y):
    """Estimate training accuracy from Gram matrices.

    Uses the fact that for Ridge regression, the predicted class for the
    training centroid of each class gives us a class-separability measure.

    More precisely: compute the predicted logits for each class centroid,
    check if the argmax matches the true class.
    """
    k = V_k.shape[1]
    C = B_task.shape[1]

    # Class centroids in original space: mu_c = B[:, c] / cc[c] (mean of X for class c)
    # Project to PCA space: mu_c_k = V_k^T @ mu_c
    centroids_k = torch.zeros(C, k, dtype=torch.float64)
    for c in range(C):
        if cc[c] > 0:
            mu_c = B_task[:, c] / cc[c]  # (D,) mean of X for class c
            centroids_k[c] = V_k.T @ mu_c  # (k,)

    # Z-score centroids
    centroids_z = (centroids_k - mean_k.unsqueeze(0)) / (std_k.unsqueeze(0) + 1e-8)

    # Predict: logits = centroids_z @ W_k + mean_y
    logits = centroids_z @ W_k + mean_y.unsqueeze(0)  # (C, C)

    # Check if diagonal dominates (class c's centroid → predicts class c)
    preds = logits.argmax(dim=1)
    correct = (preds == torch.arange(C)).float().mean().item()

    return correct


def margin_score(A, B_task, sx, cc, n, V_k, W_k, mean_k, std_k, mean_y):
    """Compute a margin-based separability score.

    For each class, compute the margin between the correct logit and the
    best competing logit at the class centroid. Average across classes.
    Positive = correct classification, larger = more confident.
    """
    k = V_k.shape[1]
    C = B_task.shape[1]

    centroids_k = torch.zeros(C, k, dtype=torch.float64)
    for c in range(C):
        if cc[c] > 0:
            centroids_k[c] = V_k.T @ (B_task[:, c] / cc[c])

    centroids_z = (centroids_k - mean_k.unsqueeze(0)) / (std_k.unsqueeze(0) + 1e-8)
    logits = centroids_z @ W_k + mean_y.unsqueeze(0)

    margins = []
    for c in range(C):
        correct_logit = logits[c, c].item()
        other_logits = torch.cat([logits[c, :c], logits[c, c+1:]])
        best_other = other_logits.max().item()
        margins.append(correct_logit - best_other)

    return np.mean(margins)


def main():
    print("PCA-Controlled Probe Analysis")
    print("=" * 60)

    rows = []

    model_dirs = sorted(PROBES.iterdir())
    model_dirs = [d for d in model_dirs if d.is_dir() and d.name in MODEL_PARAMS]

    print(f"Found {len(model_dirs)} models with saved probes")

    for model_dir in model_dirs:
        model_name = model_dir.name
        n_params = MODEL_PARAMS[model_name]

        # Find best layer from existing results
        results_file = RESULTS / f"{model_name}_per_layer_results.csv"
        if not results_file.exists():
            print(f"  {model_name}: no results file, skipping")
            continue

        rdf = pd.read_csv(results_file)

        # Load all ridge files
        ridge_files = sorted(model_dir.glob("L*_ridge.pt"))
        if not ridge_files:
            print(f"  {model_name}: no ridge probes, skipping")
            continue

        D = None

        for ridge_file in ridge_files:
            layer = int(ridge_file.stem.split("_")[0][1:])
            d = load_ridge(ridge_file)

            A = d["A"].double()
            sx = d["sx"].double()
            sx2 = d["sx2"].double()
            n = d["n"]
            D = A.shape[0]

            for task in TASKS:
                B_key = f"B_{task}"
                cc_key = f"cc_{task}"
                if B_key not in d:
                    continue

                B_task = d[B_key].double()
                cc = d[cc_key].double()

                # Full-dimensional Ridge (baseline)
                for lam in [1.0]:
                    mean = sx / n
                    var = (sx2 / n - mean**2).clamp(min=1e-8)
                    std = var.sqrt(); inv = 1.0 / std
                    A_c = A - n * mean.unsqueeze(1) @ mean.unsqueeze(0)
                    A_z = A_c * inv.unsqueeze(1) * inv.unsqueeze(0)
                    mean_y = cc / n
                    B_c = B_task - n * mean.unsqueeze(1) @ mean_y.unsqueeze(0)
                    B_z = B_c * inv.unsqueeze(1)
                    W_full = torch.linalg.solve(
                        A_z / n + lam * torch.eye(D, dtype=torch.float64), B_z / n)

                    V_full = torch.eye(D, dtype=torch.float64)
                    m = margin_score(A, B_task, sx, cc, n, V_full, W_full, mean, std, mean_y)
                    rows.append({
                        "model": model_name, "n_params": n_params,
                        "d_model": D, "layer": layer, "task": task,
                        "pca_dims": D, "lambda": lam,
                        "centroid_margin": round(m, 6),
                    })

                # PCA-truncated Ridge at various k
                V_all = pca_basis(A, sx, sx2, n, max(PCA_DIMS))

                for k in PCA_DIMS:
                    if k > D:
                        continue

                    V_k = V_all[:, -k:]  # top-k eigenvectors

                    for lam in [1.0]:
                        W_k, mean_k, std_k, mean_y = solve_ridge_pca(
                            A, B_task, sx, sx2, cc, n, V_k, lam)

                        m = margin_score(A, B_task, sx, cc, n, V_k, W_k,
                                        mean_k, std_k, mean_y)
                        rows.append({
                            "model": model_name, "n_params": n_params,
                            "d_model": D, "layer": layer, "task": task,
                            "pca_dims": k, "lambda": lam,
                            "centroid_margin": round(m, 6),
                        })

        print(f"  {model_name}: D={D}, {len(ridge_files)} layers")

    df = pd.DataFrame(rows)
    out = RESULTS / "pca_controlled_probes.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")

    # Quick summary: scaling law at each PCA dimension
    print(f"\n{'='*60}")
    print("Scaling Law Summary (best layer per model, centroid margin)")
    print(f"{'='*60}")

    from scipy import stats

    for task in TASKS:
        print(f"\n  {task}:")
        tdf = df[df["task"] == task]

        for k in [*PCA_DIMS, "full"]:
            if k == "full":
                kdf = tdf[tdf["pca_dims"] == tdf["d_model"]]
                label = "full D"
            else:
                kdf = tdf[tdf["pca_dims"] == k]
                label = f"k={k}"

            if kdf.empty:
                continue

            # Best layer per model
            best = kdf.groupby(["model", "n_params"])["centroid_margin"].max().reset_index()
            if len(best) < 3:
                continue

            x = np.log10(best["n_params"].values)
            y = best["centroid_margin"].values
            slope, intercept, r, p, se = stats.linregress(x, y)
            print(f"    {label:>8s}: R²={r**2:.3f}, slope={slope:.4f}, n={len(best)}")


if __name__ == "__main__":
    main()
