"""
language: python
# AI-Generated Code Header
# Intent: Scatter plots of y_true vs y_pred for key methods (PLS, ENS) under CV and LOBO.
# Optimization: Minimal deps; computes metrics on the fly.
# Safety: Validates file presence and available models.
"""

import os
import sys
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def require_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def summarize_metrics(df: pd.DataFrame) -> str:
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return f"RMSE={rmse:.3f}\nMAE={mae:.3f}\nR²={r2:.3f}"


def scatter_panel(ax, df: pd.DataFrame, title: str):
    ax.scatter(df["y_true"], df["y_pred"], s=18, alpha=0.5, edgecolor="none")
    # 45-degree line
    lo = min(df["y_true"].min(), df["y_pred"].min())
    hi = max(df["y_true"].max(), df["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("y_true (°Brix)")
    ax.set_ylabel("y_pred (°Brix)")
    ax.set_title(title)
    # Metrics box
    metrics_text = summarize_metrics(df)
    ax.text(0.98, 0.02, metrics_text, transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"), fontsize=9)


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(project_root, "img", "output")
    os.makedirs(out_dir, exist_ok=True)

    pred_cv = os.path.join(project_root, "output", "predictions_cv.csv")
    pred_lobo = os.path.join(project_root, "output", "predictions_lobo.csv")
    require_path(pred_cv)
    require_path(pred_lobo)

    df_cv = pd.read_csv(pred_cv)
    df_lobo = pd.read_csv(pred_lobo)

    # Focus on PLS and ENSEMBLE (ENS)
    wanted_models = ["PLS", "ENSEMBLE"]
    df_cv = df_cv[df_cv["model"].isin(wanted_models)].copy()
    df_lobo = df_lobo[df_lobo["model"].isin(wanted_models)].copy()

    if df_cv.empty or df_lobo.empty:
        raise ValueError("No predictions found for models PLS/ENSEMBLE in provided CSVs")

    sns.set(style="whitegrid", context="talk")
    # Only show ENS, left-right: CV (left) and LOBO (right)
    cv_ens = df_cv[df_cv["model"] == "ENSEMBLE"].copy()
    lobo_ens = df_lobo[df_lobo["model"] == "ENSEMBLE"].copy()
    if cv_ens.empty or lobo_ens.empty:
        raise ValueError("Missing ENSEMBLE predictions for CV or LOBO")

    # Save two separate figures
    fig_cv, ax_cv = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    scatter_panel(ax_cv, cv_ens, title="CV — ENS")
    out_cv = os.path.join(out_dir, "true_vs_pred_scatter_cv.png")
    fig_cv.suptitle("True vs Predicted: CV — ENS", fontsize=16)
    fig_cv.savefig(out_cv, dpi=200)
    print(f"Saved: {out_cv}")

    fig_lobo, ax_lobo = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    scatter_panel(ax_lobo, lobo_ens, title="LOBO — ENS")
    out_lobo = os.path.join(out_dir, "true_vs_pred_scatter_lobo.png")
    fig_lobo.suptitle("True vs Predicted: LOBO — ENS", fontsize=16)
    fig_lobo.savefig(out_lobo, dpi=200)
    print(f"Saved: {out_lobo}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


