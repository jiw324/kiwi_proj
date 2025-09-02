"""
language: python
# AI-Generated Code Header
# Intent: Residual distribution histograms by model (PLS, SVR, MPHNN, ENS) for CV and LOBO.
# Output: img/output/residual_hist.png
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def require(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def load_with_residuals(path: str):
    df = pd.read_csv(path)
    df["residual"] = df["y_pred"] - df["y_true"]
    return df


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(project_root, "img", "output")
    os.makedirs(out_dir, exist_ok=True)

    cv_path = os.path.join(project_root, "output", "predictions_cv.csv")
    lobo_path = os.path.join(project_root, "output", "predictions_lobo.csv")
    require(cv_path)
    require(lobo_path)

    df_cv = load_with_residuals(cv_path)
    df_lobo = load_with_residuals(lobo_path)

    models = ["PLS", "SVR", "MPHNN", "ENSEMBLE"]
    df_cv = df_cv[df_cv["model"].isin(models)].copy()
    df_lobo = df_lobo[df_lobo["model"].isin(models)].copy()

    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for j, (df, prot) in enumerate(((df_cv, "CV"), (df_lobo, "LOBO"))):
        for i, m in enumerate(models):
            ax = axes[j, i]
            sub = df[df["model"] == m]
            ax.hist(sub["residual"], bins=30, color="#4C72B0", alpha=0.8)
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            title = f"{prot} — {('ENS' if m=='ENSEMBLE' else m)}"
            ax.set_title(title)
            ax.set_xlabel("Residual (y_pred - y_true)")
            ax.set_ylabel("Count")
            ax.grid(axis='y', linestyle='--', alpha=0.3)

    out_path = os.path.join(out_dir, "residual_hist.png")
    fig.suptitle("Residual Distributions (CV & LOBO)", fontsize=16)
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


