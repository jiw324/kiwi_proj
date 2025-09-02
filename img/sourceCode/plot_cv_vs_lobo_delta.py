"""
language: python
# AI-Generated Code Header
# Intent: Compare CV vs LOBO metrics per model; show deltas (LOBO - CV) for RMSE/MAE/R².
# Output: img/output/cv_lobo_delta.png
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def require(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def summarize(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    agg = df.groupby("model", as_index=False).agg(
        rmse=("rmse", "mean"),
        mae=("mae", "mean"),
        r2=("r2", "mean"),
    )
    agg["protocol"] = protocol
    return agg


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(project_root, "img", "output")
    os.makedirs(out_dir, exist_ok=True)

    cv_path = os.path.join(project_root, "output", "results_cv.csv")
    lobo_path = os.path.join(project_root, "output", "results_lobo.csv")
    require(cv_path)
    require(lobo_path)

    df_cv = pd.read_csv(cv_path)
    df_lobo = pd.read_csv(lobo_path)

    cv = summarize(df_cv, "CV")
    lobo = summarize(df_lobo, "LOBO")
    combo = pd.merge(cv, lobo, on="model", suffixes=("_cv", "_lobo"))
    combo["rmse_delta"] = combo["rmse_lobo"] - combo["rmse_cv"]
    combo["mae_delta"] = combo["mae_lobo"] - combo["mae_cv"]
    combo["r2_delta"] = combo["r2_lobo"] - combo["r2_cv"]

    long = combo.melt(id_vars=["model"], value_vars=["rmse_delta", "mae_delta", "r2_delta"],
                      var_name="metric", value_name="delta")
    order = ["PLS", "SVR", "MPHNN", "ENSEMBLE"]
    long["model"] = pd.Categorical(long["model"], categories=order, ordered=True)

    sns.set(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.barplot(data=long, x="model", y="delta", hue="metric", ax=ax)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("CV vs LOBO Delta (LOBO - CV)")
    ax.set_xlabel("")
    ax.set_ylabel("Delta")
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    out_path = os.path.join(out_dir, "cv_lobo_delta.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


