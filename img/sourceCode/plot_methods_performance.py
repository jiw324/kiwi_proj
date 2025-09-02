"""
language: python
# AI-Generated Code Header
# Intent: Plot methods performance (PLS, SVR, MPHNN, ENSEMBLE) for CV and LOBO.
# Optimization: Uses pandas + seaborn; minimal deps, CPU-only.
# Safety: Validates file presence, handles missing models gracefully.
"""

import os
import sys
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_results_csv(csv_path: str, expected_models: List[str]) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Filter to only the three base methods
    df = df[df["model"].isin(expected_models)].copy()
    if df.empty:
        raise ValueError(f"No rows found for models {expected_models} in {csv_path}")
    return df


def summarize_cv(df_cv: pd.DataFrame) -> pd.DataFrame:
    # Aggregate across folds: mean and std for metrics
    agg = df_cv.groupby(["model"], as_index=False).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
    )
    agg["protocol"] = "CV"
    return agg


def summarize_lobo(df_lobo: pd.DataFrame) -> pd.DataFrame:
    # Aggregate across held-out batches
    agg = df_lobo.groupby(["model"], as_index=False).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
    )
    agg["protocol"] = "LOBO"
    return agg


def plot_bar_protocol(ax, data: pd.DataFrame, metric: str, title: str):
    sns.barplot(
        data=data,
        x="model",
        y=f"{metric}_mean",
        ax=ax,
        errorbar=None,
        order=data["model"].cat.categories,
    )
    # Add error bars using std per model
    for patch, model in zip(ax.patches, data["model"]):
        mean_val = data.loc[data["model"] == model, f"{metric}_mean"].values[0]
        std_val = data.loc[data["model"] == model, f"{metric}_std"].values[0]
        x = patch.get_x() + patch.get_width() / 2
        ax.errorbar(x, mean_val, yerr=std_val, fmt='none', ecolor='black', elinewidth=1, capsize=3)

    # Visually differentiate ENSEMBLE bars
    for patch, model in zip(ax.patches, data["model"]):
        if str(model) == "ENSEMBLE":
            patch.set_edgecolor('red')
            patch.set_linewidth(2)

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(metric.upper())
    ax.grid(axis='y', linestyle='--', alpha=0.3)


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = os.path.join(project_root, "output")
    img_out_dir = os.path.join(project_root, "img", "output")
    os.makedirs(img_out_dir, exist_ok=True)

    # Include combined ensemble as the fourth method
    expected_models = ["PLS", "SVR", "MPHNN", "ENSEMBLE"]

    cv_path = os.path.join(output_dir, "results_cv.csv")
    lobo_path = os.path.join(output_dir, "results_lobo.csv")

    df_cv = read_results_csv(cv_path, expected_models)
    df_lobo = read_results_csv(lobo_path, expected_models)

    cv_summary = summarize_cv(df_cv)
    lobo_summary = summarize_lobo(df_lobo)

    combined = pd.concat([cv_summary, lobo_summary], ignore_index=True)

    # Display label mapping: ENSEMBLE -> ENS
    label_map = {"ENSEMBLE": "ENS"}
    combined["display_model"] = combined["model"].replace(label_map)

    # Order for display
    display_order = ["PLS", "SVR", "MPHNN", "ENS"]
    combined["display_model"] = pd.Categorical(combined["display_model"], categories=display_order, ordered=True)
    combined.sort_values(["protocol", "display_model"], inplace=True)

    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    hue_order = ["CV", "LOBO"]
    metrics = [
        ("rmse", "RMSE (lower is better)"),
        ("mae", "MAE (lower is better)"),
        ("r2", "R² (higher is better)"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        # Barplot with both protocols combined
        sns.barplot(
            data=combined,
            x="display_model",
            y=f"{metric}_mean",
            hue="protocol",
            hue_order=hue_order,
            ax=ax,
            errorbar=None,
        )
        # Add error bars aligned to bars per hue
        containers = ax.containers
        for cont, prot in zip(containers, hue_order):
            dfp = (
                combined[combined["protocol"] == prot]
                .set_index("display_model")
                .reindex(display_order)
            )
            for bar, (idx, row) in zip(cont, dfp.iterrows()):
                mean_val = row[f"{metric}_mean"]
                std_val = row[f"{metric}_std"]
                x = bar.get_x() + bar.get_width() / 2
                ax.errorbar(x, mean_val, yerr=std_val, fmt='none', ecolor='black', elinewidth=1, capsize=3)

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper())
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    out_path = os.path.join(img_out_dir, "methods_performance.png")
    fig.suptitle("Methods Performance: PLS vs SVR vs MPHNN vs ENS (CV & LOBO)", fontsize=16)
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


