"""
language: python
# AI-Generated Code Header
# Intent: Plot bin-wise errors (RMSE/MAE by y_bin) for models across CV & LOBO.
# Output: img/output/error_bins_panel.png
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def require(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def melt_for_plot(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    df = df.copy()
    df["protocol"] = protocol
    long = df.melt(id_vars=["model", "y_bin", "n", "protocol"], value_vars=["rmse", "mae"],
                   var_name="metric", value_name="value")
    return long


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(project_root, "img", "output")
    os.makedirs(out_dir, exist_ok=True)

    cv_path = os.path.join(project_root, "output", "error_bins_cv.csv")
    lobo_path = os.path.join(project_root, "output", "error_bins_lobo.csv")
    require(cv_path)
    require(lobo_path)

    df_cv = pd.read_csv(cv_path)
    df_lobo = pd.read_csv(lobo_path)

    long_cv = melt_for_plot(df_cv, protocol="CV")
    long_lobo = melt_for_plot(df_lobo, protocol="LOBO")
    long = pd.concat([long_cv, long_lobo], ignore_index=True)

    # Model order and y_bin order
    model_order = ["PLS", "SVR", "MPHNN", "ENSEMBLE"]
    bin_order = ["<12", "12-14", ">14"]
    long["model"] = pd.Categorical(long["model"], categories=model_order, ordered=True)
    long["y_bin"] = pd.Categorical(long["y_bin"], categories=bin_order, ordered=True)

    sns.set(style="whitegrid", context="talk")
    g = sns.catplot(
        data=long,
        x="y_bin", y="value", hue="model", col="protocol", row="metric",
        kind="bar", height=4, aspect=1.1, sharey=False
    )
    g.set_axis_labels("Target bin (°Brix)", "Error")
    g.set_titles("{row_name} — {col_name}")
    for ax in g.axes.flat:
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    out_path = os.path.join(out_dir, "error_bins_panel.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


