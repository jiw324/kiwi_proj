"""
Plot dual-strategy performance combining enhanced CV and baseline LOBO results
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    img_out_dir = os.path.join(project_root, "img", "output")
    os.makedirs(img_out_dir, exist_ok=True)

    # Load CV results (enhanced with STACKED)
    cv_path = os.path.join(project_root, "output_enhanced", "results_cv_final.csv")
    df_cv = pd.read_csv(cv_path)
    
    # Load LOBO results (baseline with ENSEMBLE)
    lobo_path = os.path.join(project_root, "output", "results_lobo.csv")
    df_lobo = pd.read_csv(lobo_path)
    
    print(f"Loaded CV: {len(df_cv)} rows")
    print(f"Loaded LOBO: {len(df_lobo)} rows")
    
    # Summarize CV results
    cv_summary = df_cv.groupby("model").agg({
        "rmse": ["mean", "std"],
        "mae": ["mean", "std"],
        "r2": ["mean", "std"]
    }).reset_index()
    cv_summary.columns = ["model", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_mean", "r2_std"]
    cv_summary["protocol"] = "CV"
    
    # Summarize LOBO results
    lobo_summary = df_lobo.groupby("model").agg({
        "rmse": ["mean", "std"],
        "mae": ["mean", "std"],
        "r2": ["mean", "std"]
    }).reset_index()
    lobo_summary.columns = ["model", "rmse_mean", "rmse_std", "mae_mean", "mae_std", "r2_mean", "r2_std"]
    lobo_summary["protocol"] = "LOBO"
    
    # Rename models for display
    model_map = {"STACKED": "STACKED", "ENSEMBLE": "ENS", "MPHNN": "MPHNN"}
    cv_summary["display_model"] = cv_summary["model"].replace(model_map)
    lobo_summary["display_model"] = lobo_summary["model"].replace(model_map)
    
    # Combine
    combined = pd.concat([cv_summary, lobo_summary], ignore_index=True)
    
    # Create figure
    sns.set(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    metrics = [
        ("rmse", "RMSE (°Brix) - Lower is Better"),
        ("mae", "MAE (°Brix) - Lower is Better"),
        ("r2", "R² - Higher is Better"),
    ]
    
    hue_order = ["CV", "LOBO"]
    
    for ax, (metric, title) in zip(axes, metrics):
        # Create barplot
        sns.barplot(
            data=combined,
            x="display_model",
            y=f"{metric}_mean",
            hue="protocol",
            hue_order=hue_order,
            ax=ax,
            errorbar=None,
            palette={"CV": "#3498db", "LOBO": "#e74c3c"}
        )
        
        # Add error bars
        containers = ax.containers
        for cont, prot in zip(containers, hue_order):
            prot_data = combined[combined["protocol"] == prot]
            for bar, (idx, row) in zip(cont, prot_data.iterrows()):
                mean_val = row[f"{metric}_mean"]
                std_val = row[f"{metric}_std"]
                x = bar.get_x() + bar.get_width() / 2
                ax.errorbar(x, mean_val, yerr=std_val, fmt='none', 
                           ecolor='black', elinewidth=1.5, capsize=4)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Create a single legend for the entire figure (instead of per-subplot)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Protocol", fontsize=11, 
               loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=True)
    
    # Remove individual subplot legends
    for ax in axes:
        ax.get_legend().remove()
    
    fig.suptitle("Dual-Strategy Performance: Enhanced CV vs Robust LOBO", 
                 fontsize=16, fontweight='bold', y=1.05)
    
    out_path = os.path.join(img_out_dir, "methods_performance.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Saved: {out_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print("\nCV (Enhanced with STACKED):")
    cv_best = cv_summary[cv_summary["model"] == "STACKED"].iloc[0]
    print(f"  STACKED: RMSE={cv_best['rmse_mean']:.3f}+/-{cv_best['rmse_std']:.3f}, "
          f"R2={cv_best['r2_mean']:.3f}+/-{cv_best['r2_std']:.3f}")
    
    print("\nLOBO (Baseline with ENSEMBLE):")
    lobo_best = lobo_summary[lobo_summary["model"] == "ENSEMBLE"].iloc[0]
    print(f"  ENSEMBLE: RMSE={lobo_best['rmse_mean']:.3f}+/-{lobo_best['rmse_std']:.3f}, "
          f"R2={lobo_best['r2_mean']:.3f}+/-{lobo_best['r2_std']:.3f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

