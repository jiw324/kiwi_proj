"""
language: python
# AI-Generated Code Header
# **Intent:** CLI entrypoint to run kiwi NIR experiments (CV and LOBO) and save results.
# **Optimization:** Simple argparse; minimal I/O.
# **Safety:** Validates paths; prints summary locations.
"""
from __future__ import annotations

import argparse
import os

from experiment import RunConfig, run_all


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Kiwi NIR experiment runner")
    # AI-SUGGESTION: default to "input" where kiwi-*.csv reside in this repo
    p.add_argument("--input_dir", type=str, default="input")
    p.add_argument("--output_dir", type=str, default="output")
    p.add_argument("--wl_min", type=float, default=920.0)
    p.add_argument("--wl_max", type=float, default=1680.0)
    p.add_argument("--no_snv", action="store_true")
    p.add_argument("--no_savgol", action="store_true")
    p.add_argument("--savgol_window", type=int, default=21)
    p.add_argument("--savgol_polyorder", type=int, default=2)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--fast", action="store_true", help="Run faster (PLS+SVR only; smaller inner-CV)")
    p.add_argument("--pls_components", type=str, default="auto", help="Comma-separated PLS component counts or 'auto'")
    # AI-SUGGESTION: control physics models
    p.add_argument("--use_pinn", action="store_true", help="Enable PINN (BandIntegrator)")
    p.add_argument("--use_beer_pinn", action="store_true", help="Enable BeerPINN (Beer–Lambert)")
    p.add_argument("--use_mphnn", action="store_true", help="Enable MPHNN (Multi-Physics Hybrid)")
    p.add_argument("--use_ensemble", action="store_true", help="Enable Ensemble (PLS + SVR + MPHNN)")
    args = p.parse_args()

    cfg = RunConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        use_snv=not args.no_snv,
        use_savgol=not args.no_savgol,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        n_splits=args.n_splits,
        fast=args.fast,
        pls_components=args.pls_components,
        enable_pinn=args.use_pinn,
        enable_beer_pinn=args.use_beer_pinn,
        enable_mphnn=args.use_mphnn,
        enable_ensemble=args.use_ensemble,
    )
    return cfg


def main():
    cfg = parse_args()
    # AI-SUGGESTION: Expand user-relative paths and validate input existence
    cfg.input_dir = os.path.expanduser(cfg.input_dir)
    cfg.output_dir = os.path.expanduser(cfg.output_dir)
    if not os.path.isdir(cfg.input_dir):
        raise SystemExit(f"Input directory not found: {cfg.input_dir}")
    os.makedirs(cfg.output_dir, exist_ok=True)
    paths = run_all(cfg)
    print("Saved:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

