"""
CLI runner for LOBO-improved experiment with calibration transfer
"""

import argparse
from kiwi_data_loader import load_kiwi_dataset
from experiment_lobo_improved import run_lobo_improved, compare_with_baseline


def main():
    parser = argparse.ArgumentParser(description="Run LOBO-improved experiment with calibration transfer")
    
    parser.add_argument("--input_dir", type=str, default="input",
                        help="Input directory with kiwi CSV files")
    parser.add_argument("--output_dir", type=str, default="output_lobo_improved",
                        help="Output directory for results")
    
    # Calibration transfer options
    parser.add_argument("--calibration", type=str, default="pds", choices=["pds", "sbc", "none"],
                        help="Calibration transfer method (default: pds)")
    parser.add_argument("--no_calibration", action="store_true",
                        help="Disable calibration transfer")
    
    # Preprocessing options
    parser.add_argument("--enable_smote", action="store_true",
                        help="Enable SMOTE augmentation for LOBO (NOT recommended)")
    parser.add_argument("--disable_vip", action="store_true",
                        help="Disable VIP wavelength selection")
    parser.add_argument("--vip_threshold", type=float, default=0.8,
                        help="VIP threshold (0.8 = more conservative, keeps more wavelengths)")
    
    # Comparison options
    parser.add_argument("--no_comparison", action="store_true",
                        help="Skip comparison with baseline")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("LOBO IMPROVED EXPERIMENT - Configuration")
    print("="*80)
    print(f"Input:              {args.input_dir}")
    print(f"Output:             {args.output_dir}")
    print(f"Calibration:        {args.calibration if not args.no_calibration else 'DISABLED'}")
    print(f"SMOTE (LOBO):       {'ENABLED (risky!)' if args.enable_smote else 'DISABLED (recommended)'}")
    print(f"VIP Selection:      {'DISABLED' if args.disable_vip else f'ENABLED (threshold={args.vip_threshold})'}")
    print("="*80 + "\n")
    
    # Load dataset
    print(f"Loading dataset from {args.input_dir}...")
    dataset = load_kiwi_dataset(args.input_dir)
    print(f"Loaded: {len(dataset.targets)} samples, {len(dataset.wavelengths)} wavelengths\n")
    
    # Run improved LOBO
    results_df = run_lobo_improved(
        dataset,
        output_dir=args.output_dir,
        enable_calibration=not args.no_calibration,
        calibration_method=args.calibration,
        enable_smote_lobo=args.enable_smote,
        enable_vip=not args.disable_vip,
        vip_threshold=args.vip_threshold
    )
    
    # Compare with baselines
    if not args.no_comparison:
        compare_with_baseline(results_df)
    
    print("\n" + "="*80)
    print("SUCCESS - LOBO Improved Complete!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}/results_lobo_improved.csv")
    print("\nKey improvements applied:")
    print("  1. Calibration transfer (PDS/SBC) for batch effects")
    print("  2. Conservative preprocessing (no SMOTE overfitting)")
    print("  3. Relaxed VIP selection (more wavelengths for robustness)")
    print("  4. STACKED ensemble with batch-aware training")
    print("\nExpected outcome: Recover to ~1.35 RMSE (baseline ENSEMBLE level)")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()

