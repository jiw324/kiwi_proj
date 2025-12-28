"""
language: python
# AI-Generated Code Header
# Intent: CLI runner for enhanced experiment with Phase 1 + Phase 2
# Optimization: Simple argparse, defaults optimized for best performance
# Safety: Validates inputs, handles errors gracefully
"""

import argparse
from experiment_enhanced import EnhancedRunConfig, run_all_enhanced


def parse_args() -> EnhancedRunConfig:
    """Parse command-line arguments"""
    p = argparse.ArgumentParser(
        description="Run enhanced kiwi NIR experiment with Phase 1 + Phase 2 improvements"
    )
    
    # Basic options
    p.add_argument("--input_dir", type=str, default="input",
                  help="Input directory with kiwi-*.csv files")
    p.add_argument("--output_dir", type=str, default="output_enhanced",
                  help="Output directory for results")
    p.add_argument("--fast", action="store_true",
                  help="Fast mode (fewer CV folds)")
    
    # Phase 1 options
    phase1 = p.add_argument_group("Phase 1 Improvements")
    phase1.add_argument("--no_smote", action="store_true",
                       help="Disable SMOTE augmentation")
    phase1.add_argument("--smote_samples", type=int, default=150,
                       help="Target samples for high-Brix augmentation (default: 150)")
    phase1.add_argument("--no_vip", action="store_true",
                       help="Disable VIP wavelength selection")
    phase1.add_argument("--vip_threshold", type=float, default=1.0,
                       help="VIP threshold for wavelength selection (default: 1.0)")
    phase1.add_argument("--no_stacked", action="store_true",
                       help="Disable stacked ensemble")
    
    # Phase 2 options
    phase2 = p.add_argument_group("Phase 2 Improvements")
    phase2.add_argument("--no_emsc", action="store_true",
                       help="Disable EMSC preprocessing (use SNV instead)")
    phase2.add_argument("--no_stratified", action="store_true",
                       help="Disable stratified ensemble")
    phase2.add_argument("--no_calibration_transfer", action="store_true",
                       help="Disable calibration transfer for LOBO")
    phase2.add_argument("--calibration_method", type=str, default="pds",
                       choices=["pds", "sbc", "none"],
                       help="Calibration transfer method (default: pds)")
    
    # Full disable
    p.add_argument("--baseline", action="store_true",
                  help="Run baseline only (disable all enhancements)")
    
    args = p.parse_args()
    
    # Build configuration
    cfg = EnhancedRunConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fast=args.fast,
        
        # Phase 1
        enable_smote=not args.no_smote and not args.baseline,
        smote_target_samples=args.smote_samples,
        enable_vip_selection=not args.no_vip and not args.baseline,
        vip_threshold=args.vip_threshold,
        enable_stacked_ensemble=not args.no_stacked and not args.baseline,
        
        # Phase 2
        enable_emsc=not args.no_emsc and not args.baseline,
        enable_stratified_ensemble=not args.no_stratified and not args.baseline,
        enable_calibration_transfer=not args.no_calibration_transfer and not args.baseline,
        calibration_method=args.calibration_method if not args.baseline else "none"
    )
    
    return cfg


def main():
    """Main entry point"""
    cfg = parse_args()
    
    try:
        paths = run_all_enhanced(cfg)
        print("\n[SUCCESS] Experiment completed successfully!")
        print(f"Results saved to: {cfg.output_dir}")
        print(f"  CV results: {paths['cv']}")
        print(f"  LOBO results: {paths['lobo']}")
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

