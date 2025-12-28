"""
language: python
# AI-Generated Code Header
# Intent: LOBO-optimized experiment with calibration transfer and adaptive preprocessing
# Optimization: Batch-aware processing, robust cross-batch generalization
# Safety: Calibration transfer for batch effects, reduced overfitting, extensive validation
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple

from kiwi_data_loader import load_kiwi_dataset, KiwiDataset
from preprocessing import SpectralWindow, SNV, SavitzkyGolayDerivative
from preprocessing_advanced import apply_high_brix_augmentation
from feature_selection import VIPSelector
from model.advanced_ensemble import StackedEnsemble
from model.M1_pls_model import tune_pls
from model.M3_svr_model import tune_svr
from calibration_transfer import apply_lobo_calibration_transfer


def rmse(y_true, y_pred):
    """Calculate RMSE"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class LOBOPreprocessor:
    """
    LOBO-optimized preprocessing with batch-aware adaptations
    
    Key differences from CV preprocessing:
    1. More conservative VIP selection (lower threshold)
    2. No/reduced SMOTE (prevents overfitting to training batches)
    3. Calibration transfer support
    """
    
    def __init__(self, wavelengths, 
                 enable_smote: bool = False,
                 smote_threshold: float = 14.5,  # Higher threshold = less augmentation
                 smote_target: int = 100,  # Lower target = less augmentation
                 enable_vip: bool = True,
                 vip_threshold: float = 0.8,  # Lower threshold = keep more wavelengths
                 calibration_method: str = 'pds'):
        """
        Parameters:
        -----------
        wavelengths : array-like
            Original wavelengths (must match X dimensions during fit)
        enable_smote : bool
            Enable SMOTE augmentation (default False for LOBO)
        smote_threshold : float
            Brix threshold for SMOTE (higher = less augmentation)
        smote_target : int
            Target samples for high-Brix class (lower = less augmentation)
        enable_vip : bool
            Enable VIP feature selection
        vip_threshold : float
            VIP threshold (lower = keep more wavelengths for robustness)
        calibration_method : str
            Calibration transfer method: 'pds', 'sbc', or 'none'
        """
        self.wavelengths_ref = wavelengths  # Store reference wavelengths
        self.enable_smote = enable_smote
        self.smote_threshold = smote_threshold
        self.smote_target = smote_target
        self.enable_vip = enable_vip
        self.vip_threshold = vip_threshold
        self.calibration_method = calibration_method
        
        # Transformers will be initialized in fit() based on actual X dimensions
        self.window = None
        self.snv = SNV()
        self.savgol = SavitzkyGolayDerivative(window_length=21, polyorder=2, deriv=1)
        self.vip_selector = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit preprocessing pipeline"""
        # Initialize SpectralWindow with wavelengths matching X dimensions
        n_wl_input = X.shape[1]
        
        # Create wavelength array matching X dimensions
        if len(self.wavelengths_ref) == n_wl_input:
            # Wavelengths match input dimensions - use directly
            wl_for_window = self.wavelengths_ref
        else:
            # Dimension mismatch - create synthetic wavelength array
            print(f"  WARNING: wavelengths ({len(self.wavelengths_ref)}) != X.shape[1] ({n_wl_input})")
            print(f"  Creating synthetic wavelength array for SpectralWindow")
            # Assume wavelengths range from 900 to 2500 nm (typical NIR range)
            wl_for_window = np.linspace(900, 2500, n_wl_input)
        
        # Initialize and fit window
        self.window = SpectralWindow(wl_for_window, wl_min=920.0, wl_max=1680.0)
        self.window.fit(X, y)
        X_win = self.window.transform(X)
        
        print(f"  SpectralWindow: {n_wl_input} -> {X_win.shape[1]} wavelengths")
        
        # Fit other transformers
        self.snv.fit(X_win, y)
        X_snv = self.snv.transform(X_win)
        
        self.savgol.fit(X_snv, y)
        X_sg = self.savgol.transform(X_snv)
        
        # Apply SMOTE if enabled
        if self.enable_smote:
            X_aug, y_aug = apply_high_brix_augmentation(
                X_sg, y, 
                threshold=self.smote_threshold,
                target_samples=self.smote_target
            )
            print(f"  SMOTE: {len(y)} -> {len(y_aug)} samples")
        else:
            X_aug, y_aug = X_sg, y
            print(f"  SMOTE: DISABLED (LOBO robustness)")
        
        # Ensure data is 2D
        if X_aug.ndim != 2:
            raise ValueError(f"Expected 2D array after preprocessing, got {X_aug.ndim}D")
        
        # Fit VIP selector if enabled (requires sufficient samples)
        if self.enable_vip:
            min_samples_for_vip = 30  # Need at least 30 samples for reliable VIP
            if len(y_aug) >= min_samples_for_vip and X_aug.shape[1] > 20:
                self.vip_selector = VIPSelector(n_components=min(10, len(y_aug)//3), 
                                                threshold=self.vip_threshold)
                self.vip_selector.fit(X_aug, y_aug)
                n_selected = len(self.vip_selector.selected_indices_)
                print(f"  VIP: Selected {n_selected}/{X_aug.shape[1]} wavelengths (threshold={self.vip_threshold})")
            else:
                print(f"  VIP: SKIPPED (insufficient samples: {len(y_aug)} < {min_samples_for_vip})")
                self.vip_selector = None
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform spectra through pipeline"""
        if not self.is_fitted or self.window is None:
            raise ValueError("LOBOPreprocessor not fitted!")
        
        X_win = self.window.transform(X)
        X_snv = self.snv.transform(X_win)
        X_sg = self.savgol.transform(X_snv)
        
        if self.enable_vip and self.vip_selector is not None:
            X_sg = self.vip_selector.transform(X_sg)
        
        return X_sg
    
    def fit_transform(self, X, y):
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_augmented_data(self, X, y):
        """Get augmented data after SMOTE (for training)"""
        if not self.is_fitted or self.window is None:
            raise ValueError("LOBOPreprocessor not fitted!")
        
        X_win = self.window.transform(X)
        X_snv = self.snv.transform(X_win)
        X_sg = self.savgol.transform(X_snv)
        
        if self.enable_smote:
            X_aug, y_aug = apply_high_brix_augmentation(
                X_sg, y,
                threshold=self.smote_threshold,
                target_samples=self.smote_target
            )
        else:
            X_aug, y_aug = X_sg, y
        
        if self.enable_vip and self.vip_selector is not None:
            X_aug = self.vip_selector.transform(X_aug)
        
        return X_aug, y_aug


def run_lobo_improved(dataset: KiwiDataset,
                      output_dir: str = "output_lobo_improved",
                      enable_calibration: bool = True,
                      calibration_method: str = 'pds',
                      enable_smote_lobo: bool = False,
                      enable_vip: bool = True,
                      vip_threshold: float = 0.8) -> pd.DataFrame:
    """
    Run improved LOBO protocol with calibration transfer
    
    Parameters:
    -----------
    dataset : KiwiDataset
        Full dataset
    output_dir : str
        Output directory
    enable_calibration : bool
        Enable calibration transfer
    calibration_method : str
        'pds', 'sbc', or 'none'
    enable_smote_lobo : bool
        Enable SMOTE for LOBO (not recommended)
    enable_vip : bool
        Enable VIP wavelength selection
    vip_threshold : float
        VIP threshold (0.8 = keep more wavelengths than 1.0)
    
    Returns:
    --------
    results_df : DataFrame
        Results for all batches and models
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("LOBO IMPROVED EXPERIMENT")
    print("="*80)
    print(f"Dataset: {len(dataset.targets)} samples from {len(dataset.batches)} batches")
    print(f"Wavelengths: {len(dataset.wavelengths)}")
    print(f"\nConfiguration:")
    print(f"  Calibration Transfer: {calibration_method if enable_calibration else 'DISABLED'}")
    print(f"  SMOTE for LOBO: {'ENABLED' if enable_smote_lobo else 'DISABLED (recommended)'}")
    print(f"  VIP Selection: {'ENABLED' if enable_vip else 'DISABLED'}")
    if enable_vip:
        print(f"  VIP Threshold: {vip_threshold} (lower = more wavelengths)")
    print("="*80 + "\n")
    
    # Get unique batches
    batch_ids = sorted(set(dataset.batches))
    print(f"Batches: {batch_ids}\n")
    
    all_results = []
    
    for held_out_batch in batch_ids:
        print("="*80)
        print(f"HELD-OUT BATCH: {held_out_batch}")
        print("="*80 + "\n")
        
        # Split into train and test (convert batches to numpy array for element-wise comparison)
        batches_array = np.array(dataset.batches)
        train_mask = batches_array != held_out_batch
        test_mask = batches_array == held_out_batch
        
        X_train_raw = dataset.spectra[train_mask]
        y_train_raw = dataset.targets[train_mask]
        X_test_raw = dataset.spectra[test_mask]
        y_test_raw = dataset.targets[test_mask]
        
        n_train_batches = len(np.unique(batches_array[train_mask]))
        print(f"Train: {len(y_train_raw)} samples from {n_train_batches} batches")
        print(f"Test:  {len(y_test_raw)} samples from batch {held_out_batch}\n")
        
        # Safety check: ensure sufficient training samples
        if len(y_train_raw) < 20:
            print(f"WARNING: Very few training samples ({len(y_train_raw)}). Results may be unreliable.")
            print(f"         Consider excluding batch {held_out_batch} or using different protocol.\n")
        
        # Build LOBO-optimized preprocessing
        print("Building LOBO-optimized preprocessing...")
        preproc = LOBOPreprocessor(
            wavelengths=dataset.wavelengths,
            enable_smote=enable_smote_lobo,
            smote_threshold=14.5,  # Conservative
            smote_target=100,  # Conservative
            enable_vip=enable_vip,
            vip_threshold=vip_threshold,
            calibration_method=calibration_method if enable_calibration else 'none'
        )
        
        # Fit on training data
        preproc.fit(X_train_raw, y_train_raw)
        
        # Transform training and test
        X_train = preproc.transform(X_train_raw)
        X_test = preproc.transform(X_test_raw)
        
        # Get augmented training data (includes SMOTE if enabled)
        X_train_aug, y_train_aug = preproc.get_augmented_data(X_train_raw, y_train_raw)
        
        print(f"Training samples after augmentation: {len(y_train_aug)}")
        print(f"Feature dimensions: {X_train.shape[1]}\n")
        
        # Apply calibration transfer if enabled
        if enable_calibration and calibration_method != 'none':
            print(f"Applying {calibration_method.upper()} calibration transfer...")
            X_train_aug, X_test = apply_lobo_calibration_transfer(
                X_train_aug, X_test,
                method=calibration_method,
                window_size=5
            )
            print(f"  Calibration applied: train={X_train_aug.shape}, test={X_test.shape}\n")
        else:
            print("Calibration transfer: DISABLED\n")
        
        # Train base models
        print("Training base models...")
        print("  [1/3] PLS...")
        pls = tune_pls(X_train_aug, y_train_aug, cv=3)
        pls_model = pls.best_estimator_
        
        print("  [2/3] SVR...")
        svr = tune_svr(X_train_aug, y_train_aug, cv=3)
        svr_model = svr.best_estimator_
        
        print("  [3/3] STACKED ensemble...")
        stacked_model = StackedEnsemble(
            base_models={'PLS': pls_model, 'SVR': svr_model},
            cv=3
        )
        stacked_model.fit(X_train_aug, y_train_aug)
        print()
        
        # Evaluate all models
        models = [
            ("PLS", pls_model),
            ("SVR", svr_model),
            ("STACKED", stacked_model)
        ]
        
        print("Evaluation Results:")
        print("-"*60)
        for name, model in models:
            y_pred = model.predict(X_test)
            
            metrics = {
                "protocol": "LOBO_IMPROVED",
                "held_out": held_out_batch,
                "model": name,
                "rmse": rmse(y_test_raw, y_pred),
                "mae": float(mean_absolute_error(y_test_raw, y_pred)),
                "r2": float(r2_score(y_test_raw, y_pred))
            }
            all_results.append(metrics)
            
            print(f"  {name:12s}: RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}")
        print()
    
    # Save results
    df_results = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, "results_lobo_improved.csv")
    df_results.to_csv(results_path, index=False)
    
    # Summary
    print("\n" + "="*80)
    print("LOBO IMPROVED - FINAL SUMMARY")
    print("="*80 + "\n")
    
    summary = df_results.groupby("model").agg({
        "rmse": ["mean", "std"],
        "mae": ["mean", "std"],
        "r2": ["mean", "std"]
    })
    print(summary)
    print(f"\nResults saved to: {results_path}\n")
    
    return df_results


def compare_with_baseline(improved_df: pd.DataFrame, 
                          baseline_path: str = "output/results_lobo.csv",
                          baseline_enhanced_path: str = "output_enhanced/results_lobo_final.csv"):
    """Compare improved LOBO results with baseline and previous enhanced"""
    
    print("="*80)
    print("COMPARISON WITH BASELINE AND PREVIOUS ENHANCED")
    print("="*80 + "\n")
    
    try:
        # Load baseline
        baseline_df = pd.read_csv(baseline_path)
        baseline_avg = baseline_df.groupby("model")["rmse"].mean()
        
        # Load previous enhanced
        enhanced_df = pd.read_csv(baseline_enhanced_path)
        enhanced_avg = enhanced_df.groupby("model")["rmse"].mean()
        
        # Current improved
        improved_avg = improved_df.groupby("model")["rmse"].mean()
        
        print("RMSE Comparison:")
        print("-"*60)
        print(f"{'Model':<15} {'Baseline':<12} {'Enhanced':<12} {'Improved':<12}")
        print("-"*60)
        
        # Compare STACKED vs baselines
        if "STACKED" in improved_avg:
            stacked_rmse = improved_avg["STACKED"]
            
            # vs baseline PLS
            if "PLS" in baseline_avg:
                baseline_pls = baseline_avg["PLS"]
                change_pls = (stacked_rmse - baseline_pls) / baseline_pls * 100
                status_pls = "✓" if change_pls < 0 else "✗"
                print(f"{'STACKED':<15} {baseline_pls:<12.4f} {'-':<12} {stacked_rmse:<12.4f}")
                print(f"  vs Baseline PLS: {change_pls:+.1f}% {status_pls}")
            
            # vs baseline ENSEMBLE
            if "ENSEMBLE" in baseline_avg:
                baseline_ens = baseline_avg["ENSEMBLE"]
                change_ens = (stacked_rmse - baseline_ens) / baseline_ens * 100
                status_ens = "✓" if change_ens < 0 else "✗"
                print(f"  vs Baseline ENSEMBLE: {change_ens:+.1f}% {status_ens}")
            
            # vs previous enhanced STACKED
            if "STACKED" in enhanced_avg:
                enhanced_stacked = enhanced_avg["STACKED"]
                change_prev = (stacked_rmse - enhanced_stacked) / enhanced_stacked * 100
                status_prev = "✓" if change_prev < 0 else "✗"
                print(f"  vs Previous Enhanced: {change_prev:+.1f}% {status_prev}")
        
        print("-"*60)
        
        # R² comparison
        print("\nR² Comparison:")
        print("-"*60)
        baseline_r2 = baseline_df.groupby("model")["r2"].mean()
        enhanced_r2 = enhanced_df.groupby("model")["r2"].mean()
        improved_r2 = improved_df.groupby("model")["r2"].mean()
        
        print(f"{'Model':<15} {'Baseline':<12} {'Enhanced':<12} {'Improved':<12}")
        print("-"*60)
        
        if "STACKED" in improved_r2:
            stacked_r2 = improved_r2["STACKED"]
            baseline_pls_r2 = baseline_r2.get("PLS", 0)
            baseline_ens_r2 = baseline_r2.get("ENSEMBLE", 0)
            enhanced_stacked_r2 = enhanced_r2.get("STACKED", 0)
            
            print(f"{'STACKED':<15} {baseline_pls_r2:<12.4f} {enhanced_stacked_r2:<12.4f} {stacked_r2:<12.4f}")
            
            if baseline_ens_r2 > 0:
                r2_change = (stacked_r2 - baseline_ens_r2) / abs(baseline_ens_r2) * 100
                status = "✓" if stacked_r2 > baseline_ens_r2 else "✗"
                print(f"  vs Baseline ENSEMBLE: {r2_change:+.1f}% {status}")
        
        print("-"*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"Could not load comparison files: {e}\n")
    except Exception as e:
        print(f"Error during comparison: {e}\n")


if __name__ == "__main__":
    try:
        # Load dataset
        dataset = load_kiwi_dataset("input")
        
        # Run improved LOBO
        results_df = run_lobo_improved(
            dataset,
            output_dir="output_lobo_improved",
            enable_calibration=True,
            calibration_method='pds',
            enable_smote_lobo=False,  # DISABLED for robustness
            enable_vip=True,
            vip_threshold=0.8  # More conservative than 1.0
        )
        
        # Compare with baselines
        compare_with_baseline(results_df)
        
        print("\n[SUCCESS] LOBO Improved experiment complete!\n")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

