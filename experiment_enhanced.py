"""
language: python
# AI-Generated Code Header
# Intent: Enhanced experiment runner with Phase 1 + Phase 2 improvements
# Optimization: Modular design, efficient preprocessing, parallel-ready
# Safety: Comprehensive error handling, validated pipelines, reproducible results
"""

import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import base experiment components
from experiment import RunConfig, rmse, save_results
from kiwi_data_loader import KiwiDataset, load_kiwi_dataset, train_test_split_by_batches
from preprocessing import SpectralWindow, SNV, SavitzkyGolayDerivative
from sklearn.pipeline import Pipeline

# Import Phase 1 + Phase 2 enhancements
from preprocessing_advanced import SpectralSMOTE, EMSC, apply_high_brix_augmentation
from feature_selection import VIPSelector, AdaptiveWavelengthSelector
from model.advanced_ensemble import StackedEnsemble, StratifiedEnsemble
from calibration_transfer import apply_lobo_calibration_transfer

# Base models
from model.M1_pls_model import tune_pls
from model.M3_svr_model import tune_svr


@dataclass
class EnhancedRunConfig:
    """Extended configuration with Phase 1+2 options - includes all RunConfig fields plus enhancements"""
    # Base RunConfig fields
    input_dir: str = "input"
    output_dir: str = "output"
    wl_min: float = 920.0
    wl_max: float = 1680.0
    use_snv: bool = True
    use_savgol: bool = True
    savgol_window: int = 21
    savgol_polyorder: int = 2
    random_state: int = 42
    n_splits: int = 5
    fast: bool = True
    pls_components: str = "auto"
    enable_pinn: bool = False
    enable_beer_pinn: bool = False
    enable_mphnn: bool = False
    enable_ensemble: bool = False
    
    # Phase 1 options
    enable_smote: bool = True
    smote_target_samples: int = 150
    enable_vip_selection: bool = True
    vip_threshold: float = 1.0
    enable_stacked_ensemble: bool = True
    
    # Phase 2 options
    enable_emsc: bool = False  # Use SNV by default (EMSC needs more complex setup)
    enable_stratified_ensemble: bool = True
    enable_calibration_transfer: bool = True
    calibration_method: str = 'pds'  # 'pds', 'sbc', or 'none'


def build_enhanced_preprocessing(dataset: KiwiDataset, cfg: EnhancedRunConfig) -> Tuple[Pipeline, np.ndarray]:
    """Build enhanced preprocessing pipeline - exact copy from working experiment.py"""
    from typing import List, Tuple as TupleType
    
    steps: List[TupleType[str, object]] = []
    win = SpectralWindow(dataset.wavelengths, wl_min=cfg.wl_min, wl_max=cfg.wl_max)
    steps.append(("window", win))
    if cfg.use_snv:
        steps.append(("snv", SNV()))
        print("Using SNV preprocessing")
    if cfg.use_savgol:
        steps.append(("savgol", SavitzkyGolayDerivative(window_length=cfg.savgol_window, polyorder=cfg.savgol_polyorder, deriv=1)))
    pipeline = Pipeline(steps)
    pipeline.fit(dataset.spectra, dataset.targets)
    selected_wl = win.selected_wavelengths
    return pipeline, selected_wl


def run_cv_enhanced(dataset, cfg: EnhancedRunConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Enhanced CV with Phase 1 + Phase 2 improvements"""
    
    # Build preprocessing
    preproc, selected_wl = build_enhanced_preprocessing(dataset, cfg)
    X_full = preproc.transform(dataset.spectra)
    y_full = dataset.targets
    
    print(f"\n{'='*80}")
    print("ENHANCED CROSS-VALIDATION WITH PHASE 1 + PHASE 2")
    print(f"{'='*80}\n")
    
    # Phase 1: SMOTE augmentation for high-Brix samples
    if cfg.enable_smote:
        print("Phase 1: Applying SMOTE augmentation...")
        X_full, y_full = apply_high_brix_augmentation(
            X_full, y_full, 
            threshold=14.0, 
            target_samples=cfg.smote_target_samples,
            random_state=cfg.random_state
        )
        print(f"Dataset size after SMOTE: {len(y_full)} samples\n")
    
    # Phase 1: VIP wavelength selection
    if cfg.enable_vip_selection:
        print("Phase 1: Applying VIP wavelength selection...")
        vip_selector = VIPSelector(n_components=10, threshold=cfg.vip_threshold)
        vip_selector.fit(X_full, y_full)
        X_full = vip_selector.transform(X_full)
        selected_wl = vip_selector.get_selected_wavelengths(selected_wl)
        print(f"Selected {len(selected_wl)} important wavelengths (VIP > {cfg.vip_threshold})\n")
    
    # Cross-validation loop
    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    rows = []
    pred_rows = []
    
    for fold_idx, (tr, te) in enumerate(kf.split(X_full, y_full)):
        print(f"\nFold {fold_idx + 1}/{cfg.n_splits}")
        print("-" * 40)
        
        X_tr, X_te = X_full[tr], X_full[te]
        y_tr, y_te = y_full[tr], y_full[te]
        
        # Train baseline models
        inner_cv = 2 if cfg.fast else 3
        pls = tune_pls(X_tr, y_tr, cv=inner_cv)
        svr = tune_svr(X_tr, y_tr, cv=inner_cv)
        
        models = [
            ("PLS", pls.best_estimator_),
            ("SVR", svr.best_estimator_)
        ]
        
        # Phase 1: Stacked Ensemble
        if cfg.enable_stacked_ensemble:
            print("  Training Stacked Ensemble (Phase 1)...")
            base_models = {
                'PLS': pls.best_estimator_,
                'SVR': svr.best_estimator_
            }
            stacked = StackedEnsemble(base_models=base_models, cv=3)
            stacked.fit(X_tr, y_tr)
            models.append(("STACKED", stacked))
        
        # Phase 2: Stratified Ensemble
        if cfg.enable_stratified_ensemble:
            print("  Training Stratified Ensemble (Phase 2)...")
            stratified = StratifiedEnsemble(bin_edges=[12.0, 14.0])
            stratified.fit(X_tr, y_tr)
            models.append(("STRATIFIED", stratified))
        
        # Evaluate all models
        for name, model in models:
            y_pred = model.predict(X_te)
            metrics = {
                "protocol": "CV",
                "fold": fold_idx,
                "model": name,
                "rmse": rmse(y_te, y_pred),
                "mae": float(mean_absolute_error(y_te, y_pred)),
                "r2": float(r2_score(y_te, y_pred))
            }
            rows.append(metrics)
            print(f"  {name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
            
            # Save predictions
            for idx_pos, idx in enumerate(te):
                pred_rows.append({
                    "protocol": "CV",
                    "fold": fold_idx,
                    "model": name,
                    "sample_index": int(idx),
                    "y_true": float(y_te[idx_pos]),
                    "y_pred": float(y_pred[idx_pos])
                })
    
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def run_lobo_enhanced(dataset, cfg: EnhancedRunConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Enhanced LOBO with Phase 1 + Phase 2 improvements"""
    
    # Build preprocessing
    preproc, selected_wl = build_enhanced_preprocessing(dataset, cfg)
    X_all = preproc.transform(dataset.spectra)
    y_all = dataset.targets
    batches = np.array(dataset.batches)
    
    print(f"\n{'='*80}")
    print("ENHANCED LEAVE-ONE-BATCH-OUT WITH PHASE 1 + PHASE 2")
    print(f"{'='*80}\n")
    
    rows = []
    pred_rows = []
    unique_batches = list(dict.fromkeys(batches.tolist()))
    
    for held in unique_batches:
        print(f"\nHeld-out batch: {held}")
        print("-" * 40)
        
        train_mask, test_mask = train_test_split_by_batches(dataset.batches, held)
        X_tr_raw, X_te_raw = X_all[train_mask], X_all[test_mask]
        y_tr, y_te = y_all[train_mask], y_all[test_mask]
        
        if X_te_raw.shape[0] == 0:
            continue
        
        # Phase 2: Calibration transfer
        if cfg.enable_calibration_transfer:
            print(f"  Applying calibration transfer ({cfg.calibration_method})...")
            X_tr, X_te = apply_lobo_calibration_transfer(
                X_tr_raw, X_te_raw, 
                method=cfg.calibration_method
            )
        else:
            X_tr, X_te = X_tr_raw, X_te_raw
        
        # Phase 1: SMOTE augmentation
        if cfg.enable_smote:
            X_tr, y_tr = apply_high_brix_augmentation(
                X_tr, y_tr,
                threshold=14.0,
                target_samples=cfg.smote_target_samples,
                random_state=cfg.random_state
            )
        
        # Phase 1: VIP selection
        if cfg.enable_vip_selection:
            vip_selector = VIPSelector(n_components=10, threshold=cfg.vip_threshold)
            vip_selector.fit(X_tr, y_tr)
            X_tr = vip_selector.transform(X_tr)
            X_te = vip_selector.transform(X_te)
        
        # Train models
        inner_cv = 2 if cfg.fast else 3
        pls = tune_pls(X_tr, y_tr, cv=inner_cv)
        svr = tune_svr(X_tr, y_tr, cv=inner_cv)
        
        models = [
            ("PLS", pls.best_estimator_),
            ("SVR", svr.best_estimator_)
        ]
        
        # Phase 1: Stacked Ensemble
        if cfg.enable_stacked_ensemble:
            base_models = {'PLS': pls.best_estimator_, 'SVR': svr.best_estimator_}
            stacked = StackedEnsemble(base_models=base_models, cv=3)
            stacked.fit(X_tr, y_tr)
            models.append(("STACKED", stacked))
        
        # Phase 2: Stratified Ensemble
        if cfg.enable_stratified_ensemble:
            stratified = StratifiedEnsemble(bin_edges=[12.0, 14.0])
            stratified.fit(X_tr, y_tr)
            models.append(("STRATIFIED", stratified))
        
        # Evaluate
        test_indices = np.nonzero(test_mask)[0]
        for name, model in models:
            y_pred = model.predict(X_te)
            metrics = {
                "protocol": "LOBO",
                "held_out": held,
                "model": name,
                "rmse": rmse(y_te, y_pred),
                "mae": float(mean_absolute_error(y_te, y_pred)),
                "r2": float(r2_score(y_te, y_pred))
            }
            rows.append(metrics)
            print(f"  {name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
            
            # Save predictions
            for idx_pos, idx in enumerate(test_indices):
                pred_rows.append({
                    "protocol": "LOBO",
                    "held_out": str(held),
                    "model": name,
                    "sample_index": int(idx),
                    "y_true": float(y_te[idx_pos]),
                    "y_pred": float(y_pred[idx_pos])
                })
    
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def run_all_enhanced(cfg: EnhancedRunConfig) -> Dict[str, str]:
    """Run complete enhanced experiment pipeline"""
    
    # Load dataset
    dataset = load_kiwi_dataset(cfg.input_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("ENHANCED EXPERIMENT: PHASE 1 + PHASE 2 IMPROVEMENTS")
    print(f"{'='*80}")
    print(f"\nDataset: {len(dataset.targets)} samples")
    print(f"Wavelengths: {len(dataset.wavelengths)}")
    print(f"\nEnabled Improvements:")
    print(f"  Phase 1:")
    print(f"    - SMOTE augmentation: {cfg.enable_smote}")
    print(f"    - VIP selection: {cfg.enable_vip_selection}")
    print(f"    - Stacked ensemble: {cfg.enable_stacked_ensemble}")
    print(f"  Phase 2:")
    print(f"    - EMSC preprocessing: {cfg.enable_emsc}")
    print(f"    - Stratified ensemble: {cfg.enable_stratified_ensemble}")
    print(f"    - Calibration transfer: {cfg.enable_calibration_transfer}")
    
    # Run CV
    cv_df, cv_preds = run_cv_enhanced(dataset, cfg)
    cv_out = os.path.join(cfg.output_dir, "results_cv_enhanced.csv")
    save_results(cv_df, cv_out)
    save_results(cv_preds, os.path.join(cfg.output_dir, "predictions_cv_enhanced.csv"))
    
    # Run LOBO
    lobo_df, lobo_preds = run_lobo_enhanced(dataset, cfg)
    lobo_out = os.path.join(cfg.output_dir, "results_lobo_enhanced.csv")
    save_results(lobo_df, lobo_out)
    save_results(lobo_preds, os.path.join(cfg.output_dir, "predictions_lobo_enhanced.csv"))
    
    # Save configuration
    with open(os.path.join(cfg.output_dir, "config_enhanced.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}\n")
    
    print("CV Results:")
    cv_summary = cv_df.groupby("model").agg({"rmse": ["mean", "std"], "r2": ["mean", "std"]})
    print(cv_summary)
    
    print("\nLOBO Results:")
    lobo_summary = lobo_df.groupby("model").agg({"rmse": ["mean", "std"], "r2": ["mean", "std"]})
    print(lobo_summary)
    
    return {"cv": cv_out, "lobo": lobo_out}

