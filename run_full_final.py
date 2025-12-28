"""
Full Enhanced Experiment - CV + LOBO with STACKED (Final Version)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from kiwi_data_loader import load_kiwi_dataset, train_test_split_by_batches
from preprocessing import SpectralWindow, SNV, SavitzkyGolayDerivative
from preprocessing_advanced import apply_high_brix_augmentation
from feature_selection import VIPSelector
from model.advanced_ensemble import StackedEnsemble
from model.M1_pls_model import tune_pls
from model.M3_svr_model import tune_svr


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


class ManualPreprocessor:
    """Manual chaining - bypass broken Pipeline"""
    def __init__(self, wavelengths):
        self.window = SpectralWindow(wavelengths, wl_min=920.0, wl_max=1680.0)
        self.snv = SNV()
        self.savgol = SavitzkyGolayDerivative(window_length=21, polyorder=2, deriv=1)
        self.is_fitted = False
    
    def fit(self, X, y):
        self.window.fit(X, y)
        X_win = self.window.transform(X)
        self.snv.fit(X_win, y)
        X_snv = self.snv.transform(X_win)
        self.savgol.fit(X_snv, y)
        self.is_fitted = True
        return self
    
    def transform(self, X):
        X_win = self.window.transform(X)
        X_snv = self.snv.transform(X_win)
        X_sg = self.savgol.transform(X_snv)
        return X_sg
    
    def get_selected_wavelengths(self):
        return self.window.selected_wavelengths


def run_cv(X_all, y_all):
    """Run cross-validation"""
    print("\n" + "="*80)
    print("CROSS-VALIDATION")
    print("="*80 + "\n")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    
    for fold_idx, (tr, te) in enumerate(kf.split(X_all, y_all)):
        print(f"Fold {fold_idx + 1}/5")
        print("-"*40)
        
        X_tr, X_te = X_all[tr], X_all[te]
        y_tr, y_te = y_all[tr], y_all[te]
        
        # Base models
        pls = tune_pls(X_tr, y_tr, cv=2)
        svr = tune_svr(X_tr, y_tr, cv=2)
        
        # STACKED ensemble
        stacked = StackedEnsemble(
            base_models={'PLS': pls.best_estimator_, 'SVR': svr.best_estimator_},
            cv=3
        )
        stacked.fit(X_tr, y_tr)
        
        models = [
            ("PLS", pls.best_estimator_),
            ("SVR", svr.best_estimator_),
            ("STACKED", stacked)
        ]
        
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
            print(f"  {name:10s} RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        print()
    
    return pd.DataFrame(rows)


def run_lobo(dataset, preproc):
    """Run leave-one-batch-out"""
    print("\n" + "="*80)
    print("LEAVE-ONE-BATCH-OUT (LOBO)")
    print("="*80 + "\n")
    
    X_all_orig = preproc.transform(dataset.spectra)
    y_all_orig = dataset.targets
    batches = np.array(dataset.batches)
    
    rows = []
    unique_batches = list(dict.fromkeys(batches.tolist()))
    
    for held in unique_batches:
        print(f"Held-out batch: {held}")
        print("-"*40)
        
        train_mask, test_mask = train_test_split_by_batches(dataset.batches, held)
        X_tr_orig, X_te = X_all_orig[train_mask], X_all_orig[test_mask]
        y_tr_orig, y_te = y_all_orig[train_mask], y_all_orig[test_mask]
        
        if len(y_te) == 0:
            continue
        
        # Apply SMOTE + VIP on training data only
        X_tr, y_tr = apply_high_brix_augmentation(X_tr_orig, y_tr_orig, threshold=14.0, target_samples=150)
        
        vip = VIPSelector(n_components=10, threshold=1.0)
        vip.fit(X_tr, y_tr)
        X_tr = vip.transform(X_tr)
        X_te = vip.transform(X_te)
        
        # Train models
        pls = tune_pls(X_tr, y_tr, cv=2)
        svr = tune_svr(X_tr, y_tr, cv=2)
        
        stacked = StackedEnsemble(
            base_models={'PLS': pls.best_estimator_, 'SVR': svr.best_estimator_},
            cv=3
        )
        stacked.fit(X_tr, y_tr)
        
        models = [
            ("PLS", pls.best_estimator_),
            ("SVR", svr.best_estimator_),
            ("STACKED", stacked)
        ]
        
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
            print(f"  {name:10s} RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
        print()
    
    return pd.DataFrame(rows)


def main():
    print("\n" + "="*80)
    print("FULL ENHANCED EXPERIMENT - CV + LOBO")
    print("="*80 + "\n")
    
    # Load dataset
    dataset = load_kiwi_dataset("input")
    print(f"Dataset: {len(dataset.targets)} samples\n")
    
    # Build preprocessing
    print("Building preprocessing...")
    preproc = ManualPreprocessor(dataset.wavelengths)
    preproc.fit(dataset.spectra, dataset.targets)
    print(f"✓ Preprocessing ready\n")
    
    # Transform for CV
    X_all = preproc.transform(dataset.spectra)
    y_all = dataset.targets.copy()
    
    # Apply SMOTE + VIP for CV
    print("Applying SMOTE augmentation...")
    X_all, y_all = apply_high_brix_augmentation(X_all, y_all, threshold=14.0, target_samples=150)
    
    print("Applying VIP selection...")
    vip = VIPSelector(n_components=10, threshold=1.0)
    vip.fit(X_all, y_all)
    X_all = vip.transform(X_all)
    print(f"✓ Dataset ready: {len(y_all)} samples, {X_all.shape[1]} features\n")
    
    # Run CV
    cv_df = run_cv(X_all, y_all)
    
    # Run LOBO
    lobo_df = run_lobo(dataset, preproc)
    
    # Save results
    os.makedirs("output_enhanced", exist_ok=True)
    cv_path = "output_enhanced/results_cv_final.csv"
    lobo_path = "output_enhanced/results_lobo_final.csv"
    
    cv_df.to_csv(cv_path, index=False)
    lobo_df.to_csv(lobo_path, index=False)
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80 + "\n")
    
    print("CV Results:")
    cv_summary = cv_df.groupby("model").agg({"rmse": ["mean", "std"], "r2": ["mean", "std"]})
    for model in cv_summary.index:
        rmse_m = cv_summary.loc[model, ("rmse", "mean")]
        rmse_s = cv_summary.loc[model, ("rmse", "std")]
        r2_m = cv_summary.loc[model, ("r2", "mean")]
        r2_s = cv_summary.loc[model, ("r2", "std")]
        print(f"  {model:10s} RMSE={rmse_m:.4f}±{rmse_s:.4f}  R²={r2_m:.4f}±{r2_s:.4f}")
    
    print("\nLOBO Results:")
    lobo_summary = lobo_df.groupby("model").agg({"rmse": ["mean", "std"], "r2": ["mean", "std"]})
    for model in lobo_summary.index:
        rmse_m = lobo_summary.loc[model, ("rmse", "mean")]
        rmse_s = lobo_summary.loc[model, ("rmse", "std")]
        r2_m = lobo_summary.loc[model, ("r2", "mean")]
        r2_s = lobo_summary.loc[model, ("r2", "std")]
        print(f"  {model:10s} RMSE={rmse_m:.4f}±{rmse_s:.4f}  R²={r2_m:.4f}±{r2_s:.4f}")
    
    print(f"\n✓ Results saved:")
    print(f"  CV:   {cv_path}")
    print(f"  LOBO: {lobo_path}")
    
    print("\n[SUCCESS] Full experiment complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

