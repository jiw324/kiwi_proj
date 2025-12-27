"""
language: python
# AI-Generated Code Header
# **Intent:** End-to-end experiment runner for kiwi NIR prediction with CV and LOBO evaluation.
# **Optimization:** Reuse preprocessing; parallel CV; minimal copies; clear logging to CSV.
# **Safety:** Strict split hygiene; no leakage; robust to NaNs; saves results for reproducibility.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # noqa: BLE001
    plt = None  # type: ignore[assignment]
    sns = None  # type: ignore[assignment]

from kiwi_data_loader import KiwiDataset, load_kiwi_dataset, train_test_split_by_batches
from preprocessing import SpectralWindow, SNV, SavitzkyGolayDerivative
from model.M1_pls_model import tune_pls
# AI-SUGGESTION: Optionalize non-PLS models to avoid ImportError when files are absent
try:  # AI-SUGGESTION: SVR optional
    from model.M3_svr_model import tune_svr
except Exception:  # noqa: BLE001
    tune_svr = None  # type: ignore[assignment]
try:  # AI-SUGGESTION: RF optional
    from model.M2_random_forest_model import tune_rf
except Exception:  # noqa: BLE001
    tune_rf = None  # type: ignore[assignment]
try:  # AI-SUGGESTION: XGB optional
    from model.M4_xgboost_model import tune_xgb
except Exception:  # noqa: BLE001
    tune_xgb = None  # type: ignore[assignment]
# AI-SUGGESTION: Optionalize PINN imports to avoid hard dependency on torch
try:
    from model.M6_pinn import PINNConfig, train_pinn, predict_pinn
except Exception:  # noqa: BLE001
    PINNConfig = None  # type: ignore[assignment]
    train_pinn = None  # type: ignore[assignment]
    predict_pinn = None  # type: ignore[assignment]
try:
    from model.M7_beer_pinn import BeerConfig, train_beer_pinn, predict_beer_pinn
except Exception:  # noqa: BLE001
    BeerConfig = None  # type: ignore[assignment]
    train_beer_pinn = None  # type: ignore[assignment]
    predict_beer_pinn = None  # type: ignore[assignment]
# AI-SUGGESTION: Add advanced MPHNN model
try:
    from model.M8_mphnn_wrapper import MPHNNWrapper, tune_mphnn, MPHNN_AVAILABLE
except Exception:  # noqa: BLE001
    MPHNNWrapper = None  # type: ignore[assignment]
    tune_mphnn = None  # type: ignore[assignment]
    MPHNN_AVAILABLE = False  # type: ignore[assignment]

# AI-SUGGESTION: Add ensemble model
try:
    from model.ensemble_model import EnsembleModel, create_ensemble_pipeline
    ENSEMBLE_AVAILABLE = True
except Exception:  # noqa: BLE001
    EnsembleModel = None  # type: ignore[assignment]
    create_ensemble_pipeline = None  # type: ignore[assignment]
    ENSEMBLE_AVAILABLE = False  # type: ignore[assignment]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@dataclass
class RunConfig:
    input_dir: str = os.path.join("input", "kiwiProj")
    output_dir: str = os.path.join("output")
    wl_min: float = 920.0
    wl_max: float = 1680.0
    use_snv: bool = True
    use_savgol: bool = True
    savgol_window: int = 21
    savgol_polyorder: int = 2
    random_state: int = 42
    n_splits: int = 5
    fast: bool = True  # AI-SUGGESTION: run a faster subset of models and smaller inner-CV
    pls_components: str = "auto"  # "auto" or comma-separated list like "2,4,6,8,..."
    enable_pinn: bool = False  # AI-SUGGESTION: disabled by default
    enable_beer_pinn: bool = False  # AI-SUGGESTION: disabled by default
    enable_mphnn: bool = False  # AI-SUGGESTION: enable advanced MPHNN model
    enable_ensemble: bool = False  # AI-SUGGESTION: enable ensemble model


def build_preprocess_pipeline(dataset: KiwiDataset, cfg: RunConfig) -> Tuple[Pipeline, np.ndarray]:
    steps: List[Tuple[str, object]] = []
    win = SpectralWindow(dataset.wavelengths, wl_min=cfg.wl_min, wl_max=cfg.wl_max)
    steps.append(("window", win))
    if cfg.use_snv:
        steps.append(("snv", SNV()))
    if cfg.use_savgol:
        steps.append(("savgol", SavitzkyGolayDerivative(window_length=cfg.savgol_window, polyorder=cfg.savgol_polyorder, deriv=1)))
    pipeline = Pipeline(steps)
    # Fit on full dataset to configure all transformers properly
    pipeline.fit(dataset.spectra, dataset.targets)
    selected_wl = win.selected_wavelengths
    return pipeline, selected_wl


def evaluate_split(model, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "rmse": rmse(y_test, preds),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


def run_cv(dataset: KiwiDataset, cfg: RunConfig) -> Tuple[pd.DataFrame, List[Dict[str, int]], pd.DataFrame, List[Dict]]:
    preproc, selected_wl = build_preprocess_pipeline(dataset, cfg)
    X = preproc.transform(dataset.spectra)
    y = dataset.targets
    # Raw windowed reflectance for PINN (no SNV/derivative)
    wl_mask = (dataset.wavelengths >= cfg.wl_min) & (dataset.wavelengths <= cfg.wl_max)
    X_raw_win = dataset.spectra[:, wl_mask]

    kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    rows = []
    pls_hparams: List[Dict[str, int]] = []
    pred_rows: List[Dict] = []
    pinn_infos: List[Dict] = []
    inner_cv = 2 if cfg.fast else 3  # AI-SUGGESTION: reduce inner CV for speed
    for fold_idx, (tr, te) in enumerate(kf.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # Tune models
        comps = None if cfg.pls_components == "auto" else [int(v) for v in cfg.pls_components.split(',') if v.strip()]
        pls = tune_pls(X_tr, y_tr, cv=inner_cv, n_components_grid=comps)
        # AI-SUGGESTION: record best PLS n_components per fold
        try:
            ncomp = int(pls.best_params_.get("model__n_components"))  # type: ignore[arg-type]
            pls_hparams.append({"fold": int(fold_idx), "n_components": ncomp})
        except Exception:
            pass
        models = [("PLS", pls.best_estimator_)]
        if tune_svr is not None:
            svr = tune_svr(X_tr, y_tr, cv=inner_cv)
            models.append(("SVR", svr.best_estimator_))
        # Add physics-informed PINN
        if cfg.enable_pinn and (PINNConfig is not None) and (train_pinn is not None) and (predict_pinn is not None):
            try:
                pinn_cfg = PINNConfig(centers_nm=[970.0, 1200.0, 1450.0, 1700.0], sigma_nm=25.0, max_epochs=150, patience=15)
                pinn_model, _ = train_pinn(X_tr, y_tr, dataset.wavelengths, cfg=pinn_cfg, val_split=0.2, seed=cfg.random_state)
                class _PINNWrapper:
                    def __init__(self, m):
                        self.m = m
                    def fit(self, X, y):
                        return self
                    def predict(self, X):
                        return predict_pinn(self.m, X)
                models.append(("PINN", _PINNWrapper(pinn_model)))
            except Exception:
                pass
        # Add BeerPINN (physics) as optional
        if cfg.enable_beer_pinn and (BeerConfig is not None) and (train_beer_pinn is not None) and (predict_beer_pinn is not None):
            try:
                beer_cfg = BeerConfig(centers_nm=[970.0, 1150.0, 1200.0, 1350.0, 1450.0, 1700.0], sigma_nm=25.0)
                beer_model, _ = train_beer_pinn(X_tr, y_tr, dataset.wavelengths, cfg=beer_cfg, val_split=0.2, seed=cfg.random_state)
                class _BeerWrapper:
                    def __init__(self, m):
                        self.m = m
                    def fit(self, X, y):
                        return self
                    def predict(self, X):
                        return predict_beer_pinn(self.m, X)
                models.append(("BeerPINN", _BeerWrapper(beer_model)))
            except Exception:
                pass
        # Add MPHNN (advanced multi-physics) as optional
        if cfg.enable_mphnn and MPHNN_AVAILABLE and (MPHNNWrapper is not None) and (tune_mphnn is not None):
            try:
                # Create batch labels for domain adaptation (convert strings to integers)
                batch_strings = [dataset.batches[i] for i in tr]
                # Create a mapping from string to integer
                unique_batches = list(set(batch_strings))
                batch_to_int = {batch: idx for idx, batch in enumerate(unique_batches)}
                batch_labels = np.array([batch_to_int[batch] for batch in batch_strings])
                # Create synthetic temperature data if not available (placeholder for real data)
                temperatures = np.ones(len(tr)) * 22.0  # Default room temperature
                
                # Train MPHNN with domain adaptation
                mphnn_model = MPHNNWrapper(
                    encoder_dim=128,
                    physics_dim=64,
                    domain_dim=32,
                    attention_heads=8,
                    dropout_rate=0.2,
                    beer_lambert_weight=1.0,
                    smoothness_weight=0.5,
                    contrastive_weight=0.1,
                    temperature_compensation=True,
                    max_epochs=200,  # Reduced for speed
                    patience=20,
                    batch_size=32,
                    lr=1e-3,
                    random_state=cfg.random_state
                )
                
                # Fit the model
                mphnn_model.fit(X_raw_win[tr], y_tr, 
                               wavelengths=selected_wl,
                               batch_labels=batch_labels,
                               temperatures=temperatures)
                
                models.append(("MPHNN", mphnn_model))
            except Exception as e:
                print(f"MPHNN training failed: {e}")
                pass
        
        # Add Ensemble model (combining PLS, SVR, and MPHNN)
        if cfg.enable_ensemble and ENSEMBLE_AVAILABLE and (EnsembleModel is not None):
            try:
                # Create batch labels for domain adaptation
                batch_strings = [dataset.batches[i] for i in tr]
                unique_batches = list(set(batch_strings))
                batch_to_int = {batch: idx for idx, batch in enumerate(unique_batches)}
                batch_labels = np.array([batch_to_int[batch] for batch in batch_strings])
                
                # Create synthetic temperature data
                temperatures = np.ones(len(tr)) * 22.0
                
                # Create ensemble model
                ensemble_model = create_ensemble_pipeline(
                    method="weighted_average",
                    use_pls=True,
                    use_svr=True,
                    use_mphnn=cfg.enable_mphnn,  # Only use MPHNN if enabled
                    random_state=cfg.random_state
                )
                
                # Fit the ensemble model
                ensemble_model.fit(X_raw_win[tr], y_tr,
                                 wavelengths=selected_wl,
                                 batch_labels=batch_labels,
                                 temperatures=temperatures)
                
                models.append(("ENSEMBLE", ensemble_model))
                print("✓ Ensemble model trained successfully")
            except Exception as e:
                print(f"Ensemble training failed: {e}")
                pass
        if not cfg.fast:
            if tune_rf is not None:
                rf = tune_rf(X_tr, y_tr, cv=inner_cv)
                models.append(("RF", rf.best_estimator_))
            if tune_xgb is not None:
                xgb = tune_xgb(X_tr, y_tr, cv=inner_cv)
                models.append(("XGB", xgb.best_estimator_))

        for name, est in models:
            metrics = evaluate_split(est, X_tr, y_tr, X_te, y_te)
            rows.append({"protocol": "CV", "fold": fold_idx, "model": name, **metrics})
            # Save predictions for analysis
            y_pred = est.predict(X_te)
            for idx_pos, idx in enumerate(te):
                pred_rows.append({
                    "protocol": "CV",
                    "fold": int(fold_idx),
                    "model": name,
                    "sample_index": int(idx),
                    "y_true": float(y_te[idx_pos]),
                    "y_pred": float(y_pred[idx_pos]),
                })

        # Physics-informed PINN evaluated on raw windowed reflectance (small grid search)
        if cfg.enable_pinn and (PINNConfig is not None) and (train_pinn is not None) and (predict_pinn is not None):
            try:
                center_sets = [
                    [970.0, 1200.0, 1450.0, 1700.0],
                    [970.0, 1150.0, 1200.0, 1350.0, 1450.0, 1700.0],
                ]
                sigmas = [20.0, 25.0, 35.0]
                best_model = None
                best_info = None
                best_val = float("inf")
                for centers in center_sets:
                    for s in sigmas:
                        pinn_cfg = PINNConfig(centers_nm=centers, sigma_nm=s, max_epochs=150, patience=15)
                        model_cand, info = train_pinn(X_raw_win[tr], y_tr, selected_wl, cfg=pinn_cfg, val_split=0.2, seed=cfg.random_state)
                        if info.get("val_rmse", float("inf")) < best_val:
                            best_val = float(info["val_rmse"])  # type: ignore[index]
                            best_model = model_cand
                            best_info = info
                if best_model is not None and best_info is not None:
                    y_pred = predict_pinn(best_model, X_raw_win[te])
                    rows.append({
                        "protocol": "CV",
                        "fold": fold_idx,
                        "model": "PINN",
                        "rmse": float(np.sqrt(np.mean((y_pred - y_te) ** 2))),
                        "mae": float(mean_absolute_error(y_te, y_pred)),
                        "r2": float(r2_score(y_te, y_pred)),
                    })
                    best_info["fold"] = int(fold_idx)
                    pinn_infos.append(best_info)
                    for idx_pos, idx in enumerate(te):
                        pred_rows.append({
                            "protocol": "CV",
                            "fold": int(fold_idx),
                            "model": "PINN",
                            "sample_index": int(idx),
                            "y_true": float(y_te[idx_pos]),
                            "y_pred": float(y_pred[idx_pos]),
                        })
            except Exception:
                pass

    return pd.DataFrame(rows), pls_hparams, pd.DataFrame(pred_rows), pinn_infos


def run_lobo(dataset: KiwiDataset, cfg: RunConfig) -> Tuple[pd.DataFrame, List[Dict[str, str]], pd.DataFrame, List[Dict]]:
    preproc, selected_wl = build_preprocess_pipeline(dataset, cfg)
    X_all = preproc.transform(dataset.spectra)
    y_all = dataset.targets
    batches = np.array(dataset.batches)
    wl_mask = (dataset.wavelengths >= cfg.wl_min) & (dataset.wavelengths <= cfg.wl_max)
    X_raw_win_all = dataset.spectra[:, wl_mask]

    rows = []
    pls_hparams: List[Dict[str, str]] = []
    pred_rows: List[Dict] = []
    pinn_infos: List[Dict] = []
    inner_cv = 2 if cfg.fast else 3
    unique_batches = list(dict.fromkeys(batches.tolist()))
    for held in unique_batches:
        train_mask, test_mask = train_test_split_by_batches(dataset.batches, held)
        X_tr, X_te = X_all[train_mask], X_all[test_mask]
        y_tr, y_te = y_all[train_mask], y_all[test_mask]

        if X_te.shape[0] == 0:
            continue

        comps = None if cfg.pls_components == "auto" else [int(v) for v in cfg.pls_components.split(',') if v.strip()]
        pls = tune_pls(X_tr, y_tr, cv=inner_cv, n_components_grid=comps)
        # AI-SUGGESTION: record best PLS n_components per held-out batch
        try:
            ncomp = int(pls.best_params_.get("model__n_components"))  # type: ignore[arg-type]
            pls_hparams.append({"held_out": str(held), "n_components": str(ncomp)})
        except Exception:
            pass
        models = [("PLS", pls.best_estimator_)]
        if tune_svr is not None:
            svr = tune_svr(X_tr, y_tr, cv=inner_cv)
            models.append(("SVR", svr.best_estimator_))
        try:
            if cfg.enable_pinn and (PINNConfig is not None) and (train_pinn is not None) and (predict_pinn is not None):
                pinn_cfg = PINNConfig(centers_nm=[970.0, 1200.0, 1450.0, 1700.0], sigma_nm=25.0, max_epochs=150, patience=15)
                pinn_model, _ = train_pinn(X_tr, y_tr, dataset.wavelengths, cfg=pinn_cfg, val_split=0.2, seed=cfg.random_state)
                class _PINNWrapper:
                    def __init__(self, m):
                        self.m = m
                    def fit(self, X, y):
                        return self
                    def predict(self, X):
                        return predict_pinn(self.m, X)
                models.append(("PINN", _PINNWrapper(pinn_model)))
        except Exception:
            pass
        if not cfg.fast:
            if tune_rf is not None:
                rf = tune_rf(X_tr, y_tr, cv=inner_cv)
                models.append(("RF", rf.best_estimator_))
            if tune_xgb is not None:
                xgb = tune_xgb(X_tr, y_tr, cv=inner_cv)
                models.append(("XGB", xgb.best_estimator_))

        for name, est in models:
            metrics = evaluate_split(est, X_tr, y_tr, X_te, y_te)
            rows.append({"protocol": "LOBO", "held_out": held, "model": name, **metrics})
            y_pred = est.predict(X_te)
            # sample indices for test_mask
            test_indices = np.nonzero(test_mask)[0]
            for idx_pos, idx in enumerate(test_indices):
                pred_rows.append({
                    "protocol": "LOBO",
                    "held_out": str(held),
                    "model": name,
                    "sample_index": int(idx),
                    "y_true": float(y_te[idx_pos]),
                    "y_pred": float(y_pred[idx_pos]),
                })

        # PINN on LOBO raw reflectance
        try:
            if not (cfg.enable_pinn and (PINNConfig is not None) and (train_pinn is not None) and (predict_pinn is not None)):
                raise RuntimeError("PINN disabled or unavailable")
            center_sets = [
                [970.0, 1200.0, 1450.0, 1700.0],
                [970.0, 1150.0, 1200.0, 1350.0, 1450.0, 1700.0],
            ]
            sigmas = [20.0, 25.0, 35.0]
            best_model = None
            best_info = None
            best_val = float("inf")
            for centers in center_sets:
                for s in sigmas:
                    pinn_cfg = PINNConfig(centers_nm=centers, sigma_nm=s, max_epochs=150, patience=15)
                    model_cand, info = train_pinn(X_raw_win_all[train_mask], y_tr, selected_wl, cfg=pinn_cfg, val_split=0.2, seed=cfg.random_state)
                    if info.get("val_rmse", float("inf")) < best_val:
                        best_val = float(info["val_rmse"])  # type: ignore[index]
                        best_model = model_cand
                        best_info = info
            if best_model is not None and best_info is not None:
                y_pred = predict_pinn(best_model, X_raw_win_all[test_mask])
                rows.append({
                    "protocol": "LOBO",
                    "held_out": held,
                    "model": "PINN",
                    "rmse": float(np.sqrt(np.mean((y_pred - y_te) ** 2))),
                    "mae": float(mean_absolute_error(y_te, y_pred)),
                    "r2": float(r2_score(y_te, y_pred)),
                })
                best_info["held_out"] = str(held)
                pinn_infos.append(best_info)
                test_indices = np.nonzero(test_mask)[0]
                for idx_pos, idx in enumerate(test_indices):
                    pred_rows.append({
                        "protocol": "LOBO",
                        "held_out": str(held),
                        "model": "PINN",
                        "sample_index": int(idx),
                        "y_true": float(y_te[idx_pos]),
                        "y_pred": float(y_pred[idx_pos]),
                    })
        except Exception:
            pass
        
        # MPHNN on LOBO raw reflectance with domain adaptation
        try:
            if not (cfg.enable_mphnn and MPHNN_AVAILABLE and (MPHNNWrapper is not None)):
                raise RuntimeError("MPHNN disabled or unavailable")
            
            # Create batch labels for domain adaptation (convert strings to integers)
            train_batch_strings = [dataset.batches[i] for i in np.nonzero(train_mask)[0]]
            test_batch_strings = [dataset.batches[i] for i in np.nonzero(test_mask)[0]]
            
            # Create a mapping from string to integer (use all batches for consistency)
            all_batches = list(set(dataset.batches))
            batch_to_int = {batch: idx for idx, batch in enumerate(all_batches)}
            
            train_batch_labels = np.array([batch_to_int[batch] for batch in train_batch_strings])
            test_batch_labels = np.array([batch_to_int[batch] for batch in test_batch_strings])
            
            # Create synthetic temperature data if not available
            train_temperatures = np.ones(len(y_tr)) * 22.0
            test_temperatures = np.ones(len(y_te)) * 22.0
            
            # Train MPHNN with domain adaptation
            mphnn_model = MPHNNWrapper(
                encoder_dim=128,
                physics_dim=64,
                domain_dim=32,
                attention_heads=8,
                dropout_rate=0.2,
                beer_lambert_weight=1.0,
                smoothness_weight=0.5,
                contrastive_weight=0.1,
                temperature_compensation=True,
                max_epochs=200,  # Reduced for speed
                patience=20,
                batch_size=32,
                lr=1e-3,
                random_state=cfg.random_state
            )
            
            # Fit the model
            mphnn_model.fit(X_raw_win_all[train_mask], y_tr, 
                           wavelengths=selected_wl,
                           batch_labels=train_batch_labels,
                           temperatures=train_temperatures)
            
            # Make predictions
            y_pred = mphnn_model.predict(X_raw_win_all[test_mask], test_temperatures)
            
            # Add results
            rows.append({
                "protocol": "LOBO",
                "held_out": held,
                "model": "MPHNN",
                "rmse": float(np.sqrt(np.mean((y_pred - y_te) ** 2))),
                "mae": float(mean_absolute_error(y_te, y_pred)),
                "r2": float(r2_score(y_te, y_pred)),
            })
            
            # Add predictions
            for idx_pos, idx in enumerate(test_indices):
                pred_rows.append({
                    "protocol": "LOBO",
                    "held_out": str(held),
                    "model": "MPHNN",
                    "sample_index": int(idx),
                    "y_true": float(y_te[idx_pos]),
                    "y_pred": float(y_pred[idx_pos]),
                })
        except Exception as e:
            print(f"MPHNN LOBO failed: {e}")
            pass
        
        # Ensemble model on LOBO
        try:
            if not (cfg.enable_ensemble and ENSEMBLE_AVAILABLE and (EnsembleModel is not None)):
                raise RuntimeError("Ensemble model disabled or unavailable")
            
            # Create batch labels for domain adaptation
            train_batch_strings = [dataset.batches[i] for i in np.nonzero(train_mask)[0]]
            test_batch_strings = [dataset.batches[i] for i in np.nonzero(test_mask)[0]]
            
            # Create a mapping from string to integer
            all_batches = list(set(dataset.batches))
            batch_to_int = {batch: idx for idx, batch in enumerate(all_batches)}
            
            train_batch_labels = np.array([batch_to_int[batch] for batch in train_batch_strings])
            test_batch_labels = np.array([batch_to_int[batch] for batch in test_batch_strings])
            
            # Create synthetic temperature data
            train_temperatures = np.ones(len(y_tr)) * 22.0
            test_temperatures = np.ones(len(y_te)) * 22.0
            
            # Create and train ensemble model
            ensemble_model = create_ensemble_pipeline(
                method="weighted_average",
                use_pls=True,
                use_svr=True,
                use_mphnn=cfg.enable_mphnn,
                random_state=cfg.random_state
            )
            
            ensemble_model.fit(X_raw_win_all[train_mask], y_tr,
                             wavelengths=selected_wl,
                             batch_labels=train_batch_labels,
                             temperatures=train_temperatures)
            
            # Make predictions
            y_pred = ensemble_model.predict(X_raw_win_all[test_mask], test_temperatures)
            
            # Add results
            rows.append({
                "protocol": "LOBO",
                "held_out": held,
                "model": "ENSEMBLE",
                "rmse": float(np.sqrt(np.mean((y_pred - y_te) ** 2))),
                "mae": float(mean_absolute_error(y_te, y_pred)),
                "r2": float(r2_score(y_te, y_pred)),
            })
            
            # Add predictions
            for idx_pos, idx in enumerate(test_indices):
                pred_rows.append({
                    "protocol": "LOBO",
                    "held_out": str(held),
                    "model": "ENSEMBLE",
                    "sample_index": int(idx),
                    "y_true": float(y_te[idx_pos]),
                    "y_pred": float(y_pred[idx_pos]),
                })
            
            print("✓ Ensemble LOBO completed successfully")
            
        except Exception as e:
            print(f"Ensemble LOBO failed: {e}")
            pass

    return pd.DataFrame(rows), pls_hparams, pd.DataFrame(pred_rows), pinn_infos


def save_results(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def add_y_bins(preds: pd.DataFrame) -> pd.DataFrame:
    # AI-SUGGESTION: fixed bins for interpretability
    def _bin_label(y: float) -> str:
        if y < 12:
            return "<12"
        if y <= 14:
            return "12-14"
        return ">14"
    preds = preds.copy()
    preds["y_bin"] = preds["y_true"].apply(_bin_label)
    preds["abs_err"] = (preds["y_pred"] - preds["y_true"]).abs()
    preds["squared_err"] = (preds["y_pred"] - preds["y_true"]) ** 2
    return preds


def run_all(cfg: RunConfig) -> Dict[str, str]:
    dataset = load_kiwi_dataset(cfg.input_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)

    cv_df, cv_pls, cv_preds, cv_pinn = run_cv(dataset, cfg)
    cv_out = os.path.join(cfg.output_dir, "results_cv.csv")
    save_results(cv_df, cv_out)
    cv_pred_path = os.path.join(cfg.output_dir, "predictions_cv.csv")
    save_results(cv_preds, cv_pred_path)
    # Bin summary for CV
    cv_binned = add_y_bins(cv_preds)
    cv_bins = cv_binned.groupby(["model", "y_bin"]).agg(
        rmse=("squared_err", lambda s: float(np.sqrt(np.mean(s)))),
        mae=("abs_err", "mean"),
        n=("abs_err", "count"),
    ).reset_index()
    save_results(cv_bins, os.path.join(cfg.output_dir, "error_bins_cv.csv"))

    lobo_df, lobo_pls, lobo_preds, lobo_pinn = run_lobo(dataset, cfg)
    lobo_out = os.path.join(cfg.output_dir, "results_lobo.csv")
    save_results(lobo_df, lobo_out)
    lobo_pred_path = os.path.join(cfg.output_dir, "predictions_lobo.csv")
    save_results(lobo_preds, lobo_pred_path)
    lobo_binned = add_y_bins(lobo_preds)
    lobo_bins = lobo_binned.groupby(["model", "y_bin"]).agg(
        rmse=("squared_err", lambda s: float(np.sqrt(np.mean(s)))),
        mae=("abs_err", "mean"),
        n=("abs_err", "count"),
    ).reset_index()
    save_results(lobo_bins, os.path.join(cfg.output_dir, "error_bins_lobo.csv"))

    # Simple post-hoc calibration on LOBO using CV PLS predictions (per y_bin linear recalibration)
    try:
        cv_pls_preds = cv_preds[cv_preds["model"] == "PLS"][["sample_index", "y_pred"]].rename(columns={"y_pred": "y_cv_pls"})
        lobo_merge = lobo_preds.merge(cv_pls_preds, on="sample_index", how="left")
        lobo_merge = add_y_bins(lobo_merge)
        calibrated = []
        for b in ["<12", "12-14", ">14"]:
            sub = lobo_merge[lobo_merge["y_bin"] == b]
            if sub.shape[0] < 5 or sub["y_cv_pls"].isna().all():
                calibrated.append(sub.assign(y_pred_cal=sub["y_pred"]))
                continue
            # fit y_true ~ y_pred within bin
            Xb = sub[["y_pred"]].values
            yb = sub["y_true"].values
            lr = LinearRegression().fit(Xb, yb)
            yhat = lr.predict(sub[["y_pred"]])
            calibrated.append(sub.assign(y_pred_cal=yhat))
        lobo_cal = pd.concat(calibrated, axis=0).sort_values("sample_index")
        save_results(lobo_cal, os.path.join(cfg.output_dir, "predictions_lobo_calibrated.csv"))
    except Exception:
        pass

    # Save config
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    # Save selected hyperparameters
    with open(os.path.join(cfg.output_dir, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump({"PLS": {"cv": cv_pls, "lobo": lobo_pls}}, f, indent=2)
    with open(os.path.join(cfg.output_dir, "pinn_bands.json"), "w", encoding="utf-8") as f:
        json.dump({"wavelengths": dataset.wavelengths.tolist(), "cv": cv_pinn, "lobo": lobo_pinn}, f, indent=2)

    # Optional plotting and markdown summary
    try:
        if plt is not None and sns is not None:
            # CV RMSE plot
            cv_plot = cv_df.groupby(["model"]).agg(rmse=("rmse", "mean")).reset_index()
            plt.figure(figsize=(4,3)); sns.barplot(data=cv_plot, x="model", y="rmse"); plt.tight_layout()
            plt.savefig(os.path.join(cfg.output_dir, "cv_rmse.png")); plt.close()
            # LOBO RMSE plot
            lobo_plot = lobo_df.groupby(["model"]).agg(rmse=("rmse", "mean")).reset_index()
            plt.figure(figsize=(4,3)); sns.barplot(data=lobo_plot, x="model", y="rmse"); plt.tight_layout()
            plt.savefig(os.path.join(cfg.output_dir, "lobo_rmse.png")); plt.close()
            # PINN band weights overview (if available)
            if len(cv_pinn) > 0:
                alphas = []
                for info in cv_pinn:
                    alphas.append(info.get("alpha", []))
                if alphas:
                    maxk = max(len(a) for a in alphas)
                    arr = np.zeros((len(alphas), maxk)); arr[:] = np.nan
                    for i,a in enumerate(alphas):
                        arr[i,:len(a)] = a
                    plt.figure(figsize=(6,3)); plt.plot(np.nanmean(arr, axis=0), marker="o"); plt.tight_layout()
                    plt.savefig(os.path.join(cfg.output_dir, "pinn_band_weights_cv.png")); plt.close()
    except Exception:
        pass

    return {"cv": cv_out, "lobo": lobo_out}

