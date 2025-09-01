"""
language: python
# AI-Generated Code Header
# **Intent:** Ensemble model combining PLS, SVR, and MPHNN for improved performance
# **Optimization:** Multiple aggregation strategies, weighted combinations, meta-learning
# **Safety:** Robust error handling, fallback to individual models
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

try:
    from model.M1_pls_model import tune_pls
    from model.M3_svr_model import tune_svr
    from model.M8_mphnn_wrapper import MPHNNWrapper, MPHNN_AVAILABLE
    PLS_AVAILABLE = True
    SVR_AVAILABLE = True
except ImportError:
    PLS_AVAILABLE = False
    SVR_AVAILABLE = False
    MPHNN_AVAILABLE = False


class EnsembleModel(BaseEstimator, RegressorMixin):
    """Ensemble model combining PLS, SVR, and MPHNN predictions."""
    
    def __init__(self, 
                 method: str = "weighted_average",
                 use_pls: bool = True,
                 use_svr: bool = True,
                 use_mphnn: bool = True,
                 mphnn_config: Optional[Dict] = None,
                 random_state: int = 42):
        
        self.method = method
        self.use_pls = use_pls and PLS_AVAILABLE
        self.use_svr = use_svr and SVR_AVAILABLE
        self.use_mphnn = use_mphnn and MPHNN_AVAILABLE
        self.mphnn_config = mphnn_config or {}
        self.random_state = random_state
        
        # Model storage
        self.models = {}
        self.weights = {}
        self.meta_learner = None
        self.is_fitted = False
        
        # Validation
        if not any([self.use_pls, self.use_svr, self.use_mphnn]):
            raise ValueError("At least one model must be enabled")
        
        if method not in ["simple_average", "weighted_average", "stacking", "voting"]:
            raise ValueError(f"Unknown method: {method}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            wavelengths: Optional[np.ndarray] = None,
            batch_labels: Optional[np.ndarray] = None,
            temperatures: Optional[np.ndarray] = None,
            val_split: float = 0.2) -> 'EnsembleModel':
        """Fit the ensemble model."""
        
        # Split data for validation
        n_val = max(1, int(len(X) * val_split))
        val_idx = np.random.RandomState(self.random_state).choice(
            len(X), n_val, replace=False
        )
        train_idx = np.setdiff1d(np.arange(len(X)), val_idx)
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train individual models
        self._train_models(X_train, y_train, wavelengths, batch_labels, temperatures)
        
        # Get validation predictions
        val_preds = self._get_predictions(X_val, wavelengths, batch_labels, temperatures)
        
        # Compute ensemble weights/parameters
        self._compute_ensemble_params(y_val, val_preds)
        
        self.is_fitted = True
        return self
    
    def _train_models(self, X: np.ndarray, y: np.ndarray,
                      wavelengths: Optional[np.ndarray] = None,
                      batch_labels: Optional[np.ndarray] = None,
                      temperatures: Optional[np.ndarray] = None):
        """Train individual models."""
        
        # Train PLS
        if self.use_pls:
            try:
                pls_model = tune_pls(X, y, cv=3, n_components_grid=None)
                self.models['PLS'] = pls_model.best_estimator_
                print("✓ PLS model trained successfully")
            except Exception as e:
                print(f"✗ PLS training failed: {e}")
                self.use_pls = False
        
        # Train SVR
        if self.use_svr:
            try:
                svr_model = tune_svr(X, y, cv=3)
                self.models['SVR'] = svr_model.best_estimator_
                print("✓ SVR model trained successfully")
            except Exception as e:
                print(f"✗ SVR training failed: {e}")
                self.use_svr = False
        
        # Train MPHNN
        if self.use_mphnn:
            try:
                mphnn = MPHNNWrapper(
                    encoder_dim=128,
                    physics_dim=64,
                    domain_dim=32,
                    attention_heads=8,
                    dropout_rate=0.2,
                    beer_lambert_weight=1.0,
                    smoothness_weight=0.5,
                    contrastive_weight=0.1,
                    temperature_compensation=True,
                    max_epochs=200,
                    patience=20,
                    batch_size=32,
                    lr=1e-3,
                    random_state=self.random_state,
                    **self.mphnn_config
                )
                
                # Prepare batch labels and temperatures for MPHNN
                if batch_labels is not None:
                    # Convert string batch labels to integers
                    unique_batches = list(set(batch_labels))
                    batch_to_int = {batch: idx for idx, batch in enumerate(unique_batches)}
                    int_batch_labels = np.array([batch_to_int[batch] for batch in batch_labels])
                else:
                    int_batch_labels = None
                
                if temperatures is None:
                    temperatures = np.ones(len(X)) * 22.0
                
                mphnn.fit(X, y, wavelengths=wavelengths, 
                         batch_labels=int_batch_labels, 
                         temperatures=temperatures)
                self.models['MPHNN'] = mphnn
                print("✓ MPHNN model trained successfully")
            except Exception as e:
                print(f"✗ MPHNN training failed: {e}")
                self.use_mphnn = False
        
        print(f"Trained {len(self.models)} models: {list(self.models.keys())}")
    
    def _get_predictions(self, X: np.ndarray,
                         wavelengths: Optional[np.ndarray] = None,
                         batch_labels: Optional[np.ndarray] = None,
                         temperatures: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Get predictions from all models."""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if name == 'MPHNN':
                    # MPHNN needs special handling
                    if temperatures is None:
                        temp = np.ones(len(X)) * 22.0
                    else:
                        temp = temperatures
                    pred = model.predict(X, temperatures=temp)
                else:
                    # Standard sklearn models
                    pred = model.predict(X)
                
                predictions[name] = pred
            except Exception as e:
                print(f"Warning: Failed to get predictions from {name}: {e}")
                # Use zeros as fallback
                predictions[name] = np.zeros(len(X))
        
        return predictions
    
    def _compute_ensemble_params(self, y_true: np.ndarray, 
                                predictions: Dict[str, np.ndarray]):
        """Compute ensemble parameters based on validation performance."""
        
        if self.method == "simple_average":
            # Equal weights for all models
            self.weights = {name: 1.0 for name in predictions.keys()}
            
        elif self.method == "weighted_average":
            # Weight based on inverse RMSE
            self.weights = {}
            for name, pred in predictions.items():
                rmse = np.sqrt(mean_squared_error(y_true, pred))
                self.weights[name] = 1.0 / (rmse + 1e-8)  # Avoid division by zero
            
            # Normalize weights
            total_weight = sum(self.weights.values())
            self.weights = {name: w / total_weight for name, w in self.weights.items()}
            
        elif self.method == "stacking":
            # Train meta-learner
            meta_features = np.column_stack(list(predictions.values()))
            
            # Use Random Forest as meta-learner
            self.meta_learner = RandomForestRegressor(
                n_estimators=50, 
                random_state=self.random_state,
                max_depth=5
            )
            self.meta_learner.fit(meta_features, y_true)
            
        elif self.method == "voting":
            # Use multiple aggregation methods
            self.weights = {}
            for name, pred in predictions.items():
                rmse = np.sqrt(mean_squared_error(y_true, pred))
                self.weights[name] = 1.0 / (rmse + 1e-8)
            
            # Normalize weights
            total_weight = sum(self.weights.values())
            self.weights = {name: w / total_weight for name, w in self.weights.items()}
    
    def predict(self, X: np.ndarray,
                wavelengths: Optional[np.ndarray] = None,
                batch_labels: Optional[np.ndarray] = None,
                temperatures: Optional[np.ndarray] = None) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Get predictions from all models
        predictions = self._get_predictions(X, wavelengths, batch_labels, temperatures)
        
        if not predictions:
            raise RuntimeError("No models available for prediction")
        
        # Apply ensemble method
        if self.method == "simple_average":
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            
        elif self.method == "weighted_average":
            ensemble_pred = np.zeros(len(X))
            for name, pred in predictions.items():
                ensemble_pred += self.weights[name] * pred
                
        elif self.method == "stacking":
            meta_features = np.column_stack(list(predictions.values()))
            ensemble_pred = self.meta_learner.predict(meta_features)
            
        elif self.method == "voting":
            # Weighted average for voting
            ensemble_pred = np.zeros(len(X))
            for name, pred in predictions.items():
                ensemble_pred += self.weights[name] * pred
        
        return ensemble_pred
    
    def get_model_performance(self, X: np.ndarray, y: np.ndarray,
                             wavelengths: Optional[np.ndarray] = None,
                             batch_labels: Optional[np.ndarray] = None,
                             temperatures: Optional[np.ndarray] = None) -> Dict:
        """Get performance metrics for individual models and ensemble."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        # Get predictions from all models
        predictions = self._get_predictions(X, wavelengths, batch_labels, temperatures)
        
        # Calculate metrics for each model
        results = {}
        for name, pred in predictions.items():
            rmse = np.sqrt(mean_squared_error(y, pred))
            r2 = r2_score(y, pred)
            mae = np.mean(np.abs(y - pred))
            
            results[name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        
        # Calculate ensemble metrics
        ensemble_pred = self.predict(X, wavelengths, batch_labels, temperatures)
        ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
        ensemble_r2 = r2_score(y, ensemble_pred)
        ensemble_mae = np.mean(np.abs(y - ensemble_pred))
        
        results['ENSEMBLE'] = {
            'rmse': ensemble_rmse,
            'mae': ensemble_mae,
            'r2': ensemble_r2
        }
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available."""
        importance = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance[name] = np.abs(model.coef_)
            else:
                importance[name] = None
        
        return importance


def create_ensemble_pipeline(method: str = "weighted_average",
                           use_pls: bool = True,
                           use_svr: bool = True,
                           use_mphnn: bool = True,
                           mphnn_config: Optional[Dict] = None,
                           random_state: int = 42) -> EnsembleModel:
    """Create an ensemble model with specified configuration."""
    
    return EnsembleModel(
        method=method,
        use_pls=use_pls,
        use_svr=use_svr,
        use_mphnn=use_mphnn,
        mphnn_config=mphnn_config,
        random_state=random_state
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the ensemble model
    print("Testing Ensemble Model...")
    
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 50)
    y = np.random.randn(100)
    
    # Test different ensemble methods
    methods = ["simple_average", "weighted_average", "stacking"]
    
    for method in methods:
        print(f"\n--- Testing {method} ---")
        try:
            ensemble = create_ensemble_pipeline(method=method, use_mphnn=False)
            ensemble.fit(X, y)
            
            # Evaluate
            results = ensemble.get_model_performance(X, y)
            print("Results:")
            for model_name, metrics in results.items():
                print(f"  {model_name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
                
        except Exception as e:
            print(f"Failed: {e}")
    
    print("\nEnsemble model testing completed!")
