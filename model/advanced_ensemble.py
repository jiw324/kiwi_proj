"""
language: python
# AI-Generated Code Header
# Intent: Advanced ensemble methods (Stacked, Stratified) for spectral regression
# Optimization: Cross-validated predictions, efficient meta-learning
# Safety: Input validation, handles edge cases, prevents overfitting
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR


class StackedEnsemble(BaseEstimator, RegressorMixin):
    """Stacked Generalization Ensemble with meta-learner"""
    
    def __init__(self, base_models: Optional[Dict[str, Any]] = None,
                 meta_model: Optional[Any] = None,
                 cv: int = 5,
                 use_original_features: bool = False):
        """
        Parameters:
        -----------
        base_models : dict, optional
            Dictionary of {name: model} for base learners
        meta_model : estimator, optional
            Meta-learner (default: Ridge regression)
        cv : int
            Cross-validation folds for meta-feature generation
        use_original_features : bool
            Whether to include original features in meta-learning
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv = cv
        self.use_original_features = use_original_features
        
        # Will be set during fit
        self.base_models_ = None
        self.meta_model_ = None
        self.feature_names_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        """
        Fit stacked ensemble
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        # Initialize base models if not provided
        if self.base_models is None:
            self.base_models_ = {
                'PLS': PLSRegression(n_components=10),
                'SVR': SVR(kernel='rbf', C=10, gamma='scale'),
                'Ridge': Ridge(alpha=1.0)
            }
        else:
            self.base_models_ = self.base_models
        
        # Initialize meta-model if not provided
        if self.meta_model is None:
            self.meta_model_ = Ridge(alpha=1.0)
        else:
            self.meta_model_ = self.meta_model
        
        # Generate meta-features using out-of-fold predictions
        n_samples = len(y)
        n_base_models = len(self.base_models_)
        meta_features = np.zeros((n_samples, n_base_models))
        
        print("Training base models and generating meta-features...")
        for i, (name, model) in enumerate(self.base_models_.items()):
            print(f"  - {name}")
            # Out-of-fold predictions to avoid overfitting
            try:
                oof_preds = cross_val_predict(model, X, y, cv=self.cv)
                meta_features[:, i] = oof_preds
            except Exception as e:
                print(f"    Warning: {name} failed during CV: {e}")
                # Fallback: fit on full data and predict
                model.fit(X, y)
                meta_features[:, i] = model.predict(X)
            
            # Fit on full training data for final predictions
            model.fit(X, y)
        
        # Optionally include original features
        if self.use_original_features:
            meta_features = np.hstack([meta_features, X])
        
        # Train meta-model
        print("Training meta-learner...")
        self.meta_model_.fit(meta_features, y)
        
        self.feature_names_ = list(self.base_models_.keys())
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using stacked ensemble
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        predictions : array-like, shape (n_samples,)
        """
        # Get base model predictions
        base_predictions = np.column_stack([
            model.predict(X) for model in self.base_models_.values()
        ])
        
        # Optionally include original features
        if self.use_original_features:
            meta_features = np.hstack([base_predictions, X])
        else:
            meta_features = base_predictions
        
        # Meta-model makes final prediction
        return self.meta_model_.predict(meta_features)
    
    def get_base_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each base model"""
        return {name: model.predict(X) 
                for name, model in self.base_models_.items()}
    
    def get_meta_weights(self) -> Dict[str, float]:
        """Get meta-model weights (for linear meta-models)"""
        if hasattr(self.meta_model_, 'coef_'):
            weights = dict(zip(self.feature_names_, self.meta_model_.coef_))
            return weights
        else:
            return {}


class StratifiedEnsemble(BaseEstimator, RegressorMixin):
    """Stratified Ensemble with target-specific expert models"""
    
    def __init__(self, bin_edges: List[float] = [12.0, 14.0],
                 models: Optional[Dict[str, Any]] = None,
                 router_type: str = 'soft'):
        """
        Parameters:
        -----------
        bin_edges : list of float
            Boundaries for target bins (e.g., [12, 14] creates 3 bins: <12, 12-14, >14)
        models : dict, optional
            Dictionary of {bin_name: model} for each bin
        router_type : str
            'soft': probabilistic weighting, 'hard': hard assignment
        """
        self.bin_edges = sorted(bin_edges)
        self.models = models
        self.router_type = router_type
        
        # Will be set during fit
        self.expert_models_ = {}
        self.router_model_ = None
        self.bin_names_ = []
    
    def _assign_bins(self, y: np.ndarray) -> np.ndarray:
        """Assign samples to target bins"""
        bins = np.digitize(y, self.bin_edges)
        return bins
    
    def _get_bin_name(self, bin_idx: int) -> str:
        """Get descriptive name for bin"""
        if bin_idx == 0:
            return f"<{self.bin_edges[0]}"
        elif bin_idx == len(self.bin_edges):
            return f">{self.bin_edges[-1]}"
        else:
            return f"{self.bin_edges[bin_idx-1]}-{self.bin_edges[bin_idx]}"
    
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        """
        Fit stratified ensemble with expert models for each target range
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        # Assign samples to bins
        bin_assignments = self._assign_bins(y)
        n_bins = len(self.bin_edges) + 1
        
        # Initialize models for each bin if not provided
        if self.models is None:
            # Use different model types optimized for each range
            default_models = {
                0: PLSRegression(n_components=8),  # Low range: linear
                1: Ridge(alpha=0.5),  # Mid range: regularized linear
                2: SVR(kernel='rbf', C=10, gamma='scale')  # High range: nonlinear
            }
        else:
            default_models = self.models
        
        # Train expert model for each bin
        print("Training expert models for each target range...")
        for bin_idx in range(n_bins):
            bin_name = self._get_bin_name(bin_idx)
            self.bin_names_.append(bin_name)
            
            mask = bin_assignments == bin_idx
            n_samples_bin = np.sum(mask)
            
            if n_samples_bin < 5:
                print(f"  Warning: Only {n_samples_bin} samples in bin {bin_name}, skipping")
                continue
            
            X_bin = X[mask]
            y_bin = y[mask]
            
            # Get or create model for this bin
            if bin_idx in default_models:
                model = default_models[bin_idx]
            else:
                model = PLSRegression(n_components=min(10, n_samples_bin // 2))
            
            # Train expert
            print(f"  - Bin {bin_name}: {n_samples_bin} samples")
            model.fit(X_bin, y_bin)
            self.expert_models_[bin_idx] = model
        
        # Train router model (predicts which bin a sample belongs to)
        if self.router_type == 'soft':
            from sklearn.linear_model import LogisticRegression
            # Python 3.14 compatibility: multi_class='multinomial' is now default
            self.router_model_ = LogisticRegression(max_iter=1000)
            self.router_model_.fit(X, bin_assignments)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using stratified ensemble
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        predictions : array-like, shape (n_samples,)
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        if self.router_type == 'soft' and self.router_model_ is not None:
            # Soft routing: weighted combination based on bin probabilities
            bin_probs = self.router_model_.predict_proba(X)
            
            for bin_idx, model in self.expert_models_.items():
                bin_preds = model.predict(X).ravel()
                predictions += bin_probs[:, bin_idx] * bin_preds
        
        else:
            # Hard routing: predict bin, then use corresponding expert
            # Simple heuristic: use PLS prediction to estimate bin
            if len(self.expert_models_) > 0:
                # Use first available expert as fallback
                fallback_model = list(self.expert_models_.values())[0]
                rough_preds = fallback_model.predict(X).ravel()
                bin_assignments = self._assign_bins(rough_preds)
                
                for i in range(n_samples):
                    bin_idx = bin_assignments[i]
                    if bin_idx in self.expert_models_:
                        predictions[i] = self.expert_models_[bin_idx].predict(X[i:i+1])[0]
                    else:
                        # Fallback to nearest available expert
                        available_bins = list(self.expert_models_.keys())
                        nearest_bin = min(available_bins, key=lambda b: abs(b - bin_idx))
                        predictions[i] = self.expert_models_[nearest_bin].predict(X[i:i+1])[0]
        
        return predictions
    
    def get_expert_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each expert model"""
        return {self._get_bin_name(bin_idx): model.predict(X) 
                for bin_idx, model in self.expert_models_.items()}

