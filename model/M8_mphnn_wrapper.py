"""
language: python
# AI-Generated Code Header
# **Intent:** Wrapper to integrate MPHNN with existing experiment framework
# **Optimization:** Compatible with CV/LOBO protocols, sklearn-like interface
# **Safety:** Handles missing dependencies gracefully, provides fallback options
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

try:
    from model.M8_mphnn import MPHNNConfig, train_mphnn, predict_mphnn, MPHNN
    MPHNN_AVAILABLE = True
except ImportError:
    MPHNN_AVAILABLE = False
    MPHNNConfig = None
    train_mphnn = None
    predict_mphnn = None
    MPHNN = None


class MPHNNWrapper(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for the MPHNN model."""
    
    def __init__(self, 
                 encoder_dim: int = 128,
                 physics_dim: int = 64,
                 domain_dim: int = 32,
                 decoder_dim: int = 64,
                 attention_heads: int = 8,
                 dropout_rate: float = 0.2,
                 beer_lambert_weight: float = 1.0,
                 smoothness_weight: float = 0.5,
                 band_conservation_weight: float = 0.3,
                 contrastive_weight: float = 0.1,
                 temperature_compensation: bool = True,
                 max_epochs: int = 300,
                 patience: int = 25,
                 batch_size: int = 32,
                 lr: float = 1e-3,
                 random_state: int = 42):
        
        if not MPHNN_AVAILABLE:
            raise ImportError("MPHNN dependencies not available. Install PyWavelets and ensure torch is available.")
        
        self.encoder_dim = encoder_dim
        self.physics_dim = physics_dim
        self.domain_dim = domain_dim
        self.decoder_dim = decoder_dim
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.beer_lambert_weight = beer_lambert_weight
        self.smoothness_weight = smoothness_weight
        self.band_conservation_weight = band_conservation_weight
        self.contrastive_weight = contrastive_weight
        self.temperature_compensation = temperature_compensation
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        
        # Internal state
        self.model_ = None
        self.wavelengths_ = None
        self.batch_labels_ = None
        self.temperatures_ = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            wavelengths: Optional[np.ndarray] = None,
            batch_labels: Optional[np.ndarray] = None,
            temperatures: Optional[np.ndarray] = None) -> 'MPHNNWrapper':
        """Fit the MPHNN model."""
        
        if wavelengths is None:
            # Assume wavelengths are the feature dimensions if not provided
            self.wavelengths_ = np.arange(X.shape[1])
        else:
            self.wavelengths_ = wavelengths
        
        # Store batch and temperature information
        self.batch_labels_ = batch_labels
        self.temperatures_ = temperatures
        
        # Create configuration
        config = MPHNNConfig(
            encoder_dim=self.encoder_dim,
            physics_dim=self.physics_dim,
            domain_dim=self.domain_dim,
            decoder_dim=self.decoder_dim,
            attention_heads=self.attention_heads,
            dropout_rate=self.dropout_rate,
            beer_lambert_weight=self.beer_lambert_weight,
            smoothness_weight=self.smoothness_weight,
            band_conservation_weight=self.band_conservation_weight,
            contrastive_weight=self.contrastive_weight,
            temperature_compensation=self.temperature_compensation,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            lr=self.lr
        )
        
        # Train the model
        self.model_, training_info = train_mphnn(
            X, y, self.wavelengths_, config,
            batch_labels=batch_labels,
            temperatures=temperatures,
            val_split=0.2,
            seed=self.random_state
        )
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray, temperatures: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Use stored temperatures if none provided
        if temperatures is None:
            temperatures = self.temperatures_
        
        # Make predictions
        outputs = predict_mphnn(self.model_, X, temperatures)
        return outputs["predictions"]
    
    def predict_with_uncertainty(self, X: np.ndarray, temperatures: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty quantification."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Use stored temperatures if none provided
        if temperatures is None:
            temperatures = self.temperatures_
        
        # Make predictions
        outputs = predict_mphnn(self.model_, X, temperatures)
        return outputs["predictions"], outputs["uncertainties"]
    
    def get_attention_weights(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get attention weights for interpretability."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before getting attention weights")
        
        outputs = predict_mphnn(self.model_, X, self.temperatures_)
        return outputs["attention_info"]
    
    def get_physics_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get physics constraint scores."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before getting physics scores")
        
        outputs = predict_mphnn(self.model_, X, self.temperatures_)
        return outputs["physics_scores"]


def build_mphnn_pipeline(encoder_dim: int = 128, physics_dim: int = 64, 
                         domain_dim: int = 32, decoder_dim: int = 64,
                         attention_heads: int = 8, dropout_rate: float = 0.2,
                         beer_lambert_weight: float = 1.0, smoothness_weight: float = 0.5,
                         band_conservation_weight: float = 0.3, contrastive_weight: float = 0.1,
                         temperature_compensation: bool = True, max_epochs: int = 300,
                         patience: int = 25, batch_size: int = 32, lr: float = 1e-3,
                         random_state: int = 42) -> Tuple[MPHNNWrapper, Dict]:
    """Build MPHNN pipeline with hyperparameter grid."""
    
    if not MPHNN_AVAILABLE:
        raise ImportError("MPHNN dependencies not available")
    
    model = MPHNNWrapper(
        encoder_dim=encoder_dim,
        physics_dim=physics_dim,
        domain_dim=domain_dim,
        decoder_dim=decoder_dim,
        attention_heads=attention_heads,
        dropout_rate=dropout_rate,
        beer_lambert_weight=beer_lambert_weight,
        smoothness_weight=smoothness_weight,
        band_conservation_weight=band_conservation_weight,
        contrastive_weight=contrastive_weight,
        temperature_compensation=temperature_compensation,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        lr=lr,
        random_state=random_state
    )
    
    # Hyperparameter grid for tuning
    param_grid = {
        "encoder_dim": [64, 128, 256],
        "physics_dim": [32, 64, 128],
        "attention_heads": [4, 8, 16],
        "dropout_rate": [0.1, 0.2, 0.3],
        "beer_lambert_weight": [0.5, 1.0, 2.0],
        "smoothness_weight": [0.3, 0.5, 1.0],
        "contrastive_weight": [0.05, 0.1, 0.2]
    }
    
    return model, param_grid


def tune_mphnn(X: np.ndarray, y: np.ndarray, cv, 
               wavelengths: Optional[np.ndarray] = None,
               batch_labels: Optional[np.ndarray] = None,
               temperatures: Optional[np.ndarray] = None,
               random_state: int = 42) -> GridSearchCV:
    """Tune MPHNN hyperparameters using cross-validation."""
    
    if not MPHNN_AVAILABLE:
        raise ImportError("MPHNN dependencies not available")
    
    # Create a custom scorer that handles the MPHNN wrapper
    def mphnn_scorer(estimator, X, y):
        try:
            # Fit the model with additional data
            estimator.fit(X, y, wavelengths=wavelengths, 
                         batch_labels=batch_labels, temperatures=temperatures)
            # Make predictions
            y_pred = estimator.predict(X)
            # Calculate RMSE
            return -np.sqrt(mean_squared_error(y, y_pred))  # Negative for GridSearchCV
        except Exception:
            return -float('inf')  # Return worst possible score on error
    
    # Create base model
    base_model = MPHNNWrapper(random_state=random_state)
    
    # Create hyperparameter grid
    param_grid = {
        "encoder_dim": [64, 128],
        "physics_dim": [32, 64],
        "attention_heads": [4, 8],
        "dropout_rate": [0.1, 0.2],
        "beer_lambert_weight": [0.5, 1.0],
        "smoothness_weight": [0.3, 0.5],
        "contrastive_weight": [0.05, 0.1]
    }
    
    # Create GridSearchCV
    search = GridSearchCV(
        base_model, 
        param_grid=param_grid, 
        cv=cv, 
        scoring=make_scorer(mphnn_scorer, greater_is_better=False),
        n_jobs=1,  # MPHNN training is already parallelized internally
        refit=True,
        verbose=1
    )
    
    # Fit the search
    search.fit(X, y)
    
    return search


# Fallback function for when MPHNN is not available
def create_mphnn_fallback() -> Tuple[None, Dict]:
    """Create a fallback when MPHNN is not available."""
    return None, {}


# Export the main functions
__all__ = [
    "MPHNNWrapper",
    "build_mphnn_pipeline", 
    "tune_mphnn",
    "MPHNN_AVAILABLE"
]
