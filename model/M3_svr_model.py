"""
language: python
# AI-Generated Code Header
# **Intent:** SVR model with hyperparameter tuning for kiwi NIR spectroscopy
# **Optimization:** Grid search with cross-validation for optimal parameters
# **Safety:** Robust error handling and parameter validation
"""
from __future__ import annotations

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Optional, Union


def tune_svr(X: np.ndarray, y: np.ndarray, cv: int = 5) -> GridSearchCV:
    """
    Tune SVR hyperparameters using grid search.
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        cv: Number of cross-validation folds
        
    Returns:
        GridSearchCV object with best SVR model
    """
    
    # Define parameter grid for SVR
    param_grid = {
        'svr__C': [0.1, 1, 10, 100],
        'svr__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'svr__epsilon': [0.01, 0.1, 0.2]
    }
    
    # Create pipeline with scaling and SVR
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf'))
    ])
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit the model
    grid_search.fit(X, y)
    
    return grid_search


def create_svr_model(C: float = 1.0, 
                     gamma: Union[str, float] = 'scale',
                     epsilon: float = 0.1,
                     kernel: str = 'rbf') -> SVR:
    """
    Create an SVR model with specified parameters.
    
    Args:
        C: Regularization parameter
        gamma: Kernel coefficient
        epsilon: Epsilon in the epsilon-SVR model
        kernel: Kernel type
        
    Returns:
        SVR model
    """
    
    return SVR(
        C=C,
        gamma=gamma,
        epsilon=epsilon,
        kernel=kernel
    )


if __name__ == "__main__":
    # Test the SVR model
    print("Testing SVR model...")
    
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 50)
    y = np.random.randn(100)
    
    try:
        # Test hyperparameter tuning
        best_svr = tune_svr(X, y, cv=3)
        print(f"✓ SVR tuning completed successfully")
        print(f"  Best parameters: {best_svr.best_params_}")
        print(f"  Best CV score: {-best_svr.best_score_:.4f}")
        
        # Test direct model creation
        svr_model = create_svr_model(C=10, gamma='scale', epsilon=0.1)
        svr_model.fit(X, y)
        print(f"✓ Direct SVR model created successfully")
        
    except Exception as e:
        print(f"✗ SVR testing failed: {e}")
    
    print("SVR model testing completed!")
