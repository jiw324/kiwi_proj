# language: py
# AI-Generated Code Header
# **Intent:** [Base model interface providing consistent structure for all NIR spectroscopic models with standardized training, prediction, and evaluation methods.]
# **Optimization:** [Abstract base class design allows for consistent API across different ML algorithms while maintaining flexibility for model-specific implementations.]
# **Safety:** [Includes comprehensive error handling, input validation, and standardized cross-validation procedures to ensure reliable model evaluation.]

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

class BaseNIRModel(ABC):
    """
    Abstract base class for NIR spectroscopic models for kiwi sweetness prediction.
    
    Provides standardized interface for training, prediction, and evaluation
    with built-in cross-validation and error metrics calculation.
    """
    
    def __init__(self, model_name: str, random_state: int = 42):
        """
        Initialize the base model.
        
        Args:
            model_name (str): Name of the model for logging and identification
            random_state (int): Random state for reproducibility
        """
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_names = None
        
    @abstractmethod
    def _create_model(self):
        """Create and return the specific model instance."""
        pass
    
    @abstractmethod
    def _preprocess_features(self, X):
        """Apply model-specific preprocessing to features."""
        pass
    
    def fit(self, X, y):
        """
        Fit the model to training data.
        
        Args:
            X (array-like): Feature matrix (NIR spectra)
            y (array-like): Target values (sweetness)
        
        Returns:
            self: Fitted model instance
        """
        try:
            # Convert to numpy arrays if needed
            X = np.array(X)
            y = np.array(y)
            
            # Store feature information
            if hasattr(X, 'columns'):
                self.feature_names = X.columns.tolist()
            
            # AI-SUGGESTION: Validate input data shapes and types
            if X.shape[0] != len(y):
                raise ValueError(f"X and y must have same number of samples. Got X: {X.shape[0]}, y: {len(y)}")
            
            # Apply preprocessing
            X_processed = self._preprocess_features(X)
            
            # Create and fit model
            self.model = self._create_model()
            self.model.fit(X_processed, y)
            self.is_fitted = True
            
            return self
            
        except Exception as e:
            print(f"Error during {self.model_name} fitting: {e}")
            raise
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (array-like): Feature matrix
            
        Returns:
            array: Predicted values
        """
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before making predictions")
        
        try:
            X = np.array(X)
            X_processed = self._preprocess_features(X)
            return self.model.predict(X_processed)
        except Exception as e:
            print(f"Error during {self.model_name} prediction: {e}")
            raise
    
    def evaluate_with_cv(self, X, y, cv_folds: int = 5, scoring_metrics: list = None):
        """
        Evaluate model using cross-validation.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target values
            cv_folds (int): Number of cross-validation folds
            scoring_metrics (list): List of sklearn scoring metrics
            
        Returns:
            dict: Dictionary containing evaluation results
        """
        if scoring_metrics is None:
            scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        try:
            X = np.array(X)
            y = np.array(y)
            
            # AI-SUGGESTION: Use stratified approach if target has clear categorical nature,
            # but for continuous sweetness values, regular KFold is appropriate
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            # Prepare the model
            temp_model = self._create_model()
            
            results = {}
            all_predictions = []
            all_true_values = []
            
            # Perform cross-validation manually to get detailed results
            fold_scores = {metric: [] for metric in ['rmse', 'mae', 'r2']}
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit model on training fold
                temp_model_copy = self._create_model()
                X_train_processed = self._preprocess_features(X_train)
                X_val_processed = self._preprocess_features(X_val)
                
                temp_model_copy.fit(X_train_processed, y_train)
                
                # Predict on validation fold
                y_pred = temp_model_copy.predict(X_val_processed)
                
                # Calculate metrics for this fold
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                
                fold_scores['rmse'].append(rmse)
                fold_scores['mae'].append(mae)
                fold_scores['r2'].append(r2)
                
                all_predictions.extend(y_pred)
                all_true_values.extend(y_val)
            
            # Calculate overall statistics
            results[f'{self.model_name}_cv_results'] = {
                'rmse_mean': np.mean(fold_scores['rmse']),
                'rmse_std': np.std(fold_scores['rmse']),
                'mae_mean': np.mean(fold_scores['mae']),
                'mae_std': np.std(fold_scores['mae']),
                'r2_mean': np.mean(fold_scores['r2']),
                'r2_std': np.std(fold_scores['r2']),
                'cv_folds': cv_folds,
                'n_samples': len(y)
            }
            
            # Overall cross-validation predictions
            overall_rmse = np.sqrt(mean_squared_error(all_true_values, all_predictions))
            overall_mae = mean_absolute_error(all_true_values, all_predictions)
            overall_r2 = r2_score(all_true_values, all_predictions)
            
            results[f'{self.model_name}_overall'] = {
                'rmse': overall_rmse,
                'mae': overall_mae,
                'r2': overall_r2
            }
            
            return results
            
        except Exception as e:
            print(f"Error during {self.model_name} cross-validation: {e}")
            return {f'{self.model_name}_error': str(e)}
    
    def get_feature_importance(self):
        """
        Get feature importance if available.
        
        Returns:
            array or None: Feature importance scores
        """
        if not self.is_fitted:
            return None
            
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        else:
            return None 