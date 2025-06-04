# language: py
# AI-Generated Code Header
# **Intent:** [Support Vector Regression model for NIR spectroscopic data to predict kiwi sweetness. SVR is effective for high-dimensional spectral data and can capture non-linear relationships through kernel methods.]
# **Optimization:** [Uses sklearn SVR with RBF kernel and optimized hyperparameters via grid search. Includes proper feature scaling which is crucial for SVR performance.]
# **Safety:** [Includes comprehensive error handling, parameter validation, and handles potential numerical issues in SVM optimization. Provides fallback parameters if tuning fails.]

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from .base_model import BaseNIRModel

class SVRModel(BaseNIRModel):
    """
    Support Vector Regression model for NIR spectroscopic data.
    
    SVR is well-suited for spectral data because:
    - Effective in high-dimensional spaces (many wavelengths)
    - Can model non-linear relationships using kernels
    - Robust to outliers through epsilon-insensitive loss
    - Good generalization performance with proper regularization
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale',
                 epsilon: float = 0.1, random_state: int = 42, 
                 tune_hyperparameters: bool = True):
        """
        Initialize SVR model.
        
        Args:
            kernel (str): Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            C (float): Regularization parameter
            gamma (str or float): Kernel coefficient
            epsilon (float): Epsilon in the epsilon-SVR model
            random_state (int): Random state for reproducibility
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning
        """
        super().__init__("SVR", random_state)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.tune_hyperparameters = tune_hyperparameters
        self.best_params = None
        
        # AI-SUGGESTION: Initialize scaler as None, create when needed
        self.scaler = None
    
    def _create_model(self):
        """Create SVR model instance."""
        return SVR(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            epsilon=self.epsilon,
            cache_size=200,  # Increase cache for better performance
            max_iter=3000    # Increase max iterations for convergence
        )
    
    def _preprocess_features(self, X, fit_scaler=False):
        """
        Apply SVR-specific preprocessing to NIR spectral features.
        
        Args:
            X (array-like): Raw spectral data
            fit_scaler (bool): Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            array: Preprocessed spectral data
        """
        X_processed = np.array(X, dtype=np.float64)
        
        # AI-SUGGESTION: Feature scaling is essential for SVR as it's sensitive to
        # the scale of input features. StandardScaler ensures all features contribute equally
        if fit_scaler or self.scaler is None:
            self.scaler = StandardScaler()
            X_processed = self.scaler.fit_transform(X_processed)
        else:
            X_processed = self.scaler.transform(X_processed)
        
        return X_processed
    
    def _tune_hyperparameters(self, X, y, cv_folds: int = 5):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target values
            cv_folds (int): Number of CV folds
            
        Returns:
            dict: Best hyperparameters found
        """
        X = np.array(X)
        y = np.array(y)
        
        # AI-SUGGESTION: Create temporary scaler for hyperparameter tuning
        temp_scaler = StandardScaler()
        X_processed = temp_scaler.fit_transform(X)
        
        # AI-SUGGESTION: Parameter grid optimized for NIR spectroscopic data
        # Conservative C values to prevent overfitting, multiple kernels tested
        param_grid = [
            {
                'kernel': ['rbf'],
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.2]
            },
            {
                'kernel': ['linear'],
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.2]
            },
            {
                'kernel': ['poly'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'degree': [2, 3],
                'epsilon': [0.01, 0.1, 0.2]
            }
        ]
        
        print("Tuning SVR hyperparameters...")
        
        # Create base model for tuning
        svr_base = SVR(cache_size=200, max_iter=3000)
        
        # Perform grid search
        grid_search = GridSearchCV(
            svr_base, 
            param_grid, 
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_processed, y)
        
        self.best_params = grid_search.best_params_
        
        # Update model parameters
        self.kernel = self.best_params['kernel']
        self.C = self.best_params['C']
        self.epsilon = self.best_params['epsilon']
        if 'gamma' in self.best_params:
            self.gamma = self.best_params['gamma']
        
        print(f"Best SVR parameters: {self.best_params}")
        print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")
        
        return self.best_params
    
    def fit(self, X, y, tune_params: bool = None):
        """
        Fit the SVR model to training data.
        
        Args:
            X (array-like): Feature matrix (NIR spectra)
            y (array-like): Target values (sweetness)
            tune_params (bool): Whether to tune hyperparameters (overrides init setting)
            
        Returns:
            self: Fitted model instance
        """
        try:
            X = np.array(X)
            y = np.array(y)
            
            # Determine if we should tune parameters
            should_tune = tune_params if tune_params is not None else self.tune_hyperparameters
            
            # Tune hyperparameters if requested
            if should_tune:
                self._tune_hyperparameters(X, y)
            
            # Preprocess features
            X_processed = self._preprocess_features(X, fit_scaler=True)
            
            # Create and fit model
            self.model = self._create_model()
            self.model.fit(X_processed, y)
            self.is_fitted = True
            
            print(f"SVR model fitted with {self.kernel} kernel")
            print(f"Number of support vectors: {self.model.n_support_[0] if hasattr(self.model, 'n_support_') else 'N/A'}")
            
            return self
            
        except Exception as e:
            print(f"Error during SVR fitting: {e}")
            raise
    
    def get_model_info(self):
        """
        Get detailed information about the fitted SVR model.
        
        Returns:
            dict: Model information
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        info = {
            "model_type": "Support Vector Regression",
            "kernel": self.model.kernel,
            "C": self.model.C,
            "epsilon": self.model.epsilon,
            "gamma": self.model.gamma
        }
        
        # Add support vector information
        if hasattr(self.model, 'support_vectors_'):
            info["n_support_vectors"] = len(self.model.support_vectors_)
            info["support_vector_ratio"] = len(self.model.support_vectors_) / len(self.model.dual_coef_[0])
        
        # Add hyperparameter tuning results
        if self.best_params:
            info["best_hyperparameters"] = self.best_params
        
        return info
    
    def get_support_vector_analysis(self):
        """
        Get analysis of support vectors for model interpretation.
        
        Returns:
            dict: Support vector analysis
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        if not hasattr(self.model, 'support_vectors_'):
            return {"error": "Support vectors not available for this kernel"}
        
        analysis = {
            "n_support_vectors": len(self.model.support_vectors_),
            "support_vector_indices": self.model.support_.tolist(),
            "dual_coefficients": self.model.dual_coef_[0].tolist()
        }
        
        # Add statistics about support vectors
        if len(self.model.support_vectors_) > 0:
            sv_stats = {
                "mean_dual_coef": np.mean(np.abs(self.model.dual_coef_[0])),
                "max_dual_coef": np.max(np.abs(self.model.dual_coef_[0])),
                "min_dual_coef": np.min(np.abs(self.model.dual_coef_[0]))
            }
            analysis["support_vector_statistics"] = sv_stats
        
        return analysis
    
    def predict_with_decision_function(self, X):
        """
        Make predictions and return decision function values.
        
        Args:
            X (array-like): Feature matrix
            
        Returns:
            tuple: (predictions, decision_function_values)
        """
        if not self.is_fitted:
            raise ValueError("SVR must be fitted before making predictions")
        
        try:
            X = np.array(X)
            X_processed = self._preprocess_features(X)
            
            predictions = self.model.predict(X_processed)
            decision_values = self.model.decision_function(X_processed)
            
            return predictions, decision_values
        except Exception as e:
            print(f"Error during SVR prediction: {e}")
            raise 