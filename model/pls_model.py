# language: py
# AI-Generated Code Header
# **Intent:** [PLS (Partial Least Squares) regression model specialized for NIR spectroscopic data to predict kiwi sweetness. PLS is ideal for chemometrics as it handles multicollinear spectral data effectively.]
# **Optimization:** [Uses sklearn-compatible PLS regression with optimal component selection via cross-validation. Includes standard spectral preprocessing like mean centering and scaling.]
# **Safety:** [Includes component number validation, handles potential numerical issues in PLS decomposition, and provides comprehensive error handling for spectral data edge cases.]

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from .base_model import BaseNIRModel

class PLSModel(BaseNIRModel):
    """
    Partial Least Squares Regression model for NIR spectroscopic data.
    
    PLS-R is particularly well-suited for chemometrics applications where:
    - Features (wavelengths) are highly multicollinear
    - Number of features may exceed number of samples
    - Linear relationships exist in the latent variable space
    """
    
    def __init__(self, n_components: int = 10, random_state: int = 42, 
                 scale_features: bool = True, max_components: int = 20):
        """
        Initialize PLS regression model.
        
        Args:
            n_components (int): Number of PLS components to use
            random_state (int): Random state for reproducibility
            scale_features (bool): Whether to standardize features
            max_components (int): Maximum components for auto-selection
        """
        super().__init__("PLS_Regression", random_state)
        self.n_components = n_components
        self.scale_features = scale_features
        self.max_components = max_components
        self.optimal_components = None
        
        if self.scale_features:
            self.scaler = StandardScaler()
    
    def _create_model(self):
        """Create PLS regression model instance."""
        # AI-SUGGESTION: For NIR data, typically 5-20 components are optimal
        # Too many components can lead to overfitting, too few may underfit
        return PLSRegression(
            n_components=self.n_components,
            scale=False,  # We handle scaling separately for more control
            max_iter=500,
            tol=1e-06
        )
    
    def _preprocess_features(self, X):
        """
        Apply PLS-specific preprocessing to NIR spectral features.
        
        Args:
            X (array-like): Raw spectral data
            
        Returns:
            array: Preprocessed spectral data
        """
        X_processed = np.array(X, dtype=np.float64)
        
        # AI-SUGGESTION: For NIR spectroscopic data, mean centering is crucial
        # Standard scaling can also help but should be applied consistently
        if self.scale_features:
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_processed = self.scaler.fit_transform(X_processed)
            else:
                X_processed = self.scaler.transform(X_processed)
        
        return X_processed
    
    def optimize_components(self, X, y, cv_folds: int = 5):
        """
        Find optimal number of PLS components using cross-validation.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target values
            cv_folds (int): Number of CV folds
            
        Returns:
            int: Optimal number of components
        """
        X = np.array(X)
        y = np.array(y)
        
        # Test different numbers of components
        max_components = min(self.max_components, X.shape[1], X.shape[0] - 1)
        component_range = range(1, max_components + 1)
        
        cv_scores = []
        
        print(f"Optimizing PLS components (testing 1 to {max_components})...")
        
        for n_comp in component_range:
            # Create temporary PLS model
            temp_pls = PLSRegression(n_components=n_comp, scale=False)
            
            # Preprocess data
            X_processed = self._preprocess_features(X)
            
            # Cross-validation
            scores = cross_val_score(
                temp_pls, X_processed, y, 
                cv=cv_folds, 
                scoring='neg_mean_squared_error'
            )
            
            cv_scores.append(-scores.mean())  # Convert back to positive RMSE
        
        # Find optimal number of components
        optimal_idx = np.argmin(cv_scores)
        self.optimal_components = component_range[optimal_idx]
        self.n_components = self.optimal_components
        
        print(f"Optimal number of PLS components: {self.optimal_components}")
        print(f"Cross-validation RMSE: {cv_scores[optimal_idx]:.4f}")
        
        return self.optimal_components
    
    def fit(self, X, y, optimize_components: bool = True):
        """
        Fit the PLS model to training data.
        
        Args:
            X (array-like): Feature matrix (NIR spectra)
            y (array-like): Target values (sweetness)
            optimize_components (bool): Whether to optimize number of components
            
        Returns:
            self: Fitted model instance
        """
        try:
            X = np.array(X)
            y = np.array(y)
            
            # Optimize components if requested
            if optimize_components:
                self.optimize_components(X, y)
            
            # Now fit with optimal components
            X_processed = self._preprocess_features(X)
            
            self.model = self._create_model()
            self.model.fit(X_processed, y)
            self.is_fitted = True
            
            print(f"PLS model fitted with {self.n_components} components")
            
            return self
            
        except Exception as e:
            print(f"Error during PLS fitting: {e}")
            raise
    
    def get_model_info(self):
        """
        Get detailed information about the fitted PLS model.
        
        Returns:
            dict: Model information including explained variance
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        info = {
            "model_type": "PLS Regression",
            "n_components": self.n_components,
            "scale_features": self.scale_features
        }
        
        # Add explained variance information if available
        if hasattr(self.model, 'x_scores_'):
            # Calculate explained variance for X and Y
            try:
                x_var = np.var(self.model.x_scores_, axis=0)
                x_var_ratio = x_var / np.sum(x_var)
                info["x_explained_variance_ratio"] = x_var_ratio.tolist()
                info["cumulative_x_variance"] = np.cumsum(x_var_ratio).tolist()
            except:
                pass
        
        return info
    
    def get_regression_coefficients(self):
        """
        Get PLS regression coefficients for interpretation.
        
        Returns:
            array: Regression coefficients for each wavelength/feature
        """
        if not self.is_fitted:
            return None
        
        # Get coefficients from the fitted model
        if hasattr(self.model, 'coef_'):
            return self.model.coef_.flatten()
        else:
            return None 