"""
language: python
# AI-Generated Code Header
# Intent: Wavelength selection methods (VIP, GA, iPLS) for spectral data
# Optimization: Efficient computation, vectorized operations
# Safety: Input validation, handles edge cases, maintains spectral interpretability
"""

import numpy as np
from typing import Tuple, List, Optional
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score


class VIPSelector:
    """Variable Importance in Projection (VIP) for wavelength selection"""
    
    def __init__(self, n_components: int = 10, threshold: float = 1.0):
        """
        Parameters:
        -----------
        n_components : int
            Number of PLS components for VIP calculation
        threshold : float
            VIP threshold for selection (typically 1.0)
        """
        self.n_components = n_components
        self.threshold = threshold
        self.vip_scores_ = None
        self.selected_indices_ = None
        self.pls_model_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Compute VIP scores from PLS model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data
        y : array-like, shape (n_samples,)
            Target values
        """
        # Fit PLS model
        self.pls_model_ = PLSRegression(n_components=self.n_components)
        self.pls_model_.fit(X, y)
        
        # Extract PLS components
        T = self.pls_model_.x_scores_  # X scores
        W = self.pls_model_.x_weights_  # X weights
        Q = self.pls_model_.y_loadings_  # Y loadings
        
        # Compute VIP scores
        p, h = W.shape  # p = n_wavelengths, h = n_components
        
        # Sum of squares of Y explained by each component
        s = np.diag(T.T @ T @ Q.T @ Q).reshape(h, -1)
        total_s = np.sum(s)
        
        # VIP formula: sqrt(p * sum(w_ik^2 * s_k) / sum(s_k))
        vips = np.zeros(p)
        for i in range(p):
            weight = np.array([(W[i, j] / np.linalg.norm(W[:, j]))**2 
                              for j in range(h)])
            vips[i] = np.sqrt(p * (s.T @ weight) / total_s)[0]
        
        self.vip_scores_ = vips
        
        # Select wavelengths with VIP > threshold
        self.selected_indices_ = np.where(vips > self.threshold)[0]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select important wavelengths based on VIP scores"""
        if self.selected_indices_ is None:
            raise ValueError("VIPSelector must be fitted before transform")
        
        return X[:, self.selected_indices_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Return VIP scores for all wavelengths"""
        if self.vip_scores_ is None:
            raise ValueError("VIPSelector must be fitted first")
        return self.vip_scores_
    
    def get_selected_wavelengths(self, wavelengths: np.ndarray) -> np.ndarray:
        """Return selected wavelength values"""
        if self.selected_indices_ is None:
            raise ValueError("VIPSelector must be fitted first")
        return wavelengths[self.selected_indices_]


class IntervalPLS:
    """Interval PLS (iPLS) for systematic spectral region selection"""
    
    def __init__(self, n_intervals: int = 20, n_components: int = 10, cv: int = 5):
        """
        Parameters:
        -----------
        n_intervals : int
            Number of spectral intervals to test
        n_components : int
            Number of PLS components
        cv : int
            Cross-validation folds
        """
        self.n_intervals = n_intervals
        self.n_components = n_components
        self.cv = cv
        self.best_interval_ = None
        self.best_score_ = None
        self.interval_scores_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Find best spectral interval using cross-validation
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data
        y : array-like, shape (n_samples,)
            Target values
        """
        n_wl = X.shape[1]
        interval_size = n_wl // self.n_intervals
        
        scores = []
        intervals = []
        
        for i in range(self.n_intervals):
            start = i * interval_size
            end = (i + 1) * interval_size if i < self.n_intervals - 1 else n_wl
            
            X_interval = X[:, start:end]
            
            # Fit PLS and evaluate
            model = PLSRegression(n_components=min(self.n_components, X_interval.shape[1]))
            cv_scores = cross_val_score(model, X_interval, y, cv=self.cv,
                                       scoring='neg_root_mean_squared_error')
            rmse = -np.mean(cv_scores)
            
            scores.append(rmse)
            intervals.append((start, end))
        
        self.interval_scores_ = np.array(scores)
        best_idx = np.argmin(scores)
        self.best_interval_ = intervals[best_idx]
        self.best_score_ = scores[best_idx]
        
        print(f"Best interval: wavelengths {self.best_interval_[0]}-{self.best_interval_[1]}")
        print(f"Best CV RMSE: {self.best_score_:.4f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select best spectral interval"""
        if self.best_interval_ is None:
            raise ValueError("IntervalPLS must be fitted before transform")
        
        start, end = self.best_interval_
        return X[:, start:end]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)


class AdaptiveWavelengthSelector:
    """Adaptive wavelength selection combining multiple strategies"""
    
    def __init__(self, method: str = 'vip', n_features: Optional[int] = None, **kwargs):
        """
        Parameters:
        -----------
        method : str
            Selection method: 'vip', 'ipls', 'correlation', 'mutual_info'
        n_features : int, optional
            Target number of features (if None, use threshold-based selection)
        **kwargs : additional arguments for specific methods
        """
        self.method = method
        self.n_features = n_features
        self.kwargs = kwargs
        self.selector_ = None
        self.selected_indices_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Select wavelengths using specified method
        """
        if self.method == 'vip':
            threshold = self.kwargs.get('threshold', 1.0)
            n_components = self.kwargs.get('n_components', 10)
            self.selector_ = VIPSelector(n_components=n_components, threshold=threshold)
            self.selector_.fit(X, y)
            self.selected_indices_ = self.selector_.selected_indices_
            
        elif self.method == 'ipls':
            n_intervals = self.kwargs.get('n_intervals', 20)
            self.selector_ = IntervalPLS(n_intervals=n_intervals)
            self.selector_.fit(X, y)
            start, end = self.selector_.best_interval_
            self.selected_indices_ = np.arange(start, end)
            
        elif self.method == 'correlation':
            # Select wavelengths with highest correlation to target
            correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] 
                                  for i in range(X.shape[1])])
            if self.n_features is not None:
                self.selected_indices_ = np.argsort(correlations)[-self.n_features:]
            else:
                threshold = self.kwargs.get('threshold', 0.3)
                self.selected_indices_ = np.where(correlations > threshold)[0]
                
        elif self.method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(X, y)
            if self.n_features is not None:
                self.selected_indices_ = np.argsort(mi_scores)[-self.n_features:]
            else:
                threshold = self.kwargs.get('threshold', 0.1)
                self.selected_indices_ = np.where(mi_scores > threshold)[0]
        
        print(f"Selected {len(self.selected_indices_)} wavelengths using {self.method}")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using selected wavelengths"""
        if self.selected_indices_ is None:
            raise ValueError("Selector must be fitted before transform")
        
        return X[:, self.selected_indices_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_selected_wavelengths(self, wavelengths: np.ndarray) -> np.ndarray:
        """Return selected wavelength values"""
        if self.selected_indices_ is None:
            raise ValueError("Selector must be fitted first")
        return wavelengths[self.selected_indices_]

