"""
language: python
# AI-Generated Code Header
# Intent: Calibration transfer methods (PDS, DS, SBC) for cross-batch standardization
# Optimization: Efficient linear algebra, vectorized operations
# Safety: Input validation, numerical stability checks, handles rank-deficient cases
"""

import numpy as np
from typing import Tuple, Optional
from scipy.linalg import lstsq
from sklearn.base import BaseEstimator, TransformerMixin


class PiecewiseDirectStandardization(BaseEstimator, TransformerMixin):
    """Piecewise Direct Standardization (PDS) for calibration transfer"""
    
    def __init__(self, window_size: int = 5):
        """
        Parameters:
        -----------
        window_size : int
            Size of window around each wavelength for local standardization
        """
        self.window_size = window_size
        self.transfer_matrix_ = None
    
    def fit(self, X_master: np.ndarray, X_slave: np.ndarray):
        """
        Learn transfer function from slave to master instrument/batch
        
        Parameters:
        -----------
        X_master : array-like, shape (n_samples, n_wavelengths)
            Spectra from master instrument/batch (reference)
        X_slave : array-like, shape (n_samples, n_wavelengths)
            Spectra from slave instrument/batch (to be standardized)
            
        Note: X_master and X_slave must have same samples (transfer standards)
        """
        if X_master.shape != X_slave.shape:
            raise ValueError("Master and slave spectra must have same shape")
        
        n_samples, n_wl = X_master.shape
        
        if n_samples < self.window_size:
            raise ValueError(f"Need at least {self.window_size} transfer samples")
        
        # Initialize transfer matrix (F)
        F = np.zeros((n_wl, n_wl))
        
        print(f"Computing PDS transfer matrix with window_size={self.window_size}...")
        
        # For each wavelength, fit local model
        for i in range(n_wl):
            # Define window around wavelength i
            start = max(0, i - self.window_size)
            end = min(n_wl, i + self.window_size + 1)
            
            # Local spectral window from slave
            X_window = X_slave[:, start:end]
            
            # Target: corresponding wavelength in master
            y_target = X_master[:, i]
            
            # Fit linear model: master_i = F_i * slave_window
            try:
                coef, _, _, _ = lstsq(X_window, y_target)
                F[i, start:end] = coef
            except np.linalg.LinAlgError:
                # Fallback: use simple wavelength-to-wavelength mapping
                print(f"  Warning: Singular matrix at wavelength {i}, using fallback")
                F[i, i] = 1.0
        
        self.transfer_matrix_ = F
        
        return self
    
    def transform(self, X_slave: np.ndarray) -> np.ndarray:
        """
        Apply PDS transfer to standardize slave spectra
        
        Parameters:
        -----------
        X_slave : array-like, shape (n_samples, n_wavelengths)
            Slave spectra to be standardized
            
        Returns:
        --------
        X_standardized : array-like, shape (n_samples, n_wavelengths)
            Standardized spectra (in master space)
        """
        if self.transfer_matrix_ is None:
            raise ValueError("PDS must be fitted before transform")
        
        # Apply transfer: X_std = X_slave * F^T
        return X_slave @ self.transfer_matrix_.T
    
    def fit_transform(self, X_master: np.ndarray, X_slave: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X_master, X_slave)
        return self.transform(X_slave)


class DirectStandardization(BaseEstimator, TransformerMixin):
    """Direct Standardization (DS) - global transfer"""
    
    def __init__(self):
        """Simple global linear transfer"""
        self.transfer_matrix_ = None
    
    def fit(self, X_master: np.ndarray, X_slave: np.ndarray):
        """
        Learn global transfer function
        
        Parameters:
        -----------
        X_master : array-like, shape (n_samples, n_wavelengths)
            Master spectra (reference)
        X_slave : array-like, shape (n_samples, n_wavelengths)
            Slave spectra (to be standardized)
        """
        if X_master.shape != X_slave.shape:
            raise ValueError("Master and slave spectra must have same shape")
        
        # Fit global linear model: X_master = X_slave * F^T
        # Solve: F^T = (X_slave^T X_slave)^{-1} X_slave^T X_master
        try:
            self.transfer_matrix_, _, _, _ = lstsq(X_slave, X_master)
            self.transfer_matrix_ = self.transfer_matrix_.T
        except np.linalg.LinAlgError:
            # Fallback: identity
            print("Warning: Singular matrix in DS, using identity transfer")
            self.transfer_matrix_ = np.eye(X_master.shape[1])
        
        return self
    
    def transform(self, X_slave: np.ndarray) -> np.ndarray:
        """Apply DS transfer"""
        if self.transfer_matrix_ is None:
            raise ValueError("DS must be fitted before transform")
        
        return X_slave @ self.transfer_matrix_.T
    
    def fit_transform(self, X_master: np.ndarray, X_slave: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X_master, X_slave)
        return self.transform(X_slave)


class SlopeAndBiasCorrection(BaseEstimator, TransformerMixin):
    """Slope and Bias Correction (SBC) - simple wavelength-by-wavelength transfer"""
    
    def __init__(self):
        """Simple slope/bias correction per wavelength"""
        self.slopes_ = None
        self.biases_ = None
    
    def fit(self, X_master: np.ndarray, X_slave: np.ndarray):
        """
        Learn slope and bias for each wavelength
        
        Parameters:
        -----------
        X_master : array-like, shape (n_samples, n_wavelengths)
            Master spectra
        X_slave : array-like, shape (n_samples, n_wavelengths)
            Slave spectra
        """
        if X_master.shape != X_slave.shape:
            raise ValueError("Master and slave spectra must have same shape")
        
        n_wl = X_master.shape[1]
        self.slopes_ = np.zeros(n_wl)
        self.biases_ = np.zeros(n_wl)
        
        # Fit linear model per wavelength: master_i = slope_i * slave_i + bias_i
        for i in range(n_wl):
            X_slave_wl = X_slave[:, i].reshape(-1, 1)
            y_master_wl = X_master[:, i]
            
            # Add intercept
            X_with_intercept = np.hstack([np.ones((len(X_slave_wl), 1)), X_slave_wl])
            
            try:
                coef, _, _, _ = lstsq(X_with_intercept, y_master_wl)
                self.biases_[i] = coef[0]
                self.slopes_[i] = coef[1]
            except np.linalg.LinAlgError:
                self.slopes_[i] = 1.0
                self.biases_[i] = 0.0
        
        return self
    
    def transform(self, X_slave: np.ndarray) -> np.ndarray:
        """Apply SBC correction"""
        if self.slopes_ is None or self.biases_ is None:
            raise ValueError("SBC must be fitted before transform")
        
        return X_slave * self.slopes_ + self.biases_
    
    def fit_transform(self, X_master: np.ndarray, X_slave: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X_master, X_slave)
        return self.transform(X_slave)


def apply_lobo_calibration_transfer(X_train: np.ndarray, X_test: np.ndarray,
                                    method: str = 'pds',
                                    window_size: int = 5,
                                    use_subset: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply calibration transfer for LOBO scenario
    
    Parameters:
    -----------
    X_train : array-like, shape (n_train, n_wavelengths)
        Training spectra (multiple batches)
    X_test : array-like, shape (n_test, n_wavelengths)
        Test spectra (held-out batch)
    method : str
        Transfer method: 'pds', 'ds', 'sbc'
    window_size : int
        Window size for PDS
    use_subset : int, optional
        Number of training samples to use as transfer standards (if limited)
        
    Returns:
    --------
    X_train_corrected, X_test_corrected : standardized spectra
    """
    # Use subset of training as "master" and test as "slave"
    if use_subset is not None and len(X_train) > use_subset:
        # Randomly sample transfer standards from training set
        rng = np.random.RandomState(42)
        indices = rng.choice(len(X_train), use_subset, replace=False)
        X_master_subset = X_train[indices]
    else:
        X_master_subset = X_train
    
    # Create pseudo-"slave" samples from test set statistics
    # (In practice, would use a small set of transfer standards measured on both)
    # Here we approximate by matching distributions
    
    if method == 'pds':
        # For LOBO, we can't directly have transfer standards
        # Instead, use robust statistics matching
        train_mean = np.mean(X_train, axis=0)
        train_std = np.std(X_train, axis=0) + 1e-8
        test_mean = np.mean(X_test, axis=0)
        test_std = np.std(X_test, axis=0) + 1e-8
        
        # Simple standardization (mean/std matching)
        X_test_corrected = (X_test - test_mean) / test_std * train_std + train_mean
        X_train_corrected = X_train
        
        print(f"Applied {method} calibration transfer (distribution matching)")
    
    elif method == 'sbc':
        sbc = SlopeAndBiasCorrection()
        # Approximate transfer using percentile matching
        train_percentiles = np.percentile(X_train, [25, 50, 75], axis=0)
        test_percentiles = np.percentile(X_test, [25, 50, 75], axis=0)
        
        # Fit SBC on percentile matches (pseudo transfer standards)
        sbc.fit(train_percentiles.T, test_percentiles.T)
        X_test_corrected = sbc.transform(X_test)
        X_train_corrected = X_train
        
        print(f"Applied {method} calibration transfer (percentile matching)")
    
    else:
        # No transfer (baseline)
        X_train_corrected = X_train
        X_test_corrected = X_test
        print("No calibration transfer applied")
    
    return X_train_corrected, X_test_corrected

