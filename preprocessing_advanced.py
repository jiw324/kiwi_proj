"""
language: python
# AI-Generated Code Header
# Intent: Advanced preprocessing techniques (SMOTE, EMSC, EPO) for spectral data
# Optimization: Vectorized operations, efficient linear algebra
# Safety: Input validation, handles edge cases, maintains spectral integrity
"""

import numpy as np
from typing import Tuple, Optional
from scipy.linalg import lstsq
from sklearn.base import BaseEstimator, TransformerMixin


class SpectralSMOTE:
    """SMOTE augmentation specifically designed for spectral data"""
    
    def __init__(self, target_samples: int = 150, k_neighbors: int = 3, 
                 noise_level: float = 0.005, random_state: int = 42):
        """
        Parameters:
        -----------
        target_samples : int
            Target number of samples after augmentation
        k_neighbors : int
            Number of nearest neighbors for SMOTE interpolation
        noise_level : float
            Spectral noise to add (as fraction of signal std)
        random_state : int
            Random seed for reproducibility
        """
        self.target_samples = target_samples
        self.k_neighbors = k_neighbors
        self.noise_level = noise_level
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def augment(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment minority class samples using spectral-aware SMOTE
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        X_augmented, y_augmented : augmented data
        """
        n_current = len(X)
        if n_current >= self.target_samples:
            return X, y
        
        n_synthetic = self.target_samples - n_current
        
        # Generate synthetic samples via spectral interpolation
        X_synthetic = np.zeros((n_synthetic, X.shape[1]))
        y_synthetic = np.zeros(n_synthetic)
        
        for i in range(n_synthetic):
            # Randomly select a sample
            idx = self.rng.randint(0, n_current)
            sample = X[idx]
            target = y[idx]
            
            # Find k nearest neighbors (Euclidean in spectral space)
            distances = np.sum((X - sample) ** 2, axis=1)
            nearest_idx = np.argsort(distances)[1:self.k_neighbors+1]  # Exclude self
            
            # Randomly select one neighbor
            neighbor_idx = self.rng.choice(nearest_idx)
            neighbor = X[neighbor_idx]
            neighbor_target = y[neighbor_idx]
            
            # Interpolate in spectral space
            alpha = self.rng.uniform(0.3, 0.7)  # Bias toward center
            synthetic_spectrum = alpha * sample + (1 - alpha) * neighbor
            synthetic_target = alpha * target + (1 - alpha) * neighbor_target
            
            # Add small spectral noise (preserves spectral smoothness)
            noise = self.rng.normal(0, self.noise_level * np.std(synthetic_spectrum), 
                                   synthetic_spectrum.shape)
            synthetic_spectrum += noise
            
            X_synthetic[i] = synthetic_spectrum
            y_synthetic[i] = synthetic_target
        
        # Combine original and synthetic
        X_augmented = np.vstack([X, X_synthetic])
        y_augmented = np.hstack([y, y_synthetic])
        
        return X_augmented, y_augmented


class EMSC(BaseEstimator, TransformerMixin):
    """Extended Multiplicative Signal Correction for spectral preprocessing"""
    
    def __init__(self, polynomial_order: int = 2):
        """
        Parameters:
        -----------
        polynomial_order : int
            Order of polynomial baseline correction (typically 2)
        """
        self.polynomial_order = polynomial_order
        self.reference_spectrum_ = None
        self.wavelengths_ = None
    
    def fit(self, X: np.ndarray, y=None, wavelengths: Optional[np.ndarray] = None):
        """
        Fit EMSC by computing reference spectrum
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data
        wavelengths : array-like, optional
            Wavelength values for polynomial fitting
        """
        # Compute mean spectrum as reference
        self.reference_spectrum_ = np.mean(X, axis=0)
        
        if wavelengths is not None:
            self.wavelengths_ = wavelengths
        else:
            self.wavelengths_ = np.arange(X.shape[1])
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply EMSC correction to remove multiplicative and additive effects
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data
            
        Returns:
        --------
        X_corrected : array-like, shape (n_samples, n_wavelengths)
            EMSC-corrected spectra
        """
        if self.reference_spectrum_ is None:
            raise ValueError("EMSC must be fitted before transform")
        
        n_samples, n_wavelengths = X.shape
        X_corrected = np.zeros_like(X)
        
        # Normalize wavelengths to [-1, 1] for numerical stability
        wl_norm = 2 * (self.wavelengths_ - self.wavelengths_.min()) / \
                  (self.wavelengths_.max() - self.wavelengths_.min()) - 1
        
        # Build design matrix: [1, wl, wl^2, ..., reference_spectrum]
        poly_terms = [np.ones(n_wavelengths)]
        for order in range(1, self.polynomial_order + 1):
            poly_terms.append(wl_norm ** order)
        poly_terms.append(self.reference_spectrum_)
        
        X_design = np.column_stack(poly_terms)
        
        # Fit and correct each spectrum
        for i in range(n_samples):
            # Solve: spectrum = offset + polynomial + scale * reference
            coefs, _, _, _ = lstsq(X_design, X[i])
            
            # Extract components
            offset_poly = sum(coefs[j] * poly_terms[j] 
                            for j in range(self.polynomial_order + 1))
            scale = coefs[-1]
            
            # Corrected spectrum = (original - offset - polynomial) / scale
            if abs(scale) > 1e-6:  # Avoid division by zero
                X_corrected[i] = (X[i] - offset_poly) / scale
            else:
                X_corrected[i] = X[i]  # Keep original if scale is too small
        
        return X_corrected
    
    def fit_transform(self, X: np.ndarray, y=None, wavelengths: Optional[np.ndarray] = None):
        """Fit and transform in one step"""
        self.fit(X, y, wavelengths)
        return self.transform(X)


class EPO(BaseEstimator, TransformerMixin):
    """External Parameter Orthogonalization for removing unwanted variations"""
    
    def __init__(self, n_components: int = 2):
        """
        Parameters:
        -----------
        n_components : int
            Number of PLS components for modeling external parameter effects
        """
        self.n_components = n_components
        self.projection_matrix_ = None
    
    def fit(self, X: np.ndarray, external_param: np.ndarray):
        """
        Fit EPO by learning subspace of external parameter effects
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data
        external_param : array-like, shape (n_samples,) or (n_samples, n_params)
            External parameters to remove (e.g., temperature, batch ID)
        """
        from sklearn.cross_decomposition import PLSRegression
        
        # Model external parameter effects using PLS
        pls = PLSRegression(n_components=self.n_components)
        pls.fit(X, external_param)
        
        # Get PLS loading vectors (define external parameter subspace)
        subspace = pls.x_loadings_  # shape: (n_wavelengths, n_components)
        
        # Create orthogonal projection matrix (projects out subspace)
        self.projection_matrix_ = np.eye(X.shape[1]) - subspace @ subspace.T
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply EPO correction to remove external parameter effects
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_wavelengths)
            Spectral data
            
        Returns:
        --------
        X_corrected : array-like, shape (n_samples, n_wavelengths)
            EPO-corrected spectra
        """
        if self.projection_matrix_ is None:
            raise ValueError("EPO must be fitted before transform")
        
        return X @ self.projection_matrix_
    
    def fit_transform(self, X: np.ndarray, external_param: np.ndarray):
        """Fit and transform in one step"""
        self.fit(X, external_param)
        return self.transform(X)


def apply_high_brix_augmentation(X: np.ndarray, y: np.ndarray, 
                                 threshold: float = 14.0,
                                 target_samples: int = 150,
                                 random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to augment high-Brix samples
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_wavelengths)
        Spectral data
    y : array-like, shape (n_samples,)
        Target values (°Brix)
    threshold : float
        Brix threshold for augmentation (default: 14.0)
    target_samples : int
        Target number of high-Brix samples after augmentation
    random_state : int
        Random seed
        
    Returns:
    --------
    X_balanced, y_balanced : augmented dataset
    """
    # Split into high-Brix and others
    high_brix_mask = y > threshold
    X_high = X[high_brix_mask]
    y_high = y[high_brix_mask]
    X_other = X[~high_brix_mask]
    y_other = y[~high_brix_mask]
    
    print(f"Original high-Brix samples (>{threshold}): {len(y_high)}")
    
    # Augment only if needed
    if len(y_high) < target_samples:
        smote = SpectralSMOTE(target_samples=target_samples, 
                             k_neighbors=min(3, len(y_high)-1),
                             random_state=random_state)
        X_high_aug, y_high_aug = smote.augment(X_high, y_high)
        print(f"Augmented high-Brix samples: {len(y_high_aug)}")
    else:
        X_high_aug, y_high_aug = X_high, y_high
        print(f"No augmentation needed (already {len(y_high)} samples)")
    
    # Combine
    X_balanced = np.vstack([X_other, X_high_aug])
    y_balanced = np.hstack([y_other, y_high_aug])
    
    # Shuffle
    rng = np.random.RandomState(random_state)
    shuffle_idx = rng.permutation(len(y_balanced))
    
    return X_balanced[shuffle_idx], y_balanced[shuffle_idx]

