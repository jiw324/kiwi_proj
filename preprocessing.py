"""
language: python
# AI-Generated Code Header
# **Intent:** Spectral preprocessing utilities: windowing, SNV, Savitzky–Golay derivatives, scaling.
# **Optimization:** Vectorized numpy/sklearn ops; reusable sklearn-compatible transformers.
# **Safety:** Train-only fitting; robust to NaNs; clamps invalid outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class SpectralWindow(BaseEstimator, TransformerMixin):
    """Select a wavelength window by min/max bounds (inclusive)."""

    def __init__(self, wavelengths: np.ndarray, wl_min: Optional[float] = None, wl_max: Optional[float] = None):
        self.wavelengths = wavelengths.astype(float)
        self.wl_min = wl_min
        self.wl_max = wl_max
        self._mask: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None):  # noqa: D401
        mask = np.ones_like(self.wavelengths, dtype=bool)
        if self.wl_min is not None:
            mask &= self.wavelengths >= float(self.wl_min)
        if self.wl_max is not None:
            mask &= self.wavelengths <= float(self.wl_max)
        # Ensure at least 50 bands remain
        if mask.sum() < 50:
            raise ValueError("SpectralWindow left fewer than 50 wavelengths; adjust wl_min/wl_max.")
        self._mask = mask
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("SpectralWindow not fitted")
        return X[:, self._mask]

    @property
    def selected_wavelengths(self) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("SpectralWindow not fitted")
        return self.wavelengths[self._mask]


class SNV(BaseEstimator, TransformerMixin):
    """Standard Normal Variate per spectrum (row-wise z-normalization)."""

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        mean = np.nanmean(X, axis=1, keepdims=True)
        std = np.nanstd(X, axis=1, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        Z = (X - mean) / std
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


class SavitzkyGolayDerivative(BaseEstimator, TransformerMixin):
    """Apply Savitzky–Golay smoothing and first derivative along spectral axis."""

    def __init__(self, window_length: int = 21, polyorder: int = 2, deriv: int = 1):
        if window_length % 2 == 0:
            window_length += 1  # AI-SUGGESTION: force odd window length
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        wl = X.shape[1]
        w = min(self.window_length, wl - (1 - wl % 2))
        if w < self.polyorder + 2:
            w = self.polyorder + 3
            if w % 2 == 0:
                w += 1
        Y = savgol_filter(X, window_length=w, polyorder=self.polyorder, deriv=self.deriv, axis=1, mode="interp")
        return np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)


class TrainStandardScaler(StandardScaler):
    """StandardScaler wrapper that guards transform against NaNs and preserves shape."""

    def transform(self, X: np.ndarray, copy=None):  # type: ignore[override]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return super().transform(X, copy=copy)

