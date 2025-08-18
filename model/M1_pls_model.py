"""
language: python
# AI-Generated Code Header
# **Intent:** PLSRegression wrapper with hyperparameter tuning via CV.
# **Optimization:** Uses sklearn Pipeline; search over components.
# **Safety:** Nested CV-ready, reproducible random_state.
"""
from __future__ import annotations

from typing import Dict, Tuple, List, Optional

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error


def build_pls_pipeline(n_components_grid: Optional[List[int]] = None, random_state: int = 42) -> Tuple[Pipeline, Dict]:
    # AI-SUGGESTION: Centering handled by model; scaling is expected to be upstream
    pls = PLSRegression(scale=False)
    pipe = Pipeline([
        ("model", pls),
    ])
    if n_components_grid is None:
        n_components_grid = [2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40]  # AI-SUGGESTION: expanded grid
    param_grid = {"model__n_components": n_components_grid}
    return pipe, param_grid


def tune_pls(X: np.ndarray, y: np.ndarray, cv, n_components_grid: Optional[List[int]] = None) -> GridSearchCV:
    # AI-SUGGESTION: allow external control of components grid
    model, param_grid = build_pls_pipeline(n_components_grid=n_components_grid)
    rmse_scorer = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)
    search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring=rmse_scorer, n_jobs=-1, refit=True)
    search.fit(X, y)
    return search

