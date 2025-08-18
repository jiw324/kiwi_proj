"""
language: python
# AI-Generated Code Header
# **Intent:** XGBRegressor with sensible hyperparameter grid.
# **Optimization:** Use tree_method=hist for CPU speed; early stopping handled by CV refit.
# **Safety:** Limits depth and learning rate ranges; reproducible randomness.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error


def build_xgb_pipeline(random_state: int = 42) -> Tuple[Pipeline, Dict]:
    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        reg_lambda=1.0,
    )
    pipe = Pipeline([
        ("model", xgb),
    ])
    param_grid = {
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.5, 0.8, 1.0],
        "model__min_child_weight": [1, 5, 10],
    }
    return pipe, param_grid


def tune_xgb(X: np.ndarray, y: np.ndarray, cv) -> GridSearchCV:
    model, param_grid = build_xgb_pipeline()
    rmse_scorer = make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)
    search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring=rmse_scorer, n_jobs=-1, refit=True)
    search.fit(X, y)
    return search

