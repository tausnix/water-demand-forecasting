"""The eight-regressor benchmark, declared once.

Each entry records the preprocessing branch it needs. That mapping is the thing
that makes the "trees are not scaled" claim checkable rather than aspirational:
`build_pipeline()` reads it from here.

Regressors (8):
    1. ridge_yj_robust          linear, scale-sensitive branch
    2. random_forest            tree
    3. bagging_extra_tree       tree
    4. hist_gradient_boosting   tree
    5. xgboost                  tree
    6. tuned_xgboost            tree  (Optuna)
    7. keras_mlp                neural, scale-sensitive branch (TF, training-only)
    8. voting_regressor         tree  (bagging + HGB + XGBoost, weighted)

Plus three persistence baselines that cost nothing and keep us honest:
    naive_lag_1, naive_lag_24, naive_lag_168
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    BaggingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import ExtraTreeRegressor

from . import config
from .preprocessing import build_scale_preprocessor, build_tree_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str  # linear | tree | neural | ensemble
    preprocessing: str  # "scale" | "tree" | "none"
    factory: Callable[[], BaseEstimator]


# --------------------------------------------------------------------------- #
# Component factories
# --------------------------------------------------------------------------- #


def make_ridge() -> BaseEstimator:
    return Ridge(alpha=2.0)


def make_random_forest() -> BaseEstimator:
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=config.SEED,
        n_jobs=-1,
    )


def make_bagging() -> BaseEstimator:
    return BaggingRegressor(
        estimator=ExtraTreeRegressor(
            max_depth=20, min_samples_leaf=2, random_state=config.SEED
        ),
        n_estimators=70,
        max_samples=0.85,
        max_features=0.9,
        bootstrap=True,
        random_state=config.SEED,
        n_jobs=-1,
    )


def make_hist_gb() -> BaseEstimator:
    return HistGradientBoostingRegressor(
        learning_rate=0.07,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=0.5,
        min_samples_leaf=20,
        random_state=config.SEED,
    )


def make_xgboost(params: dict | None = None, device: str = "cpu") -> BaseEstimator:
    from xgboost import XGBRegressor

    base = dict(
        objective="reg:squarederror",
        n_estimators=350,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=config.SEED,
        tree_method="hist",
        n_jobs=4,
        verbosity=0,
        # CPU by default: the pickled booster must load on Streamlit Cloud,
        # which has no GPU. Training can override this explicitly.
        device=device,
    )
    if params:
        base.update(params)
    return XGBRegressor(**base)


def make_voting(xgb_params: dict | None = None) -> BaseEstimator:
    """Weighted blend. Weights [1, 1, 2] favour the strongest member (XGBoost)."""
    return VotingRegressor(
        estimators=[
            ("bag", make_bagging()),
            ("hgb", make_hist_gb()),
            ("xgb", make_xgboost(xgb_params)),
        ],
        weights=[1.0, 1.0, 2.0],
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

SPECS: dict[str, ModelSpec] = {
    "ridge_yj_robust": ModelSpec("ridge_yj_robust", "linear", "scale", make_ridge),
    "random_forest": ModelSpec("random_forest", "tree", "tree", make_random_forest),
    "bagging_extra_tree": ModelSpec("bagging_extra_tree", "tree", "tree", make_bagging),
    "hist_gradient_boosting": ModelSpec(
        "hist_gradient_boosting", "tree", "tree", make_hist_gb
    ),
    "xgboost": ModelSpec("xgboost", "tree", "tree", make_xgboost),
    "voting_regressor": ModelSpec("voting_regressor", "ensemble", "tree", make_voting),
}

# filled in at tuning time
TUNED_NAME = "tuned_xgboost"
NEURAL_NAME = "keras_mlp"

BASELINE_NAMES = ("naive_lag_1", "naive_lag_24", "naive_lag_168")
ALL_NAMES = (*BASELINE_NAMES, *SPECS, TUNED_NAME, NEURAL_NAME)


def build_pipeline(name: str, params: dict | None = None) -> Pipeline:
    """Assemble preprocessing + estimator for a registered model."""
    if name == TUNED_NAME:
        pre, est = build_tree_preprocessor(), make_xgboost(params)
    elif name == "voting_regressor":
        pre, est = build_tree_preprocessor(), make_voting(params)
    elif name == "xgboost":
        pre, est = build_tree_preprocessor(), make_xgboost(params)
    else:
        spec = SPECS[name]
        pre = (
            build_scale_preprocessor()
            if spec.preprocessing == "scale"
            else build_tree_preprocessor()
        )
        est = spec.factory()
    return Pipeline([("pre", pre), ("model", est)])


# --------------------------------------------------------------------------- #
# Persistence baselines
# --------------------------------------------------------------------------- #


def persistence(frame, lag: int) -> np.ndarray:
    """Seasonal-naive: predict y[t] = y[t - lag].

    lag=1 is calendar-persistence, lag=24 is same-hour-yesterday,
    lag=168 is same-hour-last-week. On this series lag=168 is the strong one:
    measured autocorrelation is 0.929 at 168h versus 0.901 at 1h, because demand
    oscillates *within* the day.
    """
    column = f"consumption_lag_{lag}"
    if column not in frame.columns:
        raise KeyError(f"{column} not in frame")
    return frame[column].to_numpy(dtype=float)


def naive_scores(frame) -> dict[str, dict[str, float]]:
    from .metrics import regression_metrics

    out = {}
    for lag in (1, 24, 168):
        out[f"naive_lag_{lag}"] = regression_metrics(
            frame[config.TARGET], persistence(frame, lag)
        )
    return out
