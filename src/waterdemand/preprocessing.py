"""Two preprocessing branches, with a deliberate difference.

* **Scale-sensitive branch** (Ridge, MLP): median impute -> Yeo-Johnson ->
  RobustScaler on continuous demand/weather features. Yeo-Johnson is *needed*
  here because several continuous features are skewed and the linear/neural
  models are sensitive to it.
* **Tree branch** (RF, ExtraTrees, HistGB, XGBoost): imputation and categorical
  encoding only. Trees are invariant to monotone transforms, so applying
  Yeo-Johnson there would be decoration, not modelling. We say so explicitly.

Both branches are *fitted on training rows only*. Never call `.fit()` on
validation or test data.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, RobustScaler

from . import config


def _nominal() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def _ordinal() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=config.RAIN_BANDS,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )


def _binary() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])


def build_scale_preprocessor() -> ColumnTransformer:
    """Yeo-Johnson + RobustScaler branch for Ridge and the MLP."""
    continuous_yj = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("robust", RobustScaler()),
        ]
    )
    cyclical_bounded = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("robust", RobustScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("continuous_yj", continuous_yj, config.CONTINUOUS),
            ("cyclical", cyclical_bounded, config.CYCLICAL),
            ("binary", _binary(), config.BINARY),
            ("nominal", _nominal(), config.NOMINAL),
            ("ordinal", _ordinal(), config.ORDINAL),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_tree_preprocessor() -> ColumnTransformer:
    """No scaling, no Yeo-Johnson — trees do not need them."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        [
            ("continuous", numeric, config.CONTINUOUS + config.CYCLICAL),
            ("binary", _binary(), config.BINARY),
            ("nominal", _nominal(), config.NOMINAL),
            ("ordinal", _ordinal(), config.ORDINAL),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
