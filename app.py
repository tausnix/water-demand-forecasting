"""Streamlit app — Hourly Water Demand Forecasting.

Run locally:
    streamlit run app.py

Everything here reads pre-computed artefacts. No training happens at request
time, which keeps the app inside the 1 GB RAM budget of Streamlit Community
Cloud.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from waterdemand import config, data, features, inference, train  # noqa: E402

st.set_page_config(
    page_title="Hourly Water Demand Forecasting",
    page_icon="💧",
    layout="wide",
)

ACCENT = "#0e7490"
WARN = "#b45309"


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def load_pipeline(name: str):
    return train.load_model(name)


@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    frame = data.load_raw()
    frame["timestamp"] = features.parse_timestamp(frame)
    return frame


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}


@st.cache_data(show_spinner=False)
def load_csv(path: str):
    target = Path(path)
    return pd.read_csv(target, parse_dates=["timestamp"]) if target.exists() else pd.DataFrame()


raw = load_raw()
metrics = load_json(str(config.METRICS_JSON))
split = load_json(str(config.SPLIT_JSON))
audit = load_json(str(config.REPORTS_DIR / "data_audit.json"))
predictions = load_csv(str(config.REPORTS_DIR / "locked_test_predictions.csv"))
importance = load_csv(str(config.REPORTS_DIR / "importance.csv"))

selected_name = metrics.get("selection", {}).get("selected_before_test", "voting_regressor")
available = [
    name
    for name in ("voting_regressor", "xgboost")
    if (config.MODELS_DIR / f"{name}.joblib").exists()
]

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("💧 Water Demand")
    st.caption("Hourly demand forecasting · Siemens/KIT dataset")
    st.divider()

    model_name = st.selectbox(
        "Serving model",
        available or [selected_name],
        help="The voting ensemble is the model selected on validation. XGBoost is "
        "the lighter alternative (~6 MB) — their validation MAE differs by noise.",
    )
    if metrics.get("locked_test"):
        row = next((r for r in metrics["locked_test"] if r["role"] == "selected"), None)
        if row:
            st.metric("Locked-test MAE", f"{row['mae']:.1f}")
            st.metric("Locked-test R²", f"{row['r2']:.4f}")

    st.divider()
    st.caption(
        "Test block opened once, after model selection was frozen. "
        "See the Methodology tab."
    )


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Hourly Water Demand Forecasting")
st.caption(
    "One-hour-ahead aggregate demand for a mixed residential / commercial / "
    "industrial district — 45,864 hourly observations, 2016–2021."
)

if not (config.MODELS_DIR / f"{model_name}.joblib").exists():
    st.error(
        "No trained model found. Run `python scripts/train_all.py` first — "
        "the app deliberately does not train at request time."
    )
    st.stop()

tab_forecast, tab_backtest, tab_models, tab_drivers, tab_method = st.tabs(
    ["🔮 Forecast", "📈 Backtest", "🏁 Models", "🔍 Drivers", "🧪 Methodology"]
)


# --------------------------------------------------------------------------- #
# 1. Forecast
# --------------------------------------------------------------------------- #
with tab_forecast:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Forecast target")
        latest = raw["timestamp"].max()
        chosen_date = st.date_input(
            "Target date",
            value=(latest - pd.Timedelta(days=7)).date(),
            min_value=raw["timestamp"].min().date(),
            max_value=latest.date(),
        )
        chosen_hour = st.slider("Target hour", 0, 23, 12)
        target_ts = pd.Timestamp(chosen_date) + pd.Timedelta(hours=chosen_hour)

        history = raw[raw["timestamp"] < target_ts]
        st.caption(f"History available: {len(history):,} hours up to {target_ts - pd.Timedelta(hours=1)}")

    with right:
        if len(history) < config.WARMUP_HOURS:
            st.warning(
                f"Need at least {config.WARMUP_HOURS} hours of history before "
                f"{target_ts}. Pick a later date."
            )
        else:
            pipeline = load_pipeline(model_name)
            result = inference.forecast_next_hour(history, pipeline)
            anchors = result["persistence"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Predicted demand", f"{result['prediction']:,.0f} m³")
            c2.metric("Last observed", f"{result['last_observed']:,.0f} m³")
            if anchors.get("lag_168") is not None:
                delta = result["prediction"] - anchors["lag_168"]
                c3.metric(
                    "vs same hour last week",
                    f"{anchors['lag_168']:,.0f} m³",
                    delta=f"{delta:+,.0f}",
                    delta_color="inverse",
                )

            actual_row = raw[raw["timestamp"] == target_ts]
            if not actual_row.empty:
                actual = float(actual_row["Consumption"].iloc[0])
                error = result["prediction"] - actual
                c4.metric(
                    "Actual (held out)",
                    f"{actual:,.0f} m³",
                    delta=f"{error:+,.0f} error",
                    delta_color="off",
                )

            window = history.tail(24 * 7)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=window["timestamp"],
                    y=window["Consumption"],
                    name="Observed",
                    line=dict(color="#64748b", width=1.4),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[target_ts],
                    y=[result["prediction"]],
                    name="Forecast",
                    mode="markers",
                    marker=dict(color=ACCENT, size=12, symbol="diamond"),
                )
            )
            fig.update_layout(
                height=340,
                margin=dict(l=0, r=0, t=30, b=0),
                title="Last 7 days + forecast",
                legend=dict(orientation="h", y=1.1),
                yaxis_title="m³ / hour",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Persistence anchors used by the feature builder"):
                st.json(anchors)
                st.caption(
                    "lag_1 is a weak anchor on this series: intraday oscillation "
                    "means same-hour-last-week (lag_168) tracks demand better."
                )


# --------------------------------------------------------------------------- #
# 2. Backtest
# --------------------------------------------------------------------------- #
with tab_backtest:
    if predictions.empty:
        st.info("Run `python scripts/train_all.py` to generate locked-test predictions.")
    else:
        st.subheader("Locked-test window")
        start, end = predictions["timestamp"].min(), predictions["timestamp"].max()
        st.caption(f"{len(predictions):,} unseen hours — {start:%d %b %Y} → {end:%d %b %Y}")

        span = st.slider(
            "Window (hours from the start of the test block)",
            24,
            len(predictions),
            min(24 * 14, len(predictions)),
            step=24,
        )
        view = predictions.head(span)

        column = model_name if model_name in view.columns else selected_name
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=view["timestamp"], y=view["actual"], name="Actual",
                       line=dict(color="#334155", width=1.6))
        )
        fig.add_trace(
            go.Scatter(x=view["timestamp"], y=view[column], name="Predicted",
                       line=dict(color=ACCENT, width=1.3))
        )
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="m³ / hour", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            err = (view["actual"] - view[column]).abs()
            by_hour = (
                pd.DataFrame({"hour": view["timestamp"].dt.hour, "abs_error": err})
                .groupby("hour", as_index=False)["abs_error"].mean()
            )
            f2 = go.Figure(go.Bar(x=by_hour["hour"], y=by_hour["abs_error"],
                                  marker_color=ACCENT))
            f2.update_layout(height=300, title="MAE by hour of day",
                             xaxis_title="hour", yaxis_title="MAE (m³)",
                             margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(f2, use_container_width=True)

        with right:
            resid = view["actual"] - view[column]
            f3 = go.Figure(go.Histogram(x=resid, nbinsx=50, marker_color=ACCENT))
            f3.add_vrect(x0=resid.quantile(0.05), x1=resid.quantile(0.95),
                         fillcolor=WARN, opacity=0.12, line_width=0)
            f3.update_layout(height=300, title="Residuals (actual − predicted)",
                             xaxis_title="m³", yaxis_title="hours",
                             margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(f3, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean error (bias)", f"{resid.mean():+.1f} m³")
        c2.metric("90% error band", f"{resid.quantile(.05):.0f} … {resid.quantile(.95):.0f}")
        c3.metric("Under-forecast rate", f"{(resid > 0).mean():.0%}")

        st.caption(
            "Under-forecasting matters operationally: it is the direction that "
            "risks a storage shortfall."
        )


# --------------------------------------------------------------------------- #
# 3. Models
# --------------------------------------------------------------------------- #
with tab_models:
    st.subheader("Every candidate, in the order it was evaluated")

    validation = pd.DataFrame(metrics.get("validation", []))
    locked = pd.DataFrame(metrics.get("locked_test", []))

    if not validation.empty:
        st.markdown("**Validation block (used for selection)**")
        table = validation[["model", "stage", "mae", "rmse", "r2", "smape_pct"]].copy()
        table.columns = ["model", "stage", "MAE", "RMSE", "R²", "sMAPE %"]
        st.dataframe(
            table.sort_values("MAE").style.format(
                {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}", "sMAPE %": "{:.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        f = go.Figure(
            go.Bar(
                x=table.sort_values("MAE")["model"],
                y=table.sort_values("MAE")["MAE"],
                marker_color=[
                    ACCENT if m == selected_name else "#94a3b8"
                    for m in table.sort_values("MAE")["model"]
                ],
            )
        )
        f.update_layout(height=340, yaxis_title="Validation MAE (m³)",
                        margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(f, use_container_width=True)

    if not locked.empty:
        st.markdown("**Locked test — opened once, after selection was frozen**")
        lt = locked[["model", "role", "mae", "rmse", "r2", "smape_pct"]].copy()
        lt.columns = ["model", "role", "MAE", "RMSE", "R²", "sMAPE %"]
        st.dataframe(
            lt.style.format(
                {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}", "sMAPE %": "{:.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        skill = metrics.get("skill_vs_naive_lag_168")
        if skill:
            st.success(
                f"**{skill:.1%} MAE reduction** versus the seasonal-naive baseline "
                "(same hour, previous week). That is the honest headline — "
                "not the 4× improvement you would get by comparing against lag-1."
            )

    cv = pd.DataFrame(metrics.get("cv", []))
    if not cv.empty:
        st.markdown("**TimeSeriesSplit cross-validation (refit inside every fold)**")
        st.dataframe(cv, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# 4. Drivers
# --------------------------------------------------------------------------- #
with tab_drivers:
    st.subheader("What moves the forecast")
    if importance.empty:
        st.info(
            "Run `python scripts/make_report.py` to compute permutation importance. "
            "It is pre-computed because the calculation does not belong in a "
            "request path."
        )
    else:
        top = importance.head(15).iloc[::-1]
        f = go.Figure(
            go.Bar(
                x=top["importance"],
                y=top["feature"],
                orientation="h",
                marker_color=ACCENT,
            )
        )
        f.update_layout(
            height=480,
            xaxis_title="increase in MAE when the feature is shuffled (m³)",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(f, use_container_width=True)
        st.caption(
            "Permutation importance on the validation block. Demand history "
            "dominates, as expected for a one-hour horizon; weather contributes "
            "little because it is only available lagged."
        )


# --------------------------------------------------------------------------- #
# 5. Methodology
# --------------------------------------------------------------------------- #
with tab_method:
    st.subheader("Split")
    if split:
        c1, c2, c3 = st.columns(3)
        c1.metric("Train", f"{split['train_rows']:,}")
        c2.metric("Validation", f"{split['validation_rows']:,}")
        c3.metric("Test", f"{split['test_rows']:,}")
        st.caption(
            f"Train ends {split['train_end']} · validation "
            f"{split['validation_start']} → {split['validation_end']} · "
            f"test {split['test_start']} → {split['test_end']}"
        )

    st.subheader("Data")
    if audit:
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw rows", f"{audit['rows']:,}")
        c2.metric("Usable rows", f"{len(raw) - config.WARMUP_HOURS:,}")
        c3.metric("Missing", sum(audit["missing"].values()))
        st.caption(
            f"Source: {audit['source']} · Zenodo {audit['zenodo_record']} · "
            f"sha256 `{audit['sha256'][:16]}…` · synthetic data: **no**"
        )
        with st.expander("Missing values by column"):
            st.json(audit["missing"])

    st.subheader("Leakage controls")
    st.markdown(
        """
| Control | How it is enforced |
|---|---|
| Demand history | every lag is `y.shift(k)` with `k ≥ 1`; rolling stats run on `y.shift(1)` |
| Weather | only lag 1 and lag 24; the target hour's observation is never a feature |
| Preprocessing | imputer, Yeo-Johnson, scaler and encoders are fitted inside each CV fold |
| Model selection | frozen to `reports/selection.json` before the test block is read |
| Tuning | Optuna and Keras Tuner search over **training rows only** |
| Regression tests | `tests/test_no_leakage.py` — 13 assertions, including a truncate-the-future test |
        """
    )
    st.caption(
        "Known limitation: this is a research prototype. In production the "
        "weather inputs must come from a forecast feed, and the model should be "
        "re-fitted on a rolling window as the demand trend drifts."
    )
