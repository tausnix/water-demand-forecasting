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

# --------------------------------------------------------------------------- #
# Cyber-teal SCADA palette
# --------------------------------------------------------------------------- #
ACCENT = "#06B6D4"                     # cyan — forecast / selected model
ACCENT_SOFT = "rgba(6, 182, 212, 0.16)"  # confidence band fill
ACTUAL = "#94A3B8"                     # slate-400 — observations
MUTED = "#475569"                      # de-emphasised models
WARN = "#F59E0B"                       # amber — error band, bias
GRID = "rgba(148, 163, 184, 0.14)"
TEXT = "#F9FAFB"
TEXT_DIM = "#94A3B8"

BACKTEST_HOURS = 72


def style_chart(
    fig: go.Figure,
    *,
    height: int = 340,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    show_legend: bool = True,
) -> go.Figure:
    """Apply the control-room look shared by every chart in the app."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=4, r=4, t=42 if title else 12, b=4),
        title=dict(text=title, font=dict(size=13, color=TEXT_DIM)) if title else None,
        font=dict(color=TEXT, size=12),
        xaxis_title=x_title,
        yaxis_title=y_title,
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="#111827", bordercolor=ACCENT, font=dict(color=TEXT, size=12)
        ),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def kpi_card(column, label: str, value: str, note: str | None = None) -> None:
    """One bordered KPI tile in the top row."""
    with column:
        with st.container(border=True):
            st.metric(label, value)
            if note:
                st.caption(note)


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
    if not target.exists():
        return pd.DataFrame()
    frame = pd.read_csv(target)
    # Not every artefact is time-indexed (importance.csv is not), so only parse
    # the timestamp when the column is actually present.
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame


raw = load_raw()
metrics = load_json(str(config.METRICS_JSON))
split = load_json(str(config.SPLIT_JSON))
audit = load_json(str(config.REPORTS_DIR / "data_audit.json"))
predictions = load_csv(str(config.REPORTS_DIR / "locked_test_predictions.csv"))
importance = load_csv(str(config.REPORTS_DIR / "importance.csv"))

selected_name = metrics.get("selection", {}).get("selected_before_test", "voting_regressor")
available = sorted(path.stem for path in config.MODELS_DIR.glob("*.joblib"))
if selected_name in available:  # keep the reported model first in the selector
    available = [selected_name] + [m for m in available if m != selected_name]

locked_selected = next(
    (r for r in metrics.get("locked_test", []) if r.get("role") == "selected"), None
)
skill = metrics.get("skill_vs_naive_lag_168")


@st.cache_data(show_spinner=False, max_entries=8)
def forecast_at(model: str, target: str) -> dict:
    """Cached single-hour forecast, keyed on model + target timestamp.

    Building features over the full 45k-hour history is the expensive part of a
    request, so it must not run twice per script execution or on every rerun.
    """
    ts = pd.Timestamp(target)
    history = raw[raw["timestamp"] < ts]
    return inference.forecast_next_hour(history, load_pipeline(model), ts)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("💧 Water Demand")
    st.caption("SCADA control room · hourly forecasting")
    st.divider()

    model_name = st.selectbox(
        "Serving model",
        available or [selected_name],
        help="The voting ensemble is the model selected on validation. XGBoost is "
        "the lighter alternative (~6 MB) — their validation MAE differs by noise.",
    )
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


# --------------------------------------------------------------------------- #
# Top KPI row — the four numbers that matter at a glance
# --------------------------------------------------------------------------- #
latest_ts = raw["timestamp"].max()
next_ts = latest_ts + pd.Timedelta(hours=1)
next_hour = forecast_at(model_name, next_ts.isoformat())

k1, k2, k3, k4 = st.columns(4)
kpi_card(
    k1,
    "Next Hour Forecast",
    f"{next_hour['prediction']:,.0f} m³",
    f"for {next_ts:%d %b %Y, %H:%M}",
)
kpi_card(
    k2,
    "Locked-Test R²",
    f"{locked_selected['r2']:.4f}" if locked_selected else "—",
    "unseen 6,880 hours",
)
kpi_card(
    k3,
    "Locked-Test MAE",
    f"{locked_selected['mae']:.2f} m³" if locked_selected else "—",
    "selected before the test was read",
)
kpi_card(
    k4,
    "Skill vs. Naive",
    f"+{skill:.1%}" if skill else "—",
    "vs same hour last week",
)

st.write("")

tab_forecast, tab_backtest, tab_models, tab_drivers, tab_method = st.tabs(
    ["🔮 Forecast", "📈 Backtest", "🏁 Models", "🔍 Drivers", "🧪 Methodology"]
)


# --------------------------------------------------------------------------- #
# 1. Forecast
# --------------------------------------------------------------------------- #
with tab_forecast:
    if predictions.empty:
        st.info("Run `python scripts/train_all.py` to generate locked-test predictions.")
    else:
        # Backtest artefacts are recorded for the model selected on validation;
        # fall back when the sidebar is switched to a model with no saved column.
        column = model_name if model_name in predictions.columns else selected_name
        view = predictions.tail(BACKTEST_HOURS).copy()

        # Empirical 90% band from the whole locked-test residual distribution,
        # not from the 72 hours on screen (which would be circular).
        resid_all = predictions["actual"] - predictions[column]
        lo, hi = resid_all.quantile(0.05), resid_all.quantile(0.95)

        left, right = st.columns([1.6, 1], gap="large")

        with left:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=view["timestamp"],
                    y=view[column] + lo,
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=view["timestamp"],
                    y=view[column] + hi,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=ACCENT_SOFT,
                    name=f"90% band ({lo:+.0f} … {hi:+.0f} m³)",
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=view["timestamp"],
                    y=view["actual"],
                    name="Actual",
                    line=dict(color=ACTUAL, width=1.5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=view["timestamp"],
                    y=view[column],
                    name="Predicted",
                    line=dict(color=ACCENT, width=1.6),
                )
            )
            style_chart(
                fig,
                height=360,
                title=f"{BACKTEST_HOURS}-hour backtest — actual vs predicted",
                y_title="m³ / hour",
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"Final {BACKTEST_HOURS} hours of the locked test "
                f"({view['timestamp'].min():%d %b} → {view['timestamp'].max():%d %b %Y}). "
                "The cyan band is the empirical 90% residual interval."
            )

        with right:
            if importance.empty:
                st.info("Run `python scripts/make_report.py` for feature importance.")
            else:
                top6 = importance.head(6).iloc[::-1]
                f_drv = go.Figure(
                    go.Bar(
                        x=top6["importance"],
                        y=top6["feature"],
                        orientation="h",
                        marker=dict(color=ACCENT, line=dict(width=0)),
                        hovertemplate="%{y}: +%{x:.1f} m³ MAE<extra></extra>",
                    )
                )
                style_chart(
                    f_drv,
                    height=210,
                    title="Top-6 feature drivers",
                    x_title="Δ MAE (m³)",
                    show_legend=False,
                )
                st.plotly_chart(f_drv, width="stretch")

            if not predictions.empty:
                heat = predictions.copy()
                heat["hour"] = heat["timestamp"].dt.hour
                heat["day"] = heat["timestamp"].dt.day_name().str[:3]
                heat["residual"] = heat["actual"] - heat[column]
                pivot = heat.pivot_table(
                    index="day", columns="hour", values="residual", aggfunc="mean"
                ).reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

                f_heat = go.Figure(
                    go.Heatmap(
                        z=pivot.values,
                        x=[f"{h:02d}" for h in pivot.columns],
                        y=pivot.index,
                        colorscale="RdBu",
                        zmid=0,
                        colorbar=dict(
                            title=dict(text="m³", font=dict(size=10)),
                            thickness=8,
                            len=0.9,
                            tickfont=dict(size=9),
                        ),
                        hovertemplate="%{y} %{x}:00 — %{z:+.0f} m³<extra></extra>",
                    )
                )
                style_chart(
                    f_heat,
                    height=250,
                    title="Hourly residual heatmap (actual − predicted)",
                    show_legend=False,
                )
                st.plotly_chart(f_heat, width="stretch")
                st.caption(
                    "All 6,880 test hours. Blue = over-forecast, red = "
                    "under-forecast. The bias is structural, not hour-specific: "
                    "the test block sits above the training range."
                )

    st.divider()

    # Opt-in: building features over the full history is the expensive step, so
    # the inspector must not run on every page load. The KPI row already shows
    # the live next-hour forecast.
    if st.toggle("Inspect a specific hour", value=False):
        with st.expander("Hour inspector", expanded=True):
            c_date, c_hour = st.columns(2)
            with c_date:
                chosen_date = st.date_input(
                    "Target date",
                    value=(latest_ts - pd.Timedelta(days=7)).date(),
                    min_value=raw["timestamp"].min().date(),
                    max_value=latest_ts.date(),
                )
            with c_hour:
                chosen_hour = st.slider("Target hour", 0, 23, 12)

            target_ts = pd.Timestamp(chosen_date) + pd.Timedelta(hours=chosen_hour)

            if (target_ts - raw["timestamp"].min()) < pd.Timedelta(
                hours=config.WARMUP_HOURS
            ):
                st.warning(
                    f"Need at least {config.WARMUP_HOURS} hours of history before "
                    f"{target_ts}. Pick a later date."
                )
            else:
                result = forecast_at(model_name, target_ts.isoformat())
                anchors = result["persistence"]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Predicted demand", f"{result['prediction']:,.0f} m³")
                c2.metric("Last observed", f"{result['last_observed']:,.0f} m³")
                if anchors.get("lag_168") is not None:
                    c3.metric(
                        "Same hour last week",
                        f"{anchors['lag_168']:,.0f} m³",
                        delta=f"{result['prediction'] - anchors['lag_168']:+,.0f}",
                        delta_color="inverse",
                    )

                actual_row = raw[raw["timestamp"] == target_ts]
                if not actual_row.empty:
                    actual = float(actual_row["Consumption"].iloc[0])
                    c4.metric(
                        "Actual (held out)",
                        f"{actual:,.0f} m³",
                        delta=f"{result['prediction'] - actual:+,.0f} error",
                        delta_color="off",
                    )

                window = raw[
                    (raw["timestamp"] < target_ts)
                    & (raw["timestamp"] >= target_ts - pd.Timedelta(hours=24 * 7))
                ]
                f_ctx = go.Figure()
                f_ctx.add_trace(
                    go.Scatter(
                        x=window["timestamp"],
                        y=window["Consumption"],
                        name="Observed",
                        line=dict(color=ACTUAL, width=1.4),
                    )
                )
                f_ctx.add_trace(
                    go.Scatter(
                        x=[target_ts],
                        y=[result["prediction"]],
                        name="Forecast",
                        mode="markers",
                        marker=dict(color=ACCENT, size=12, symbol="diamond"),
                    )
                )
                style_chart(
                    f_ctx,
                    height=300,
                    title=f"History and forecast to {target_ts:%d %b %Y, %H:%M}",
                    y_title="m³ / hour",
                )
                st.plotly_chart(f_ctx, width="stretch")

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
        if column != model_name:
            st.caption(
                f"Recorded predictions exist for `{selected_name}` only — showing "
                "those regardless of the sidebar selection."
            )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=view["timestamp"],
                y=view["actual"],
                name="Actual",
                line=dict(color=ACTUAL, width=1.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=view["timestamp"],
                y=view[column],
                name="Predicted",
                line=dict(color=ACCENT, width=1.4),
            )
        )
        style_chart(fig, height=380, y_title="m³ / hour")
        st.plotly_chart(fig, width="stretch")

        left, right = st.columns(2)
        with left:
            err = (view["actual"] - view[column]).abs()
            by_hour = (
                pd.DataFrame({"hour": view["timestamp"].dt.hour, "abs_error": err})
                .groupby("hour", as_index=False)["abs_error"].mean()
            )
            f2 = go.Figure(
                go.Bar(
                    x=by_hour["hour"],
                    y=by_hour["abs_error"],
                    marker=dict(color=ACCENT, line=dict(width=0)),
                )
            )
            style_chart(
                f2,
                height=300,
                title="MAE by hour of day",
                x_title="hour",
                y_title="MAE (m³)",
                show_legend=False,
            )
            st.plotly_chart(f2, width="stretch")

        with right:
            resid = view["actual"] - view[column]
            f3 = go.Figure(
                go.Histogram(
                    x=resid,
                    nbinsx=50,
                    marker=dict(color=ACCENT, line=dict(width=0)),
                )
            )
            f3.add_vrect(
                x0=resid.quantile(0.05),
                x1=resid.quantile(0.95),
                fillcolor=WARN,
                opacity=0.12,
                line_width=0,
            )
            style_chart(
                f3,
                height=300,
                title="Residuals (actual − predicted)",
                x_title="m³",
                y_title="hours",
                show_legend=False,
            )
            st.plotly_chart(f3, width="stretch")

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
            width="stretch",
            hide_index=True,
        )

        ordered = table.sort_values("MAE")
        f = go.Figure(
            go.Bar(
                x=ordered["model"],
                y=ordered["MAE"],
                marker=dict(
                    color=[ACCENT if m == selected_name else MUTED for m in ordered["model"]],
                    line=dict(width=0),
                ),
            )
        )
        style_chart(
            f, height=340, y_title="Validation MAE (m³)", show_legend=False
        )
        st.plotly_chart(f, width="stretch")

    if not locked.empty:
        st.markdown("**Locked test — opened once, after selection was frozen**")
        lt = locked[["model", "role", "mae", "rmse", "r2", "smape_pct"]].copy()
        lt.columns = ["model", "role", "MAE", "RMSE", "R²", "sMAPE %"]
        st.dataframe(
            lt.style.format(
                {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}", "sMAPE %": "{:.2f}"}
            ),
            width="stretch",
            hide_index=True,
        )
        if skill:
            st.success(
                f"**{skill:.1%} MAE reduction** versus the seasonal-naive baseline "
                "(same hour, previous week). That is the honest headline — "
                "not the 4× improvement you would get by comparing against lag-1."
            )

    cv = pd.DataFrame(metrics.get("cv", []))
    if not cv.empty:
        st.markdown("**TimeSeriesSplit cross-validation (refit inside every fold)**")
        st.dataframe(cv, width="stretch", hide_index=True)


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
                marker=dict(color=ACCENT, line=dict(width=0)),
                hovertemplate="%{y}: +%{x:.1f} m³ MAE<extra></extra>",
            )
        )
        style_chart(
            f,
            height=480,
            x_title="increase in MAE when the feature is shuffled (m³)",
            show_legend=False,
        )
        st.plotly_chart(f, width="stretch")
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
