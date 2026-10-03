"""
Phase 5: Streamlit dashboard for the traffic-flow predictor.

Features
--------
* Sidebar: lookback selector, horizon selector (15/30/60 min), "simulate live" toggle.
* Plotly chart: actual vs. predicted with a visually separated forecast region.
* Colour-coded congestion indicator (Low / Medium / High).
* Metrics panel with MAE / RMSE / R2 for the selected horizon.

Caching
-------
* `st.cache_resource` for model + scaler loading (heavy, unhashable).
* `st.cache_data` for the traffic data, metrics JSON and series predictions.

Live mode
---------
Streamlit's built-in `st.fragment(run_every=...)` acts as the refresh loop and
advances the mock stream one step per tick into an `st.empty()` placeholder.
No extra dependency (streamlit-autorefresh) is required.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    SAMPLING_INTERVAL_MIN,
    HORIZONS_MIN,
    HORIZONS_STEPS,
    LOOKBACK_OPTIONS,
)
from src.congestion import get_congestion_level
from src.data_pipeline import add_time_features, make_features

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_PATH = Path("data/raw/traffic.csv")
MODELS_DIR = Path("models")

# How much history to draw on the main chart (one step = 5 minutes).
DISPLAY_WINDOWS = {
    "12 hours": 12 * 12,
    "1 day": 24 * 12,
    "2 days": 48 * 12,
}

CONGESTION_COLOURS = {
    "Low": "#2e7d32",     # green
    "Medium": "#f9a825",  # amber
    "High": "#c62828",    # red
}

HORIZON_STEPS_BY_MIN = dict(zip(HORIZONS_MIN, HORIZONS_STEPS))

st.set_page_config(
    page_title="Traffic Flow Predictor",
    page_icon="🚦",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_traffic_data(path: str) -> pd.DataFrame:
    """Reads the raw CSV, fills sensor gaps and adds calendar features."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["vehicle_count"] = df["vehicle_count"].ffill().bfill()
    df = add_time_features(df)
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    with open(MODELS_DIR / "metrics.json", "r") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model_and_scaler(lookback: int, horizon_min: int):
    """Loads the shipped XGBoost model for a fixed (lookback, horizon) pair."""
    model_path = MODELS_DIR / f"xgb_lb{lookback}_h{horizon_min}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model {model_path}. Run `python -m src.model` to train the "
            f"fixed lookbacks {LOOKBACK_OPTIONS}."
        )
    model = joblib.load(model_path)
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    return model, scaler


@st.cache_data(show_spinner=False)
def compute_series_predictions(lookback: int, horizon_min: int, path: str):
    """
    Runs the model at every origin so we can overlay a continuous
    "predicted vs actual" line and read the forecast for the latest window.

    Returns
    -------
    preds : np.ndarray
        Traffic-count predictions (inverse-scaled), aligned by index with
        `target_idx`.
    target_idx : np.ndarray
        The df row index each prediction refers to (origin + horizon steps).
    """
    df = load_traffic_data(path)
    model, scaler = load_model_and_scaler(lookback, horizon_min)
    horizon_steps = HORIZON_STEPS_BY_MIN[horizon_min]

    scaled = scaler.transform(df[["vehicle_count"]]).ravel()
    feature_df = df.copy()
    feature_df["vehicle_count_scaled"] = scaled

    X, origins = make_features(feature_df, lookback)
    preds_scaled = model.predict(X)
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    target_idx = origins + horizon_steps
    return preds, target_idx


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def _forecast_value(preds, lookback, now_idx, fallback):
    """Prediction whose window ends at `now_idx` (target = now + horizon)."""
    origin_in_preds = now_idx - (lookback - 1)
    if 0 <= origin_in_preds < len(preds):
        return float(preds[origin_in_preds])
    return float(fallback)


def render_congestion_card(label, predicted, capacity, horizon_min, now_ts):
    colour = CONGESTION_COLOURS.get(label, "#607d8b")
    text_colour = "#1b1b1b" if label == "Medium" else "#ffffff"
    pct = (predicted / capacity * 100) if capacity > 0 else 0.0
    st.markdown(
        f"""
        <div style="
            background:{colour};
            color:{text_colour};
            padding:1.1rem 1.3rem;
            border-radius:0.8rem;
            box-shadow:0 2px 8px rgba(0,0,0,0.25);
            margin-bottom:0.5rem;">
          <div style="font-size:0.85rem;letter-spacing:0.05em;text-transform:uppercase;opacity:0.9;">
            Congestion &mdash; {horizon_min} min ahead
          </div>
          <div style="font-size:2.4rem;font-weight:800;line-height:1.2;">{label}</div>
          <div style="font-size:0.95rem;opacity:0.95;">
            Predicted <b>{predicted:,.0f}</b> veh / 5 min
            &nbsp;·&nbsp; {pct:,.0f}% of 95th-pct capacity ({capacity:,.0f})
            &nbsp;·&nbsp; from {now_ts:%Y-%m-%d %H:%M}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_figure(df, preds, target_idx, lookback, horizon_min, now_idx, display_steps, capacity):
    horizon_steps = HORIZON_STEPS_BY_MIN[horizon_min]
    ts = df["timestamp"].values
    vals = df["vehicle_count"].values
    vis_start = max(0, now_idx - display_steps + 1)
    now_ts = df["timestamp"].iloc[now_idx]
    forecast_ts = now_ts + pd.Timedelta(minutes=horizon_min)

    fig = go.Figure()

    # --- Forecast region (visually separated with a shaded band) ------------ #
    fig.add_vrect(
        x0=now_ts,
        x1=forecast_ts,
        fillcolor="#ff7f0e",
        opacity=0.13,
        line_width=0,
        annotation_text=f"Forecast ({horizon_min} min)",
        annotation_position="top left",
        annotation_font_size=12,
    )
    fig.add_vline(x=now_ts, line_width=1, line_dash="dot", line_color="#888888")

    # --- Actuals ------------------------------------------------------------ #
    fig.add_trace(
        go.Scatter(
            x=ts[vis_start:now_idx + 1],
            y=vals[vis_start:now_idx + 1],
            mode="lines",
            name="Actual",
            line=dict(color="#1f77b4", width=2),
        )
    )

    # --- Backtest predictions aligned to their target timestamps ----------- #
    mask = (target_idx >= vis_start) & (target_idx <= now_idx)
    if mask.any():
        fig.add_trace(
            go.Scatter(
                x=ts[target_idx[mask]],
                y=preds[mask],
                mode="lines",
                name=f"Predicted ({horizon_min} min ahead)",
                line=dict(color="#ff7f0e", width=1.7, dash="dash"),
            )
        )

    # --- The single forecast point beyond "now" ----------------------------- #
    forecast_val = _forecast_value(preds, lookback, now_idx, vals[now_idx])
    fig.add_trace(
        go.Scatter(
            x=[now_ts, forecast_ts],
            y=[vals[now_idx], forecast_val],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#d62728", width=2.6),
            marker=dict(size=10, symbol="diamond", color="#d62728"),
        )
    )

    # --- Capacity reference ------------------------------------------------- #
    if capacity > 0:
        fig.add_hline(
            y=capacity,
            line_width=1,
            line_dash="longdash",
            line_color="#2e7d32",
            annotation_text="95th-pct capacity",
            annotation_position="bottom right",
            annotation_font_size=11,
        )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Time",
        yaxis_title="Vehicles per 5-min interval",
        hovermode="x unified",
    )
    return fig


def render_metrics_panel(metrics, lookback, horizon_min, scale):
    xgb = metrics["xgboost"]
    pers = metrics["persistence"]

    st.subheader(
        f"Model metrics — {horizon_min} min horizon "
        f"(lookback {lookback * SAMPLING_INTERVAL_MIN} min)"
    )
    c1, c2, c3 = st.columns(3)
    # MAE/RMSE were computed in standardized units; convert back to vehicles.
    c1.metric("MAE (veh/5min)", f"{xgb['MAE'] * scale:,.1f}", help="Mean Absolute Error (test set)")
    c2.metric("RMSE (veh/5min)", f"{xgb['RMSE'] * scale:,.1f}", help="Root Mean Squared Error (test set)")
    c3.metric("R²", f"{xgb['R2']:.3f}", help="Coefficient of determination (test set)")

    with st.expander("Baseline comparison & standardized metrics"):
        st.markdown(
            "Metrics are evaluated on the chronological **test split** (last 15% of data). "
            "The persistence baseline simply repeats the last observed count."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "XGBoost": {
                        "MAE (std)": xgb["MAE"],
                        "RMSE (std)": xgb["RMSE"],
                        "R²": xgb["R2"],
                    },
                    "Persistence": {
                        "MAE (std)": pers["MAE"],
                        "RMSE (std)": pers["RMSE"],
                        "R²": pers["R2"],
                    },
                }
            )
        )


def render_view(now_idx, slots, df, preds, target_idx, lookback, horizon_min,
                capacity, display_steps):
    """Draws the congestion card + chart + caption into the given placeholders."""
    indicator_slot, chart_slot, caption_slot = slots
    now_ts = df["timestamp"].iloc[now_idx]
    actual_now = float(df["vehicle_count"].iloc[now_idx])
    forecast_val = _forecast_value(preds, lookback, now_idx, actual_now)
    label, _ = get_congestion_level(forecast_val, capacity)

    with indicator_slot.container():
        left, right = st.columns([1, 2])
        with left:
            render_congestion_card(label, forecast_val, capacity, horizon_min, now_ts)
        with right:
            delta = forecast_val - actual_now
            st.metric(
                f"Current count ({now_ts:%H:%M})",
                f"{actual_now:,.0f} veh / 5 min",
                delta=f"{delta:+,.0f} vs forecast",
            )
            st.metric(
                "Forecast 95th-% capacity utilisation",
                f"{(forecast_val / capacity * 100) if capacity > 0 else 0:,.0f}%",
            )

    with chart_slot.container():
        fig = build_figure(
            df, preds, target_idx, lookback, horizon_min, now_idx, display_steps, capacity
        )
        st.plotly_chart(fig, width="stretch")

    with caption_slot.container():
        st.caption(
            f"Showing row {now_idx + 1:,} of {len(df):,} "
            f"({now_ts:%Y-%m-%d %H:%M}). Each model was trained for a fixed lookback; "
            f"the slider selects among pre-trained models."
        )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.header("Controls")

lookback = st.sidebar.select_slider(
    "Lookback window (sliding history used as model input)",
    options=LOOKBACK_OPTIONS,
    value=LOOKBACK_OPTIONS[1],
    format_func=lambda steps: f"{steps} steps ({steps * SAMPLING_INTERVAL_MIN} min)",
)
horizon_min = st.sidebar.selectbox("Prediction horizon", HORIZONS_MIN, index=1)
display_label = st.sidebar.selectbox("Chart history", list(DISPLAY_WINDOWS.keys()), index=1)
live_mode = st.sidebar.toggle("Simulate live", value=False)
live_interval = st.sidebar.slider(
    "Live refresh (seconds)", min_value=1, max_value=10, value=3, disabled=not live_mode
)

st.sidebar.markdown(
    f"""
    ---
    **Lookback design**

    Models are trained once for the fixed lookbacks
    `{LOOKBACK_OPTIONS}` (30 min / 1 h / 2 h) and the slider picks among them;
    we do not retrain on the fly. See the README for details.
    """
)

# --------------------------------------------------------------------------- #
# Load data + model outputs
# --------------------------------------------------------------------------- #
try:
    df = load_traffic_data(str(DATA_PATH))
except FileNotFoundError:
    st.error(
        f"Could not find {DATA_PATH}. Generate it first with:\n\n"
        "`python -m src.mock_stream`"
    )
    st.stop()

if len(df) <= max(LOOKBACK_OPTIONS):
    st.error("Not enough data to build features. Generate more mock data.")
    st.stop()

capacity = float(df["vehicle_count"].quantile(0.95))
display_steps = DISPLAY_WINDOWS[display_label]

try:
    preds, target_idx = compute_series_predictions(lookback, horizon_min, str(DATA_PATH))
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

metrics = load_metrics()[f"lookback_{lookback}"][f"horizon_{horizon_min}_mins"]
_scale = load_model_and_scaler(lookback, horizon_min)[1].scale_[0]

# --------------------------------------------------------------------------- #
# Main layout
# --------------------------------------------------------------------------- #
st.title("🚦 Traffic Flow Predictor")
st.markdown(
    f"Forecasting vehicle counts {horizon_min} minutes ahead · "
    f"capacity (95th percentile) ≈ **{capacity:,.0f} veh / 5 min**."
)

render_metrics_panel(metrics, lookback, horizon_min, _scale)
st.divider()

if live_mode:
    live_start = max(lookback - 1, len(df) - 1 - (24 * 12))  # 1 day of ticks
    if st.session_state.get("_live_lookback") != lookback:
        st.session_state._live_cursor = live_start
        st.session_state._live_lookback = lookback
    if "_live_cursor" not in st.session_state:
        st.session_state._live_cursor = live_start

    st.info(
        f"Live simulation running every {live_interval}s — advancing one 5-minute "
        "step per tick. Toggle **Simulate live** off to pause."
    )

    @st.fragment(run_every=live_interval)
    def live_fragment():
        cursor = int(st.session_state._live_cursor)
        # st.empty() placeholders so the fragment rewrites in place each tick.
        render_view(
            cursor,
            (st.empty(), st.empty(), st.empty()),
            df, preds, target_idx, lookback, horizon_min, capacity, display_steps,
        )
        nxt = cursor + 1
        st.session_state._live_cursor = nxt if nxt <= len(df) - 1 else live_start

    live_fragment()
else:
    render_view(
        len(df) - 1,
        (st.empty(), st.empty(), st.empty()),
        df, preds, target_idx, lookback, horizon_min, capacity, display_steps,
    )
