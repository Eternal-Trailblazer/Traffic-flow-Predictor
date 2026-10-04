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
advances the mock stream one step per tick. The header and KPI cards are
rendered directly rather than through `st.empty()` slots: an empty's subtree is
replaced wholesale each rerun, which destroyed and rebuilt them every tick and
was the main source of the flicker. The chart sits in a fixed-height box that
reserves its space across ticks.
No extra dependency (streamlit-autorefresh) is required.
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    DATA_PATH,
    DATASET_NAME,
    MODELS_DIR,
    SAMPLING_INTERVAL_MIN,
    STEPS_PER_HOUR,
    HORIZONS_MIN,
    HORIZONS_STEPS,
    LOOKBACK_OPTIONS,
    format_duration,
    metrics_filename,
    model_filename,
    scaler_filename,
)
from src.congestion import get_congestion_level
from src.data_pipeline import add_time_features, make_features

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# How much history to draw on the main chart (one step = SAMPLING_INTERVAL_MIN).
DISPLAY_WINDOWS = {
    "12 hours": 12 * STEPS_PER_HOUR,
    "1 day": 24 * STEPS_PER_HOUR,
    "2 days": 48 * STEPS_PER_HOUR,
}

# Congestion semantics (label -> colour).
CONGESTION_COLOURS = {
    "Low": "#22c55e",     # green
    "Medium": "#f59e0b",  # amber
    "High": "#ef4444",    # red
}
CONGESTION_TINT = {  # translucent version used for the capacity bar
    "Low": "rgba(34,197,94,0.35)",
    "Medium": "rgba(245,158,11,0.35)",
    "High": "rgba(239,68,68,0.35)",
}

# Chart trace colours — deliberately distinct from the congestion palette so a
# prediction line is never mistaken for a congestion state.
TRACE_COLOURS = {
    "actual": "#38bdf8",     # sky
    "predicted": "#a78bfa",  # violet
    "forecast": "#fb7185",   # rose
    "capacity": "#4ade80",   # green
}

HORIZON_STEPS_BY_MIN = dict(zip(HORIZONS_MIN, HORIZONS_STEPS))

st.set_page_config(
    page_title="Traffic Flow Predictor",
    page_icon="🚦",
    layout="wide",
)

# Which theme are we rendering into? (light/dark) so chart + cards match it.
IS_DARK = (st.context.theme.type or "dark") != "light"


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def _theme_vars() -> str:
    if IS_DARK:
        return """
            --card-bg: rgba(255,255,255,0.035);
            --card-border: rgba(255,255,255,0.09);
            --muted: #8b98ab;
            --text: #e5ecf5;
            --grid: rgba(255,255,255,0.07);
            --hover-border: rgba(56,189,248,0.55);
        """
    return """
        --card-bg: rgba(15,23,42,0.03);
        --card-border: rgba(15,23,42,0.12);
        --muted: #5b6676;
        --text: #0f172a;
        --grid: rgba(15,23,42,0.09);
        --hover-border: rgba(2,132,199,0.55);
    """


PAGE_CSS = f"""
<style>
:root {{
    --accent: #38bdf8;
    --low: #22c55e;
    --medium: #f59e0b;
    --high: #ef4444;
    {_theme_vars()}
}}

/* ---- page rhythm -------------------------------------------------------- */
.block-container {{
    /* Streamlit's fixed header overlays the top 60px — keep clear of it. */
    padding-top: calc(60px + 1.4rem);
    padding-bottom: 2.5rem;
    max-width: 1280px;
}}
h1, h2, h3 {{ letter-spacing: -0.02em; }}

/* ---- hero header -------------------------------------------------------- */
.tfp-hero {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--card-border);
}}
.tfp-title {{
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1.15;
}}
.tfp-sub {{
    font-size: 0.92rem;
    color: var(--muted);
    margin-top: 0.3rem;
}}
.tfp-live {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.32rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    border: 1px solid var(--card-border);
    background: var(--card-bg);
    color: var(--muted);
    white-space: nowrap;
}}
.tfp-live.on {{
    border-color: rgba(34,197,94,0.55);
    color: #22c55e;
    background: rgba(34,197,94,0.10);
}}
.tfp-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: currentColor;
}}
.tfp-live.on .tfp-dot {{ animation: tfpPulse 1.6s ease-in-out infinite; }}
@keyframes tfpPulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%      {{ opacity: 0.35; transform: scale(0.7); }}
}}

/* ---- KPI cards ---------------------------------------------------------- */
.tfp-card {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 0.85rem 1rem 0.95rem;
    min-height: 132px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
}}
.tfp-card:hover {{
    transform: translateY(-2px);
    border-color: var(--hover-border);
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}}
.tfp-card.accent {{ border-left: 3px solid var(--accent-col, var(--accent)); }}
.tfp-label {{
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.tfp-value {{
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1.1;
    margin: 0.35rem 0 0.15rem;
    white-space: nowrap;
}}
.tfp-unit {{
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--muted);
    margin-left: 0.3rem;
    white-space: nowrap;
}}
.tfp-sub {{
    font-size: 0.78rem;
    color: var(--muted);
    line-height: 1.35;
}}
.tfp-up   {{ color: #ef4444; font-weight: 700; }}
.tfp-down {{ color: #22c55e; font-weight: 700; }}

/* capacity utilisation bar */
.tfp-bar {{
    height: 6px;
    border-radius: 3px;
    background: var(--card-border);
    overflow: hidden;
    margin: 0.5rem 0 0.4rem;
}}
.tfp-bar > span {{
    display: block;
    height: 100%;
    border-radius: 3px;
    transition: width 0.35s ease;
}}

/* ---- metrics panel ------------------------------------------------------ */
div[data-testid="stMetric"] {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 0.85rem 1rem;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted) !important;
}}
[data-testid="stMetricValue"] {{ font-size: 1.6rem !important; font-weight: 800; }}
[data-testid="stMetricDelta"] {{ font-size: 0.82rem !important; }}

/* ---- chart repaint smoothness -------------------------------------------- */
/* The chart element is remounted by Streamlit each tick (its id is derived
   from the figure JSON). The reserved box prevents layout collapse; this
   short fade softens the hard remove/insert swap. */
[data-testid="stElementContainer"]:has(.js-plotly-plot) {{
    animation: tfpChartIn 0.2s ease-out;
}}
@keyframes tfpChartIn {{
    from {{ opacity: 0.55; }}
    to   {{ opacity: 1; }}
}}

/* ---- sidebar ------------------------------------------------------------ */
[data-testid="stSidebar"] {{
    border-right: 1px solid var(--card-border);
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 1.6rem;
}}
[data-testid="stSidebar"] label {{
    font-weight: 600;
}}
.tfp-side-note {{
    font-size: 0.78rem;
    color: var(--muted);
    line-height: 1.5;
}}

/* section eyebrow above the chart / metrics */
.tfp-section {{
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0.4rem 0 0.15rem;
}}
</style>
"""


def kpi_card(label: str, value: str, sub: str = "", *, unit: str = "",
             accent: str | None = None, bar: tuple[int, str] | None = None) -> str:
    """Builds the HTML for one dashboard KPI card."""
    cls = "tfp-card accent" if accent else "tfp-card"
    accent_style = f"--accent-col:{accent};" if accent else ""
    bar_html = ""
    if bar is not None:
        pct, colour = bar
        bar_html = (
            f'<div class="tfp-bar"><span style="width:{max(0.0, min(100.0, pct)):.1f}%;'
            f'background:{colour}"></span></div>'
        )
    unit_html = f'<span class="tfp-unit">{unit}</span>' if unit else ""
    return f"""
    <div class="{cls}" style="{accent_style}">
      <div class="tfp-label">{label}</div>
      <div class="tfp-value">{value}{unit_html}</div>
      {bar_html}
      <div class="tfp-sub">{sub}</div>
    </div>
    """


def render_header(horizon_min: int, capacity: float, live_mode: bool,
                  live_interval: int, now_ts: pd.Timestamp) -> None:
    """Compact hero header: title, context line and a live/static status pill."""
    if live_mode:
        status = (
            f'<span class="tfp-live on"><span class="tfp-dot"></span>'
            f'LIVE · {live_interval}s</span>'
        )
    else:
        status = (
            f'<span class="tfp-live"><span class="tfp-dot"></span>'
            f'STATIC · {now_ts:%H:%M}</span>'
        )
    st.markdown(
        f"""
        <div class="tfp-hero">
          <div>
            <div class="tfp-title">🚦 Traffic Flow Predictor</div>
            <div class="tfp-sub">
              Forecasting vehicle counts <b>{format_duration(horizon_min)} ahead</b>
              &nbsp;·&nbsp; capacity (95th percentile) ≈ <b>{capacity:,.0f}</b> veh / {format_duration(SAMPLING_INTERVAL_MIN)}
              &nbsp;·&nbsp; dataset <b>{DATASET_NAME}</b>
            </div>
          </div>
          <div>{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
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
    with open(MODELS_DIR / metrics_filename(), "r") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model_and_scaler(lookback: int, horizon_min: int):
    """Loads the shipped XGBoost model for a fixed (lookback, horizon) pair."""
    model_path = MODELS_DIR / model_filename(lookback, horizon_min)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model {model_path.name}. Run `python -m src.model` to train "
            f"the fixed lookbacks {LOOKBACK_OPTIONS} for this dataset."
        )
    model = joblib.load(model_path)
    scaler = joblib.load(MODELS_DIR / scaler_filename())
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


def render_kpi_row(now_idx, df, preds, lookback, horizon_min, capacity):
    """Four KPI cards: congestion state, forecast, now, capacity."""
    now_ts = df["timestamp"].iloc[now_idx]
    actual_now = float(df["vehicle_count"].iloc[now_idx])
    forecast_val = _forecast_value(preds, lookback, now_idx, actual_now)
    label, _ = get_congestion_level(forecast_val, capacity)

    pct = (forecast_val / capacity * 100) if capacity > 0 else 0.0
    colour = CONGESTION_COLOURS.get(label, "#64748b")
    delta = forecast_val - actual_now
    delta_cls = "tfp-up" if delta > 0 else ("tfp-down" if delta < 0 else "")
    delta_txt = ("▲" if delta > 0 else ("▼" if delta < 0 else "•")) + f"{abs(delta):,.0f} vs now"

    c1, c2, c3, c4 = st.columns([1.25, 1, 1, 1])

    with c1:
        st.markdown(
            kpi_card(
                "Congestion",
                label,
                f"in {format_duration(horizon_min)} · {forecast_val:,.0f} veh · {pct:,.0f}% of capacity",
                accent=colour,
                bar=(pct, colour),
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            kpi_card(
                "Forecast",
                f"{forecast_val:,.0f}",
                f'in {format_duration(horizon_min)} · <span class="{delta_cls}">{delta_txt}</span>',
                unit=f"veh/{format_duration(SAMPLING_INTERVAL_MIN)}",
                accent=TRACE_COLOURS["forecast"],
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            kpi_card(
                "Current",
                f"{actual_now:,.0f}",
                f"{now_ts:%H:%M} · row {now_idx + 1:,} of {len(df):,}",
                unit=f"veh/{format_duration(SAMPLING_INTERVAL_MIN)}",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            kpi_card(
                "Capacity",
                f"{capacity:,.0f}",
                "95th percentile · reference line on chart",
                unit=f"veh/{format_duration(SAMPLING_INTERVAL_MIN)}",
                accent=TRACE_COLOURS["capacity"],
            ),
            unsafe_allow_html=True,
        )


def build_figure(df, preds, target_idx, lookback, horizon_min, now_idx,
                 display_steps, capacity):
    ts = df["timestamp"].values
    vals = df["vehicle_count"].values
    vis_start = max(0, now_idx - display_steps + 1)
    now_ts = df["timestamp"].iloc[now_idx]
    forecast_ts = now_ts + pd.Timedelta(minutes=horizon_min)

    grid = "rgba(255,255,255,0.07)" if IS_DARK else "rgba(15,23,42,0.09)"
    axis_text = "#8b98ab" if IS_DARK else "#5b6676"

    fig = go.Figure()

    # --- Forecast region (visually separated with a shaded band) ------------ #
    fig.add_vrect(
        x0=now_ts,
        x1=forecast_ts,
        fillcolor=TRACE_COLOURS["predicted"],
        opacity=0.14,
        line_width=0,
        annotation_text=f"Forecast ({format_duration(horizon_min)})",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color=axis_text,
    )
    fig.add_vline(
        x=now_ts,
        line_width=1,
        line_dash="dot",
        line_color=axis_text,
        annotation_text="now",
        annotation_position="bottom right",
        annotation_font_size=10,
        annotation_font_color=axis_text,
    )

    # --- Actuals ------------------------------------------------------------ #
    fig.add_trace(
        go.Scatter(
            x=ts[vis_start:now_idx + 1],
            y=vals[vis_start:now_idx + 1],
            mode="lines",
            name="Actual",
            line=dict(color=TRACE_COLOURS["actual"], width=2),
            hovertemplate="%{x|%b %d %H:%M} · %{y:.0f} veh<extra>Actual</extra>",
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
                name=f"Predicted ({format_duration(horizon_min)} ahead)",
                line=dict(color=TRACE_COLOURS["predicted"], width=1.8, dash="dash"),
                hovertemplate="%{x|%b %d %H:%M} · %{y:.0f} veh<extra>Predicted</extra>",
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
            line=dict(color=TRACE_COLOURS["forecast"], width=2.6),
            marker=dict(size=10, symbol="diamond", color=TRACE_COLOURS["forecast"]),
            hovertemplate="%{x|%b %d %H:%M} · %{y:.0f} veh<extra>Forecast</extra>",
        )
    )

    # --- Capacity reference ------------------------------------------------- #
    if capacity > 0:
        fig.add_hline(
            y=capacity,
            line_width=1,
            line_dash="longdash",
            line_color=TRACE_COLOURS["capacity"],
            annotation_text="95th-pct capacity",
            annotation_position="bottom right",
            annotation_font_size=10,
            annotation_font_color=axis_text,
        )

    fig.update_layout(
        template="plotly_dark" if IS_DARK else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(l=64, r=8, t=56, b=8),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="left",
            x=0,
            font=dict(size=12),
        ),
        hovermode="x unified",
        hoverlabel=dict(font_size=13),
        # Keeps zoom/pan stable while live ticks re-render the figure.
        uirevision="tfp-chart",
        font=dict(color=axis_text),
    )
    fig.update_xaxes(
        gridcolor=grid,
        zeroline=False,
        showline=False,
        title="",
        linecolor=grid,
        tickfont=dict(color=axis_text),
    )
    fig.update_yaxes(
        gridcolor=grid,
        zeroline=False,
        showline=False,
        title=f"Vehicles / {format_duration(SAMPLING_INTERVAL_MIN)}",
        title_font=dict(color=axis_text, size=12),
        tickfont=dict(color=axis_text),
    )
    return fig


def render_metrics_panel(metrics, lookback, horizon_min, scale):
    """MAE / RMSE / R² cards with deltas against the persistence baseline."""
    xgb = metrics["xgboost"]
    pers = metrics["persistence"]

    st.markdown(
        f'<div class="tfp-section">Model performance · {format_duration(horizon_min)} horizon'
        f' · lookback {format_duration(lookback * SAMPLING_INTERVAL_MIN)}</div>',
        unsafe_allow_html=True,
    )

    # MAE/RMSE are "lower is better" (inverse), R² is "higher is better".
    mae_delta = (xgb["MAE"] - pers["MAE"]) * scale
    rmse_delta = (xgb["RMSE"] - pers["RMSE"]) * scale
    r2_delta = xgb["R2"] - pers["R2"]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "MAE", f"{xgb['MAE'] * scale:,.1f}",
        delta=f"{mae_delta:+,.1f} vs baseline",
        delta_color="inverse",
        help="Mean Absolute Error on the chronological test split, in vehicles.",
    )
    c2.metric(
        "RMSE", f"{xgb['RMSE'] * scale:,.1f}",
        delta=f"{rmse_delta:+,.1f} vs baseline",
        delta_color="inverse",
        help="Root Mean Squared Error on the chronological test split, in vehicles.",
    )
    c3.metric(
        "R²", f"{xgb['R2']:.3f}",
        delta=f"{r2_delta:+.3f} vs baseline",
        help="Coefficient of determination on the chronological test split.",
    )

    with st.expander("Baseline comparison & standardized metrics"):
        st.markdown(
            "Metrics are evaluated on the chronological **test split** (last 15% of data). "
            "The persistence baseline simply repeats the last observed count."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "XGBoost (std)": [xgb["MAE"], xgb["RMSE"], xgb["R2"]],
                    "Persistence (std)": [pers["MAE"], pers["RMSE"], pers["R2"]],
                },
                index=["MAE", "RMSE", "R²"],
            ),
            width="stretch",
        )


def render_view(now_idx, df, preds, target_idx, lookback, horizon_min,
                capacity, display_steps):
    """
    Draws the KPI row + chart into the page (or the live fragment).

    The KPI cards are written straight into the page instead of through an
    `st.empty()` slot: Streamlit replaces an empty's entire subtree on every
    rerun, so wrapping them tore them down and rebuilt them each tick and held
    up the chart's repaint (measured blinks up to 223 ms).

    The chart is still remounted once per tick — Streamlit derives its element
    id from the figure JSON, so any new data is a new element — but the
    fixed-height box reserves its space, so nothing below reflows. Measured
    34-43 ms repaint with zero layout movement.
    """
    render_kpi_row(now_idx, df, preds, lookback, horizon_min, capacity)

    st.markdown(
        '<div class="tfp-section">Traffic flow · actual vs predicted</div>',
        unsafe_allow_html=True,
    )
    fig = build_figure(
        df, preds, target_idx, lookback, horizon_min, now_idx, display_steps, capacity
    )
    # The chart lives in a fixed-height keyed box so its space is reserved, and
    # is written into an st.empty() slot inside it. Streamlit re-identifies the
    # Plotly element every tick (its id is derived from the figure JSON), so
    # React unmounts/remounts it regardless of placeholder. The reserved box
    # keeps that swap from collapsing the layout below it; the empty() is the
    # in-place update slot. Measured ~40ms repaint, no layout movement.
    _chart_box = st.container(key="chart-box", height=510, border=False)
    with _chart_box:
        _chart_slot = st.empty()
        with _chart_slot.container():
            st.plotly_chart(
                fig,
                width="stretch",
                config={
                    "displayModeBar": "hover",
                    "displaylogo": False,
                    "responsive": True,
                },
            )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.markdown('<div class="tfp-section">Controls</div>', unsafe_allow_html=True)

lookback = st.sidebar.segmented_control(
    "Lookback window",
    options=LOOKBACK_OPTIONS,
    default=LOOKBACK_OPTIONS[1],
    selection_mode="single",
    format_func=lambda steps: format_duration(steps * SAMPLING_INTERVAL_MIN),
    width="stretch",
    help="History window used as model input. Models are pre-trained for each "
         "of these stops — switching swaps models rather than retraining.",
) or LOOKBACK_OPTIONS[1]
horizon_min = st.sidebar.segmented_control(
    "Prediction horizon",
    options=HORIZONS_MIN,
    default=HORIZONS_MIN[1],
    selection_mode="single",
    format_func=format_duration,
    width="stretch",
) or HORIZONS_MIN[1]
display_label = st.sidebar.segmented_control(
    "Chart history",
    options=list(DISPLAY_WINDOWS.keys()),
    default="1 day",
    selection_mode="single",
    width="stretch",
) or "1 day"

live_mode = st.sidebar.toggle("Simulate live", value=False)
live_interval = 3
if live_mode:
    live_interval = st.sidebar.slider("Refresh (seconds)", 1, 10, 3)

with st.sidebar.expander("About the lookback"):
    st.markdown(
        f"""
        <div class="tfp-side-note">
        The lookback changes the model <b>input shape</b>, so it cannot be
        retrained on the fly. We train once for the fixed lookbacks
        <code>{LOOKBACK_OPTIONS}</code> ({ " / ".join(format_duration(s * SAMPLING_INTERVAL_MIN) for s in LOOKBACK_OPTIONS) })
        and the selector picks among those pre-trained models.
        </div>
        """,
        unsafe_allow_html=True,
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
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# Header is static content: it sits outside the live fragment so it never
# updates on each tick.
_ts = df["timestamp"].iloc[len(df) - 1]
render_header(horizon_min, capacity, live_mode, live_interval, _ts)

if live_mode:
    live_start = max(lookback - 1, len(df) - 1 - (24 * STEPS_PER_HOUR))  # 1 day of ticks
    if st.session_state.get("_live_lookback") != lookback:
        st.session_state._live_cursor = live_start
        st.session_state._live_lookback = lookback
    if "_live_cursor" not in st.session_state:
        st.session_state._live_cursor = live_start

    @st.fragment(run_every=live_interval)
    def live_fragment():
        cursor = int(st.session_state._live_cursor)
        render_view(
            cursor,
            df, preds, target_idx, lookback, horizon_min, capacity, display_steps,
        )
        nxt = cursor + 1
        st.session_state._live_cursor = nxt if nxt <= len(df) - 1 else live_start

    live_fragment()
else:
    render_view(
        len(df) - 1,
        df, preds, target_idx, lookback, horizon_min, capacity, display_steps,
    )

st.divider()
render_metrics_panel(metrics, lookback, horizon_min, _scale)
