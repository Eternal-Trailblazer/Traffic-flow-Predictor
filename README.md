# 🚦 Traffic Flow Predictor

A traffic-flow forecasting project: synthetic 5-minute traffic counts, a windowed
XGBoost model, a congestion classifier, and a Streamlit dashboard that forecasts
vehicle counts 15 / 30 / 60 minutes ahead.

The pipeline is deliberately simple and honest — every design decision and its
trade-offs are documented below.

---

## Project structure

```
traffic-flow-predictor/
├── app.py                     # Phase 5: Streamlit dashboard
├── src/
│   ├── config.py              # Shared constants (horizons, lookbacks, thresholds)
│   ├── mock_stream.py         # Phase 1: synthetic data generator + stream generator
│   ├── data_pipeline.py       # Phase 2: time features + windowing / feature building
│   ├── model.py               # Phases 3–4: train, evaluate, persist models
│   └── congestion.py          # Congestion level + colour mapping
├── tests/
│   ├── test_windowing.py          # Windowing / leakage tests
│   ├── test_congestion.py         # Congestion boundary tests
│   └── test_inference_features.py # Training/inference feature parity
├── .streamlit/config.toml     # Dashboard colour theme (dark, sky-blue accent)
├── models/                    # Trained artifacts (generated)
├── data/raw/traffic.csv       # Generated dataset
├── pytest.ini                 # pytest config (puts repo root on sys.path)
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 1. Generate the data

```bash
python -m src.mock_stream
```

Creates `data/raw/traffic.csv`: 90 days of 5-minute counts with weekday
double-peaks, a flatter weekend curve, injected accidents/events and ~0.5%
missing sensor values.

## 2. Train the models

```bash
python -m src.model
```

Trains one XGBoost model per **(lookback, horizon)** combination among the fixed
lookbacks `[6, 12, 24]` (30 min / 1 h / 2 h) and horizons `[15, 30, 60]` minutes.
Artifacts:

* `models/xgb_lb{lookback}_h{horizon}.joblib` — one model per pair
* `models/scaler.joblib` — `StandardScaler` fitted on the training split only
* `models/metrics.json` — MAE / RMSE / R² for XGBoost and the persistence baseline

## 3. Run the dashboard

```bash
streamlit run app.py
```

---

## Dashboard features (Phase 5)

* **Sidebar controls**
  * **Lookback window** — sliding history fed to the model (see design note below).
  * **Prediction horizon** — 15 / 30 / 60 minutes.
  * **Chart history** — 12 h / 1 day / 2 days of context.
  * **Simulate live** — streams the mock data forward one 5-minute step per tick.
  * **Live refresh** — seconds between ticks (1–10 s).
* **Main Plotly chart** — actual counts (sky blue), model predictions aligned to
  their target timestamps (dashed violet), and the forecast point (rose
  diamond). The **forecast region is shaded and separated** from history by a
  dotted "now" line and a green capacity reference line.
* **KPI cards** — congestion state (with a capacity-utilisation bar), the
  forecast value, the current count and observed capacity.
* **Congestion indicator** — colour-coded (🟢 Low / 🟡 Medium / 🔴 High)
  computed from the model's forecast relative to the road's observed capacity.
* **Metrics panel** — MAE / RMSE / R² for the selected horizon (XGBoost), each
  with a green/red delta against the persistence baseline, plus a
  persistence-baseline comparison table.

### Colour theme

`.streamlit/config.toml` defines a dark palette (near-black background, sky-blue
accent) and the dashboard's custom CSS + Plotly layout pick it up automatically.
The chart and card colours adapt to the active Streamlit theme via
`st.context.theme.type`, so switching the theme in the header menu keeps charts
and cards consistent.

### Caching

| Concern                | Mechanism            |
| ---------------------- | -------------------- |
| Traffic data           | `st.cache_data`      |
| Metrics JSON           | `st.cache_data`      |
| Series predictions     | `st.cache_data`      |
| Model + scaler loading | `st.cache_resource`  |

### Live mode

Live mode uses Streamlit's built-in `st.fragment(run_every=…)` as the refresh
loop, advancing the mock stream one step per tick. No extra dependency
(`streamlit-autorefresh`) is required. When the cursor reaches the end of the
data it wraps back to one day before the end.

Ticks render without visible flicker:

* The header and KPI cards are written **directly** rather than into
  `st.empty()` placeholders. An empty's subtree is replaced wholesale each
  rerun, which destroyed and rebuilt those blocks every tick and delayed the
  chart's repaint.
* The chart sits in a **fixed-height box** (`st.container(height=…)`), so its
  space stays reserved while Streamlit repaints it — nothing below reflows.
* A short CSS **fade-in** softens the chart's swap.

Streamlit derives a chart's element id from the figure JSON, so the chart is
remounted once per tick by design. Measured on the live loop: ~34–43 ms repaint
per tick and zero layout movement (was up to 223 ms plus layout jumps).

---

## Lookback design (important, honest trade-off)

The lookback window changes the **model input**, so a model trained with one
lookback cannot be used with another. Retraining every time the selection changes
would make the dashboard slow and non-reproducible. The simplest honest design —
used here — is:

> **Train models once for a few fixed lookbacks (6, 12, 24 steps) and let the
> selector pick among the pre-trained models.** The selector does not retrain.

This keeps the app instant while remaining honest about what is happening. If a
different lookback is needed, add it to `LOOKBACK_OPTIONS` in `src/config.py`
and re-run `python -m src.model`.

---

## Congestion levels

`src/congestion.py` maps a predicted count to a label + colour using the road's
observed **95th-percentile capacity** (`capacity_95th`):

| Ratio of predicted / capacity | Level  | Colour |
| ----------------------------- | ------ | ------ |
| `< 0.40`                      | Low    | green  |
| `0.40 – 0.70`                 | Medium | amber  |
| `≥ 0.70`                      | High   | red    |

Thresholds live in `CONGESTION_THRESHOLDS` in `src/config.py` and are intended to
be tuned. A non-positive capacity falls back safely to **Low**.

---

## Metrics

Metrics are always evaluated on the chronological **test split** (last 15% of the
data) — never on training data. On the dashboard, MAE/RMSE are shown in vehicle
counts by multiplying the standardized error by the scaler's standard deviation;
R² is scale-independent.

Example (lookback 12, horizon 30 min): MAE ≈ 15.3 veh, RMSE ≈ 37.7 veh, R² ≈ 0.89.

---

## Tests

```bash
python -m pytest -q
```

Covers windowing/leakage, congestion boundary values, and training/inference
feature parity (so the dashboard builds features exactly like training does).
