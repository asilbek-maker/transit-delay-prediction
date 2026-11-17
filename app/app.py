"""Streamlit app: predict a transit trip's arrival delay before it departs.

Run locally:   streamlit run app/app.py
Deploy:        Streamlit Community Cloud → main file path = app/app.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The pickled pipelines reference src.features.add_features, so the module
# must be importable before joblib.load().
from src.features import RUSH_HOURS, SEASON_BY_MONTH  # noqa: E402

MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="Transit Delay Predictor", page_icon="🚌", layout="wide")


@st.cache_resource
def load_artifacts():
    """Load both pipelines and the metadata once per server process."""
    reg = joblib.load(MODEL_DIR / "delay_regressor.joblib")
    clf = joblib.load(MODEL_DIR / "delay_classifier.joblib")
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    return reg, clf, meta


try:
    reg_model, clf_model, meta = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found in `models/`. Run the notebook first and copy "
             "`delay_regressor.joblib`, `delay_classifier.joblib` and `metadata.json` there.")
    st.stop()

cats, ranges = meta["categories"], meta["numeric_ranges"]
threshold_min = meta["delay_threshold_min"]
decision_threshold = meta["classification_threshold"]


def num_slider(label: str, col: str, step: float = 1.0):
    """Slider bounded by the range seen in training (no silent extrapolation)."""
    r = ranges[col]
    return st.slider(label, float(r["min"]), float(r["max"]), float(r["median"]), step)


st.title("🚌 Transit Delay Predictor")
st.caption(f"Predicts arrival delay and the chance of arriving more than "
           f"{threshold_min} min late, using only information available before departure.")

col_trip, col_weather, col_event = st.columns(3)
with col_trip:
    st.subheader("Trip")
    transport = st.selectbox("Transport type", cats["transport_type"])
    route = st.selectbox("Route", cats["route_id"])
    origin = st.selectbox("Origin station", cats["origin_station"])
    destination = st.selectbox("Destination station", cats["destination_station"], index=1)
    trip_date = st.date_input("Date", value=date.today())
    dep_time = st.time_input("Scheduled departure", value=time(8, 0), step=timedelta(minutes=15))
    dr = meta["duration_range"]
    duration = st.slider("Scheduled trip duration (min)", dr["min"], dr["max"], dr["median"])
    holiday = st.checkbox("Public holiday")

with col_weather:
    st.subheader("Weather")
    weather = st.selectbox("Condition", cats["weather_condition"])
    temperature = num_slider("Temperature (°C)", "temperature_C", 0.5)
    humidity = num_slider("Humidity (%)", "humidity_percent")
    wind = num_slider("Wind speed (km/h)", "wind_speed_kmh")
    precip = num_slider("Precipitation (mm)", "precipitation_mm", 0.1)

with col_event:
    st.subheader("Events & traffic")
    event_options = cats["event_type"]
    event = st.selectbox("Nearby event", event_options,
                         index=event_options.index("None") if "None" in event_options else 0)
    attendance = 0.0
    if event != "None":
        attendance = num_slider("Expected attendance", "event_attendance_est", 100.0)
    congestion = num_slider("Traffic congestion index", "traffic_congestion_index")

# Build exactly one raw row with the same columns the pipeline was trained on.
dep_dt = datetime.combine(trip_date, dep_time)
arr_dt = dep_dt + timedelta(minutes=int(duration))
row = {
    "transport_type": transport, "route_id": route,
    "origin_station": origin, "destination_station": destination,
    "scheduled_departure": dep_dt.strftime("%H:%M:%S"),
    "scheduled_arrival": arr_dt.strftime("%H:%M:%S"),
    "weather_condition": weather, "temperature_C": temperature,
    "humidity_percent": humidity, "wind_speed_kmh": wind, "precipitation_mm": precip,
    "event_type": event, "event_attendance_est": attendance,
    "traffic_congestion_index": congestion, "holiday": int(holiday),
    "peak_hour": int(dep_time.hour in RUSH_HOURS),
    "weekday": trip_date.weekday(), "season": SEASON_BY_MONTH[trip_date.month],
}
X = pd.DataFrame([row])[meta["raw_input_columns"]]

st.divider()
delay_min = float(reg_model.predict(X)[0])
p_delayed = float(clf_model.predict_proba(X)[0, 1])
is_late = p_delayed >= decision_threshold

m1, m2, m3 = st.columns(3)
m1.metric("Predicted arrival delay", f"{delay_min:.1f} min")
m2.metric(f"P(delay > {threshold_min} min)", f"{p_delayed:.0%}")
m3.metric("Verdict", "⚠️ Likely delayed" if is_late else "✅ Likely on time",
          help=f"Flagged when probability ≥ {decision_threshold:.2f} "
               "(threshold tuned for F1 on training-period CV).")

with st.expander("Model details & limitations"):
    reg_m, clf_m = meta["metrics"]["regression"], meta["metrics"]["classification"]
    st.markdown(
        f"- **Regression model:** {meta['models']['regression']} — test MAE "
        f"{reg_m['test_MAE']:.2f} min\n"
        f"- **Classification model:** {meta['models']['classification']} — test ROC-AUC "
        f"{clf_m['test_ROC_AUC']:.3f}\n"
        f"- Trained on {meta['train_period'][0][:10]} → {meta['train_period'][1][:10]}.\n"
        "- **Trained on a synthetic Kaggle dataset.** Predictions demonstrate the pipeline, "
        "not real-world transit behaviour."
    )
    st.dataframe(X.T.rename(columns={0: "model input"}))
