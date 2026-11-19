"""Transit Delay Predictor: Streamlit front-end for the trained pipelines.

Run locally:   streamlit run app/app.py
Deploy:        Streamlit Community Cloud -> main file path = app/app.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The pickled pipelines reference src.features.add_features, so the module
# must be importable before joblib.load().
from src.features import RUSH_HOURS, SEASON_BY_MONTH  # noqa: E402
from src.collectors.seoul_bus import QuotaExceeded, fetch_route  # noqa: E402
from src.live_bus import (geocode, load_stop_index, nearest_stops,  # noqa: E402
                          search_stops, summarize_stop)

try:  # browser geolocation component (optional: the app still works without it)
    from streamlit_js_eval import get_geolocation
except ImportError:  # pragma: no cover
    get_geolocation = None

KST = timezone(timedelta(hours=9))
APP_DAILY_CALL_CAP = 300   # protects the quota shared with the scheduled data collector

MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPO_URL = "https://github.com/asilbek-maker/transit-delay-prediction"
NOTEBOOK_URL = f"{REPO_URL}/blob/main/notebooks/transit_delay_prediction.ipynb"

BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#6b6a66"

# Readable groups for the per-prediction explanation. One-hot columns are
# matched by prefix and summed, so "weather_condition_Rain" counts as Weather.
FEATURE_GROUPS = {
    "Time of day": ["dep_hour", "hour_sin", "hour_cos", "is_rush_hour", "peak_hour"],
    "Day type": ["weekday", "is_weekend", "holiday", "is_day_off"],
    "Weather": ["temperature_C", "humidity_percent", "wind_speed_kmh", "precipitation_mm",
                "is_rain", "is_snow", "is_extreme_temp", "is_heavy_precip",
                "is_high_wind", "bad_weather_score"],
    "Event": ["log_event_attendance", "has_event", "is_large_event", "event_in_rush"],
    "Traffic": ["traffic_congestion_index", "congestion_x_rush"],
    "Trip length": ["sched_duration_min"],
    "Route": ["route_id"],
    "Origin station": ["origin_station"],
    "Destination station": ["destination_station"],
}
PREFIX_GROUPS = {"weather_condition_": "Weather", "event_type_": "Event",
                 "transport_type_": "Transport mode", "season_": "Day type"}

st.set_page_config(page_title="Transit Delay Predictor", page_icon="🚌",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2.5rem; max-width: 1250px;}
      [data-testid="stMetricValue"] {font-size: 1.85rem; font-weight: 650;}
      [data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #52514e;}
      .hero-sub {color: #52514e; font-size: 1.05rem; margin-top: -0.6rem;}
      .pill {display: inline-block; padding: 2px 10px; margin: 0 6px 6px 0;
             border-radius: 999px; background: #eef3fb; color: #1d4f8f;
             font-size: 0.8rem; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load both pipelines and the metadata once per server process."""
    reg = joblib.load(MODEL_DIR / "delay_regressor.joblib")
    clf = joblib.load(MODEL_DIR / "delay_classifier.joblib")
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    return reg, clf, meta


@st.cache_data
def load_results(name: str) -> pd.DataFrame:
    """Parse a markdown results table written by the notebook."""
    lines = [ln.strip() for ln in (ROOT / "reports" / name).read_text().splitlines()
             if ln.strip().startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines[2:]]
    table = pd.DataFrame(rows, columns=header).set_index("model")
    return table.apply(pd.to_numeric, errors="coerce")


@st.cache_data
def stop_index() -> dict:
    return load_stop_index()


def bus_key() -> str:
    """Key from Streamlit secrets (Cloud) or the environment (local)."""
    try:
        key = st.secrets.get("DATA_GO_KR_KEY", "")
    except Exception:  # noqa: BLE001 - no secrets file configured
        key = ""
    return str(key or os.environ.get("DATA_GO_KR_KEY", "")).strip()


@st.cache_resource
def geocode_clock() -> dict:
    return {"last": 0.0}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_geocode(query: str):
    """Nominatim allows 1 request/second: cache for a day and space out calls."""
    import time as _time
    if os.environ.get("TRANSIT_OFFLINE") == "1":      # tests: never call external services
        return None
    clock = geocode_clock()
    wait = 1.1 - (_time.monotonic() - clock["last"])
    if wait > 0:
        _time.sleep(wait)
    clock["last"] = _time.monotonic()
    return geocode(query)


@st.cache_resource
def call_budget() -> dict:
    return {"day": None, "used": 0}


@st.cache_data(ttl=120, show_spinner=False)
def live_route(route_id: str) -> tuple[list[dict], str]:
    """All stops of one route, cached 2 min and capped per day for the whole app."""
    budget = call_budget()
    today = datetime.now(KST).date().isoformat()
    if budget["day"] != today:
        budget.update(day=today, used=0)
    if budget["used"] >= APP_DAILY_CALL_CAP:
        return [], "cap"
    budget["used"] += 1
    try:
        rows = fetch_route(bus_key(), route_id)
    except QuotaExceeded:
        return [], "quota"
    return rows, datetime.now(KST).strftime("%H:%M:%S")


try:
    reg_model, clf_model, meta = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found in `models/`. Run the notebook first.")
    st.stop()

cats, ranges = meta["categories"], meta["numeric_ranges"]
THRESHOLD_MIN = meta["delay_threshold_min"]
DECISION_THRESHOLD = meta["classification_threshold"]
INPUT_COLS = meta["raw_input_columns"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def feature_group(name: str) -> str:
    for group, cols in FEATURE_GROUPS.items():
        if name in cols:
            return group
    for prefix, group in PREFIX_GROUPS.items():
        if name.startswith(prefix):
            return group
    return "Other"


def build_row(inputs: dict, dep_dt: datetime) -> dict:
    """One raw input row with exactly the columns the pipeline was trained on."""
    arr_dt = dep_dt + timedelta(minutes=int(inputs["duration"]))
    has_event = inputs["event"] != "None"
    return {
        "transport_type": inputs["transport"], "route_id": inputs["route"],
        "origin_station": inputs["origin"], "destination_station": inputs["destination"],
        "scheduled_departure": dep_dt.strftime("%H:%M:%S"),
        "scheduled_arrival": arr_dt.strftime("%H:%M:%S"),
        "weather_condition": inputs["weather"], "temperature_C": inputs["temperature"],
        "humidity_percent": inputs["humidity"], "wind_speed_kmh": inputs["wind"],
        "precipitation_mm": inputs["precip"], "event_type": inputs["event"],
        "event_attendance_est": inputs["attendance"] if has_event else 0.0,
        "traffic_congestion_index": inputs["congestion"], "holiday": int(inputs["holiday"]),
        "peak_hour": int(dep_dt.hour in RUSH_HOURS), "weekday": dep_dt.weekday(),
        "season": SEASON_BY_MONTH[dep_dt.month],
    }


def explain(X: pd.DataFrame) -> pd.DataFrame | None:
    """Exact TreeSHAP contributions from LightGBM, grouped into readable factors."""
    model = clf_model.named_steps["model"]
    if not hasattr(model, "booster_"):          # only LightGBM exposes pred_contrib
        return None
    feats = clf_model.named_steps["features"].transform(X)
    mat = np.asarray(clf_model.named_steps["preprocess"].transform(feats), dtype=float)
    contrib = np.asarray(model.predict(mat, pred_contrib=True))[0, :-1]   # drop bias
    names = clf_model.named_steps["preprocess"].get_feature_names_out()
    grouped = pd.Series(contrib, index=names).groupby(
        [feature_group(n) for n in names]).sum()
    out = pd.DataFrame({"factor": grouped.index, "effect": grouped.values})
    out["direction"] = np.where(out["effect"] > 0, "Raises delay risk", "Lowers delay risk")
    return out.sort_values("effect", key=np.abs, ascending=False)


def hour_sweep(inputs: dict, trip_date: date, minute: int) -> pd.DataFrame:
    """Same trip, every departure hour of the day: a what-if curve."""
    rows = [build_row(inputs, datetime.combine(trip_date, time(h, minute))) for h in range(24)]
    X = pd.DataFrame(rows)[INPUT_COLS]
    return pd.DataFrame({"hour": range(24),
                         "p_delay": clf_model.predict_proba(X)[:, 1],
                         "delay_min": reg_model.predict(X)})


def num_slider(label: str, col: str, step: float = 1.0, key: str | None = None):
    """Slider bounded by the training range (no silent extrapolation)."""
    r = ranges[col]
    return st.slider(label, float(r["min"]), float(r["max"]), float(r["median"]), step, key=key)


# --------------------------------------------------------------------------
# Sidebar: inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔮 Demo model inputs")
    st.caption("These drive the **Prediction (demo model)** tab. Everything here is known "
               "**before departure**: timetable, forecast, events.")
    with st.form("trip_form"):
        st.markdown("##### 🚏 Trip")
        transport = st.selectbox("Transport type", cats["transport_type"], key="transport")
        route = st.selectbox("Route", cats["route_id"], key="route")
        c1, c2 = st.columns(2)
        origin = c1.selectbox("From", cats["origin_station"], key="origin")
        destination = c2.selectbox("To", cats["destination_station"], index=1, key="destination")
        c1, c2 = st.columns(2)
        trip_date = c1.date_input("Date", value=date.today(), key="date")
        dep_time = c2.time_input("Departure", value=time(8, 0),
                                 step=timedelta(minutes=15), key="dep_time")
        dr = meta["duration_range"]
        duration = st.slider("Scheduled duration (min)", dr["min"], dr["max"], dr["median"],
                             key="duration")
        holiday = st.toggle("Public holiday", key="holiday")

        st.markdown("##### 🌦️ Weather forecast")
        weather = st.selectbox("Condition", cats["weather_condition"], key="weather")
        temperature = num_slider("Temperature (°C)", "temperature_C", 0.5, key="temperature")
        precip = num_slider("Precipitation (mm)", "precipitation_mm", 0.1, key="precip")
        c1, c2 = st.columns(2)
        with c1:
            humidity = num_slider("Humidity (%)", "humidity_percent", key="humidity")
        with c2:
            wind = num_slider("Wind (km/h)", "wind_speed_kmh", key="wind")

        st.markdown("##### 🎟️ Events & traffic")
        events = cats["event_type"]
        event = st.selectbox("Nearby event", events, key="event",
                             index=events.index("None") if "None" in events else 0)
        att = ranges["event_attendance_est"]
        attendance = st.slider("Expected attendance (ignored if no event)", 0,
                               int(att["max"]), 5_000, 500, key="attendance")
        congestion = num_slider("Traffic congestion index", "traffic_congestion_index",
                                key="congestion")
        st.form_submit_button("Predict delay", type="primary")

inputs = dict(transport=transport, route=route, origin=origin, destination=destination,
              duration=duration, holiday=holiday, weather=weather, temperature=temperature,
              humidity=humidity, wind=wind, precip=precip, event=event,
              attendance=float(attendance), congestion=congestion)
dep_dt = datetime.combine(trip_date, dep_time)
arr_dt = dep_dt + timedelta(minutes=int(duration))
X = pd.DataFrame([build_row(inputs, dep_dt)])[INPUT_COLS]

delay_min = float(reg_model.predict(X)[0])
p_delayed = float(clf_model.predict_proba(X)[0, 1])
is_late = p_delayed >= DECISION_THRESHOLD
expected_arrival = arr_dt + timedelta(minutes=delay_min)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
head_l, head_r = st.columns([4, 1], vertical_alignment="center")
with head_l:
    st.title("Transit Delay Predictor")
    st.markdown(
        '<p class="hero-sub">Forecasts a trip\'s arrival delay before it leaves, '
        "from timetable, weather and event data.</p>"
        '<span class="pill">Random Forest + LightGBM</span>'
        '<span class="pill">Time-aware CV</span>'
        '<span class="pill">Optuna-tuned</span>'
        '<span class="pill">TreeSHAP explanations</span>',
        unsafe_allow_html=True,
    )
with head_r:
    st.link_button("View code on GitHub", REPO_URL)
    st.link_button("Open the notebook", NOTEBOOK_URL)

tab_live, tab_pred, tab_perf, tab_insights, tab_about = st.tabs(
    ["🚏 Live bus lookup", "🔮 Prediction (demo model)", "📊 Model performance",
     "🔍 Data insights", "ℹ️ About & model card"])

# --------------------------------------------------------------------------
# Tab 0: live bus lookup (real Seoul data)
# --------------------------------------------------------------------------
STATUS_ICON = {"On schedule": "✅", "Delayed": "⚠️", "Bunched": "🔁", "No bus": "⏸️"}

def _use_gps():
    st.session_state["gps_on"] = True
    st.session_state["stop_query"] = ""


def _typed_departure():
    st.session_state["gps_on"] = False


def render_live_stop(stop: dict) -> None:
    """Live cards for every tracked route serving one physical stop."""
    if not bus_key():
        st.warning("Live data isn't configured on this deployment: add the "
                   "`DATA_GO_KR_KEY` secret to enable it.", icon="🔑")
        return
    statuses, updated = [], None
    for r in stop["routes"]:
        rows, stamp = live_route(r["route_id"])
        if stamp in ("cap", "quota"):
            st.warning("Live lookups have hit today's limit. Please try again tomorrow.", icon="⏳")
            return
        updated = stamp
        row = next((x for x in rows if int(x.get("staOrd", -1)) == r["ord"]), None)
        if row is not None:
            statuses.append((r, summarize_stop(row, r["term"])))
    if not statuses:
        st.info("No live data for this stop right now (outside service hours?).")
        return
    late = [s for _, s in statuses if s.status == "Delayed"]
    headline = (f"⚠️ {len(late)} of {len(statuses)} route(s) running late" if late
                else f"✅ All {len(statuses)} route(s) on schedule")
    walk = (f" · {stop['distance_m']} m away ({stop['walk_min']} min walk)"
            if "distance_m" in stop else "")
    st.markdown(f"#### {stop['stNm']} (#{stop['arsId']}){walk}")
    st.markdown(f"**{headline}** · updated {updated} KST · refreshes every 2 minutes")
    for r, s in statuses:
        with st.container(border=True):
            st.markdown(f"**Route {s.route}** → towards {r['next_stop']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Next bus", "Arriving" if s.next_min == 0
                      else (f"{s.next_min:.0f} min" if s.next_min else "—"))
            c2.metric("Following bus", f"{s.second_min:.0f} min" if s.second_min else "—")
            c3.metric("Status", f"{STATUS_ICON.get(s.status, 'ℹ️')} {s.status}")
            c4.metric("Delay vs schedule", f"+{s.delay_min:.0f} min" if s.delay_min else "0 min")
            st.caption(f"Scheduled every {s.term_min} min · {s.crowding}"
                       + (f" · {s.message}" if s.message else ""))
    if st.button("Refresh now", key="refresh_live"):
        live_route.clear()
        st.rerun()


with tab_live:
    st.subheader("Where are you departing from?")
    index = stop_index()
    route_names = ", ".join(r["rtNm"] for r in index["routes"].values())
    st.caption(f"Live data from Seoul (서울특별시 버스도착정보) for trunk routes {route_names}. "
               "A bus is **delayed** when it is further away than its scheduled interval "
               "(+2 min tolerance) and **bunched** when two arrive much closer together than "
               "scheduled.")

    col_txt, col_gps = st.columns([3, 1], vertical_alignment="bottom")
    with col_txt:
        query = st.text_input("Departure: place, address or bus stop",
                              placeholder="e.g. 코엑스, 강남역, 테헤란로 152, 신설동, 23218",
                              key="stop_query", on_change=_typed_departure)
    with col_gps:
        st.button("📍 Use my location", key="use_gps", on_click=_use_gps,
                  disabled=get_geolocation is None,
                  help="Your browser will ask for permission. The location is only used to "
                       "find nearby stops and is not stored.")

    origin, origin_label, candidates = None, None, []
    if st.session_state.get("gps_on") and get_geolocation is not None:
        loc = get_geolocation(component_key="geo")
        coords = loc.get("coords") if isinstance(loc, dict) else None
        if coords and coords.get("latitude") is not None:
            origin = (float(coords["latitude"]), float(coords["longitude"]))
            origin_label = f"your location (±{float(coords.get('accuracy') or 0):.0f} m)"
        elif isinstance(loc, dict) and loc.get("error"):
            st.warning("Location access was blocked by the browser. Type where you're "
                       "departing from instead.", icon="📍")
        else:
            st.caption("📍 Waiting for your browser… please allow location access.")
    elif query:
        candidates = search_stops(index, query)
        if not candidates:
            with st.spinner("Looking up the place…"):
                found = cached_geocode(query.strip())
            if found:
                origin, origin_label = found[:2], found[2]
            else:
                st.info(f"Couldn't find “{query}” in Seoul. Try a station, landmark or address.")

    if origin is not None:
        candidates = nearest_stops(index, origin[0], origin[1], k=6)
        if not candidates:
            st.info(f"No stop on routes {route_names} within 2 km of {origin_label}. "
                    "Live lookups cover these 7 trunk routes only.")
        else:
            st.markdown(f"**Nearest stops to {origin_label}**")
            pts = pd.DataFrame(
                [{"lat": origin[0], "lon": origin[1], "color": ORANGE, "size": 40}]
                + [{"lat": c["lat"], "lon": c["lon"], "color": BLUE, "size": 25}
                   for c in candidates])
            st.map(pts, latitude="lat", longitude="lon", color="color", size="size",
                   zoom=15, height=260)
            st.caption("Orange = departure · blue = nearby stops on tracked routes")

    if candidates:
        def _label(c: dict) -> str:
            dist = (f"  ·  {c['distance_m']} m, {c['walk_min']} min walk"
                    if "distance_m" in c else "")
            return (f"{c['stNm']}  ·  #{c['arsId']}{dist}  ·  routes "
                    f"{', '.join(r['rtNm'] for r in c['routes'])}")
        choice = st.selectbox(f"{len(candidates)} stop(s): pick one", range(len(candidates)),
                              format_func=lambda i: _label(candidates[i]), key="stop_choice")
        render_live_stop(candidates[choice])

# --------------------------------------------------------------------------
# Tab 1: prediction
# --------------------------------------------------------------------------
with tab_pred:
    st.markdown(f"**{transport} · {route}** — {origin} → {destination} · "
                f"{dep_dt:%a %d %b %Y, %H:%M} · scheduled {int(duration)} min")

    k1, k2, k3, k4 = st.columns(4)
    with k1.container(border=True):
        st.metric("Predicted arrival delay", f"{delay_min:.1f} min")
        st.caption("Regression model (Random Forest)")
    with k2.container(border=True):
        st.metric("Expected arrival", f"{expected_arrival:%H:%M}")
        st.caption(f"Scheduled {arr_dt:%H:%M}")
    with k3.container(border=True):
        st.metric(f"Chance of > {THRESHOLD_MIN} min late", f"{p_delayed:.0%}")
        st.progress(min(max(p_delayed, 0.0), 1.0))
    with k4.container(border=True):
        st.metric("Verdict", "⚠️ Likely late" if is_late else "✅ Likely on time")
        st.caption(f"Flagged at ≥ {DECISION_THRESHOLD:.0%} (threshold tuned on CV)")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Why this prediction?")
        contrib = explain(X)
        if contrib is None:
            st.info("Per-prediction explanations are available for LightGBM models only.")
        else:
            bars = alt.Chart(contrib).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("effect:Q", title="← lowers risk   |   raises risk → (log-odds)"),
                y=alt.Y("factor:N", sort=alt.EncodingSortField("effect", op="sum",
                                                               order="descending"), title=None),
                color=alt.Color("direction:N", title=None,
                                scale=alt.Scale(domain=["Raises delay risk", "Lowers delay risk"],
                                                range=[ORANGE, BLUE]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=[alt.Tooltip("factor:N", title="Factor"),
                         alt.Tooltip("effect:Q", title="Effect (log-odds)", format="+.3f")],
            )
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=MUTED).encode(x="x:Q")
            st.altair_chart((bars + zero).properties(height=320))
            st.caption("Exact TreeSHAP values from the LightGBM classifier, summed per factor.")

    with right:
        st.subheader("Delay risk through the day")
        sweep = hour_sweep(inputs, trip_date, dep_time.minute)
        base = alt.Chart(sweep).encode(
            x=alt.X("hour:Q", title="Departure hour", scale=alt.Scale(domain=[0, 23]),
                    axis=alt.Axis(tickCount=12)),
            y=alt.Y("p_delay:Q", title=f"P(late > {THRESHOLD_MIN} min)",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("hour:Q", title="Hour"),
                     alt.Tooltip("p_delay:Q", title="P(late)", format=".0%"),
                     alt.Tooltip("delay_min:Q", title="Predicted delay (min)", format=".1f")],
        )
        line = base.mark_line(color=BLUE, strokeWidth=2) + base.mark_point(
            color=BLUE, filled=True, size=40)
        threshold = alt.Chart(pd.DataFrame({"y": [DECISION_THRESHOLD]})).mark_rule(
            color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
        current = alt.Chart(sweep[sweep["hour"] == dep_dt.hour]).mark_point(
            color=ORANGE, filled=True, size=160).encode(x="hour:Q", y="p_delay:Q")
        st.altair_chart((line + threshold + current).properties(height=320))
        st.caption("Same trip at every hour. Orange = your departure; dashed = alert threshold.")

    with st.expander("Model input (exact row sent to the pipeline)"):
        st.dataframe(X.T.rename(columns={0: "value"}).astype(str))

# --------------------------------------------------------------------------
# Tab 2: model performance (honest)
# --------------------------------------------------------------------------
with tab_perf:
    reg_res = load_results("results_regression.md")
    clf_res = load_results("results_classification.md")
    perm = meta["permutation_test"]
    reg_m, clf_m = meta["metrics"]["regression"], meta["metrics"]["classification"]

    st.warning(
        "**Key finding: this dataset contains no learnable signal.** The tuned models do not "
        "beat a mean predictor on held-out data, and a permutation test cannot distinguish "
        "them from models trained on shuffled labels. The dataset is synthetic; the pipeline "
        "is built so that real data can be swapped in.",
        icon="🔬",
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1.container(border=True):
        st.metric("Test MAE: best model", f"{reg_m['test_MAE']:.2f} min")
        st.caption(f"Mean predictor: {reg_res.loc['Baseline', 'test_MAE']:.2f} min")
    with m2.container(border=True):
        st.metric("Test ROC-AUC: best model", f"{clf_m['test_ROC_AUC']:.3f}")
        st.caption("Random guessing: 0.500")
    with m3.container(border=True):
        st.metric("Permutation test p (regression)", f"{perm['regression']['p_value']:.2f}")
        st.caption("p ≥ 0.05 → indistinguishable from chance")
    with m4.container(border=True):
        st.metric("Permutation test p (classification)",
                  f"{perm['classification']['p_value']:.2f}")
        st.caption("30 shuffled-label refits")

    fmt = {c: st.column_config.NumberColumn(format="%.3f") for c in
           set(reg_res.columns) | set(clf_res.columns)}
    t1, t2 = st.columns(2, gap="large")
    with t1:
        st.subheader("Regression: delay minutes")
        st.dataframe(reg_res[["cv_MAE", "cv_MAE_std", "test_MAE", "test_RMSE", "test_R2"]],
                     column_config=fmt)
    with t2:
        st.subheader("Classification: late > 5 min")
        st.dataframe(clf_res[["cv_ROC_AUC", "cv_ROC_AUC_std", "test_ROC_AUC", "test_PR_AUC",
                              "test_F1"]], column_config=fmt)
    st.caption("Sorted by cross-validation score. Model selection never used the test set.")
    st.image(str(FIG_DIR / "08_model_comparison.png"))

    with st.expander("Why is an F1 of 0.87 meaningless here?"):
        st.markdown(
            "About 77% of test trips are more than 5 minutes late, so flagging **every** trip "
            "as late already scores F1 ≈ 0.87. That is why ROC-AUC (ranking quality, "
            "threshold-free) is the primary metric, and it sits at chance level."
        )

# --------------------------------------------------------------------------
# Tab 3: data insights
# --------------------------------------------------------------------------
with tab_insights:
    insights = [
        ("01_delay_distribution.png", "Delays are spread almost uniformly",
         "Arrival delays run from −3 to +29 min with a flat shape, and departure delay is "
         "unrelated to arrival delay (r = −0.01). Both are fingerprints of synthetic data."),
        ("03_delay_by_hour.png", "No rush-hour effect",
         "Every hourly mean sits inside the others' 95% confidence intervals."),
        ("04_delay_by_weekday_weather_event.png", "Weather and events don't move delays",
         "Rain, snow, storms and clear skies average 13–14 min alike; so do all event types."),
        ("07_correlation_heatmap.png", "Every correlation is ≈ 0",
         "All |ρ| ≤ 0.05, between features and with the target: columns look independent."),
        ("10_classification_shap_summary.png", "SHAP explains the model, not the world",
         "Stations and routes top the SHAP ranking only because high-cardinality features "
         "give trees the most room to fit noise."),
        ("11_regression_permutation_importance.png", "Held-out importance confirms it",
         "Shuffling the top input raises test MAE by just 0.03 min (0.4%)."),
    ]
    for i in range(0, len(insights), 2):
        cols = st.columns(2, gap="large")
        for col, (fname, title, text) in zip(cols, insights[i:i + 2]):
            with col.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(text)
                st.image(str(FIG_DIR / fname))

# --------------------------------------------------------------------------
# Tab 4: about & model card
# --------------------------------------------------------------------------
with tab_about:
    a1, a2 = st.columns([3, 2], gap="large")
    with a1:
        st.subheader("How it works")
        st.markdown(
            """
1. **Problem framing:** two tasks. Regression for delay minutes (MAE) and
   classification for late > 5 min (ROC-AUC, PR-AUC).
2. **Leakage control:** only pre-departure inputs. Departure delay and the
   provided `delayed` flag are excluded and asserted in code.
3. **Pipeline:** `add_features` → `ColumnTransformer` (median imputation, scaling,
   one-hot, cross-fitted `TargetEncoder` for routes/stations) → model, so every
   statistic is learned on training folds only.
4. **Validation:** chronological 80/20 hold-out + 5-fold `TimeSeriesSplit`;
   selection by CV score only.
5. **Models:** mean / logistic baselines → Random Forest, XGBoost, LightGBM, each
   tuned with Optuna (30 TPE trials).
6. **Signal check:** a 30-shuffle permutation test, plus SHAP and held-out
   permutation importance.
            """
        )
        st.subheader("Tech stack")
        st.markdown("Python · pandas · scikit-learn · LightGBM · XGBoost · Optuna · SHAP · "
                    "Altair · Streamlit · GitHub Actions CI")
    with a2:
        st.subheader("Model card")
        with st.container(border=True):
            st.markdown(
                f"""
- **Regression model:** {meta['models']['regression']}
- **Classification model:** {meta['models']['classification']}
- **Training data:** {meta['train_period'][0][:10]} → {meta['train_period'][1][:10]},
  2,000 synthetic trips (Kaggle)
- **Decision threshold:** {DECISION_THRESHOLD:.3f} (max F1 on out-of-fold predictions)
- **Random seed:** {meta['seed']}
- **Intended use:** portfolio demonstration of a leakage-safe delay-forecasting pipeline
- **Not for:** real operational decisions. Predictions reflect synthetic data.
                """
            )

st.divider()
st.caption(f"Built by asilbek-maker · [Source code]({REPO_URL}) · "
           "Data: Kaggle *Public Transport Delays with Weather & Events* (synthetic)")
