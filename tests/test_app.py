"""Smoke tests: saved models load, and the Streamlit app renders without errors."""
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "app.py")
META = json.loads((ROOT / "models" / "metadata.json").read_text())


def test_saved_models_predict_on_a_raw_row():
    reg = joblib.load(ROOT / "models" / "delay_regressor.joblib")
    clf = joblib.load(ROOT / "models" / "delay_classifier.joblib")
    cats = META["categories"]
    row = {
        "transport_type": cats["transport_type"][0], "route_id": cats["route_id"][0],
        "origin_station": cats["origin_station"][0],
        "destination_station": cats["destination_station"][1],
        "scheduled_departure": "08:00:00", "scheduled_arrival": "08:40:00",
        "weather_condition": cats["weather_condition"][0], "temperature_C": 15.0,
        "humidity_percent": 60.0, "wind_speed_kmh": 20.0, "precipitation_mm": 2.0,
        "event_type": "None", "event_attendance_est": 0.0, "traffic_congestion_index": 50.0,
        "holiday": 0, "peak_hour": 1, "weekday": 2, "season": "Winter",
    }
    X = pd.DataFrame([row])[META["raw_input_columns"]]
    assert -60 < reg.predict(X)[0] < 600
    assert 0.0 <= clf.predict_proba(X)[0, 1] <= 1.0


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    return at


def test_app_renders_without_exceptions(app):
    assert not app.exception, app.exception
    assert len(app.tabs) == 4
    labels = [m.label for m in app.metric]
    assert "Predicted arrival delay" in labels
    assert any(label.startswith("Chance of") for label in labels)


@pytest.mark.parametrize("event, weather", [("Concert", "Storm"), ("None", "Snow")])
def test_app_updates_for_other_inputs(app, event, weather):
    app.selectbox(key="event").select(event)
    app.selectbox(key="weather").select(weather)
    app.toggle(key="holiday").set_value(True)
    app.run()
    assert not app.exception, app.exception
