"""Unit tests for the shared feature engineering and preprocessing."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

from src.features import (LEAKAGE_COLUMNS, RAW_INPUT_COLUMNS, add_features,
                          clock_to_minutes, make_pipeline)


def make_row(**overrides) -> dict:
    """A valid raw trip; override any field per test."""
    row = {
        "transport_type": "Bus", "route_id": "Route_1", "origin_station": "Station_1",
        "destination_station": "Station_2", "scheduled_departure": "08:00:00",
        "scheduled_arrival": "08:40:00", "weather_condition": "Clear",
        "temperature_C": 15.0, "humidity_percent": 60.0, "wind_speed_kmh": 10.0,
        "precipitation_mm": 0.0, "event_type": "None", "event_attendance_est": 0.0,
        "traffic_congestion_index": 50.0, "holiday": 0, "peak_hour": 1, "weekday": 2,
        "season": "Winter",
    }
    row.update(overrides)
    return row


def features_for(**overrides) -> pd.Series:
    return add_features(pd.DataFrame([make_row(**overrides)])).iloc[0]


def test_no_leakage_columns_in_model_inputs():
    assert not set(LEAKAGE_COLUMNS) & set(RAW_INPUT_COLUMNS)


def test_clock_parsing_handles_both_formats():
    minutes = clock_to_minutes(pd.Series(["05:02:00", "7:30", "23:59:59"]))
    assert minutes.tolist() == [302, 450, 1439]


def test_duration_wraps_past_midnight():
    f = features_for(scheduled_departure="23:50:00", scheduled_arrival="00:20:00")
    assert f["sched_duration_min"] == 30


@pytest.mark.parametrize("dep, rush", [("08:15:00", 1), ("12:00:00", 0), ("17:45:00", 1)])
def test_rush_hour_flag(dep, rush):
    assert features_for(scheduled_departure=dep)["is_rush_hour"] == rush


def test_cyclical_hour_encoding_is_on_unit_circle():
    f = features_for(scheduled_departure="18:00:00")
    assert np.isclose(f["hour_sin"] ** 2 + f["hour_cos"] ** 2, 1.0)


def test_weekend_and_day_off():
    assert features_for(weekday=6)["is_weekend"] == 1
    assert features_for(weekday=2, holiday=1)["is_day_off"] == 1
    assert features_for(weekday=2, holiday=0)["is_day_off"] == 0


def test_weather_flags():
    f = features_for(weather_condition="Storm", temperature_C=-5.0, precipitation_mm=15.0,
                     wind_speed_kmh=50.0)
    assert (f["is_rain"], f["is_extreme_temp"], f["is_heavy_precip"], f["is_high_wind"]) == (1, 1, 1, 1)
    assert f["bad_weather_score"] == 4


def test_event_flags_ignore_attendance_without_event():
    no_event = features_for(event_type="None", event_attendance_est=500.0)
    assert no_event["has_event"] == 0
    concert = features_for(event_type="Concert", event_attendance_est=20_000.0)
    assert concert["has_event"] == 1 and concert["is_large_event"] == 1
    assert concert["event_in_rush"] == 1          # 08:00 departure is rush hour


def test_add_features_does_not_mutate_input():
    df = pd.DataFrame([make_row()])
    before = df.copy()
    add_features(df)
    pd.testing.assert_frame_equal(df, before)


def _random_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        h, m = rng.integers(0, 24), rng.choice([0, 15, 30, 45])
        rows.append(make_row(
            transport_type=rng.choice(["Bus", "Tram", "Metro"]),
            route_id=f"Route_{rng.integers(1, 6)}",
            origin_station=f"Station_{rng.integers(1, 10)}",
            destination_station=f"Station_{rng.integers(1, 10)}",
            scheduled_departure=f"{h:02d}:{m:02d}:00",
            scheduled_arrival=f"{(h + 1) % 24:02d}:{m:02d}:00",
            weather_condition=rng.choice(["Clear", "Rain", "Snow"]),
            temperature_C=float(rng.uniform(-5, 35)),
            event_type=rng.choice(["None", "Concert"]),
            weekday=int(rng.integers(0, 7)),
        ))
    return pd.DataFrame(rows)[RAW_INPUT_COLUMNS]


def test_pipeline_fits_and_predicts_end_to_end():
    X = _random_frame()
    y_reg = pd.Series(np.random.default_rng(1).normal(10, 5, len(X)))
    reg = make_pipeline(DummyRegressor(), "regression").fit(X, y_reg)
    assert reg.predict(X).shape == (len(X),)

    y_clf = (y_reg > 10).astype(int)
    clf = make_pipeline(LogisticRegression(max_iter=500), "classification").fit(X, y_clf)
    proba = clf.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_handles_unseen_categories():
    X = _random_frame()
    y = pd.Series(np.random.default_rng(2).normal(10, 5, len(X)))
    pipe = make_pipeline(DummyRegressor(), "regression").fit(X, y)
    unseen = pd.DataFrame([make_row(route_id="Route_999", weather_condition="Hail")])
    assert np.isfinite(pipe.predict(unseen[RAW_INPUT_COLUMNS])).all()
