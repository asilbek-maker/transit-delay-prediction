"""Feature engineering and preprocessing for transit delay prediction.

Design rule: every feature here must be computable *before the trip departs*
(timetable, weather forecast, scheduled events). The same functions are used
inside the training Pipeline and by the Streamlit app, so training-time and
serving-time features can never drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)

# --------------------------------------------------------------------------
# Business / domain constants (documented here instead of magic numbers)
# --------------------------------------------------------------------------
DELAY_THRESHOLD_MIN = 5            # "delayed" = arrives more than 5 min late
RUSH_HOURS = set(range(7, 10)) | set(range(16, 20))   # 07-09h and 16-19h
WET_CONDITIONS = {"Rain", "Storm", "Drizzle", "Thunderstorm"}
SNOW_CONDITIONS = {"Snow", "Sleet", "Blizzard"}
EXTREME_COLD_C = 0.0
EXTREME_HEAT_C = 30.0
HEAVY_PRECIP_MM = 10.0
HIGH_WIND_KMH = 40.0
LARGE_EVENT_ATTENDANCE = 10_000
SEASON_BY_MONTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

# --------------------------------------------------------------------------
# Column contracts
# --------------------------------------------------------------------------
TARGET_REG = "actual_arrival_delay_min"
TARGET_CLF = "is_delayed"

# Known only during/after the trip -> must never be used as a feature.
LEAKAGE_COLUMNS = ["actual_departure_delay_min", "delayed", TARGET_REG]

# Raw columns the model pipeline expects as input (pre-departure info only).
RAW_INPUT_COLUMNS = [
    "transport_type", "route_id", "origin_station", "destination_station",
    "scheduled_departure", "scheduled_arrival",
    "weather_condition", "temperature_C", "humidity_percent",
    "wind_speed_kmh", "precipitation_mm",
    "event_type", "event_attendance_est",
    "traffic_congestion_index", "holiday", "peak_hour", "weekday", "season",
]

NUMERIC_FEATURES = [
    "temperature_C", "humidity_percent", "wind_speed_kmh", "precipitation_mm",
    "log_event_attendance", "traffic_congestion_index", "sched_duration_min",
    "dep_hour", "hour_sin", "hour_cos", "weekday", "bad_weather_score",
    "congestion_x_rush",
]
BINARY_FEATURES = [
    "is_rush_hour", "peak_hour", "is_weekend", "holiday", "is_day_off",
    "is_rain", "is_snow", "is_extreme_temp", "is_heavy_precip",
    "is_high_wind", "has_event", "is_large_event", "event_in_rush",
]
LOW_CARD_CATEGORICAL = [
    "transport_type", "weather_condition", "event_type", "season",
]
HIGH_CARD_CATEGORICAL = ["route_id", "origin_station", "destination_station"]


def clock_to_minutes(values: pd.Series) -> pd.Series:
    """Convert 'HH:MM' or 'HH:MM:SS' strings to minutes after midnight."""
    parts = values.astype(str).str.strip().str.split(":", expand=True)
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else 0
    return hours * 60 + minutes


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered, leakage-free features added."""
    out = df.copy()

    # ---- Time features ----------------------------------------------------
    dep_min = clock_to_minutes(out["scheduled_departure"])
    arr_min = clock_to_minutes(out["scheduled_arrival"])
    out["dep_hour"] = (dep_min // 60).astype(float)
    # Cyclical encoding: 23:00 and 00:00 are neighbours, not 23 units apart.
    out["hour_sin"] = np.sin(2 * np.pi * dep_min / 1440)
    out["hour_cos"] = np.cos(2 * np.pi * dep_min / 1440)
    # Modulo handles trips that cross midnight.
    out["sched_duration_min"] = (arr_min - dep_min) % 1440
    out["is_rush_hour"] = out["dep_hour"].isin(RUSH_HOURS).astype(int)
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)          # Mon=0
    out["is_day_off"] = ((out["is_weekend"] == 1) | (out["holiday"] == 1)).astype(int)

    # ---- Weather flags ----------------------------------------------------
    weather = out["weather_condition"].fillna("Unknown")
    out["is_rain"] = weather.isin(WET_CONDITIONS).astype(int)
    out["is_snow"] = weather.isin(SNOW_CONDITIONS).astype(int)
    out["is_extreme_temp"] = (
        (out["temperature_C"] < EXTREME_COLD_C) | (out["temperature_C"] > EXTREME_HEAT_C)
    ).astype(int)
    out["is_heavy_precip"] = (out["precipitation_mm"] > HEAVY_PRECIP_MM).astype(int)
    out["is_high_wind"] = (out["wind_speed_kmh"] > HIGH_WIND_KMH).astype(int)
    out["bad_weather_score"] = out[
        ["is_rain", "is_snow", "is_extreme_temp", "is_heavy_precip", "is_high_wind"]
    ].sum(axis=1)

    # ---- Event proximity --------------------------------------------------
    # The dataset links an event to each trip record (no coordinates), so
    # "proximity" = an event affecting this trip, scaled by its size.
    out["has_event"] = (out["event_type"].fillna("None") != "None").astype(int)
    attendance = out["event_attendance_est"].clip(lower=0)
    out["log_event_attendance"] = np.log1p(attendance)   # tames the heavy tail
    out["is_large_event"] = (attendance >= LARGE_EVENT_ATTENDANCE).astype(int)
    out["event_in_rush"] = out["has_event"] * out["is_rush_hour"]

    # ---- Interactions -----------------------------------------------------
    out["congestion_x_rush"] = out["traffic_congestion_index"] * out["is_rush_hour"]
    return out


def build_preprocessor(task: str, seed: int = 42) -> ColumnTransformer:
    """ColumnTransformer: impute + scale numerics, encode categoricals.

    All statistics (medians, encodings, scales) are learned inside the
    Pipeline from the training fold only, which prevents test-set leakage.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),   # needed for linear models; harmless for trees
    ])
    low_card = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # Routes/stations have many levels: one-hot would create ~100 sparse
    # columns for 2k rows. TargetEncoder cross-fits internally, so a row's
    # own target never leaks into its encoding.
    high_card = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("target_enc", TargetEncoder(
            target_type="continuous" if task == "regression" else "binary",
            random_state=seed,
        )),
    ])
    binary = SimpleImputer(strategy="most_frequent")
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("bin", binary, BINARY_FEATURES),
            ("cat", low_card, LOW_CARD_CATEGORICAL),
            ("hicard", high_card, HIGH_CARD_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(model, task: str, seed: int = 42) -> Pipeline:
    """Full raw-input -> prediction pipeline (features + preprocessing + model)."""
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("preprocess", build_preprocessor(task, seed)),
        ("model", model),
    ])
