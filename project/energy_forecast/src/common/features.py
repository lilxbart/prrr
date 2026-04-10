"""Feature engineering – календарные, лаговые и скользящие признаки.

Вся логика реализована в одной функции и используется как на этапе
обучения, так и на этапе инференса, чтобы не допустить расхождения
(training-serving skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATE_COL, TARGET, NUMERIC_FEATURES, LAG_HOURS, ROLLING_WINDOWS


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df[DATE_COL]
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["quarter"] = ts.dt.quarter
    df["year"] = ts.dt.year
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # циклические признаки
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    df = df.copy()
    for lag in LAG_HOURS:
        df[f"lag_{lag}"] = df[target].shift(lag)
    for w in ROLLING_WINDOWS:
        df[f"rolling_mean_{w}"] = df[target].shift(1).rolling(window=w).mean()
        df[f"rolling_std_{w}"] = df[target].shift(1).rolling(window=w).std()
    return df


def build_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Полный конвейер feature engineering."""
    df = add_calendar_features(df)
    df = add_lag_features(df)
    if dropna:
        df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLUMNS = (
    [f"lag_{lag}" for lag in LAG_HOURS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
    + NUMERIC_FEATURES
    + [
        "hour", "dayofweek", "day", "month", "quarter",
        "is_weekend", "hour_sin", "hour_cos", "month_sin", "month_cos",
    ]
)


def get_feature_columns() -> list[str]:
    return list(FEATURE_COLUMNS)
