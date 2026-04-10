"""Feature engineering для дневных данных энергопотребления.

Формирует:
  * календарные признаки (день недели, месяц, квартал, год, выходные,
    начало/конец месяца, циклические sin/cos);
  * лаговые значения ENERGY (1, 2, 3, 7, 14, 28, 365 дней назад);
  * скользящие mean/std по окнам 7/14/28 дней;
  * все погодные признаки и HOLIDAY из дневной агрегации;
  * три варианта целевой переменной — суммарное потребление на
    горизонтах 1, 7 и 30 дней вперёд.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DATE_COL, TARGET, NUMERIC_FEATURES, LAG_DAYS, ROLLING_WINDOWS, HORIZONS,
)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df[DATE_COL]
    df["dayofweek"] = ts.dt.dayofweek
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["quarter"] = ts.dt.quarter
    df["year"] = ts.dt.year
    df["weekofyear"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = ts.dt.is_month_start.astype(int)
    df["is_month_end"] = ts.dt.is_month_end.astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    df = df.copy()
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df[target].shift(lag)
    for w in ROLLING_WINDOWS:
        df[f"rolling_mean_{w}"] = df[target].shift(1).rolling(window=w).mean()
        df[f"rolling_std_{w}"] = df[target].shift(1).rolling(window=w).std()
    return df


def add_targets(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Добавить будущие суммарные потребления на горизонтах HORIZONS."""
    df = df.copy()
    for h in HORIZONS:
        # сумма энергии на горизонте h дней вперёд (сдвиг: начиная со следующего дня)
        df[f"target_{h}"] = (
            df[target].shift(-1).rolling(window=h, min_periods=h).sum().shift(-(h - 1))
        )
    return df


def build_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_targets(df)
    if dropna:
        df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLUMNS = (
    [f"lag_{lag}" for lag in LAG_DAYS]
    + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
    + NUMERIC_FEATURES
    + [
        "dayofweek", "day", "month", "quarter", "weekofyear",
        "is_weekend", "is_month_start", "is_month_end",
        "dow_sin", "dow_cos", "month_sin", "month_cos",
    ]
)


def get_feature_columns() -> list[str]:
    return list(FEATURE_COLUMNS)


def target_column(horizon: int) -> str:
    return f"target_{horizon}"
