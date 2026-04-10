"""Загрузка, очистка и агрегация данных энергопотребления.

Основной рабочий уровень — посуточные агрегаты: сумма ENERGY за сутки,
средние значения погодных признаков, максимум признака выходного дня.
Почасовой уровень признан бесполезным для прогнозирования на горизонты
«день / неделя / месяц» и в проекте больше не используется.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

from .config import DATA_DIR, DATE_COL, TARGET


def load_raw(building: str = "A", data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    path = data_dir / f"db_building_{building}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Не найден датасет: {path}")
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%m/%d/%Y %H:%M")
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def clean_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Почасовая очистка: удаление дубликатов и интерполяция пропусков."""
    df = df.copy()
    df = df.drop_duplicates(subset=[DATE_COL])
    df = df.set_index(DATE_COL)
    df = df.interpolate(method="time", limit_direction="both")
    df = df.reset_index()
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def aggregate_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Свести почасовые данные к посуточным агрегатам.

    * ENERGY — суммарное потребление за сутки (kWh/день)
    * температуры — среднее/мин/макс
    * осадки — сумма
    * градусо-дни, влажность, облучённость — среднее
    * HOLIDAY — максимум (признак «был ли праздник в сутках»)
    """
    df = df_hourly.copy().set_index(DATE_COL)
    agg = pd.DataFrame({
        TARGET:   df[TARGET].resample("D").sum(),
        "T2M":    df["T2M"].resample("D").mean(),
        "T2M_MIN": df["T2M_MIN"].resample("D").min(),
        "T2M_MAX": df["T2M_MAX"].resample("D").max(),
        "HDD18_3": df["HDD18_3"].resample("D").mean(),
        "CDD0":    df["CDD0"].resample("D").mean(),
        "CDD10":   df["CDD10"].resample("D").mean(),
        "PRECTOT": df["PRECTOT"].resample("D").sum(),
        "RH2M":    df["RH2M"].resample("D").mean(),
        "ALLSKY":  df["ALLSKY"].resample("D").mean(),
        "HOLIDAY": df["HOLIDAY"].resample("D").max(),
    }).reset_index()
    # Если в сутках < 20 наблюдений — отбрасываем (неполный день)
    counts = df[TARGET].resample("D").count().values
    agg = agg[counts >= 20].reset_index(drop=True)
    return agg


def load_building_daily(building: str = "A", data_dir: Path | None = None) -> pd.DataFrame:
    """Главная точка входа: вернуть чистые посуточные данные по зданию."""
    raw = load_raw(building=building, data_dir=data_dir)
    cleaned = clean_hourly(raw)
    daily = aggregate_daily(cleaned)
    return daily
