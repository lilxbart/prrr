"""Загрузка и базовая очистка данных энергопотребления."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

from .config import DATA_DIR, DATE_COL, TARGET


def load_raw(building: str = "A", data_dir: Path | None = None) -> pd.DataFrame:
    """Считать исходный CSV, привести временную метку к datetime."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    path = data_dir / f"db_building_{building}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Не найден датасет: {path}")
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%m/%d/%Y %H:%M")
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Простое восстановление пропусков.

    В исходных данных встречаются пропуски ENERGY и некоторых погодных
    столбцов. Для временного ряда разумно заполнить их интерполяцией
    по времени (forward/back fill + линейная).
    """
    df = df.copy()
    df = df.drop_duplicates(subset=[DATE_COL])
    df = df.set_index(DATE_COL)
    df = df.interpolate(method="time", limit_direction="both")
    df = df.reset_index()
    # если после интерполяции остались NaN – дропаем
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def load_building(building: str = "A", data_dir: Path | None = None) -> pd.DataFrame:
    return clean(load_raw(building=building, data_dir=data_dir))
