"""Стратегия валидации: time-based split и TimeSeriesSplit (из ЛР2)."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .config import DATE_COL, TEST_YEAR, VAL_MONTHS


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def time_based_split(df: pd.DataFrame) -> SplitResult:
    """Разбить по времени: test = последний год, val = полгода перед ним."""
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    test_start = pd.Timestamp(year=TEST_YEAR, month=1, day=1)
    val_start = test_start - pd.DateOffset(months=VAL_MONTHS)

    train = df[df[DATE_COL] < val_start].copy()
    val = df[(df[DATE_COL] >= val_start) & (df[DATE_COL] < test_start)].copy()
    test = df[df[DATE_COL] >= test_start].copy()
    return SplitResult(train=train, val=val, test=test)
