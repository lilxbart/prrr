"""Общие настройки проекта прогнозирования энергопотребления."""
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", ROOT_DIR / "reports"))
MLRUNS_DIR = Path(os.getenv("MLRUNS_DIR", ROOT_DIR / "mlruns"))

for p in (MODELS_DIR, REPORTS_DIR, MLRUNS_DIR):
    p.mkdir(parents=True, exist_ok=True)

TARGET = "ENERGY"
DATE_COL = "DATE"
DEFAULT_BUILDING = "A"

# Признаки, которые используются моделью
NUMERIC_FEATURES = [
    "HDD18_3", "CDD0", "CDD10", "PRECTOT", "RH2M",
    "T2M", "T2M_MIN", "T2M_MAX", "ALLSKY", "HOLIDAY",
]

# Лаговые признаки (часы назад)
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]  # до недели назад
ROLLING_WINDOWS = [3, 24, 168]

RANDOM_STATE = 42

# Стратегия разбиения (из ЛР2): TimeSeriesSplit, последний год – тест
TEST_YEAR = 2020  # последний полный год используем как hold-out тест
VAL_MONTHS = 6    # полгода до теста – валидация
