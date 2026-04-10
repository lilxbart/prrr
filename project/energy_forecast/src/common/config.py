"""Общие настройки проекта прогнозирования энергопотребления."""
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", ROOT_DIR / "reports"))
MLRUNS_DIR = Path(os.getenv("MLRUNS_DIR", ROOT_DIR / "mlruns"))

for p in (MODELS_DIR, REPORTS_DIR, MLRUNS_DIR, DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)

TARGET = "ENERGY"
DATE_COL = "DATE"
DEFAULT_BUILDING = "A"

# Поддерживаемые горизонты прогноза (в сутках)
HORIZONS = [1, 7, 30]
HORIZON_LABELS = {1: "день", 7: "неделя", 30: "месяц"}

# Признаки внешних факторов (после агрегации до дневного уровня)
NUMERIC_FEATURES = [
    "HDD18_3", "CDD0", "CDD10", "PRECTOT", "RH2M",
    "T2M", "T2M_MIN", "T2M_MAX", "ALLSKY", "HOLIDAY",
]

# Лаговые признаки (дни назад) для дневных данных
LAG_DAYS = [1, 2, 3, 7, 14, 28, 365]
ROLLING_WINDOWS = [7, 14, 28]

RANDOM_STATE = 42

# Стратегия разбиения: тест = последний год, валидация – 6 месяцев до теста
TEST_YEAR = 2020
VAL_MONTHS = 6
