# Energy Forecast Service

ML-сервис прогнозирования почасового энергопотребления здания.
Реализован в рамках лабораторных работ 3 и 4 по дисциплине
«Проектирование интеллектуальных информационных систем», МТУСИ,
гр. БВТ2201.

Задача (см. отчёты ЛР1–ЛР2): по историческим почасовым наблюдениям
энергопотребления и погодным признакам (NASA POWER) предсказать
значение ENERGY (kWh) на ближайший горизонт.

## Что внутри

```
energy_forecast/
├── data/                       датасеты Building A и Building B
├── src/
│   ├── common/                 конфиг, загрузка, feature engineering, модели, метрики, tracker
│   ├── training/               train.py (эксперименты) и analysis.py (разбор ошибок)
│   ├── inference/              HTTP inference-сервис
│   └── gateway/                API Gateway + HTML UI + кэш + /metrics
├── models/                     финальный model.pkl
├── reports/                    experiments.json, analysis_summary.txt, plots/
├── tests/                      интеграционные тесты (27 проверок)
├── docker/                     Dockerfile.inference, Dockerfile.gateway, Dockerfile.training
├── docker-compose.yml
└── requirements.txt
```

## Быстрый старт (локально, без Docker)

Минимальные зависимости: `python 3.10+`, `numpy`, `pandas`, `matplotlib`.

```bash
# 1. Обучение + эксперименты + сохранение model.pkl
python -m src.training.train
# 2. Разбор ошибок финальной модели и построение графиков
python -m src.training.analysis
# 3. Запуск inference-сервиса (порт 8001)
python -m src.inference.app
# 4. В отдельном терминале – API Gateway + UI (порт 8000)
python -m src.gateway.app
# 5. Интеграционные тесты
python -m tests.test_integration
```

После п. 4 открыть в браузере `http://localhost:8000/` – HTML-форма
с выбором здания, момента времени и кнопкой «Получить прогноз».

## Запуск одной командой (Docker)

```bash
# собрать образы и запустить inference+gateway
docker compose up --build

# (опционально) обучить модель внутри контейнера
docker compose run --rm --profile training training
```

Сервис становится доступен по адресу `http://localhost:8000/`.
Inference-сервис – внутри сети, напрямую он доступен на `:8001`.

## API

`POST /predict` (Gateway)

```json
{
  "building": "A",
  "timestamp": "2020-06-15T10:00:00",
  "horizon": 1
}
```

ответ:

```json
{
  "building": "A",
  "timestamp": "2020-06-15T10:00:00",
  "horizon": 1,
  "prediction": 151.233,
  "actual_energy": 143.353,
  "abs_error": 7.88,
  "model": "gbr_tuned",
  "cache_hit": false,
  "latency_ms": 209.41
}
```

`GET /health` – проверка состояния, `GET /metrics` – Prometheus-формат
(requests_total, error counters, cache hit rate, latency).

`POST /predict` (Inference, низкоуровневый): принимает готовый вектор
признаков `{"features": [...]}` и возвращает значение модели.

## Минимальные требования

* Docker 20.10+, docker compose plugin v2
* при локальном запуске – Python ≥ 3.10, numpy, pandas, matplotlib

## Результаты (финальная модель)

| Модель             | val MAE | test MAE | test RMSE | test MAPE | test R² |
|--------------------|---------|----------|-----------|-----------|---------|
| naive_lag24        | 27.59   | 25.29    | 54.38     | 15.64%    | 0.498   |
| linear_regression  | 13.84   | 12.51    | 20.78     |  8.10%    | 0.927   |
| ridge (α=1)        | 13.84   | 12.51    | 20.78     |  8.10%    | 0.927   |
| decision_tree      | 11.30   | 10.12    | 19.68     |  6.38%    | 0.934   |
| gbr_default        | 10.34   |  9.74    | 16.36     |  6.76%    | 0.955   |
| **gbr_tuned (final)** | **10.08** | **9.54** | **16.05** | **6.52%** | **0.956** |

Финальная модель – собственный градиентный бустинг на решающих
деревьях (150 деревьев, learning_rate=0.05, max_depth=5).

## Источники

* Дата-сет: Building Energy Consumption Datasets — Mendeley Data
* NASA POWER — https://power.larc.nasa.gov/
* Отчёты ЛР1 и ЛР2 (в корне проекта) с описанием бизнес-постановки,
  архитектуры и стратегии валидации.
