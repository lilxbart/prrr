# Energy Forecast Service

ML-сервис прогнозирования **суммарного суточного / недельного /
месячного** энергопотребления здания. Реализован в рамках
лабораторных работ 3 и 4 по дисциплине «Проектирование
интеллектуальных информационных систем», МТУСИ, гр. БВТ2201.

Задача (см. отчёты ЛР1–ЛР2): по историческим данным здания
(2016–2020, Building Energy Consumption Datasets + NASA POWER) и
календарным признакам предсказать суммарное ENERGY (kWh) на
горизонтах **1 день / 7 дней / 30 дней** вперёд. Почасовой прогноз
в этой версии проекта признан бесполезным для бизнес-задачи и
не используется.

## Что внутри

```
energy_forecast/
├── data/                       датасеты Building A/B, SQLite БД пользователей
├── src/
│   ├── common/                 config, data_loader, features, models,
│   │                           metrics, splits, tracker, db, auth
│   ├── training/               train.py (мульти-горизонт) и analysis.py
│   ├── inference/              HTTP inference-сервис (stdlib)
│   └── gateway/                API gateway + HTML UI + авторизация
├── models/                     финальный артефакт model.pkl (словарь моделей)
├── reports/                    experiments.jsonl, analysis_summary.txt, plots/
├── tests/                      интеграционные тесты (68 проверок)
├── docker/                     Dockerfile для inference/gateway/training
├── docker-compose.yml          минимальный стек для локального запуска
└── requirements.txt            numpy, pandas, matplotlib
```

## Архитектура

Многосервисная, повторяющая схему из ЛР2:

1. **Gateway** (`src/gateway/app.py`, порт 8000) — публичный фронт:
   HTML-страницы (индекс, регистрация, вход, личный кабинет), JSON-API
   `/predict`, личная история прогнозов, общий график фактического
   потребления. Содержит in-memory LRU-кэш прогнозов (заменитель Redis)
   и SQLite-хранилище пользователей/сессий/истории (заменитель
   PostgreSQL).
2. **Inference** (`src/inference/app.py`, порт 8001) — внутренняя
   служба прогноза. Загружает словарь моделей
   `{1: model, 7: model, 30: model}` из `models/model.pkl` и отдаёт
   ответ на `POST /predict` с обязательным полем `horizon`.
3. **Training** (`src/training/train.py`) — офлайн-обучение.
   Запускается вручную или через `docker compose run training`.

Все сервисы экспортируют метрики в формате Prometheus по `/metrics`.

## Модель

* Уровень данных — **суточные агрегаты**: ENERGY — сумма за сутки,
  погода — средние/мин/макс/сумма.
* Признаки (35 шт): лаги ENERGY (1, 2, 3, 7, 14, 28, 365 дней),
  rolling mean/std (7/14/28 дней), календарные признаки и
  циклические sin/cos, все погодные признаки (T2M*, HDD, CDD, RH, ALLSKY,
  PRECTOT, HOLIDAY).
* Для каждого горизонта обучается отдельная модель. В train.py семь
  экспериментов на горизонт: naive_last, naive_season, linear,
  ridge×2, decision_tree, gbr_default, gbr_tuned. Реализация
  моделей — собственная на numpy (без sklearn / LightGBM), чтобы
  сервис работал в окружении без выхода в PyPI.
* Итоговые результаты на тестовой выборке (полный 2020 год) —
  см. `reports/analysis_summary.txt` и раздел 5 отчёта ЛР3.

## Авторизация и история прогнозов

Пользователь регистрируется через форму `/register`, пароль
хэшируется `hashlib.scrypt` (N=16384, r=8, p=1). После входа
устанавливается HttpOnly-cookie `session` с токеном из
`secrets.token_urlsafe(32)`. Токены и TTL хранятся в SQLite.
Для каждого запроса `/predict` сохраняется строка в таблице
`prediction_history` с ссылкой на `user_id`; на странице
`/dashboard` эта история отображается вместе со средней абсолютной
ошибкой и latency.

## Запуск локально

```bash
# 1. обучить модели (один раз)
python -m src.training.train            # ~40 сек на CPU
python -m src.training.analysis         # графики ошибок по горизонтам

# 2. запустить inference-сервис
INFERENCE_PORT=8001 python -m src.inference.app &

# 3. запустить gateway
INFERENCE_URL=http://localhost:8001 GATEWAY_PORT=8000 \
    python -m src.gateway.app

# 4. открыть http://localhost:8000
```

## Запуск в Docker

```bash
# собрать образы
docker compose build

# обучить и положить модель в общий том (однократно)
docker compose --profile training run --rm training

# запустить стек
docker compose up -d inference gateway

# снять стек
docker compose down
```

## Тесты

```bash
python -m tests.test_integration
```

Покрывает feature-пайплайн, разбиение без утечек, собственные
ML-модели, примитивы авторизации (scrypt, сессии), HTTP inference
(все три горизонта и 422 на невалидных запросах), e2e-gateway с
регистрацией/входом/логаутом, кэш и Prometheus-метрики.

## HTTP API

Публичные:

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | публичная главная |
| GET/POST | `/register` | форма и обработчик регистрации |
| GET/POST | `/login` | форма и обработчик входа |
| POST | `/logout` | завершить сессию |
| GET | `/health` | статус gateway |
| GET | `/metrics` | Prometheus-метрики |
| GET | `/history?building=A&limit=120` | суточное фактическое потребление |

Требуют авторизации (cookie `session`):

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/dashboard` | личный кабинет с графиком и таблицей |
| POST | `/predict` | `{"building":"A","date":"2020-06-15","horizon":7}` |
| GET | `/my-history?limit=50` | личная история прогнозов + агрегаты |

Внутренний inference (порт 8001):

```
POST /predict  body {"horizon": 7, "features": [...35...]}
POST /predict_raw body {"horizon": 7, "history": [{DATE,ENERGY,...}]}
GET  /health
GET  /metrics
```
