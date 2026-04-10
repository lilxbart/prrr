"""Inference-сервис на stdlib http.server (мульти-горизонтный).

Загружает словарь моделей `{1: model, 7: model, 30: model}` из
`models/model.pkl`, отдаёт прогнозы через POST /predict с обязательным
полем `horizon` (1, 7 или 30). Также реализует /health, /info и
/metrics в формате Prometheus. Реализация использует только stdlib —
окружение без выхода в PyPI.

POST /predict body:
  {"horizon": 7, "features": [..35 float..]}

Для удобства интеграции с UI реализован также POST /predict_raw,
который принимает «сырые» суточные данные (DATE + ENERGY история +
погода) и сам строит признаки на лету.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import MODELS_DIR, HORIZONS, DATE_COL, TARGET
from src.common.features import build_features, get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [inference] %(levelname)s %(message)s",
)
log = logging.getLogger("inference")


# --------------------------- Загрузка моделей ------------------------------

class ModelHolder:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.models: dict[int, dict] = {}
        self.feature_columns: list[str] = []
        self.loaded_at: float = 0.0
        self.trained_at: str = ""
        self.building: str = "A"
        self.load()

    def load(self) -> None:
        t0 = time.time()
        with self.path.open("rb") as f:
            bundle = pickle.load(f)
        self.models = bundle["models"]
        self.feature_columns = bundle["feature_columns"]
        self.trained_at = bundle.get("trained_at", "")
        self.building = bundle.get("building", "A")
        self.loaded_at = time.time()
        names = {h: a["model_name"] for h, a in self.models.items()}
        log.info(
            "Модели загружены за %.3f сек: %s (%d признаков)",
            self.loaded_at - t0, names, len(self.feature_columns),
        )

    def predict(self, horizon: int, features: list[float]) -> dict:
        if horizon not in self.models:
            raise ValueError(
                f"неизвестный горизонт {horizon}; допустимы: {sorted(self.models)}"
            )
        if len(features) != len(self.feature_columns):
            raise ValueError(
                f"ожидалось {len(self.feature_columns)} признаков, "
                f"получено {len(features)}"
            )
        x = np.asarray(features, dtype=float).reshape(1, -1)
        art = self.models[horizon]
        y = art["model"].predict(x)
        return {
            "prediction": float(y[0]),
            "horizon": horizon,
            "model": art["model_name"],
            "val_metrics": art.get("val_metrics"),
            "test_metrics": art.get("test_metrics"),
        }


# --------------------------- Метрики ---------------------------------------

class Metrics:
    def __init__(self):
        self.lock = Lock()
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency = 0.0
        self.predict_count = 0
        self.predict_latency = 0.0
        self.predict_by_h: dict[int, int] = {}
        self.started_at = time.time()

    def record(self, path: str, status: int, latency: float,
               horizon: int | None = None) -> None:
        with self.lock:
            self.total_requests += 1
            self.total_latency += latency
            if status >= 500:
                self.total_errors += 1
            if path.startswith("/predict"):
                self.predict_count += 1
                self.predict_latency += latency
                if horizon is not None:
                    self.predict_by_h[horizon] = self.predict_by_h.get(horizon, 0) + 1

    def prometheus(self) -> str:
        with self.lock:
            avg = self.total_latency / self.total_requests if self.total_requests else 0
            avg_pred = (
                self.predict_latency / self.predict_count if self.predict_count else 0
            )
            uptime = time.time() - self.started_at
            lines = [
                "# HELP inference_uptime_seconds Uptime of inference service",
                "# TYPE inference_uptime_seconds gauge",
                f"inference_uptime_seconds {uptime:.1f}",
                "# HELP inference_requests_total Total HTTP requests",
                "# TYPE inference_requests_total counter",
                f"inference_requests_total {self.total_requests}",
                "# HELP inference_errors_total HTTP 5xx errors",
                "# TYPE inference_errors_total counter",
                f"inference_errors_total {self.total_errors}",
                "# HELP inference_request_latency_avg_seconds Mean request latency",
                "# TYPE inference_request_latency_avg_seconds gauge",
                f"inference_request_latency_avg_seconds {avg:.6f}",
                "# HELP inference_predict_total Total predictions",
                "# TYPE inference_predict_total counter",
                f"inference_predict_total {self.predict_count}",
                "# HELP inference_predict_latency_avg_seconds Mean prediction latency",
                "# TYPE inference_predict_latency_avg_seconds gauge",
                f"inference_predict_latency_avg_seconds {avg_pred:.6f}",
                "# HELP inference_predict_by_horizon_total Predictions per horizon",
                "# TYPE inference_predict_by_horizon_total counter",
            ]
            for h, c in sorted(self.predict_by_h.items()):
                lines.append(
                    f'inference_predict_by_horizon_total{{horizon="{h}"}} {c}'
                )
        return "\n".join(lines) + "\n"


METRICS = Metrics()
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "model.pkl"))
MODEL = ModelHolder(MODEL_PATH)


# --------------------------- Predict-raw helper ----------------------------

def predict_from_raw_history(records: list[dict], horizon: int) -> dict:
    """Построить признаки из суточной истории и сделать предсказание.

    records – список словарей с ключами DATE (YYYY-MM-DD), ENERGY и
    всеми погодными признаками. Последняя строка — «текущий день»,
    относительно которого делается прогноз на horizon дней вперёд.
    Для построения лаговых признаков требуется не менее 366 последних
    точек.
    """
    if horizon not in MODEL.models:
        raise ValueError(f"неизвестный горизонт {horizon}")
    if len(records) < 370:
        raise ValueError(
            f"для прогноза нужно минимум 370 суток истории, получено {len(records)}"
        )
    df = pd.DataFrame(records)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df_feat = build_features(df, dropna=False)
    last = df_feat.iloc[[-1]].copy()
    feats = get_feature_columns()
    if last[feats].isnull().any(axis=None):
        missing = last[feats].columns[last[feats].isnull().any()].tolist()
        raise ValueError(f"не хватает данных для признаков: {missing}")
    return MODEL.predict(horizon, last[feats].values[0].tolist())


# --------------------------- HTTP Handler ----------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "energy-inference/2.0"

    def log_message(self, fmt, *args) -> None:  # noqa: N802
        log.info("%s - %s", self.client_address[0], fmt % args)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str, ctype: str = "text/plain") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        t0 = time.time()
        status = 200
        try:
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "horizons": sorted(MODEL.models),
                        "models": {
                            str(h): a["model_name"] for h, a in MODEL.models.items()
                        },
                        "n_features": len(MODEL.feature_columns),
                        "trained_at": MODEL.trained_at,
                        "loaded_at": MODEL.loaded_at,
                    },
                )
            elif self.path == "/metrics":
                self._send_text(200, METRICS.prometheus())
            elif self.path in ("/", "/info"):
                self._send_json(
                    200,
                    {
                        "service": "energy-inference",
                        "horizons": sorted(MODEL.models),
                        "feature_columns": MODEL.feature_columns,
                        "building": MODEL.building,
                    },
                )
            else:
                status = 404
                self._send_json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            status = 500
            log.exception("Ошибка обработки GET %s: %s", self.path, exc)
            self._send_json(500, {"error": str(exc)})
        finally:
            METRICS.record(self.path, status, time.time() - t0)

    def do_POST(self) -> None:  # noqa: N802
        t0 = time.time()
        status = 200
        horizon: int | None = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                status = 400
                self._send_json(400, {"error": f"invalid json: {exc}"})
                return

            if self.path == "/predict":
                horizon = int(payload.get("horizon", 0))
                if "features" not in payload:
                    status = 400
                    self._send_json(400, {"error": "missing 'features'"})
                    return
                try:
                    result = MODEL.predict(horizon, payload["features"])
                except ValueError as exc:
                    status = 422
                    self._send_json(422, {"error": str(exc)})
                    return
                result["latency_ms"] = round((time.time() - t0) * 1000, 2)
                self._send_json(200, result)

            elif self.path == "/predict_raw":
                horizon = int(payload.get("horizon", 0))
                records = payload.get("history") or []
                try:
                    result = predict_from_raw_history(records, horizon)
                except ValueError as exc:
                    status = 422
                    self._send_json(422, {"error": str(exc)})
                    return
                result["latency_ms"] = round((time.time() - t0) * 1000, 2)
                self._send_json(200, result)

            else:
                status = 404
                self._send_json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            status = 500
            log.exception("Ошибка обработки POST %s: %s", self.path, exc)
            self._send_json(500, {"error": str(exc)})
        finally:
            METRICS.record(self.path, status, time.time() - t0, horizon=horizon)


def serve(host: str = "0.0.0.0", port: int = 8001) -> None:
    srv = ThreadingHTTPServer((host, port), Handler)
    log.info("Inference service listening on %s:%d", host, port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        srv.shutdown()


if __name__ == "__main__":
    port = int(os.getenv("INFERENCE_PORT", "8001"))
    serve(port=port)
