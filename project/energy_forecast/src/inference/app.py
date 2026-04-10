"""Inference-сервис на stdlib http.server.

Минимальный REST-сервис, загружающий финальную модель из MODELS_DIR
и отдающий предсказания по POST /predict. Реализует также /health и
/metrics (счётчики запросов, средняя latency, экспортируются в
текстовом формате, совместимом с Prometheus).

Используется stdlib (без FastAPI/uvicorn), чтобы сервис мог быть
запущен в среде без выхода во внешний PyPI. В production-контейнере
(Dockerfile.inference) рекомендуется заменить этот модуль на FastAPI.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [inference] %(levelname)s %(message)s",
)
log = logging.getLogger("inference")


# --------------------------- Загрузка модели -------------------------------

class ModelHolder:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.model = None
        self.feature_columns: list[str] = []
        self.model_name: str = ""
        self.loaded_at: float = 0.0
        self.load()

    def load(self) -> None:
        t0 = time.time()
        with self.path.open("rb") as f:
            bundle = pickle.load(f)
        self.model = bundle["model"]
        self.feature_columns = bundle["feature_columns"]
        self.model_name = bundle["model_name"]
        self.loaded_at = time.time()
        log.info(
            "Модель '%s' загружена за %.3f сек (%d признаков)",
            self.model_name, self.loaded_at - t0, len(self.feature_columns),
        )

    def predict(self, features: list[float]) -> float:
        if len(features) != len(self.feature_columns):
            raise ValueError(
                f"ожидалось {len(self.feature_columns)} признаков, "
                f"получено {len(features)}"
            )
        x = np.asarray(features, dtype=float).reshape(1, -1)
        y = self.model.predict(x)
        return float(y[0])


# --------------------------- Метрики ---------------------------------------

class Metrics:
    def __init__(self):
        self.lock = Lock()
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency = 0.0
        self.predict_count = 0
        self.predict_latency = 0.0
        self.started_at = time.time()

    def record(self, path: str, status: int, latency: float) -> None:
        with self.lock:
            self.total_requests += 1
            self.total_latency += latency
            if status >= 500:
                self.total_errors += 1
            if path == "/predict":
                self.predict_count += 1
                self.predict_latency += latency

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
            ]
        return "\n".join(lines) + "\n"


METRICS = Metrics()
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "model.pkl"))
MODEL = ModelHolder(MODEL_PATH)


# --------------------------- HTTP Handler ----------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "energy-inference/1.0"

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
                        "model": MODEL.model_name,
                        "n_features": len(MODEL.feature_columns),
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
                        "model": MODEL.model_name,
                        "feature_columns": MODEL.feature_columns,
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
                if "features" not in payload:
                    status = 400
                    self._send_json(400, {"error": "missing 'features'"})
                    return
                try:
                    value = MODEL.predict(payload["features"])
                except ValueError as exc:
                    status = 422
                    self._send_json(422, {"error": str(exc)})
                    return
                self._send_json(
                    200,
                    {
                        "prediction": round(value, 3),
                        "model": MODEL.model_name,
                        "latency_ms": round((time.time() - t0) * 1000, 2),
                    },
                )
            else:
                status = 404
                self._send_json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            status = 500
            log.exception("Ошибка обработки POST %s: %s", self.path, exc)
            self._send_json(500, {"error": str(exc)})
        finally:
            METRICS.record(self.path, status, time.time() - t0)


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
