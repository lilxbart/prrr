"""API Gateway – пользовательский вход в систему прогнозирования.

Реализует:
  * POST /predict  – прогноз энергопотребления для здания на заданный момент
  * GET  /history  – исторические ряды (для графика в UI)
  * GET  /health   – статус
  * GET  /metrics  – Prometheus-метрики
  * GET  /         – простой HTML-UI

Имитирует архитектуру из ЛР2: приходящий запрос сначала проверяется в
in-memory кэше (аналог Redis), иначе собирается вектор признаков из
«БД» (CSV) и отправляется в Inference-сервис по HTTP.

Stdlib-only: http.server + urllib.request.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import DATE_COL, TARGET, DATA_DIR
from src.common.data_loader import load_building
from src.common.features import build_features, get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [gateway] %(levelname)s %(message)s",
)
log = logging.getLogger("gateway")

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8001")
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "300"))
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8000"))


# ------------------------- Хранилище данных --------------------------------

class DataStore:
    """Загружает историю обоих зданий в память и предоставляет готовые
    feature-таблицы. В production-контейнере заменяется на PostgreSQL.
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.cache: dict[str, pd.DataFrame] = {}

    def features_for(self, building: str) -> pd.DataFrame:
        if building in self.cache:
            return self.cache[building]
        df = load_building(building, data_dir=self.data_dir)
        df = build_features(df)
        self.cache[building] = df
        log.info("Признаки для здания %s готовы: %d строк", building, len(df))
        return df


STORE = DataStore()


# ------------------------- Кэш предсказаний --------------------------------

class PredictionCache:
    def __init__(self, ttl: int = CACHE_TTL_SEC, maxsize: int = 1024):
        self.ttl = ttl
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._store: dict[str, tuple[float, dict]] = {}

    def get(self, key: str) -> dict | None:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            ts, value = item
            if time.time() - ts > self.ttl:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            if len(self._store) >= self.maxsize:
                # простейший LRU: удалить самый старый
                oldest = min(self._store.items(), key=lambda kv: kv[1][0])[0]
                self._store.pop(oldest, None)
            self._store[key] = (time.time(), value)

    def size(self) -> int:
        with self._lock:
            return len(self._store)


CACHE = PredictionCache()


# ------------------------- Метрики ----------------------------------------

class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.inference_fail = 0
        self.started_at = time.time()

    def record_request(self, status: int, latency: float, cache_hit: bool) -> None:
        with self.lock:
            self.total_requests += 1
            self.total_latency += latency
            if status >= 500:
                self.total_errors += 1
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def record_inference_fail(self) -> None:
        with self.lock:
            self.inference_fail += 1

    def prometheus(self) -> str:
        with self.lock:
            avg = self.total_latency / self.total_requests if self.total_requests else 0
            hit_rate = (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) else 0
            )
            uptime = time.time() - self.started_at
            lines = [
                "# TYPE gateway_uptime_seconds gauge",
                f"gateway_uptime_seconds {uptime:.1f}",
                "# TYPE gateway_requests_total counter",
                f"gateway_requests_total {self.total_requests}",
                "# TYPE gateway_errors_total counter",
                f"gateway_errors_total {self.total_errors}",
                "# TYPE gateway_request_latency_avg_seconds gauge",
                f"gateway_request_latency_avg_seconds {avg:.6f}",
                "# TYPE gateway_cache_hits_total counter",
                f"gateway_cache_hits_total {self.cache_hits}",
                "# TYPE gateway_cache_misses_total counter",
                f"gateway_cache_misses_total {self.cache_misses}",
                "# TYPE gateway_cache_hit_rate gauge",
                f"gateway_cache_hit_rate {hit_rate:.4f}",
                "# TYPE gateway_inference_failures_total counter",
                f"gateway_inference_failures_total {self.inference_fail}",
                "# TYPE gateway_cache_size gauge",
                f"gateway_cache_size {CACHE.size()}",
            ]
        return "\n".join(lines) + "\n"


METRICS = Metrics()


# ------------------------- Работа с инференсом -----------------------------

def call_inference(features: list[float], timeout: float = 5.0) -> dict:
    req = urllib.request.Request(
        f"{INFERENCE_URL}/predict",
        data=json.dumps({"features": features}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ------------------------- Подготовка запроса ------------------------------

def lookup_features(building: str, timestamp: str) -> tuple[list[float], dict]:
    df = STORE.features_for(building)
    try:
        ts = pd.to_datetime(timestamp)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"некорректная timestamp: {timestamp!r}") from exc
    mask = df[DATE_COL] == ts
    if not mask.any():
        # Ищем ближайший доступный
        nearest = (df[DATE_COL] - ts).abs().idxmin()
        row = df.loc[nearest]
    else:
        row = df.loc[mask].iloc[0]
    feats = get_feature_columns()
    return [float(row[f]) for f in feats], {
        "actual_timestamp": str(row[DATE_COL]),
        "actual_energy": float(row[TARGET]),
    }


# ------------------------- HTTP Handler -----------------------------------

INDEX_HTML = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Energy Forecast Service</title>
<style>
  body {font-family: -apple-system, Arial, sans-serif; max-width: 780px; margin: 2em auto; color:#222}
  h1 {color:#244}
  form {background:#f4f6fa; padding:1em 1.4em; border-radius:8px}
  label {display:block; margin:.6em 0 .2em; font-weight:600}
  input, select {padding:.45em .6em; border:1px solid #bbd; border-radius:4px; width: 260px}
  button {margin-top:1em; padding:.55em 1.4em; background:#2563eb; color:#fff;
          border:none; border-radius:4px; cursor:pointer; font-weight:600}
  .result {background:#eefbe8; padding:1em 1.3em; border-radius:6px; margin-top:1em}
  .err {background:#fce8e8; padding:1em 1.3em; border-radius:6px; margin-top:1em}
  code {background:#eee; padding:2px 4px; border-radius:3px}
  small {color:#556}
</style>
</head>
<body>
<h1>Прогноз энергопотребления здания</h1>
<p>Выберите здание и момент времени. Сервис вернёт прогноз ENERGY
(kWh) финальной модели и покажет фактическое значение из исторических
данных для сравнения.</p>
<form id="f">
  <label>Здание</label>
  <select name="building"><option>A</option><option>B</option></select>
  <label>Момент времени (ISO)</label>
  <input name="timestamp" value="2020-06-15T10:00:00">
  <label>Горизонт, часов</label>
  <input name="horizon" value="1" type="number" min="1" max="24">
  <div><button type="submit">Получить прогноз</button></div>
</form>
<div id="out"></div>
<p><small>API:
<code>POST /predict</code> · <code>GET /health</code> ·
<code>GET /metrics</code></small></p>
<script>
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  const body = {
    building: f.get('building'),
    timestamp: f.get('timestamp'),
    horizon: Number(f.get('horizon'))
  };
  const out = document.getElementById('out');
  out.innerHTML = 'Запрашиваем прогноз…';
  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'ошибка сервиса');
    out.innerHTML = `<div class="result">
      <b>Прогноз:</b> ${j.prediction} kWh<br>
      <b>Факт (из истории):</b> ${j.actual_energy?.toFixed(2) ?? '—'} kWh<br>
      <b>Абс. ошибка:</b> ${j.abs_error?.toFixed(2) ?? '—'} kWh<br>
      <b>Модель:</b> ${j.model}<br>
      <b>Кэш:</b> ${j.cache_hit ? 'попадание' : 'промах'} ·
      <b>Latency:</b> ${j.latency_ms} мс
    </div>`;
  } catch (err) {
    out.innerHTML = `<div class="err">Ошибка: ${err.message}</div>`;
  }
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "energy-gateway/1.0"

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

    # ---- GET ----

    def do_GET(self) -> None:  # noqa: N802
        t0 = time.time()
        status = 200
        cache_hit = False
        try:
            if self.path == "/" or self.path.startswith("/index"):
                self._send_text(200, INDEX_HTML, ctype="text/html")
            elif self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "cache_size": CACHE.size(),
                        "inference_url": INFERENCE_URL,
                    },
                )
            elif self.path == "/metrics":
                self._send_text(200, METRICS.prometheus())
            elif self.path.startswith("/history"):
                # /history?building=A&limit=168
                from urllib.parse import urlparse, parse_qs
                q = parse_qs(urlparse(self.path).query)
                building = (q.get("building") or ["A"])[0]
                limit = int((q.get("limit") or ["168"])[0])
                df = STORE.features_for(building).tail(limit)
                payload = {
                    "building": building,
                    "points": [
                        {"ts": str(r[DATE_COL]), "energy": float(r[TARGET])}
                        for _, r in df.iterrows()
                    ],
                }
                self._send_json(200, payload)
            else:
                status = 404
                self._send_json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            status = 500
            log.exception("Ошибка GET %s: %s", self.path, exc)
            self._send_json(500, {"error": str(exc)})
        finally:
            METRICS.record_request(status, time.time() - t0, cache_hit)

    # ---- POST ----

    def do_POST(self) -> None:  # noqa: N802
        t0 = time.time()
        status = 200
        cache_hit = False
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
                building = str(payload.get("building", "A")).upper()
                timestamp = str(payload.get("timestamp", ""))
                horizon = int(payload.get("horizon", 1))
                if building not in ("A", "B"):
                    status = 400
                    self._send_json(400, {"error": "building должен быть A или B"})
                    return
                if not timestamp:
                    status = 400
                    self._send_json(400, {"error": "требуется timestamp"})
                    return

                cache_key = f"{building}|{timestamp}|{horizon}"
                cached = CACHE.get(cache_key)
                if cached:
                    cache_hit = True
                    resp = {**cached, "cache_hit": True,
                            "latency_ms": round((time.time() - t0) * 1000, 2)}
                    self._send_json(200, resp)
                    return

                try:
                    features, meta = lookup_features(building, timestamp)
                except ValueError as exc:
                    status = 422
                    self._send_json(422, {"error": str(exc)})
                    return

                try:
                    infer_resp = call_inference(features)
                except Exception as exc:  # noqa: BLE001
                    METRICS.record_inference_fail()
                    status = 502
                    log.exception("Inference недоступен: %s", exc)
                    self._send_json(502, {"error": f"inference error: {exc}"})
                    return

                prediction = float(infer_resp["prediction"])
                result = {
                    "building": building,
                    "timestamp": timestamp,
                    "horizon": horizon,
                    "prediction": round(prediction, 3),
                    "actual_energy": round(meta["actual_energy"], 3),
                    "abs_error": round(abs(prediction - meta["actual_energy"]), 3),
                    "model": infer_resp.get("model"),
                }
                CACHE.set(cache_key, result)
                self._send_json(
                    200,
                    {**result, "cache_hit": False,
                     "latency_ms": round((time.time() - t0) * 1000, 2)},
                )
            else:
                status = 404
                self._send_json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            status = 500
            log.exception("Ошибка POST %s: %s", self.path, exc)
            self._send_json(500, {"error": str(exc)})
        finally:
            METRICS.record_request(status, time.time() - t0, cache_hit)


def serve(host: str = "0.0.0.0", port: int = GATEWAY_PORT) -> None:
    srv = ThreadingHTTPServer((host, port), Handler)
    log.info("Gateway listening on %s:%d (inference=%s)", host, port, INFERENCE_URL)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        srv.shutdown()


if __name__ == "__main__":
    serve()
