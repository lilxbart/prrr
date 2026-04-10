"""API Gateway с аутентификацией и веб-интерфейсом.

Архитектура:
  * stdlib http.server (без FastAPI/uvicorn, чтобы работать без PyPI);
  * SQLite-хранилище пользователей, сессий и истории прогнозов
    (замена PostgreSQL/Redis в рамках демонстрационной установки);
  * внутренний LRU-кэш прогнозов (замена Redis);
  * HTTP-клиент к inference-сервису (urllib);
  * простой HTML/JS UI с графиком факт vs прогноз (Chart.js через CDN,
    а при его недоступности — SVG-fallback, который генерируется на
    сервере из точек истории).

Публичные эндпоинты:
  * GET  /                — публичная главная страница
  * GET  /register        — форма регистрации
  * POST /register        — создать пользователя
  * GET  /login           — форма входа
  * POST /login           — выдать cookie-сессию
  * POST /logout          — завершить сессию
  * GET  /dashboard       — личный кабинет (требует авторизации)
  * POST /predict         — прогноз {building, timestamp, horizon}
  * GET  /history         — история ENERGY из хранилища данных
  * GET  /my-history      — личная история прогнозов пользователя
  * GET  /health · /metrics
"""
from __future__ import annotations

import html
import json
import logging
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import (
    DATE_COL, TARGET, DATA_DIR, HORIZONS, HORIZON_LABELS,
)
from src.common.data_loader import load_building_daily
from src.common.features import build_features, get_feature_columns
from src.common import auth, db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [gateway] %(levelname)s %(message)s",
)
log = logging.getLogger("gateway")

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8001")
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "300"))
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8000"))

db.init_db()


# ------------------------- Хранилище данных --------------------------------

class DataStore:
    """Суточные признаки по зданиям, кэшируются в памяти."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self._feats: dict[str, pd.DataFrame] = {}
        self._daily: dict[str, pd.DataFrame] = {}
        self._lock = threading.Lock()

    def features_for(self, building: str) -> pd.DataFrame:
        with self._lock:
            if building in self._feats:
                return self._feats[building]
            df = load_building_daily(building, data_dir=self.data_dir)
            self._daily[building] = df.copy()
            df_feat = build_features(df)
            self._feats[building] = df_feat
            log.info(
                "Здание %s: %d суток, %d строк после FE",
                building, len(df), len(df_feat),
            )
            return df_feat

    def daily(self, building: str) -> pd.DataFrame:
        if building not in self._daily:
            self.features_for(building)
        return self._daily[building]


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
        self.logins_ok = 0
        self.logins_fail = 0
        self.registrations = 0
        self.started_at = time.time()

    def record_request(self, status: int, latency: float, cache_hit: bool) -> None:
        with self.lock:
            self.total_requests += 1
            self.total_latency += latency
            if status >= 500:
                self.total_errors += 1
            if cache_hit:
                self.cache_hits += 1

    def record_cache_miss(self) -> None:
        with self.lock:
            self.cache_misses += 1

    def record_inference_fail(self) -> None:
        with self.lock:
            self.inference_fail += 1

    def record_login(self, ok: bool) -> None:
        with self.lock:
            if ok:
                self.logins_ok += 1
            else:
                self.logins_fail += 1

    def record_registration(self) -> None:
        with self.lock:
            self.registrations += 1

    def prometheus(self) -> str:
        with self.lock:
            avg = self.total_latency / self.total_requests if self.total_requests else 0
            denom = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / denom if denom else 0
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
                "# TYPE gateway_cache_size gauge",
                f"gateway_cache_size {CACHE.size()}",
                "# TYPE gateway_inference_failures_total counter",
                f"gateway_inference_failures_total {self.inference_fail}",
                "# TYPE gateway_login_success_total counter",
                f"gateway_login_success_total {self.logins_ok}",
                "# TYPE gateway_login_failure_total counter",
                f"gateway_login_failure_total {self.logins_fail}",
                "# TYPE gateway_registrations_total counter",
                f"gateway_registrations_total {self.registrations}",
                "# TYPE gateway_users_total gauge",
                f"gateway_users_total {db.count_users()}",
            ]
        return "\n".join(lines) + "\n"


METRICS = Metrics()


# ------------------------- Работа с инференсом -----------------------------

def call_inference(horizon: int, features: list[float],
                   timeout: float = 5.0) -> dict:
    req = urllib.request.Request(
        f"{INFERENCE_URL}/predict",
        data=json.dumps({"horizon": horizon, "features": features}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def lookup_features(building: str, date_str: str) -> tuple[list[float], dict]:
    df = STORE.features_for(building)
    try:
        ts = pd.to_datetime(date_str)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"некорректная дата: {date_str!r}") from exc
    ts = pd.Timestamp(ts.date())
    mask = df[DATE_COL] == ts
    if not mask.any():
        nearest = (df[DATE_COL] - ts).abs().idxmin()
        row = df.loc[nearest]
    else:
        row = df.loc[mask].iloc[0]
    feats = get_feature_columns()
    return [float(row[f]) for f in feats], {
        "actual_date": str(row[DATE_COL].date()),
    }


def actual_future_sum(building: str, start_date: str, horizon: int) -> float | None:
    """Фактическое суммарное потребление за horizon суток, начиная со следующего дня."""
    daily = STORE.daily(building)
    ts = pd.Timestamp(pd.to_datetime(start_date).date())
    idx = daily.index[daily[DATE_COL] == ts]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    j = i + 1 + horizon
    if j > len(daily):
        return None
    window = daily.iloc[i + 1 : j][TARGET].values
    if len(window) < horizon:
        return None
    return float(window.sum())


# ------------------------- HTML шаблоны -----------------------------------

BASE_CSS = """
body{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:960px;
  margin:2em auto;color:#1f2330;padding:0 1em;background:#f6f8fb}
h1,h2{color:#1a365d}
a{color:#2563eb;text-decoration:none}
a:hover{text-decoration:underline}
nav{padding:.6em 0;border-bottom:1px solid #d5dbe6;margin-bottom:1em;
  display:flex;gap:1.2em;align-items:center}
nav .spacer{flex:1}
form.card,.card{background:#fff;padding:1.1em 1.4em;border-radius:10px;
  box-shadow:0 1px 3px rgba(10,20,60,.06);margin-bottom:1.2em}
label{display:block;margin:.6em 0 .2em;font-weight:600;color:#314158}
input,select{padding:.5em .65em;border:1px solid #c4cbd9;border-radius:6px;
  font-size:1em;min-width:220px;background:#fff}
button{margin-top:1em;padding:.6em 1.4em;background:#2563eb;color:#fff;
  border:none;border-radius:6px;cursor:pointer;font-weight:600;font-size:1em}
button:hover{background:#1d4fd4}
.result{background:#eefbe8;padding:1em 1.3em;border-radius:8px;margin-top:1em;
  border-left:4px solid #4caf50}
.err{background:#fce8e8;padding:.8em 1.2em;border-radius:8px;margin-top:1em;
  color:#8a2a2a;border-left:4px solid #e53e3e}
table{width:100%;border-collapse:collapse;margin-top:.5em;background:#fff}
th,td{padding:.55em .75em;border-bottom:1px solid #e6eaf3;text-align:left}
th{background:#f0f4fa;font-size:.9em;color:#314158}
small{color:#5a6578}
.badge{display:inline-block;padding:.15em .6em;background:#e5edff;color:#1d4fd4;
  border-radius:10px;font-size:.85em;font-weight:600}
.metrics{display:flex;gap:1em;flex-wrap:wrap}
.metric{background:#fff;padding:.8em 1.1em;border-radius:8px;flex:1;min-width:150px;
  box-shadow:0 1px 3px rgba(10,20,60,.06)}
.metric .v{font-size:1.6em;font-weight:700;color:#1a365d}
.metric .l{font-size:.8em;color:#5a6578;text-transform:uppercase;letter-spacing:.5px}
#chart{width:100%;height:320px}
"""


def _nav(user: auth.UserCtx | None) -> str:
    if user:
        right = (
            f'<span class="badge">{html.escape(user.username)}</span>'
            f'<form method="POST" action="/logout" style="display:inline;margin:0">'
            f'<button type="submit" style="background:#64748b;margin:0;'
            f'padding:.3em .9em">Выйти</button></form>'
        )
        links = (
            '<a href="/dashboard">Личный кабинет</a>'
            '<a href="/history">История</a>'
        )
    else:
        right = (
            '<a href="/login">Вход</a> · '
            '<a href="/register">Регистрация</a>'
        )
        links = ""
    return (
        f'<nav><a href="/"><b>⚡ Energy Forecast</b></a>'
        f'{links}<div class="spacer"></div>{right}</nav>'
    )


def _page(title: str, body: str, user: auth.UserCtx | None) -> str:
    return (
        f"<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        f"<style>{BASE_CSS}</style></head><body>"
        f"{_nav(user)}{body}</body></html>"
    )


def index_page(user: auth.UserCtx | None) -> str:
    horizons_str = ", ".join(
        f"{h} д. ({HORIZON_LABELS[h]})" for h in HORIZONS
    )
    if user:
        cta = (
            '<p><a href="/dashboard"><button type="button">Перейти в личный кабинет</button></a></p>'
        )
    else:
        cta = (
            '<p><a href="/register"><button type="button">Зарегистрироваться</button></a> '
            '<a href="/login"><button type="button" style="background:#64748b">Войти</button></a></p>'
        )
    body = f"""
<h1>Прогноз энергопотребления здания</h1>
<div class="card">
<p>Сервис прогнозирует суммарное потребление электроэнергии зданием на
горизонты: <b>{horizons_str}</b>. Модели обучены на 5 годах суточных данных
(2016–2020) и учитывают температуру, влажность, осадки, солнечную
радиацию и календарные признаки.</p>
<p>Для работы с прогнозами требуется учётная запись — сервис сохраняет
вашу личную историю и метрики точности.</p>
{cta}
<p><small>API: <code>POST /predict</code> · <code>GET /metrics</code> ·
<code>GET /health</code></small></p>
</div>
"""
    return _page("Energy Forecast", body, user)


def register_page(error: str = "") -> str:
    err_html = f'<div class="err">{html.escape(error)}</div>' if error else ""
    body = f"""
<h1>Регистрация</h1>
<form class="card" method="POST" action="/register">
<label>Имя пользователя</label>
<input name="username" required minlength="3" maxlength="32" autofocus>
<label>Пароль</label>
<input name="password" type="password" required minlength="6" maxlength="128">
<label>Повторите пароль</label>
<input name="password2" type="password" required minlength="6" maxlength="128">
<div><button type="submit">Создать аккаунт</button></div>
</form>
{err_html}
<p><small>Уже есть аккаунт? <a href="/login">Войти</a></small></p>
"""
    return _page("Регистрация", body, None)


def login_page(error: str = "") -> str:
    err_html = f'<div class="err">{html.escape(error)}</div>' if error else ""
    body = f"""
<h1>Вход</h1>
<form class="card" method="POST" action="/login">
<label>Имя пользователя</label>
<input name="username" required autofocus>
<label>Пароль</label>
<input name="password" type="password" required>
<div><button type="submit">Войти</button></div>
</form>
{err_html}
<p><small>Нет аккаунта? <a href="/register">Зарегистрироваться</a></small></p>
"""
    return _page("Вход", body, None)


def dashboard_page(user: auth.UserCtx) -> str:
    options = "".join(
        f'<option value="{h}">{h} суток — {HORIZON_LABELS[h]}</option>'
        for h in HORIZONS
    )
    default_date = "2020-06-15"
    body = f"""
<h1>Личный кабинет</h1>
<div class="metrics">
  <div class="metric"><div class="v" id="m-total">—</div><div class="l">Прогнозов</div></div>
  <div class="metric"><div class="v" id="m-err">—</div><div class="l">Средняя |ошибка|</div></div>
  <div class="metric"><div class="v" id="m-lat">—</div><div class="l">Latency, мс</div></div>
</div>

<h2>Новый прогноз</h2>
<form class="card" id="f">
  <label>Здание</label>
  <select name="building"><option>A</option><option>B</option></select>
  <label>Дата старта (с которой строим прогноз)</label>
  <input name="date" type="date" value="{default_date}">
  <label>Горизонт</label>
  <select name="horizon">{options}</select>
  <div><button type="submit">Получить прогноз</button></div>
</form>
<div id="out"></div>

<h2>График истории (суточное потребление)</h2>
<div class="card">
<canvas id="chart"></canvas>
<div><small>Последние 120 суток выбранного здания</small></div>
</div>

<h2>Моя история прогнозов</h2>
<div class="card">
<table id="hist">
  <thead><tr><th>Время</th><th>Здание</th><th>Горизонт</th>
  <th>Прогноз, kWh</th><th>Факт, kWh</th><th>|ошибка|</th><th>lat, мс</th></tr></thead>
  <tbody></tbody>
</table>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
async function refreshStats() {{
  const r = await fetch('/my-history?limit=100');
  if (!r.ok) return;
  const j = await r.json();
  document.getElementById('m-total').textContent = j.stats.total;
  document.getElementById('m-err').textContent =
    j.stats.avg_abs_error ? j.stats.avg_abs_error.toFixed(1) : '—';
  document.getElementById('m-lat').textContent =
    j.stats.avg_latency_ms ? j.stats.avg_latency_ms.toFixed(1) : '—';
  const tb = document.querySelector('#hist tbody');
  tb.innerHTML = '';
  for (const h of j.history) {{
    const tr = document.createElement('tr');
    tr.innerHTML =
      `<td>${{new Date(h.created_at*1000).toLocaleString('ru-RU')}}</td>`+
      `<td>${{h.building}}</td><td>${{h.horizon}} д.</td>`+
      `<td>${{h.prediction.toFixed(1)}}</td>`+
      `<td>${{h.actual !== null ? h.actual.toFixed(1) : '—'}}</td>`+
      `<td>${{h.abs_error !== null ? h.abs_error.toFixed(1) : '—'}}</td>`+
      `<td>${{h.latency_ms.toFixed(0)}}</td>`;
    tb.appendChild(tr);
  }}
}}

let chart;
async function refreshChart(building) {{
  const r = await fetch(`/history?building=${{building}}&limit=120`);
  const j = await r.json();
  const labels = j.points.map(p => p.ts);
  const data = j.points.map(p => p.energy);
  const ctx = document.getElementById('chart').getContext('2d');
  if (chart) chart.destroy();
  if (typeof Chart === 'undefined') {{
    // fallback: нарисовать простой SVG-like текст
    document.getElementById('chart').outerHTML =
      '<div>Не удалось загрузить Chart.js. Данные: '+
      data.length+' точек.</div>';
    return;
  }}
  chart = new Chart(ctx, {{
    type: 'line',
    data: {{ labels, datasets: [{{
      label: `ENERGY, kWh (здание ${{building}})`,
      data, borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,.15)',
      fill: true, tension: .2, pointRadius: 0,
    }}] }},
    options: {{ responsive: true, maintainAspectRatio: false,
      scales: {{ x: {{ ticks: {{ maxTicksLimit: 10 }} }} }} }}
  }});
}}

document.getElementById('f').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const f = new FormData(e.target);
  const body = {{
    building: f.get('building'),
    date: f.get('date'),
    horizon: Number(f.get('horizon')),
  }};
  const out = document.getElementById('out');
  out.innerHTML = 'Запрашиваем прогноз…';
  try {{
    const r = await fetch('/predict', {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify(body)
    }});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'ошибка сервиса');
    out.innerHTML = `<div class="result">
      <b>Прогноз на ${{j.horizon}} суток</b> (старт ${{j.date}}, здание ${{j.building}}):<br>
      сумма ENERGY = <b>${{j.prediction.toFixed(1)}} kWh</b><br>
      ${{j.actual !== null ? '<b>Факт:</b> '+j.actual.toFixed(1)+' kWh, <b>|ошибка|:</b> '+j.abs_error.toFixed(1)+'<br>' : ''}}
      <b>Модель:</b> ${{j.model}} ·
      <b>Кэш:</b> ${{j.cache_hit ? 'попадание' : 'промах'}} ·
      <b>Latency:</b> ${{j.latency_ms.toFixed(0)}} мс
    </div>`;
    refreshStats();
  }} catch (err) {{
    out.innerHTML = `<div class="err">Ошибка: ${{err.message}}</div>`;
  }}
}});

document.querySelector('[name=building]').addEventListener('change', e => {{
  refreshChart(e.target.value);
}});

refreshChart('A');
refreshStats();
</script>
"""
    return _page("Личный кабинет", body, user)


# ------------------------- HTTP Handler -----------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "energy-gateway/2.0"

    def log_message(self, fmt, *args) -> None:  # noqa: N802
        log.info("%s - %s", self.client_address[0], fmt % args)

    # ---- low-level helpers ----

    def _send_json(self, status: int, payload: dict,
                   extra_headers: list[tuple[str, str]] | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra_headers or []):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str,
                   ctype: str = "text/plain",
                   extra_headers: list[tuple[str, str]] | None = None) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra_headers or []):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str,
                  extra_headers: list[tuple[str, str]] | None = None) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        for k, v in (extra_headers or []):
            self.send_header(k, v)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _current_user(self) -> auth.UserCtx | None:
        cookies = auth.parse_cookies(self.headers.get("Cookie"))
        return auth.resolve_session(cookies.get(auth.SESSION_COOKIE))

    def _current_token(self) -> str | None:
        return auth.parse_cookies(self.headers.get("Cookie")).get(auth.SESSION_COOKIE)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or 0)
        return self.rfile.read(length) if length else b""

    def _parse_form(self) -> dict[str, str]:
        raw = self._read_body().decode("utf-8", errors="replace")
        parsed = parse_qs(raw, keep_blank_values=True)
        return {k: v[0] for k, v in parsed.items()}

    # ---- GET ----

    def do_GET(self) -> None:  # noqa: N802
        t0 = time.time()
        status = 200
        cache_hit = False
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            user = self._current_user()

            if path == "/":
                self._send_text(200, index_page(user), ctype="text/html")
            elif path == "/register":
                self._send_text(200, register_page(), ctype="text/html")
            elif path == "/login":
                self._send_text(200, login_page(), ctype="text/html")
            elif path == "/dashboard":
                if not user:
                    self._redirect("/login")
                    return
                self._send_text(200, dashboard_page(user), ctype="text/html")
            elif path == "/health":
                self._send_json(200, {
                    "status": "ok",
                    "cache_size": CACHE.size(),
                    "inference_url": INFERENCE_URL,
                    "users": db.count_users(),
                    "horizons": HORIZONS,
                })
            elif path == "/metrics":
                self._send_text(200, METRICS.prometheus())
            elif path == "/history":
                q = parse_qs(parsed.query)
                building = (q.get("building") or ["A"])[0].upper()
                limit = int((q.get("limit") or ["120"])[0])
                if building not in ("A", "B"):
                    status = 400
                    self._send_json(400, {"error": "building должен быть A или B"})
                    return
                daily = STORE.daily(building).tail(limit)
                self._send_json(200, {
                    "building": building,
                    "points": [
                        {"ts": str(r[DATE_COL].date()), "energy": float(r[TARGET])}
                        for _, r in daily.iterrows()
                    ],
                })
            elif path == "/my-history":
                if not user:
                    status = 401
                    self._send_json(401, {"error": "требуется авторизация"})
                    return
                q = parse_qs(parsed.query)
                limit = int((q.get("limit") or ["50"])[0])
                rows = db.list_history(user.id, limit=limit)
                hist = [
                    {
                        "id": r["id"],
                        "building": r["building"],
                        "timestamp": r["timestamp"],
                        "horizon": r["horizon"],
                        "prediction": r["prediction"],
                        "actual": r["actual"],
                        "abs_error": r["abs_error"],
                        "latency_ms": r["latency_ms"],
                        "created_at": r["created_at"],
                    }
                    for r in rows
                ]
                total = db.count_history(user.id)
                errs = [h["abs_error"] for h in hist if h["abs_error"] is not None]
                lats = [h["latency_ms"] for h in hist]
                stats = {
                    "total": total,
                    "avg_abs_error": sum(errs) / len(errs) if errs else None,
                    "avg_latency_ms": sum(lats) / len(lats) if lats else None,
                }
                self._send_json(200, {"history": hist, "stats": stats})
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
            parsed = urlparse(self.path)
            path = parsed.path
            user = self._current_user()

            if path == "/register":
                form = self._parse_form()
                username = form.get("username", "").strip()
                password = form.get("password", "")
                password2 = form.get("password2", "")
                if password != password2:
                    status = 400
                    self._send_text(400, register_page("Пароли не совпадают"),
                                    ctype="text/html")
                    return
                try:
                    uctx = auth.register_user(username, password)
                except auth.AuthError as exc:
                    status = 400
                    self._send_text(400, register_page(str(exc)),
                                    ctype="text/html")
                    return
                METRICS.record_registration()
                token = auth.create_session_token(uctx.id)
                self._redirect(
                    "/dashboard",
                    extra_headers=[("Set-Cookie", auth.session_cookie_header(token))],
                )

            elif path == "/login":
                form = self._parse_form()
                username = form.get("username", "").strip()
                password = form.get("password", "")
                try:
                    uctx = auth.authenticate(username, password)
                except auth.AuthError as exc:
                    METRICS.record_login(ok=False)
                    status = 401
                    self._send_text(401, login_page(str(exc)),
                                    ctype="text/html")
                    return
                METRICS.record_login(ok=True)
                token = auth.create_session_token(uctx.id)
                self._redirect(
                    "/dashboard",
                    extra_headers=[("Set-Cookie", auth.session_cookie_header(token))],
                )

            elif path == "/logout":
                token = self._current_token()
                auth.revoke_session(token)
                self._redirect(
                    "/",
                    extra_headers=[("Set-Cookie", auth.clear_cookie_header())],
                )

            elif path == "/predict":
                if not user:
                    status = 401
                    self._send_json(401, {"error": "требуется авторизация"})
                    return
                raw = self._read_body()
                try:
                    payload = json.loads(raw or b"{}")
                except json.JSONDecodeError as exc:
                    status = 400
                    self._send_json(400, {"error": f"invalid json: {exc}"})
                    return
                building = str(payload.get("building", "A")).upper()
                date_str = str(payload.get("date", ""))
                horizon = int(payload.get("horizon", 1))
                if building not in ("A", "B"):
                    status = 400
                    self._send_json(400, {"error": "building должен быть A или B"})
                    return
                if horizon not in HORIZONS:
                    status = 400
                    self._send_json(400, {"error": f"horizon ∈ {HORIZONS}"})
                    return
                if not date_str:
                    status = 400
                    self._send_json(400, {"error": "требуется date"})
                    return

                cache_key = f"{building}|{date_str}|{horizon}"
                cached = CACHE.get(cache_key)
                if cached is not None:
                    cache_hit = True
                    result = {**cached, "cache_hit": True,
                              "latency_ms": (time.time() - t0) * 1000}
                else:
                    METRICS.record_cache_miss()
                    try:
                        features, meta = lookup_features(building, date_str)
                    except ValueError as exc:
                        status = 422
                        self._send_json(422, {"error": str(exc)})
                        return
                    try:
                        infer_resp = call_inference(horizon, features)
                    except Exception as exc:  # noqa: BLE001
                        METRICS.record_inference_fail()
                        status = 502
                        log.exception("Inference недоступен: %s", exc)
                        self._send_json(502, {"error": f"inference error: {exc}"})
                        return
                    prediction = float(infer_resp["prediction"])
                    actual = actual_future_sum(building, date_str, horizon)
                    abs_err = abs(prediction - actual) if actual is not None else None
                    result = {
                        "building": building,
                        "date": date_str,
                        "horizon": horizon,
                        "prediction": prediction,
                        "actual": actual,
                        "abs_error": abs_err,
                        "model": infer_resp.get("model"),
                        "horizon_label": HORIZON_LABELS[horizon],
                    }
                    CACHE.set(cache_key, result)
                    result = {**result, "cache_hit": False,
                              "latency_ms": (time.time() - t0) * 1000}

                db.add_history(
                    user_id=user.id,
                    building=result["building"],
                    timestamp=result["date"],
                    horizon=result["horizon"],
                    prediction=float(result["prediction"]),
                    actual=result.get("actual"),
                    abs_error=result.get("abs_error"),
                    latency_ms=float(result["latency_ms"]),
                )
                self._send_json(200, result)

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
