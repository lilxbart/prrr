"""Интеграционные тесты: пайплайн признаков, inference-сервис, gateway.

Запуск: python -m tests.test_integration  (из корня проекта)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.common.data_loader import load_building
from src.common.features import build_features, get_feature_columns
from src.common.splits import time_based_split
from src.common.metrics import compute_metrics
from src.common import models as custom_models
from src.inference import app as inf_app
from src.gateway import app as gw_app


RESULTS = {"passed": 0, "failed": 0, "errors": []}


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        RESULTS["passed"] += 1
        print(f"  ✓ {name}")
    else:
        RESULTS["failed"] += 1
        RESULTS["errors"].append((name, detail))
        print(f"  ✗ {name}  {detail}")


# --------------------------- 1. Feature pipeline ---------------------------

def test_feature_pipeline() -> None:
    print("[1] Feature pipeline")
    df = load_building("A")
    check("raw dataset loaded", len(df) > 40_000, f"rows={len(df)}")
    check("no NaN in ENERGY after cleaning", df["ENERGY"].isna().sum() == 0)

    feat = build_features(df)
    check("feature frame non-empty", len(feat) > 0)
    feats = get_feature_columns()
    check("all feature columns present", all(c in feat.columns for c in feats))
    check("no NaN in features", feat[feats].isna().sum().sum() == 0)


# --------------------------- 2. Train/val/test split ----------------------

def test_splits_no_leakage() -> None:
    print("[2] Temporal splits (ЛР2 стратегия)")
    df = build_features(load_building("A"))
    s = time_based_split(df)
    check("splits ordered", s.train["DATE"].max() < s.val["DATE"].min() < s.test["DATE"].min())
    check("train size > 0", len(s.train) > 0)
    check("val size > 0", len(s.val) > 0)
    check("test year == 2020", s.test["DATE"].dt.year.unique().tolist() == [2020])


# --------------------------- 3. Custom models -----------------------------

def test_custom_models_fit_predict() -> None:
    print("[3] Custom model implementations")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 5))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 1.5]) + rng.normal(scale=0.1, size=300)
    for name, model in [
        ("LinearRegression", custom_models.LinearRegression()),
        ("Ridge", custom_models.Ridge(alpha=1.0)),
        ("DecisionTree", custom_models.DecisionTreeRegressor(max_depth=5, min_samples_leaf=10)),
        ("GBR", custom_models.GradientBoostingRegressor(n_estimators=20, max_depth=3, min_samples_leaf=10)),
    ]:
        model.fit(X, y)
        pred = model.predict(X)
        r2 = compute_metrics(y, pred)["R2"]
        check(f"{name} fits synthetic data (R2>{0.7})", r2 > 0.7, f"R2={r2:.3f}")


# --------------------------- 4. Inference service ------------------------

def start_inference_server() -> int:
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", port), inf_app.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    # примитивный sleep для полного старта
    time.sleep(0.2)
    return port


def test_inference_service() -> None:
    print("[4] Inference HTTP service")
    port = start_inference_server()
    base = f"http://127.0.0.1:{port}"

    # health
    with urllib.request.urlopen(f"{base}/health") as r:
        body = json.loads(r.read())
    check("health returns ok", body.get("status") == "ok")
    check("model name present", "model" in body)
    n_features = body["n_features"]

    # /predict
    feats = [0.0] * n_features
    req = urllib.request.Request(
        f"{base}/predict",
        data=json.dumps({"features": feats}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        pred = json.loads(r.read())
    check("predict returns number", isinstance(pred.get("prediction"), (int, float)))

    # invalid features
    try:
        req = urllib.request.Request(
            f"{base}/predict",
            data=json.dumps({"features": [0.0]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req).read()
        check("predict with wrong size 422", False, "no error raised")
    except urllib.error.HTTPError as exc:
        check("predict with wrong size 422", exc.code == 422, f"code={exc.code}")

    # metrics
    with urllib.request.urlopen(f"{base}/metrics") as r:
        body = r.read().decode()
    check("metrics contain inference_requests_total",
          "inference_requests_total" in body)


# --------------------------- 5. Gateway end-to-end -----------------------

def start_gateway(inference_port: int) -> int:
    os.environ["INFERENCE_URL"] = f"http://127.0.0.1:{inference_port}"
    gw_app.INFERENCE_URL = f"http://127.0.0.1:{inference_port}"
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", port), gw_app.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    return port


def test_gateway_end_to_end() -> None:
    print("[5] Gateway end-to-end")
    inf_port = start_inference_server()
    gw_port = start_gateway(inf_port)
    base = f"http://127.0.0.1:{gw_port}"

    with urllib.request.urlopen(f"{base}/health") as r:
        check("gateway /health", json.loads(r.read()).get("status") == "ok")

    payload = {"building": "A", "timestamp": "2020-06-15T10:00:00", "horizon": 1}
    req = urllib.request.Request(
        f"{base}/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        result = json.loads(r.read())
    check("prediction returned", "prediction" in result, str(result))
    check("actual energy returned", "actual_energy" in result)
    check("cache miss on first call", result.get("cache_hit") is False)
    first_pred = result["prediction"]

    # второй запрос -> должен попасть в кэш
    with urllib.request.urlopen(req) as r:
        cached = json.loads(r.read())
    check("cache hit on second call", cached.get("cache_hit") is True)
    check("cached prediction matches", cached["prediction"] == first_pred)

    # /history
    with urllib.request.urlopen(f"{base}/history?building=B&limit=24") as r:
        hist = json.loads(r.read())
    check("history returns points", len(hist.get("points", [])) == 24)

    # /metrics
    with urllib.request.urlopen(f"{base}/metrics") as r:
        m = r.read().decode()
    check("gateway metrics have cache_hits", "gateway_cache_hits_total" in m)

    # корректность прогноза: абс ошибка не должна быть катастрофической
    check(
        "prediction within 3x stdev of actual",
        result["abs_error"] < 200,
        f"abs_error={result['abs_error']}",
    )


def main() -> int:
    tests = [
        test_feature_pipeline,
        test_splits_no_leakage,
        test_custom_models_fit_predict,
        test_inference_service,
        test_gateway_end_to_end,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            RESULTS["failed"] += 1
            RESULTS["errors"].append((t.__name__, repr(exc)))
            print(f"  !! {t.__name__} raised {exc!r}")

    print("\n=============================")
    print(f"PASSED: {RESULTS['passed']}  FAILED: {RESULTS['failed']}")
    if RESULTS["failed"]:
        for name, detail in RESULTS["errors"]:
            print(f" - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
