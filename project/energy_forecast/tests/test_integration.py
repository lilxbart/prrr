"""Интеграционные тесты: пайплайн признаков, мульти-горизонтный
inference, gateway с авторизацией и личной историей прогнозов.

Запуск: python -m tests.test_integration  (из корня проекта)
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Используем отдельную тестовую БД в /tmp — чтобы не затирать
# production app.db и чтобы избежать ограничений WAL на mount-точке.
import tempfile
_TEST_DB = Path(tempfile.gettempdir()) / f"energy_test_{os.getpid()}.db"
if _TEST_DB.exists():
    _TEST_DB.unlink()
os.environ["DB_PATH"] = str(_TEST_DB)

import numpy as np

from src.common.data_loader import load_building_daily
from src.common.features import build_features, get_feature_columns, target_column
from src.common.splits import time_based_split
from src.common.metrics import compute_metrics
from src.common import models as custom_models
from src.common import db, auth
from src.common.config import HORIZONS
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
    print("[1] Feature pipeline (daily)")
    df = load_building_daily("A")
    check("daily dataset loaded", 1000 < len(df) < 2500, f"rows={len(df)}")
    check("no NaN in ENERGY", df["ENERGY"].isna().sum() == 0)

    feat = build_features(df)
    check("feature frame non-empty", len(feat) > 0)
    feats = get_feature_columns()
    check("all feature columns present", all(c in feat.columns for c in feats))
    check("no NaN in features", feat[feats].isna().sum().sum() == 0)
    for h in HORIZONS:
        tc = target_column(h)
        check(f"target column present: {tc}", tc in feat.columns)
        check(f"target {tc} no NaN", feat[tc].isna().sum() == 0)


# --------------------------- 2. Train/val/test split ----------------------

def test_splits_no_leakage() -> None:
    print("[2] Temporal splits (ЛР2 стратегия)")
    df = build_features(load_building_daily("A"))
    s = time_based_split(df)
    check(
        "splits ordered",
        s.train["DATE"].max() < s.val["DATE"].min() < s.test["DATE"].min(),
    )
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
        ("DecisionTree", custom_models.DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=10)),
        ("GBR", custom_models.GradientBoostingRegressor(
            n_estimators=20, max_depth=3, min_samples_leaf=10)),
    ]:
        model.fit(X, y)
        pred = model.predict(X)
        r2 = compute_metrics(y, pred)["R2"]
        check(f"{name} fits synthetic data (R2>0.7)", r2 > 0.7, f"R2={r2:.3f}")


# --------------------------- 4. Auth primitives ---------------------------

def test_auth_primitives() -> None:
    print("[4] Auth primitives")
    db.init_db()
    # password hashing
    h1, salt = auth.hash_password("secret123")
    h2, _ = auth.hash_password("secret123", salt=salt)
    check("password hash deterministic with same salt", h1 == h2)
    check("verify ok", auth.verify_password("secret123", h1, salt))
    check("verify wrong fails", not auth.verify_password("wrong", h1, salt))

    # register / authenticate
    try:
        u = auth.register_user("alice_test", "pass12345")
        check("register returns user", u.username == "alice_test")
    except auth.AuthError as exc:
        check("register returns user", False, repr(exc))

    try:
        auth.register_user("alice_test", "pass12345")
        check("duplicate register rejected", False)
    except auth.AuthError:
        check("duplicate register rejected", True)

    try:
        auth.register_user("x", "short")
        check("short password rejected", False)
    except auth.AuthError:
        check("short password rejected", True)

    try:
        auth.authenticate("alice_test", "wrong")
        check("wrong password rejected", False)
    except auth.AuthError:
        check("wrong password rejected", True)

    ctx = auth.authenticate("alice_test", "pass12345")
    token = auth.create_session_token(ctx.id)
    resolved = auth.resolve_session(token)
    check("session resolves to user",
          resolved is not None and resolved.username == "alice_test")
    auth.revoke_session(token)
    check("session revoked", auth.resolve_session(token) is None)


# --------------------------- 5. Inference service ------------------------

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_inference_server() -> int:
    port = _free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), inf_app.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    return port


def test_inference_service() -> None:
    print("[5] Inference HTTP service")
    port = start_inference_server()
    base = f"http://127.0.0.1:{port}"

    # health
    with urllib.request.urlopen(f"{base}/health") as r:
        body = json.loads(r.read())
    check("health returns ok", body.get("status") == "ok")
    check("horizons in health", set(body["horizons"]) == set(HORIZONS))
    n_features = body["n_features"]

    # valid /predict per horizon
    for h in HORIZONS:
        feats = [0.0] * n_features
        req = urllib.request.Request(
            f"{base}/predict",
            data=json.dumps({"horizon": h, "features": feats}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as r:
            pred = json.loads(r.read())
        check(f"predict h={h} returns number",
              isinstance(pred.get("prediction"), (int, float)))
        check(f"predict h={h} echoes horizon", pred.get("horizon") == h)

    # unknown horizon
    try:
        req = urllib.request.Request(
            f"{base}/predict",
            data=json.dumps({"horizon": 99, "features": [0.0] * n_features}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req).read()
        check("unknown horizon rejected", False)
    except urllib.error.HTTPError as exc:
        check("unknown horizon rejected", exc.code == 422)

    # invalid features length
    try:
        req = urllib.request.Request(
            f"{base}/predict",
            data=json.dumps({"horizon": 1, "features": [0.0]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req).read()
        check("wrong feature size rejected", False)
    except urllib.error.HTTPError as exc:
        check("wrong feature size rejected", exc.code == 422)

    # metrics
    with urllib.request.urlopen(f"{base}/metrics") as r:
        body = r.read().decode()
    check("metrics contain inference_predict_total",
          "inference_predict_total" in body)
    check("metrics contain per-horizon counters",
          "inference_predict_by_horizon_total" in body)


# --------------------------- 6. Gateway end-to-end -----------------------

def start_gateway(inference_port: int) -> int:
    os.environ["INFERENCE_URL"] = f"http://127.0.0.1:{inference_port}"
    gw_app.INFERENCE_URL = f"http://127.0.0.1:{inference_port}"
    port = _free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), gw_app.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    return port


class _Client:
    """HTTP-клиент, умеющий хранить cookie между запросами."""

    def __init__(self, base: str):
        self.base = base
        self.cookies: dict[str, str] = {}

    def _merge_set_cookie(self, resp) -> None:
        for hdr, val in resp.getheaders():
            if hdr.lower() == "set-cookie":
                part = val.split(";", 1)[0]
                if "=" in part:
                    k, v = part.split("=", 1)
                    self.cookies[k.strip()] = v.strip()

    def _cookie_header(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items())

    def request(self, method: str, path: str, data: bytes | None = None,
                ctype: str | None = None, allow_redirect: bool = False):
        req = urllib.request.Request(f"{self.base}{path}", data=data, method=method)
        if ctype:
            req.add_header("Content-Type", ctype)
        if self.cookies:
            req.add_header("Cookie", self._cookie_header())

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **kw):  # noqa: D401
                return None

        if not allow_redirect:
            opener = urllib.request.build_opener(NoRedirect)
        else:
            opener = urllib.request.build_opener()
        try:
            resp = opener.open(req)
        except urllib.error.HTTPError as e:
            self._merge_set_cookie(e)
            return e
        self._merge_set_cookie(resp)
        return resp

    def get(self, path: str):
        return self.request("GET", path)

    def post_form(self, path: str, form: dict):
        from urllib.parse import urlencode
        body = urlencode(form).encode()
        return self.request("POST", path, data=body,
                            ctype="application/x-www-form-urlencoded")

    def post_json(self, path: str, obj: dict):
        body = json.dumps(obj).encode()
        return self.request("POST", path, data=body,
                            ctype="application/json")


def test_gateway_end_to_end() -> None:
    print("[6] Gateway end-to-end with auth")
    inf_port = start_inference_server()
    gw_port = start_gateway(inf_port)
    client = _Client(f"http://127.0.0.1:{gw_port}")

    # public pages
    r = client.get("/")
    check("index 200", r.status == 200)
    r = client.get("/register")
    check("register page 200", r.status == 200)
    r = client.get("/login")
    check("login page 200", r.status == 200)

    # /predict without auth -> 401
    r = client.post_json("/predict", {"building": "A", "date": "2020-06-15", "horizon": 1})
    check("predict without auth rejected", r.status == 401)

    # register
    username = f"bob_{int(time.time()*1000)}"
    r = client.post_form("/register", {
        "username": username, "password": "pass12345", "password2": "pass12345",
    })
    check("register redirect", r.status == 303, f"status={r.status}")
    check("session cookie set", auth.SESSION_COOKIE in client.cookies)

    # dashboard accessible
    r = client.get("/dashboard")
    check("dashboard 200 after register", r.status == 200)

    # protected prediction for each horizon
    first_pred = None
    for h in HORIZONS:
        r = client.post_json("/predict", {
            "building": "A", "date": "2020-06-15", "horizon": h,
        })
        check(f"authorized predict h={h}", r.status == 200)
        result = json.loads(r.read())
        check(f"prediction present h={h}", "prediction" in result)
        check(f"horizon echoed h={h}", result["horizon"] == h)
        if h == 1:
            first_pred = result["prediction"]

    # cache hit on repeat
    r = client.post_json("/predict", {
        "building": "A", "date": "2020-06-15", "horizon": 1,
    })
    cached = json.loads(r.read())
    check("cache hit on second call", cached.get("cache_hit") is True)
    check("cached prediction matches", cached["prediction"] == first_pred)

    # /history public
    r = client.get("/history?building=B&limit=30")
    hist = json.loads(r.read())
    check("history returns 30 points", len(hist.get("points", [])) == 30)

    # /my-history requires auth and lists entries
    r = client.get("/my-history?limit=10")
    mh = json.loads(r.read())
    check("my-history returns entries", len(mh["history"]) >= len(HORIZONS))
    check("my-history stats total>0", mh["stats"]["total"] > 0)

    # logout
    r = client.post_form("/logout", {})
    check("logout redirect", r.status == 303)
    r = client.get("/dashboard")
    check("dashboard requires auth after logout", r.status == 303)

    # wrong login
    r = client.post_form("/login", {
        "username": username, "password": "wrong",
    })
    check("login with wrong password fails", r.status == 401)

    # correct login
    r = client.post_form("/login", {
        "username": username, "password": "pass12345",
    })
    check("login ok", r.status == 303)
    r = client.get("/dashboard")
    check("dashboard again after login", r.status == 200)

    # metrics
    r = client.get("/metrics")
    m = r.read().decode()
    check("gateway metrics have login_success", "gateway_login_success_total" in m)
    check("gateway metrics have users_total", "gateway_users_total" in m)


def main() -> int:
    tests = [
        test_feature_pipeline,
        test_splits_no_leakage,
        test_custom_models_fit_predict,
        test_auth_primitives,
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
            import traceback
            traceback.print_exc()

    print("\n=============================")
    print(f"PASSED: {RESULTS['passed']}  FAILED: {RESULTS['failed']}")
    if RESULTS["failed"]:
        for name, detail in RESULTS["errors"]:
            print(f" - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
