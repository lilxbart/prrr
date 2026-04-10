"""Главный скрипт обучения: baseline + эксперименты + финальная модель.

Для воспроизводимости используется локальный JSON-трекер
(src.common.tracker), совместимый по смыслу с MLflow. В
production-контейнере (см. Dockerfile.training) этот трекер можно
подменить на реальный mlflow.client без изменения интерфейса.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import (
    MODELS_DIR, REPORTS_DIR, TARGET, DATE_COL, RANDOM_STATE, DEFAULT_BUILDING,
)
from src.common.data_loader import load_building
from src.common.features import build_features, get_feature_columns
from src.common.splits import time_based_split
from src.common.metrics import compute_metrics
from src.common.models import (
    LinearRegression, Ridge, DecisionTreeRegressor, GradientBoostingRegressor,
)
from src.common.tracker import Tracker


def make_xy(df: pd.DataFrame):
    feats = get_feature_columns()
    X = df[feats].values
    y = df[TARGET].values
    return X, y, feats


def _evaluate(model, splits, tracker: Tracker, run_name: str, params: dict):
    X_tr, y_tr, _ = make_xy(splits.train)
    X_val, y_val, _ = make_xy(splits.val)
    X_te, y_te, _ = make_xy(splits.test)

    tracker.start_run(run_name)
    tracker.log_params({
        "model": run_name,
        "train_rows": len(splits.train),
        "val_rows": len(splits.val),
        "test_rows": len(splits.test),
        **params,
    })
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_sec = time.time() - t0
    tracker.log_metric("fit_sec", fit_sec)

    pred_val = model.predict(X_val)
    pred_te = model.predict(X_te)
    m_val = compute_metrics(y_val, pred_val)
    m_te = compute_metrics(y_te, pred_te)
    tracker.log_metrics({f"val_{k}": v for k, v in m_val.items()})
    tracker.log_metrics({f"test_{k}": v for k, v in m_te.items()})
    tracker.end_run()

    print(
        f"[{run_name}]  fit={fit_sec:5.1f}s  VAL MAE={m_val['MAE']:6.2f}  "
        f"TEST MAE={m_te['MAE']:6.2f}  RMSE={m_te['RMSE']:6.2f}  "
        f"MAPE={m_te['MAPE']:5.2f}%  R2={m_te['R2']:.3f}"
    )
    return {
        "model": model,
        "val_metrics": m_val,
        "test_metrics": m_te,
    }


def run_all(building: str = DEFAULT_BUILDING, save_final: bool = True,
            quick: bool = False) -> dict:
    """Запустить полный пайплайн экспериментов.

    Parameters
    ----------
    quick : bool
        Если True, градиентный бустинг обучается с меньшим числом деревьев
        (удобно для smoke-test).
    """
    tracker = Tracker()

    df = load_building(building)
    df_feat = build_features(df)
    splits = time_based_split(df_feat)
    print(
        f"Загружено {len(df)} строк. После FE: {len(df_feat)}.\n"
        f"Train={len(splits.train)} Val={len(splits.val)} Test={len(splits.test)}"
    )
    print(
        f"Период: train {splits.train[DATE_COL].min()} .. {splits.train[DATE_COL].max()} | "
        f"val {splits.val[DATE_COL].min()} .. {splits.val[DATE_COL].max()} | "
        f"test {splits.test[DATE_COL].min()} .. {splits.test[DATE_COL].max()}"
    )

    results: dict = {}

    # ---------------- Baseline 0: наивный прогноз (lag_24) -----------------
    tracker.start_run("naive_lag24")
    tracker.log_params({"model": "naive_lag24"})
    naive_val = splits.val["lag_24"].values
    naive_te = splits.test["lag_24"].values
    m_val = compute_metrics(splits.val[TARGET].values, naive_val)
    m_te = compute_metrics(splits.test[TARGET].values, naive_te)
    tracker.log_metrics({f"val_{k}": v for k, v in m_val.items()})
    tracker.log_metrics({f"test_{k}": v for k, v in m_te.items()})
    tracker.end_run()
    print(
        f"[naive_lag24]                VAL MAE={m_val['MAE']:6.2f}  "
        f"TEST MAE={m_te['MAE']:6.2f}  RMSE={m_te['RMSE']:6.2f}  "
        f"MAPE={m_te['MAPE']:5.2f}%  R2={m_te['R2']:.3f}"
    )
    results["naive_lag24"] = {"val_metrics": m_val, "test_metrics": m_te}

    # ---------------- Baseline 1: линейная регрессия -----------------------
    results["linear_regression"] = _evaluate(
        LinearRegression(), splits, tracker, "linear_regression", {}
    )

    # ---------------- Эксперимент: Ridge -----------------------------------
    results["ridge_1.0"] = _evaluate(
        Ridge(alpha=1.0), splits, tracker, "ridge_1.0", {"alpha": 1.0}
    )
    results["ridge_10"] = _evaluate(
        Ridge(alpha=10.0), splits, tracker, "ridge_10", {"alpha": 10.0}
    )

    # ---------------- Эксперимент: решающее дерево -------------------------
    results["decision_tree"] = _evaluate(
        DecisionTreeRegressor(max_depth=10, min_samples_leaf=50, max_bins=48,
                              random_state=RANDOM_STATE),
        splits, tracker, "decision_tree",
        {"max_depth": 10, "min_samples_leaf": 50, "max_bins": 48},
    )

    # ---------------- Эксперимент: градиентный бустинг ---------------------
    gbr_params_default = dict(
        n_estimators=60 if quick else 150,
        learning_rate=0.08,
        max_depth=4,
        min_samples_leaf=80,
        subsample=0.9,
        max_bins=48,
        feature_subsample=0.8,
        random_state=RANDOM_STATE,
    )
    results["gbr_default"] = _evaluate(
        GradientBoostingRegressor(**gbr_params_default),
        splits, tracker, "gbr_default", gbr_params_default,
    )

    gbr_params_tuned = dict(
        n_estimators=80 if quick else 250,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=60,
        subsample=0.9,
        max_bins=64,
        feature_subsample=0.8,
        random_state=RANDOM_STATE,
    )
    results["gbr_tuned"] = _evaluate(
        GradientBoostingRegressor(**gbr_params_tuned),
        splits, tracker, "gbr_tuned", gbr_params_tuned,
    )

    # ---------------- Выбор финальной модели -------------------------------
    scored = {
        k: v["val_metrics"]["MAE"]
        for k, v in results.items()
        if isinstance(v, dict) and "model" in v
    }
    best_name = min(scored, key=scored.get)
    best = results[best_name]
    print(f"\n>>> Лучшая модель по val MAE: {best_name}  "
          f"(val MAE={scored[best_name]:.2f})")

    if save_final and "model" in best:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "model.pkl"
        with model_path.open("wb") as f:
            pickle.dump(
                {
                    "model": best["model"],
                    "feature_columns": get_feature_columns(),
                    "model_name": best_name,
                    "trained_at": datetime.utcnow().isoformat(),
                    "val_metrics": best["val_metrics"],
                    "test_metrics": best["test_metrics"],
                    "framework": "custom-numpy",
                },
                f,
            )
        print(f"Финальная модель сохранена: {model_path}")

        summary = {
            k: {
                "val_metrics": v.get("val_metrics"),
                "test_metrics": v.get("test_metrics"),
            }
            for k, v in results.items()
        }
        summary["_final"] = best_name
        (REPORTS_DIR / "experiments.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Сводка экспериментов: {REPORTS_DIR / 'experiments.json'}")

    return {"results": results, "best_name": best_name, "splits": splits}


if __name__ == "__main__":
    run_all(quick=("--quick" in sys.argv))
