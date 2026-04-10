"""Обучение мульти-горизонтных моделей энергопотребления.

Для каждого горизонта h ∈ {1, 7, 30} (день / неделя / месяц) обучается
отдельная модель: baseline-моделей — наивный «последнее значение * h»
и линейная регрессия, затем эксперименты с Ridge, решающим деревом и
градиентным бустингом (собственная реализация на numpy). Все прогоны
логируются в локальный JSON-трекер. Итоговые модели сохраняются в
`models/model.pkl` как словарь `{1: artefact, 7: artefact, 30: artefact}`.
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
    HORIZONS, HORIZON_LABELS,
)
from src.common.data_loader import load_building_daily
from src.common.features import build_features, get_feature_columns, target_column
from src.common.splits import time_based_split
from src.common.metrics import compute_metrics
from src.common.models import (
    LinearRegression, Ridge, DecisionTreeRegressor, GradientBoostingRegressor,
)
from src.common.tracker import Tracker


def make_xy(df: pd.DataFrame, y_col: str):
    feats = get_feature_columns()
    X = df[feats].values
    y = df[y_col].values
    return X, y, feats


def _evaluate(model, splits, y_col: str, tracker: Tracker, run_name: str,
              params: dict, horizon: int):
    X_tr, y_tr, _ = make_xy(splits.train, y_col)
    X_val, y_val, _ = make_xy(splits.val, y_col)
    X_te, y_te, _ = make_xy(splits.test, y_col)

    tracker.start_run(run_name)
    tracker.log_params({
        "model": run_name,
        "horizon": horizon,
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
        f"[h={horizon:>2}d {run_name:<20}] fit={fit_sec:5.1f}s "
        f"VAL MAE={m_val['MAE']:9.2f}  TEST MAE={m_te['MAE']:9.2f}  "
        f"RMSE={m_te['RMSE']:9.2f}  MAPE={m_te['MAPE']:5.2f}%  "
        f"R2={m_te['R2']:.3f}"
    )
    return {
        "model": model,
        "val_metrics": m_val,
        "test_metrics": m_te,
    }


def _experiments_for_horizon(splits, horizon: int, tracker: Tracker,
                             quick: bool) -> dict:
    y_col = target_column(horizon)
    results: dict = {}

    # --- Baseline 0: «последнее значение × horizon» ------------------------
    # предсказание на h дней = последнее наблюдение ENERGY * h
    tracker.start_run(f"h{horizon}_naive_last")
    tracker.log_params({"model": "naive_last", "horizon": horizon})
    naive_val = splits.val["lag_1"].values * horizon
    naive_te = splits.test["lag_1"].values * horizon
    m_val = compute_metrics(splits.val[y_col].values, naive_val)
    m_te = compute_metrics(splits.test[y_col].values, naive_te)
    tracker.log_metrics({f"val_{k}": v for k, v in m_val.items()})
    tracker.log_metrics({f"test_{k}": v for k, v in m_te.items()})
    tracker.end_run()
    print(
        f"[h={horizon:>2}d naive_last          ]              "
        f"VAL MAE={m_val['MAE']:9.2f}  TEST MAE={m_te['MAE']:9.2f}  "
        f"RMSE={m_te['RMSE']:9.2f}  MAPE={m_te['MAPE']:5.2f}%  "
        f"R2={m_te['R2']:.3f}"
    )
    results["naive_last"] = {"val_metrics": m_val, "test_metrics": m_te}

    # --- Baseline 1: сезонный наивный (lag_7 * h/7 округлено) ---------------
    if horizon >= 7:
        tracker.start_run(f"h{horizon}_naive_season")
        tracker.log_params({"model": "naive_season", "horizon": horizon})
        # среднее за последнюю неделю * horizon
        naive_val = splits.val["rolling_mean_7"].values * horizon
        naive_te = splits.test["rolling_mean_7"].values * horizon
        m_val = compute_metrics(splits.val[y_col].values, naive_val)
        m_te = compute_metrics(splits.test[y_col].values, naive_te)
        tracker.log_metrics({f"val_{k}": v for k, v in m_val.items()})
        tracker.log_metrics({f"test_{k}": v for k, v in m_te.items()})
        tracker.end_run()
        print(
            f"[h={horizon:>2}d naive_season        ]              "
            f"VAL MAE={m_val['MAE']:9.2f}  TEST MAE={m_te['MAE']:9.2f}  "
            f"RMSE={m_te['RMSE']:9.2f}  MAPE={m_te['MAPE']:5.2f}%  "
            f"R2={m_te['R2']:.3f}"
        )
        results["naive_season"] = {"val_metrics": m_val, "test_metrics": m_te}

    # --- Линейная регрессия -------------------------------------------------
    results["linear_regression"] = _evaluate(
        LinearRegression(), splits, y_col, tracker,
        f"h{horizon}_linear_regression", {}, horizon,
    )

    # --- Ridge --------------------------------------------------------------
    results["ridge_1.0"] = _evaluate(
        Ridge(alpha=1.0), splits, y_col, tracker,
        f"h{horizon}_ridge_1.0", {"alpha": 1.0}, horizon,
    )
    results["ridge_10"] = _evaluate(
        Ridge(alpha=10.0), splits, y_col, tracker,
        f"h{horizon}_ridge_10", {"alpha": 10.0}, horizon,
    )

    # --- Решающее дерево ----------------------------------------------------
    results["decision_tree"] = _evaluate(
        DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, max_bins=48,
                              random_state=RANDOM_STATE),
        splits, y_col, tracker, f"h{horizon}_decision_tree",
        {"max_depth": 8, "min_samples_leaf": 10, "max_bins": 48}, horizon,
    )

    # --- Градиентный бустинг (default) --------------------------------------
    gbr_default_params = dict(
        n_estimators=40 if quick else 120,
        learning_rate=0.08,
        max_depth=4,
        min_samples_leaf=15,
        subsample=0.9,
        max_bins=48,
        feature_subsample=0.8,
        random_state=RANDOM_STATE,
    )
    results["gbr_default"] = _evaluate(
        GradientBoostingRegressor(**gbr_default_params),
        splits, y_col, tracker, f"h{horizon}_gbr_default",
        gbr_default_params, horizon,
    )

    # --- Градиентный бустинг (tuned) ----------------------------------------
    gbr_tuned_params = dict(
        n_estimators=60 if quick else 200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=10,
        subsample=0.9,
        max_bins=64,
        feature_subsample=0.8,
        random_state=RANDOM_STATE,
    )
    results["gbr_tuned"] = _evaluate(
        GradientBoostingRegressor(**gbr_tuned_params),
        splits, y_col, tracker, f"h{horizon}_gbr_tuned",
        gbr_tuned_params, horizon,
    )

    return results


def run_all(building: str = DEFAULT_BUILDING, save_final: bool = True,
            quick: bool = False) -> dict:
    tracker = Tracker()

    df = load_building_daily(building)
    df_feat = build_features(df)
    splits = time_based_split(df_feat)
    print(
        f"Дней в сутках: {len(df)}. После FE: {len(df_feat)}.\n"
        f"Train={len(splits.train)} Val={len(splits.val)} Test={len(splits.test)}"
    )
    print(
        f"Период: train {splits.train[DATE_COL].min().date()} .. "
        f"{splits.train[DATE_COL].max().date()} | "
        f"val {splits.val[DATE_COL].min().date()} .. "
        f"{splits.val[DATE_COL].max().date()} | "
        f"test {splits.test[DATE_COL].min().date()} .. "
        f"{splits.test[DATE_COL].max().date()}"
    )

    all_results: dict = {}
    final_bundle: dict = {}

    for h in HORIZONS:
        print(f"\n===== Горизонт: {h} суток ({HORIZON_LABELS[h]}) =====")
        res = _experiments_for_horizon(splits, h, tracker, quick)
        all_results[h] = res

        scored = {
            k: v["val_metrics"]["MAE"]
            for k, v in res.items()
            if isinstance(v, dict) and "model" in v
        }
        best_name = min(scored, key=scored.get)
        best = res[best_name]
        print(f">>> h={h}d лучшая модель: {best_name} "
              f"(val MAE={scored[best_name]:.2f})")

        final_bundle[h] = {
            "model": best["model"],
            "model_name": best_name,
            "val_metrics": best["val_metrics"],
            "test_metrics": best["test_metrics"],
        }

    if save_final:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "model.pkl"
        with model_path.open("wb") as f:
            pickle.dump(
                {
                    "horizons": HORIZONS,
                    "models": final_bundle,
                    "feature_columns": get_feature_columns(),
                    "trained_at": datetime.utcnow().isoformat(),
                    "framework": "custom-numpy",
                    "building": building,
                },
                f,
            )
        print(f"\nФинальные модели сохранены: {model_path}")

        summary = {
            str(h): {
                "_final": final_bundle[h]["model_name"],
                "runs": {
                    k: {
                        "val_metrics": v.get("val_metrics"),
                        "test_metrics": v.get("test_metrics"),
                    }
                    for k, v in all_results[h].items()
                },
            }
            for h in HORIZONS
        }
        (REPORTS_DIR / "experiments.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Сводка экспериментов: {REPORTS_DIR / 'experiments.json'}")

    return {"results": all_results, "final": final_bundle, "splits": splits}


if __name__ == "__main__":
    run_all(quick=("--quick" in sys.argv))
