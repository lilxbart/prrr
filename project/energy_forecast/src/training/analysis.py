"""Мульти-горизонтный анализ ошибок финальных моделей.

Для каждого горизонта (1, 7, 30 дней) строятся:
  * график «факт vs прогноз» на тестовой выборке;
  * гистограмма остатков;
  * MAE по месяцу и по дню недели;
  * важности признаков модели.
Дополнительно сохраняется общий файл `reports/analysis_summary.txt`
и список 10 худших предсказаний каждого горизонта.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import (
    MODELS_DIR, REPORTS_DIR, TARGET, DATE_COL, DEFAULT_BUILDING,
    HORIZONS, HORIZON_LABELS,
)
from src.common.data_loader import load_building_daily
from src.common.features import build_features, target_column
from src.common.splits import time_based_split
from src.common.metrics import compute_metrics


def _plot_fact_vs_pred(test: pd.DataFrame, horizon: int, plots_dir: Path):
    n = min(60, len(test))
    sample = test.iloc[:n]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sample[DATE_COL], sample["y_true"], label="факт", color="#1f77b4")
    ax.plot(sample[DATE_COL], sample["y_pred"], label="прогноз",
            color="#ff7f0e", alpha=0.85)
    ax.set_title(
        f"Факт vs прогноз, горизонт {horizon} д. "
        f"({HORIZON_LABELS[horizon]}), первые {n} точек теста"
    )
    ax.set_ylabel("Сумма ENERGY на горизонте, kWh")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plots_dir / f"h{horizon}_fact_vs_pred.png", dpi=120)
    plt.close(fig)


def _plot_error_hist(test: pd.DataFrame, horizon: int, plots_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(test["error"], bins=40, color="#4c72b0", edgecolor="white")
    ax.axvline(0, color="k", linestyle="--")
    ax.set_title(f"Распределение остатков, h={horizon} д.")
    ax.set_xlabel("Ошибка, kWh")
    fig.tight_layout()
    fig.savefig(plots_dir / f"h{horizon}_error_hist.png", dpi=120)
    plt.close(fig)


def _plot_mae_by_month(test: pd.DataFrame, horizon: int, plots_dir: Path):
    by_month = test.groupby(test[DATE_COL].dt.month)["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_month.index, by_month.values, color="#c44e52")
    ax.set_title(f"MAE по месяцу старта прогноза, h={horizon} д.")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("MAE, kWh")
    fig.tight_layout()
    fig.savefig(plots_dir / f"h{horizon}_mae_by_month.png", dpi=120)
    plt.close(fig)
    return by_month


def _plot_mae_by_dow(test: pd.DataFrame, horizon: int, plots_dir: Path):
    by_dow = test.groupby(test[DATE_COL].dt.dayofweek)["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    ax.bar([labels[i] for i in by_dow.index], by_dow.values, color="#55a868")
    ax.set_title(f"MAE по дню недели старта прогноза, h={horizon} д.")
    ax.set_ylabel("MAE, kWh")
    fig.tight_layout()
    fig.savefig(plots_dir / f"h{horizon}_mae_by_dow.png", dpi=120)
    plt.close(fig)
    return by_dow


def _plot_feature_importance(model, feats: list[str], horizon: int,
                             plots_dir: Path):
    imp = None
    if getattr(model, "feature_importances_", None) is not None:
        imp = pd.Series(model.feature_importances_, index=feats).sort_values()
        title = f"Топ-15 важностей признаков, h={horizon} д."
    elif getattr(model, "coef_", None) is not None:
        imp = pd.Series(np.abs(model.coef_), index=feats).sort_values()
        title = f"Топ-15 коэффициентов модели (|β|), h={horizon} д."
    if imp is None:
        return
    top = imp.tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, color="#8172b2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(plots_dir / f"h{horizon}_feature_importance.png", dpi=120)
    plt.close(fig)


def analyze(building: str = DEFAULT_BUILDING) -> dict:
    with (MODELS_DIR / "model.pkl").open("rb") as f:
        bundle = pickle.load(f)
    feats = bundle["feature_columns"]
    models = bundle["models"]

    df = load_building_daily(building)
    df_feat = build_features(df)
    splits = time_based_split(df_feat)

    plots_dir = REPORTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_horizon = {}
    summary_lines = [f"Здание: {building}", f"Признаков: {len(feats)}", ""]

    for h in HORIZONS:
        art = models[h]
        model = art["model"]
        y_col = target_column(h)

        test = splits.test[[DATE_COL, y_col] + feats].copy()
        test = test.rename(columns={y_col: "y_true"})
        test["y_pred"] = model.predict(test[feats].values)
        test["error"] = test["y_true"] - test["y_pred"]
        test["abs_error"] = np.abs(test["error"])

        metrics = compute_metrics(test["y_true"].values, test["y_pred"].values)
        per_horizon[h] = {
            "model_name": art["model_name"],
            "metrics": metrics,
        }

        print(
            f"h={h:>2}d ({HORIZON_LABELS[h]})  model={art['model_name']:<20} "
            f"MAE={metrics['MAE']:9.2f}  RMSE={metrics['RMSE']:9.2f}  "
            f"MAPE={metrics['MAPE']:5.2f}%  R2={metrics['R2']:.3f}"
        )

        _plot_fact_vs_pred(test, h, plots_dir)
        _plot_error_hist(test, h, plots_dir)
        by_month = _plot_mae_by_month(test, h, plots_dir)
        by_dow = _plot_mae_by_dow(test, h, plots_dir)
        _plot_feature_importance(model, feats, h, plots_dir)

        worst = test.nlargest(10, "abs_error")[
            [DATE_COL, "y_true", "y_pred", "abs_error"]
        ]
        worst.to_csv(REPORTS_DIR / f"h{h}_worst_cases.csv", index=False)

        summary_lines += [
            f"--- Горизонт {h} д. ({HORIZON_LABELS[h]}) ---",
            f"Финальная модель: {art['model_name']}",
            f"MAE  = {metrics['MAE']:.2f} kWh",
            f"RMSE = {metrics['RMSE']:.2f} kWh",
            f"MAPE = {metrics['MAPE']:.2f}%",
            f"R²   = {metrics['R2']:.3f}",
            "MAE по месяцу:",
            by_month.round(2).to_string(),
            "",
        ]

    (REPORTS_DIR / "analysis_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    print(f"Графики сохранены: {plots_dir}")
    return per_horizon


if __name__ == "__main__":
    analyze()
