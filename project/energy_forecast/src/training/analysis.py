"""Анализ ошибок финальной модели и построение графиков."""
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

from src.common.config import MODELS_DIR, REPORTS_DIR, TARGET, DATE_COL, DEFAULT_BUILDING
from src.common.data_loader import load_building
from src.common.features import build_features
from src.common.splits import time_based_split
from src.common.metrics import compute_metrics


def analyze(building: str = DEFAULT_BUILDING) -> dict:
    with (MODELS_DIR / "model.pkl").open("rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feats = bundle["feature_columns"]

    df = load_building(building)
    df_feat = build_features(df)
    splits = time_based_split(df_feat)
    test = splits.test.copy()
    y_true = test[TARGET].values
    y_pred = model.predict(test[feats].values)
    test["pred"] = y_pred
    test["error"] = y_true - y_pred
    test["abs_error"] = np.abs(test["error"])

    metrics = compute_metrics(y_true, y_pred)
    print(
        f"Финальная модель: {bundle['model_name']}  "
        f"TEST MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
        f"MAPE={metrics['MAPE']:.2f}%  R2={metrics['R2']:.3f}"
    )

    plots_dir = REPORTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Факт vs прогноз (2 недели теста)
    sample = test.iloc[:24 * 14]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sample[DATE_COL], sample[TARGET], label="факт", color="#1f77b4")
    ax.plot(sample[DATE_COL], sample["pred"], label="прогноз",
            color="#ff7f0e", alpha=0.85)
    ax.set_title("Факт vs прогноз (первые 14 суток теста)")
    ax.set_ylabel("ENERGY, kWh")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plots_dir / "fact_vs_pred.png", dpi=120)
    plt.close(fig)

    # 2. Распределение остатков
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(test["error"], bins=60, color="#4c72b0", edgecolor="white")
    ax.axvline(0, color="k", linestyle="--")
    ax.set_title("Распределение остатков (факт − прогноз)")
    ax.set_xlabel("Ошибка, kWh")
    fig.tight_layout()
    fig.savefig(plots_dir / "error_hist.png", dpi=120)
    plt.close(fig)

    # 3. MAE по часу суток
    by_hour = test.groupby(test[DATE_COL].dt.hour)["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_hour.index, by_hour.values, color="#55a868")
    ax.set_title("Средняя абсолютная ошибка по часу суток")
    ax.set_xlabel("Час")
    ax.set_ylabel("MAE, kWh")
    fig.tight_layout()
    fig.savefig(plots_dir / "mae_by_hour.png", dpi=120)
    plt.close(fig)

    # 4. MAE по месяцу
    by_month = test.groupby(test[DATE_COL].dt.month)["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_month.index, by_month.values, color="#c44e52")
    ax.set_title("Средняя абсолютная ошибка по месяцу")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("MAE, kWh")
    fig.tight_layout()
    fig.savefig(plots_dir / "mae_by_month.png", dpi=120)
    plt.close(fig)

    # 5. Важность признаков
    imp = None
    if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
        imp = pd.Series(model.feature_importances_, index=feats).sort_values()
        top = imp.tail(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top.index, top.values, color="#8172b2")
        ax.set_title("Топ-15 важностей признаков")
        fig.tight_layout()
        fig.savefig(plots_dir / "feature_importance.png", dpi=120)
        plt.close(fig)
    elif hasattr(model, "coef_") and model.coef_ is not None:
        imp = pd.Series(np.abs(model.coef_), index=feats).sort_values()
        top = imp.tail(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top.index, top.values, color="#8172b2")
        ax.set_title("Топ-15 коэффициентов модели (|β|)")
        fig.tight_layout()
        fig.savefig(plots_dir / "feature_importance.png", dpi=120)
        plt.close(fig)

    worst = test.nlargest(10, "abs_error")[[DATE_COL, TARGET, "pred", "abs_error"]]
    worst.to_csv(REPORTS_DIR / "worst_cases.csv", index=False)

    summary_lines = [
        f"Модель: {bundle['model_name']}",
        f"Тестовый период: {test[DATE_COL].min()} .. {test[DATE_COL].max()}",
        f"Объём теста: {len(test)} наблюдений",
        f"MAE  = {metrics['MAE']:.3f} kWh",
        f"RMSE = {metrics['RMSE']:.3f} kWh",
        f"MAPE = {metrics['MAPE']:.3f}%",
        f"R2   = {metrics['R2']:.3f}",
        "",
        "MAE по часу суток:",
        by_hour.round(2).to_string(),
        "",
        "MAE по месяцу:",
        by_month.round(2).to_string(),
    ]
    (REPORTS_DIR / "analysis_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    print(f"Графики сохранены: {plots_dir}")
    return {"metrics": metrics, "by_hour": by_hour, "by_month": by_month}


if __name__ == "__main__":
    analyze()
