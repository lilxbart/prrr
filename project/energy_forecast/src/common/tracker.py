"""Локальный трекер экспериментов в JSON.

Интерфейс имитирует MLflow: start_run / log_param / log_metric / end_run.
Результаты сохраняются в reports/experiments.jsonl, что обеспечивает
воспроизводимость и аудит без необходимости поднимать MLflow-сервер.

В продакшн-контейнере tracker можно заменить на mlflow одной строкой
(см. src/training/train.py).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

from .config import REPORTS_DIR


LOG_PATH = REPORTS_DIR / "experiments.jsonl"


@dataclass
class Run:
    run_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    start_ts: float = field(default_factory=time.time)
    end_ts: float = 0.0

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
            "duration_sec": round(self.end_ts - self.start_ts, 3),
            "finished_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.end_ts or time.time())
            ),
        }


class Tracker:
    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._current: Run | None = None

    def start_run(self, run_name: str) -> Run:
        self._current = Run(run_name=run_name)
        return self._current

    def log_param(self, key: str, value: Any) -> None:
        if self._current:
            self._current.params[key] = value

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._current:
            self._current.params.update(params)

    def log_metric(self, key: str, value: float) -> None:
        if self._current:
            self._current.metrics[key] = float(value)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        if self._current:
            for k, v in metrics.items():
                self._current.metrics[k] = float(v)

    def set_tag(self, key: str, value: str) -> None:
        if self._current:
            self._current.tags[key] = value

    def end_run(self) -> None:
        if not self._current:
            return
        self._current.end_ts = time.time()
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._current.to_dict(), ensure_ascii=False) + "\n")
        self._current = None
