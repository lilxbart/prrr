"""SQLite-хранилище пользователей, сессий и истории прогнозов.

Используется только stdlib `sqlite3`, база лежит по пути из переменной
окружения `DB_PATH` (по умолчанию `data/app.db`). Этого достаточно для
демонстрации регистрации и персональной истории прогнозов; в
production-архитектуре (ЛР1–2) этот модуль заменяется на PostgreSQL
тем же контрактом функций.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable

from .config import DATA_DIR

DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "app.db"))
_LOCK = threading.Lock()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT NOT NULL UNIQUE,
    pw_hash    BLOB NOT NULL,
    salt       BLOB NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS prediction_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    building    TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    horizon     INTEGER NOT NULL,
    prediction  REAL NOT NULL,
    actual      REAL,
    abs_error   REAL,
    latency_ms  REAL,
    created_at  REAL NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_history_user ON prediction_history(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # WAL-mode не работает на некоторых FS/моунтах; в этом случае
    # откатываемся на обычный journal-режим.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:  # noqa: BLE001
        conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with _LOCK, get_conn() as conn:
        conn.executescript(SCHEMA)


# ----------------------------- users ---------------------------------------

def create_user(username: str, pw_hash: bytes, salt: bytes) -> int:
    with _LOCK, get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users(username, pw_hash, salt, created_at) VALUES(?,?,?,?)",
            (username, pw_hash, salt, time.time()),
        )
        return int(cur.lastrowid)


def get_user_by_username(username: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM users WHERE username=?", (username,))
        return cur.fetchone()


def get_user_by_id(user_id: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM users WHERE id=?", (user_id,))
        return cur.fetchone()


def count_users() -> int:
    with get_conn() as conn:
        cur = conn.execute("SELECT COUNT(*) AS c FROM users")
        return int(cur.fetchone()["c"])


# ----------------------------- sessions ------------------------------------

def create_session(token: str, user_id: int, ttl_sec: int) -> None:
    now = time.time()
    with _LOCK, get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions(token,user_id,created_at,expires_at) VALUES(?,?,?,?)",
            (token, user_id, now, now + ttl_sec),
        )


def get_session(token: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM sessions WHERE token=? AND expires_at>?",
            (token, time.time()),
        )
        return cur.fetchone()


def delete_session(token: str) -> None:
    with _LOCK, get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))


def cleanup_expired_sessions() -> int:
    with _LOCK, get_conn() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE expires_at<?", (time.time(),))
        return cur.rowcount


# ----------------------------- history -------------------------------------

def add_history(
    user_id: int,
    building: str,
    timestamp: str,
    horizon: int,
    prediction: float,
    actual: float | None,
    abs_error: float | None,
    latency_ms: float,
) -> int:
    with _LOCK, get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO prediction_history
               (user_id, building, timestamp, horizon, prediction, actual,
                abs_error, latency_ms, created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (user_id, building, timestamp, horizon, prediction, actual,
             abs_error, latency_ms, time.time()),
        )
        return int(cur.lastrowid)


def list_history(user_id: int, limit: int = 50) -> list[sqlite3.Row]:
    with get_conn() as conn:
        cur = conn.execute(
            """SELECT id, building, timestamp, horizon, prediction, actual,
                      abs_error, latency_ms, created_at
               FROM prediction_history
               WHERE user_id=?
               ORDER BY created_at DESC
               LIMIT ?""",
            (user_id, int(limit)),
        )
        return list(cur.fetchall())


def count_history(user_id: int) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT COUNT(*) AS c FROM prediction_history WHERE user_id=?",
            (user_id,),
        )
        return int(cur.fetchone()["c"])


def global_history_stats() -> dict:
    with get_conn() as conn:
        cur = conn.execute(
            """SELECT COUNT(*) AS total,
                      AVG(abs_error) AS avg_err,
                      AVG(latency_ms) AS avg_latency
               FROM prediction_history"""
        )
        row = cur.fetchone()
        return {
            "total": int(row["total"] or 0),
            "avg_abs_error": float(row["avg_err"] or 0.0),
            "avg_latency_ms": float(row["avg_latency"] or 0.0),
        }
