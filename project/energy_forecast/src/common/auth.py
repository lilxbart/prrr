"""Аутентификация: хэширование паролей, сессии, cookie.

Используется только stdlib: `hashlib.scrypt` для паролей, `secrets` для
токенов, простые функции парсинга cookie. Сессионный токен хранится в
HttpOnly-cookie `session`.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import time
from dataclasses import dataclass

from . import db

SESSION_COOKIE = "session"
SESSION_TTL_SEC = 7 * 24 * 3600  # 7 дней

# Параметры scrypt (соответствуют рекомендациям OWASP для CPU-cost N=2**14)
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{3,32}$")


class AuthError(Exception):
    pass


@dataclass
class UserCtx:
    id: int
    username: str


# ----------------------------- passwords ------------------------------------

def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN,
    )
    return dk, salt


def verify_password(password: str, pw_hash: bytes, salt: bytes) -> bool:
    dk, _ = hash_password(password, salt=salt)
    return hmac.compare_digest(dk, pw_hash)


# ----------------------------- validation -----------------------------------

def validate_username(username: str) -> None:
    if not _USERNAME_RE.match(username or ""):
        raise AuthError(
            "Имя пользователя: 3–32 символа, буквы/цифры/._-"
        )


def validate_password(password: str) -> None:
    if not password or len(password) < 6:
        raise AuthError("Пароль: минимум 6 символов")
    if len(password) > 128:
        raise AuthError("Пароль слишком длинный")


# ----------------------------- high-level API ------------------------------

def register_user(username: str, password: str) -> UserCtx:
    validate_username(username)
    validate_password(password)
    if db.get_user_by_username(username) is not None:
        raise AuthError("Пользователь уже существует")
    pw_hash, salt = hash_password(password)
    uid = db.create_user(username, pw_hash, salt)
    return UserCtx(id=uid, username=username)


def authenticate(username: str, password: str) -> UserCtx:
    row = db.get_user_by_username(username)
    if row is None:
        raise AuthError("Неверное имя пользователя или пароль")
    if not verify_password(password, row["pw_hash"], row["salt"]):
        raise AuthError("Неверное имя пользователя или пароль")
    return UserCtx(id=int(row["id"]), username=row["username"])


def create_session_token(user_id: int, ttl_sec: int = SESSION_TTL_SEC) -> str:
    token = secrets.token_urlsafe(32)
    db.create_session(token, user_id, ttl_sec)
    return token


def resolve_session(token: str | None) -> UserCtx | None:
    if not token:
        return None
    sess = db.get_session(token)
    if sess is None:
        return None
    user = db.get_user_by_id(int(sess["user_id"]))
    if user is None:
        return None
    return UserCtx(id=int(user["id"]), username=user["username"])


def revoke_session(token: str | None) -> None:
    if token:
        db.delete_session(token)


# ----------------------------- cookie helpers ------------------------------

def parse_cookies(header_value: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not header_value:
        return out
    for part in header_value.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def session_cookie_header(token: str, max_age: int = SESSION_TTL_SEC) -> str:
    return (
        f"{SESSION_COOKIE}={token}; HttpOnly; SameSite=Lax; Path=/; "
        f"Max-Age={max_age}"
    )


def clear_cookie_header() -> str:
    return f"{SESSION_COOKIE}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0"
