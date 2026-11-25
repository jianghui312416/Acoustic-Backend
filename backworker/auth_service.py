"""密码、令牌与用户认证逻辑。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from . import auth_config, auth_database

PBKDF2_ITERATIONS = 260_000
PBKDF2_PREFIX = "pbkdf2_sha256"
DEFAULT_TOKEN_TTL_HOURS = 12


class AuthError(Exception):
    def __init__(self, message: str, status_code: int = 400, code: str = "auth_error"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{PBKDF2_PREFIX}${PBKDF2_ITERATIONS}${_b64encode(salt)}${_b64encode(digest)}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        prefix, iterations, salt, digest = stored.split("$", 3)
        if prefix != PBKDF2_PREFIX:
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            _b64decode(salt),
            int(iterations),
        )
        return hmac.compare_digest(candidate, _b64decode(digest))
    except (ValueError, TypeError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _token_ttl_hours() -> float:
    return auth_config.token_ttl_hours()


def issue_token(
    user: dict[str, Any],
    user_agent: str = "",
    ip: str = "",
) -> dict[str, Any]:
    token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=_token_ttl_hours())
    auth_database.create_token(token_hash(token), int(user["id"]), expires_at, user_agent, ip)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_at": expires_at.isoformat(),
        "expires_in": int((expires_at - datetime.now(timezone.utc)).total_seconds()),
    }


def authenticate(username: str, password: str) -> dict[str, Any]:
    user = auth_database.get_user_by_username(username, include_hash=True)
    if not user or not verify_password(password, user.get("password_hash")):
        raise AuthError("用户名或密码错误。", 401, "invalid_credentials")
    status = str(user.get("status") or "").lower()
    if status == "pending":
        raise AuthError("账号正在等待管理员审核，请稍后再试。", 403, "pending_approval")
    if status == "rejected":
        raise AuthError("账号注册申请已被拒绝，请联系管理员。", 403, "rejected")
    if status != "active":
        raise AuthError("账号已被停用，请联系管理员。", 403, "disabled")
    auth_database.touch_login(int(user["id"]))
    user.pop("password_hash", None)
    for key in ("created_at", "updated_at", "approved_at", "last_login_at"):
        value = user.get(key)
        if isinstance(value, datetime):
            user[key] = value.astimezone(timezone.utc).isoformat()
    return user


def resolve_token(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    user = auth_database.resolve_token(token_hash(token))
    if not user:
        return None
    if str(user.get("status") or "").lower() != "active":
        return None
    return user


def validate_registration(username: str, email: str | None, password: str) -> None:
    username = (username or "").strip()
    if not 3 <= len(username) <= 32:
        raise AuthError("用户名长度必须为 3 到 32 个字符。", 400, "invalid_username")
    if not all(ch.isalnum() or ch in "._-" for ch in username):
        raise AuthError("用户名只能包含字母、数字、点、下划线和短横线。", 400, "invalid_username")
    if email:
        email = email.strip()
        if "@" not in email or len(email) > 255:
            raise AuthError("邮箱格式无效。", 400, "invalid_email")
    if len(password or "") < 8:
        raise AuthError("密码至少需要 8 个字符。", 400, "weak_password")
    if len(password) > 128:
        raise AuthError("密码长度不能超过 128 个字符。", 400, "invalid_password")
