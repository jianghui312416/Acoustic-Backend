"""认证配置读取：统一使用 config/data_config.json 的 AUTH 段，不依赖 env 文件。"""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any, Mapping

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config" / "data_config.json"

DEFAULTS: dict[str, Any] = {
    "database": {
        "host": "127.0.0.1",
        "port": 5432,
        "dbname": "acoustic_auth",
        "user": "postgres",
        "password": "",
        "connect_timeout": 5,
        "pool_max": 10,
    },
    "admin": {
        "username": "admin",
        "password": "admin123456",
        "email": "",
    },
    "token_ttl_hours": 12,
    "registration_auto_approve": False,
    "max_concurrent_jobs": 4,
    "job_queue_wait_seconds": 600,
}

_LOCK = threading.Lock()
_CACHE: dict[str, Any] | None = None
_CACHE_MTIME: int | None = None


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _read_data_config() -> dict[str, Any]:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def settings() -> dict[str, Any]:
    """返回 AUTH 配置；修改 data_config.json 后自动重新读取。"""
    global _CACHE, _CACHE_MTIME
    try:
        mtime = CONFIG_PATH.stat().st_mtime_ns
    except OSError:
        mtime = 0
    with _LOCK:
        if _CACHE is not None and _CACHE_MTIME == mtime:
            return _CACHE
        config = _read_data_config()
        section = config.get("AUTH") if isinstance(config.get("AUTH"), Mapping) else {}
        merged = _deep_merge(copy.deepcopy(DEFAULTS), section)
        services = config.get("DATA_SERVICES") if isinstance(config.get("DATA_SERVICES"), Mapping) else {}
        api1 = services.get("api1") if isinstance(services.get("api1"), Mapping) else {}
        # 数据库主机/账号为空时继承 api1，避免额外维护一份连接信息。
        database = merged.setdefault("database", {})
        for key in ("host", "port", "user", "password"):
            if not database.get(key) and api1.get(key) not in (None, ""):
                database[key] = api1[key]
        _CACHE = merged
        _CACHE_MTIME = mtime
        return merged


def database_options() -> dict[str, Any]:
    database = settings().get("database") or {}
    return {
        "host": str(database.get("host") or "127.0.0.1"),
        "port": int(database.get("port") or 5432),
        "dbname": str(database.get("dbname") or "acoustic_auth"),
        "user": str(database.get("user") or "postgres"),
        "password": str(database.get("password") or ""),
        "connect_timeout": int(database.get("connect_timeout") or 5),
    }


def pool_max() -> int:
    database = settings().get("database") or {}
    return max(1, int(database.get("pool_max") or 10))


def admin_credentials() -> tuple[str, str, str | None]:
    admin = settings().get("admin") or {}
    username = str(admin.get("username") or "admin")
    password = str(admin.get("password") or "admin123456")
    email = str(admin.get("email") or "").strip() or None
    return username, password, email


def token_ttl_hours() -> float:
    try:
        value = float(settings().get("token_ttl_hours") or 12)
    except (TypeError, ValueError):
        value = 12.0
    return value if value > 0 else 12.0


def registration_auto_approve() -> bool:
    return bool(settings().get("registration_auto_approve"))


def max_concurrent_jobs() -> int:
    try:
        return max(1, int(settings().get("max_concurrent_jobs") or 4))
    except (TypeError, ValueError):
        return 4


def job_queue_wait_seconds() -> float:
    try:
        value = float(settings().get("job_queue_wait_seconds") or 600)
    except (TypeError, ValueError):
        value = 600.0
    return value if value > 0 else 600.0
