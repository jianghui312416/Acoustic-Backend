"""在计算调用链中传递当前登录用户，便于登记任务归属。"""

from __future__ import annotations

import logging
from contextvars import ContextVar

from . import auth_database

logger = logging.getLogger(__name__)

_CURRENT_USER_ID: ContextVar[int | None] = ContextVar("acoustic_auth_user_id", default=None)


def set_current_user_id(user_id: int | None):
    return _CURRENT_USER_ID.set(user_id)


def reset_current_user_id(token) -> None:
    try:
        _CURRENT_USER_ID.reset(token)
    except (ValueError, RuntimeError):
        pass


def current_user_id() -> int | None:
    return _CURRENT_USER_ID.get()


def register_job(job_id: str | None, plugin: str | None, status: str = "completed", user_id: int | None = None) -> None:
    if not job_id:
        return
    owner = user_id if user_id is not None else current_user_id()
    try:
        auth_database.register_job(job_id, owner, plugin)
        if status:
            auth_database.set_job_status(job_id, status)
    except Exception as exc:
        logger.warning("登记任务归属失败 job=%s: %s", job_id, exc)
