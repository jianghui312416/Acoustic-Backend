"""全局 HTTP 认证中间件。

无论前端是否启用了路由守卫，所有业务接口都会在这里校验访问令牌，
避免绕过 Vue 页面直接调用后端接口。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from . import auth_database, auth_service

logger = logging.getLogger(__name__)

PUBLIC_PATHS = {
    "/",
    "/health",
    "/favicon.ico",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/auth/health",
    "/auth/login",
    "/auth/register",
}

PUBLIC_PREFIXES = ("/docs", "/redoc", "/openapi.json", "/static")

LOG_PREFIXES = (
    "/file/",
    "/download_results/",
    "/download_job_logs/",
    "/job_diagnostics/",
    "/array-channel/",
    "/reverberation/",
    "/ws/",
)
LOG_EXACT = {"/", "/compute"}

# 工作区刷新会批量读取这些任务元数据文件，属于高频内部轮询，不计入管理员操作日志。
WORKSPACE_META_FILES = ("params.json", "job_status.json")


def _is_workspace_meta_read(path: str, method: str) -> bool:
    if method != "GET" or not path.startswith("/file/"):
        return False
    return path.rsplit("/", 1)[-1] in WORKSPACE_META_FILES


def _cors_headers(request: Request) -> dict[str, str]:
    origin = request.headers.get("origin") or "*"
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Bellhop-Explicit-Submit",
        "Access-Control-Max-Age": "86400",
        "Vary": "Origin",
    }


def extract_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
        if token:
            return token
    for name in ("access_token", "token"):
        value = request.query_params.get(name)
        if value:
            return value.strip()
    return None


def is_public_path(path: str, method: str) -> bool:
    """``GET /`` 只是给浏览器探测服务使用，其余统一入口仍需登录。"""
    if method == "OPTIONS":
        return True
    if path in PUBLIC_PATHS and not (path == "/" and method == "POST"):
        return True
    return any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES)


def _should_log(path: str, method: str) -> bool:
    if method == "OPTIONS":
        return False
    if _is_workspace_meta_read(path, method):
        return False
    return path in LOG_EXACT or any(path.startswith(prefix) for prefix in LOG_PREFIXES)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path or "/"
        method = request.method.upper()

        if is_public_path(path, method):
            return await call_next(request)

        token = extract_token(request)
        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "未登录或登录已过期。", "code": "unauthorized"},
                headers={**_cors_headers(request), "WWW-Authenticate": "Bearer"},
            )

        try:
            user = await run_in_threadpool(auth_service.resolve_token, token)
        except auth_database.AuthDatabaseError as exc:
            logger.exception("认证数据库不可用")
            return JSONResponse(
                status_code=503,
                content={"detail": f"认证服务暂不可用：{exc}", "code": "auth_unavailable"},
                headers=_cors_headers(request),
            )

        if not user:
            return JSONResponse(
                status_code=401,
                content={"detail": "登录已失效，请重新登录。", "code": "token_invalid"},
                headers={**_cors_headers(request), "WWW-Authenticate": "Bearer"},
            )

        request.state.user = user
        request.state.auth_token = token

        if path.startswith("/admin") and str(user.get("role")) != "admin":
            return JSONResponse(
                status_code=403,
                content={"detail": "需要管理员权限。", "code": "forbidden"},
                headers=_cors_headers(request),
            )

        response = await call_next(request)

        if _should_log(path, method):
            detail: dict[str, Any] = {"method": method, "path": path, "status": response.status_code}
            try:
                await run_in_threadpool(
                    auth_database.log_operation,
                    int(user["id"]),
                    str(user.get("username")),
                    "api_request",
                    path,
                    detail,
                    (request.client.host if request.client else "")[:64],
                )
            except Exception as exc:
                logger.debug("记录接口操作日志失败：%s", exc)
        return response
