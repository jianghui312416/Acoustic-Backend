"""注册、登录、个人信息与管理员审核接口。"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from . import auth_config, auth_database, auth_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Authentication"])

USER_STATUSES = {"pending", "active", "rejected", "disabled"}
USER_ROLES = {"user", "admin"}


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    email: str | None = Field(default=None, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=64)


class LoginRequest(BaseModel):
    username: str = Field(..., max_length=64)
    password: str = Field(..., max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., max_length=128)
    new_password: str = Field(..., min_length=8, max_length=128)


class RoleRequest(BaseModel):
    role: Literal["user", "admin"]


def _db_unavailable(exc: Exception) -> HTTPException:
    logger.exception("认证数据库操作失败")
    return HTTPException(status_code=503, detail=f"认证服务暂不可用：{exc}")


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "")[:64]


def _audit(
    request: Request,
    user: dict[str, Any] | None,
    action: str,
    target: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    try:
        auth_database.log_operation(
            int(user["id"]) if user else None,
            str(user.get("username")) if user else None,
            action,
            target,
            detail,
            _client_ip(request),
        )
    except Exception as exc:  # 审计日志失败不应阻断主流程
        logger.warning("写入操作日志失败：%s", exc)


def current_user(request: Request) -> dict[str, Any]:
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="未登录或登录已过期。",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def admin_user(request: Request) -> dict[str, Any]:
    user = current_user(request)
    if str(user.get("role")) != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限。")
    return user


@router.get("/auth/health", summary="认证服务健康检查")
def auth_health():
    try:
        auth_database.initialize()
        return {"ok": True, "database": "up"}
    except auth_database.AuthDatabaseError as exc:
        return {"ok": False, "database": "down", "message": str(exc)}
    except Exception as exc:
        return {"ok": False, "database": "down", "message": str(exc)}


@router.post("/auth/register", status_code=201, summary="注册账号")
def register(body: RegisterRequest, request: Request):
    username = body.username.strip()
    email = (body.email or "").strip() or None
    try:
        auth_service.validate_registration(username, email, body.password)
        if auth_database.get_user_by_username(username):
            raise HTTPException(status_code=409, detail="用户名已被注册。")
        if email and auth_database.get_user_by_email(email):
            raise HTTPException(status_code=409, detail="邮箱已被注册。")
        auto_approve = auth_config.registration_auto_approve()
        user = auth_database.create_user(
            username=username,
            email=email,
            password_hash=auth_service.hash_password(body.password),
            display_name=(body.display_name or "").strip() or username,
            status="active" if auto_approve else "pending",
            role="user",
        )
    except auth_service.AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except HTTPException:
        raise
    except auth_database.AuthDatabaseError as exc:
        raise _db_unavailable(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"注册失败：{exc}") from exc

    _audit(request, None, "register", username, {"email": email})
    return {
        "ok": True,
        "message": "注册成功，请等待管理员审核。" if user["status"] == "pending" else "注册成功。",
        "user": user,
    }


@router.post("/auth/login", summary="登录并签发访问令牌")
def login(body: LoginRequest, request: Request):
    try:
        user = auth_service.authenticate(body.username.strip(), body.password)
        token = auth_service.issue_token(
            user,
            request.headers.get("user-agent", ""),
            _client_ip(request),
        )
    except auth_service.AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except auth_database.AuthDatabaseError as exc:
        raise _db_unavailable(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"登录失败：{exc}") from exc
    _audit(request, user, "login")
    return {"ok": True, **token, "user": user}


@router.post("/auth/logout", summary="退出登录并吊销令牌")
def logout(request: Request):
    user = current_user(request)
    token = getattr(request.state, "auth_token", None)
    try:
        if token:
            auth_database.delete_token(auth_service.token_hash(token))
    except auth_database.AuthDatabaseError as exc:
        raise _db_unavailable(exc) from exc
    _audit(request, user, "logout")
    return {"ok": True}


@router.get("/auth/me", summary="读取当前登录用户")
def me(request: Request):
    user = current_user(request)
    return {"ok": True, "user": user}


@router.post("/auth/change-password", summary="修改当前账号密码")
def change_password(body: ChangePasswordRequest, request: Request):
    user = current_user(request)
    try:
        row = auth_database.get_user_by_id(int(user["id"]), include_hash=True)
        if not row or not auth_service.verify_password(body.current_password, row.get("password_hash")):
            raise HTTPException(status_code=400, detail="当前密码不正确。")
        if body.current_password == body.new_password:
            raise HTTPException(status_code=400, detail="新密码不能与当前密码相同。")
        auth_database.update_password(int(user["id"]), auth_service.hash_password(body.new_password))
        token = getattr(request.state, "auth_token", None)
        auth_database.delete_user_tokens(
            int(user["id"]),
            auth_service.token_hash(token) if token else None,
        )
    except HTTPException:
        raise
    except auth_database.AuthDatabaseError as exc:
        raise _db_unavailable(exc) from exc
    _audit(request, user, "change_password")
    return {"ok": True, "message": "密码已更新。"}


@router.get("/admin/users", summary="管理员查看用户列表")
def admin_list_users(
    request: Request,
    status: str | None = None,
    search: str | None = None,
    limit: int = 200,
):
    admin_user(request)
    if status and status not in USER_STATUSES:
        raise HTTPException(status_code=400, detail="状态筛选值无效。")
    try:
        users = auth_database.list_users(status=status, search=search, limit=limit)
        stats = auth_database.count_users()
    except auth_database.AuthDatabaseError as exc:
        raise _db_unavailable(exc) from exc
    return {"ok": True, "users": users, "stats": stats}


@router.get("/admin/stats", summary="管理员查看系统运行统计")
def admin_stats(request: Request):
    admin_user(request)
    try:
        return {
            "ok": True,
            "users": auth_database.count_users(),
            "jobs": auth_database.job_statistics(),
        }
    except auth_database.AuthDatabaseError as exc:
        raise _db_unavailable(exc) from exc


@router.get("/admin/operation-logs", summary="管理员查看操作日志")
def admin_logs(request: Request, limit: int | None = None):
    admin_user(request)
    try:
        return {"ok": True, "logs": auth_database.list_operation_logs(limit)}
    except auth_database.AuthDatabaseError as exc:
        raise _db_unavailable(exc) from exc


def _target_user(request: Request, user_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
    admin = admin_user(request)
    try:
        target = auth_database.get_user_by_id(user_id)
    except auth_database.AuthDatabaseError as exc:
        raise _db_unavailable(exc) from exc
    if not target:
        raise HTTPException(status_code=404, detail="用户不存在。")
    return admin, target


@router.post("/admin/users/{user_id}/approve", summary="审核通过注册申请")
def admin_approve(user_id: int, request: Request):
    admin, target = _target_user(request, user_id)
    try:
        user = auth_database.update_user_status(user_id, "active", int(admin["id"]))
    except auth_database.AuthDatabaseError as exc:
        raise _db_unavailable(exc) from exc
    _audit(request, admin, "approve_user", str(user_id), {"username": target["username"]})
    return {"ok": True, "user": user}


@router.post("/admin/users/{user_id}/reject", summary="拒绝注册申请")
def admin_reject(user_id: int, request: Request):
    admin, target = _target_user(request, user_id)
    if int(target["id"]) == int(admin["id"]):
        raise HTTPException(status_code=400, detail="不能拒绝自己的账号。")
    try:
        user = auth_database.update_user_status(user_id, "rejected", int(admin["id"]))
        auth_database.delete_user_tokens(user_id)
    except auth_database.AuthDatabaseError as exc:
        raise _db_unavailable(exc) from exc
    _audit(request, admin, "reject_user", str(user_id), {"username": target["username"]})
    return {"ok": True, "user": user}


@router.post("/admin/users/{user_id}/disable", summary="停用账号")
def admin_disable(user_id: int, request: Request):
    admin, target = _target_user(request, user_id)
    if int(target["id"]) == int(admin["id"]):
        raise HTTPException(status_code=400, detail="不能停用自己的账号。")
    try:
        user = auth_database.update_user_status(user_id, "disabled", int(admin["id"]))
        auth_database.delete_user_tokens(user_id)
    except auth_database.AuthDatabaseError as exc:
        raise _db_unavailable(exc) from exc
    _audit(request, admin, "disable_user", str(user_id), {"username": target["username"]})
    return {"ok": True, "user": user}


@router.post("/admin/users/{user_id}/enable", summary="启用账号")
def admin_enable(user_id: int, request: Request):
    admin, target = _target_user(request, user_id)
    try:
        user = auth_database.update_user_status(user_id, "active", int(admin["id"]))
    except auth_database.AuthDatabaseError as exc:
        raise _db_unavailable(exc) from exc
    _audit(request, admin, "enable_user", str(user_id), {"username": target["username"]})
    return {"ok": True, "user": user}


@router.post("/admin/users/{user_id}/role", summary="设置用户角色")
def admin_set_role(user_id: int, body: RoleRequest, request: Request):
    admin, target = _target_user(request, user_id)
    if int(target["id"]) == int(admin["id"]) and body.role != "admin":
        raise HTTPException(status_code=400, detail="不能取消自己的管理员角色。")
    try:
        user = auth_database.update_user_role(user_id, body.role)
        if body.role != "admin":
            auth_database.delete_user_tokens(user_id)
    except auth_database.AuthDatabaseError as exc:
        raise _db_unavailable(exc) from exc
    _audit(request, admin, "set_role", str(user_id), {"role": body.role})
    return {"ok": True, "user": user}
