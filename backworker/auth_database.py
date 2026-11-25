"""PostgreSQL 用户、令牌、任务归属与操作日志存储。

认证表统一使用 ``auth_`` 前缀，连接信息读取 ``config/data_config.json``
的 ``AUTH`` 段，使用独立数据库 ``acoustic_auth``，不污染 api1 环境数据表。
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from . import auth_config

logger = logging.getLogger(__name__)


class AuthDatabaseError(RuntimeError):
    """认证数据库不可用或查询失败。"""


_POOL_LOCK = threading.Lock()
_POOL = None
_POOL_KEY: tuple[Any, ...] | None = None
_SCHEMA_READY = False
_CONFIGURED = False

SCHEMA_SQL = (
    """
    CREATE TABLE IF NOT EXISTS auth_users (
        id BIGSERIAL PRIMARY KEY,
        username VARCHAR(64) NOT NULL UNIQUE,
        email VARCHAR(255) UNIQUE,
        password_hash TEXT NOT NULL,
        role VARCHAR(16) NOT NULL DEFAULT 'user',
        status VARCHAR(16) NOT NULL DEFAULT 'pending',
        display_name VARCHAR(128),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        approved_at TIMESTAMPTZ,
        approved_by BIGINT REFERENCES auth_users(id) ON DELETE SET NULL,
        last_login_at TIMESTAMPTZ
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS auth_tokens (
        token_hash CHAR(64) PRIMARY KEY,
        user_id BIGINT NOT NULL REFERENCES auth_users(id) ON DELETE CASCADE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMPTZ NOT NULL,
        user_agent TEXT,
        ip VARCHAR(64)
    )
    """,
    "CREATE INDEX IF NOT EXISTS auth_tokens_user_idx ON auth_tokens (user_id)",
    "CREATE INDEX IF NOT EXISTS auth_tokens_expires_idx ON auth_tokens (expires_at)",
    """
    CREATE TABLE IF NOT EXISTS auth_jobs (
        job_id VARCHAR(64) PRIMARY KEY,
        user_id BIGINT REFERENCES auth_users(id) ON DELETE SET NULL,
        plugin VARCHAR(64),
        status VARCHAR(24) NOT NULL DEFAULT 'running',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS auth_jobs_user_idx ON auth_jobs (user_id)",
    """
    CREATE TABLE IF NOT EXISTS auth_operation_logs (
        id BIGSERIAL PRIMARY KEY,
        user_id BIGINT REFERENCES auth_users(id) ON DELETE SET NULL,
        username VARCHAR(64),
        action VARCHAR(64) NOT NULL,
        target VARCHAR(160),
        detail JSONB,
        ip VARCHAR(64),
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS auth_logs_user_idx ON auth_operation_logs (user_id)",
    "CREATE INDEX IF NOT EXISTS auth_logs_created_idx ON auth_operation_logs (created_at DESC)",
)


def _database_options() -> dict[str, Any]:
    """认证库连接参数：data_config.json 的 AUTH.database。"""
    return auth_config.database_options()


def _connect_kwargs(options: Mapping[str, Any]) -> dict[str, Any]:
    if options.get("dsn"):
        return {"dsn": options["dsn"], "connect_timeout": 5}
    return dict(options)


def _ensure_database() -> None:
    """认证库不存在时自动创建；表仍由 initialize() 建在独立库中。"""
    options = _database_options()
    if options.get("dsn"):
        try:
            import psycopg2.extensions

            parsed = dict(psycopg2.extensions.parse_dsn(options["dsn"]))
        except Exception:
            return
        target = str(parsed.get("dbname") or "").strip()
        admin = {k: v for k, v in parsed.items() if k != "dbname"}
    else:
        target = str(options.get("dbname") or "").strip()
        if not target or target in {"postgres", "template1"}:
            return
        admin = dict(options)
        admin["dbname"] = "postgres"
    if not target:
        return

    import psycopg2
    from psycopg2 import sql

    try:
        conn = psycopg2.connect(**admin)
    except Exception as exc:
        raise AuthDatabaseError(
            f"无法连接 PostgreSQL 维护库来创建 {target}：{exc}。"
            "请确认 AUTH_DB_* 配置，或先手动创建该数据库。"
        ) from exc
    conn.autocommit = True
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target,))
            if cursor.fetchone():
                return
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target)))
            logger.info("已创建认证数据库 %s", target)
    finally:
        conn.close()


def _get_pool():
    global _POOL, _POOL_KEY
    options = _database_options()
    key = tuple(sorted(options.items()))
    with _POOL_LOCK:
        if _POOL is not None and _POOL_KEY == key:
            return _POOL
        if _POOL is not None:
            try:
                _POOL.closeall()
            except Exception:
                pass
        try:
            import psycopg2.pool
        except ImportError as exc:  # pragma: no cover - 由 requirements 保证
            raise AuthDatabaseError("缺少 psycopg2，无法连接 PostgreSQL。") from exc
        try:
            _POOL = psycopg2.pool.ThreadedConnectionPool(
                1,
                auth_config.pool_max(),
                **{k: v for k, v in _connect_kwargs(options).items() if k != "connect_timeout"},
                connect_timeout=int(_connect_kwargs(options).get("connect_timeout", 5)),
            )
        except Exception as exc:
            _POOL = None
            raise AuthDatabaseError(f"无法连接认证数据库：{exc}") from exc
        _POOL_KEY = key
        return _POOL


@contextmanager
def connection() -> Iterator[Any]:
    pool = _get_pool()
    conn = None
    try:
        conn = pool.getconn()
    except Exception as exc:
        raise AuthDatabaseError(f"获取认证数据库连接失败：{exc}") from exc
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            pool.putconn(conn)
        except Exception:
            pass


def _row(cursor: Any) -> dict[str, Any] | None:
    row = cursor.fetchone()
    return dict(row) if row else None


def _rows(cursor: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in cursor.fetchall()]


def initialize() -> None:
    """创建认证表，并在空库中生成初始管理员。"""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    _ensure_database()
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(918273645)")
            for statement in SCHEMA_SQL:
                cursor.execute(statement)
            cursor.execute("SELECT COUNT(*) FROM auth_users")
            count = int(cursor.fetchone()[0])
            if count == 0:
                from . import auth_service

                username, password, email = auth_config.admin_credentials()
                cursor.execute(
                    """
                    INSERT INTO auth_users
                        (username, email, password_hash, role, status, display_name, approved_at)
                    VALUES (%s, %s, %s, 'admin', 'active', %s, NOW())
                    ON CONFLICT (username) DO NOTHING
                    """,
                    (username, email, auth_service.hash_password(password), "管理员"),
                )
                logger.warning(
                    "已创建初始管理员账号 %s，请登录后立即修改密码。", username
                )
    _SCHEMA_READY = True


def _serialize_user(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    data = dict(row)
    data.pop("password_hash", None)
    for key in ("created_at", "updated_at", "approved_at", "last_login_at"):
        value = data.get(key)
        if isinstance(value, datetime):
            data[key] = value.astimezone(timezone.utc).isoformat()
    return data


def get_user_by_username(username: str, *, include_hash: bool = False) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor(cursor_factory=_dict_cursor()) as cursor:
            cursor.execute("SELECT * FROM auth_users WHERE lower(username) = lower(%s)", (username,))
            row = _row(cursor)
    return row if include_hash else _serialize_user(row)


def get_user_by_email(email: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor(cursor_factory=_dict_cursor()) as cursor:
            cursor.execute("SELECT * FROM auth_users WHERE lower(email) = lower(%s)", (email,))
            return _serialize_user(_row(cursor))


def get_user_by_id(user_id: int, *, include_hash: bool = False) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor(cursor_factory=_dict_cursor()) as cursor:
            cursor.execute("SELECT * FROM auth_users WHERE id = %s", (int(user_id),))
            row = _row(cursor)
    return row if include_hash else _serialize_user(row)


def create_user(
    username: str,
    email: str | None,
    password_hash: str,
    display_name: str | None = None,
    status: str = "pending",
    role: str = "user",
) -> dict[str, Any]:
    with connection() as conn:
        with conn.cursor(cursor_factory=_dict_cursor()) as cursor:
            cursor.execute(
                """
                INSERT INTO auth_users (username, email, password_hash, role, status, display_name)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (username, email, password_hash, role, status, display_name),
            )
            return _serialize_user(_row(cursor))


def list_users(status: str | None = None, search: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    clauses: list[str] = []
    args: list[Any] = []
    if status:
        clauses.append("status = %s")
        args.append(status)
    if search:
        clauses.append("(lower(username) LIKE %s OR lower(coalesce(email,'')) LIKE %s)")
        token = f"%{search.lower()}%"
        args.extend([token, token])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    args.append(max(1, min(1000, int(limit))))
    with connection() as conn:
        with conn.cursor(cursor_factory=_dict_cursor()) as cursor:
            cursor.execute(
                f"SELECT * FROM auth_users {where} ORDER BY created_at DESC LIMIT %s",
                tuple(args),
            )
            return [_serialize_user(item) for item in _rows(cursor)]


def update_user_status(user_id: int, status: str, approved_by: int | None = None) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor(cursor_factory=_dict_cursor()) as cursor:
            cursor.execute(
                """
                UPDATE auth_users
                SET status = %s,
                    updated_at = NOW(),
                    approved_at = CASE WHEN %s = 'active' THEN NOW() ELSE approved_at END,
                    approved_by = CASE WHEN %s = 'active' THEN %s ELSE approved_by END
                WHERE id = %s
                RETURNING *
                """,
                (status, status, status, approved_by, int(user_id)),
            )
            return _serialize_user(_row(cursor))


def update_user_role(user_id: int, role: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor(cursor_factory=_dict_cursor()) as cursor:
            cursor.execute(
                "UPDATE auth_users SET role = %s, updated_at = NOW() WHERE id = %s RETURNING *",
                (role, int(user_id)),
            )
            return _serialize_user(_row(cursor))


def update_password(user_id: int, password_hash: str) -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE auth_users SET password_hash = %s, updated_at = NOW() WHERE id = %s",
                (password_hash, int(user_id)),
            )


def touch_login(user_id: int) -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE auth_users SET last_login_at = NOW() WHERE id = %s", (int(user_id),))


def count_users() -> dict[str, int]:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT status, COUNT(*) FROM auth_users GROUP BY status"
            )
            result = {str(status): int(count) for status, count in cursor.fetchall()}
    result["total"] = sum(result.values())
    return result


def create_token(token_hash: str, user_id: int, expires_at: datetime, user_agent: str, ip: str) -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO auth_tokens (token_hash, user_id, expires_at, user_agent, ip)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (token_hash, int(user_id), expires_at, user_agent[:1000], ip[:64]),
            )


def delete_token(token_hash: str) -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM auth_tokens WHERE token_hash = %s", (token_hash,))


def delete_user_tokens(user_id: int, keep_hash: str | None = None) -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            if keep_hash:
                cursor.execute(
                    "DELETE FROM auth_tokens WHERE user_id = %s AND token_hash <> %s",
                    (int(user_id), keep_hash),
                )
            else:
                cursor.execute("DELETE FROM auth_tokens WHERE user_id = %s", (int(user_id),))


def delete_expired_tokens() -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM auth_tokens WHERE expires_at < NOW()")


def resolve_token(token_hash: str) -> dict[str, Any] | None:
    with connection() as conn:
        with conn.cursor(cursor_factory=_dict_cursor()) as cursor:
            cursor.execute(
                """
                SELECT u.*, t.expires_at AS token_expires_at
                FROM auth_tokens t
                JOIN auth_users u ON u.id = t.user_id
                WHERE t.token_hash = %s AND t.expires_at > NOW()
                """,
                (token_hash,),
            )
            row = _row(cursor)
    if not row:
        return None
    expires = row.pop("token_expires_at", None)
    user = _serialize_user(row)
    if expires is not None:
        user["token_expires_at"] = expires.astimezone(timezone.utc).isoformat()
    return user


def register_job(job_id: str, user_id: int | None, plugin: str | None = None) -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO auth_jobs (job_id, user_id, plugin, status)
                VALUES (%s, %s, %s, 'running')
                ON CONFLICT (job_id) DO UPDATE
                SET user_id = COALESCE(auth_jobs.user_id, EXCLUDED.user_id),
                    plugin = COALESCE(EXCLUDED.plugin, auth_jobs.plugin),
                    updated_at = NOW()
                """,
                (job_id, user_id, plugin),
            )


def set_job_status(job_id: str, status: str) -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE auth_jobs SET status = %s, updated_at = NOW() WHERE job_id = %s",
                (status, job_id),
            )


def job_owner(job_id: str) -> int | None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT user_id FROM auth_jobs WHERE job_id = %s", (job_id,))
            row = cursor.fetchone()
    return int(row[0]) if row and row[0] is not None else None


def job_ids_for_user(user_id: int) -> set[str]:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT job_id FROM auth_jobs WHERE user_id = %s", (int(user_id),))
            return {str(row[0]) for row in cursor.fetchall()}


def count_active_jobs(user_id: int) -> int:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM auth_jobs WHERE user_id = %s AND status IN ('queued','running')",
                (int(user_id),),
            )
            return int(cursor.fetchone()[0])


def job_statistics() -> dict[str, int]:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT status, COUNT(*) FROM auth_jobs GROUP BY status")
            return {str(status): int(count) for status, count in cursor.fetchall()}


def delete_job(job_id: str) -> None:
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM auth_jobs WHERE job_id = %s", (job_id,))


def log_operation(
    user_id: int | None,
    username: str | None,
    action: str,
    target: str | None = None,
    detail: Mapping[str, Any] | None = None,
    ip: str | None = None,
) -> None:
    try:
        from psycopg2.extras import Json
    except ImportError:
        Json = None
    payload = Json(dict(detail or {})) if Json else json.dumps(dict(detail or {}))
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO auth_operation_logs (user_id, username, action, target, detail, ip)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, username, action, target, payload, ip),
            )


def list_operation_logs(limit: int | None = None) -> list[dict[str, Any]]:
    with connection() as conn:
        with conn.cursor(cursor_factory=_dict_cursor()) as cursor:
            if limit is None:
                cursor.execute(
                    "SELECT * FROM auth_operation_logs ORDER BY created_at DESC"
                )
            else:
                cursor.execute(
                    "SELECT * FROM auth_operation_logs ORDER BY created_at DESC LIMIT %s",
                    (max(1, int(limit)),),
                )
            items = _rows(cursor)
    for item in items:
        created = item.get("created_at")
        if isinstance(created, datetime):
            item["created_at"] = created.astimezone(timezone.utc).isoformat()
    return items


def _dict_cursor():
    from psycopg2.extras import RealDictCursor

    return RealDictCursor
