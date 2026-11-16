"""结构化数据目录表的只读查询。

本模块只负责 PostgreSQL 查询和字段整理；门户协议包装留给 API 层处理。
连接信息只读取 ``DATA_SERVICES.api1``，该服务与其他环境数据服务互相独立。
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Mapping

_LOCAL = threading.local()
_CONNECT_ATTEMPTS = 2
_RETRY_DELAY_S = 0.5


def _text(value: Any, default: str = "") -> str:
    if value in (None, ""):
        value = default
    return str(value).strip()


def catalog_mode(config: Mapping[str, Any]) -> str:
    """返回环境文件目录模式：file 或 postgres。"""
    section = config.get("DATA_CATALOG")
    raw = section.get("mode", "postgres") if isinstance(section, Mapping) else "postgres"
    mode = _text(raw, "postgres").lower()
    if mode not in {"file", "postgres"}:
        raise ValueError("DATA_CATALOG.mode 仅支持 file 或 postgres。")
    return mode


def _catalog_config(config: Mapping[str, Any]) -> dict[str, Any]:
    services = config.get("DATA_SERVICES")
    if not isinstance(services, Mapping) or not isinstance(services.get("api1"), Mapping):
        raise ValueError("data_config.json 缺少 DATA_SERVICES.api1 配置。")
    catalog = dict(services["api1"])
    return {
        "host": _text(catalog.get("host"), "127.0.0.1"),
        "port": int(catalog.get("port", 5432)),
        "dbname": _text(catalog.get("dbname"), "Ocean_meta"),
        "user": _text(catalog.get("user"), "postgres"),
        "password": _text(catalog.get("password")),
        "connect_timeout": int(catalog.get("connect_timeout", 5)),
    }


def _connection_cache() -> dict[tuple[Any, ...], Any]:
    cache = getattr(_LOCAL, "connections", None)
    if cache is None:
        cache = {}
        _LOCAL.connections = cache
    return cache


def _cache_key(options: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        options["host"],
        options["port"],
        options["dbname"],
        options["user"],
        options["password"],
    )


def _discard(config: Mapping[str, Any]) -> None:
    cache = _connection_cache()
    connection = cache.pop(_cache_key(_catalog_config(config)), None)
    if connection is not None:
        try:
            connection.close()
        except Exception:
            pass


def _connect(config: Mapping[str, Any]):
    if catalog_mode(config) != "postgres":
        raise RuntimeError(
            "DATA_CATALOG.mode=file，已禁用 datasetFileCatalog/PostgreSQL 目录查询。"
        )
    import psycopg2

    options = _catalog_config(config)
    key = _cache_key(options)
    cache = _connection_cache()
    connection = cache.get(key)
    if connection is not None and not getattr(connection, "closed", True):
        return connection
    last_error = None
    for attempt in range(_CONNECT_ATTEMPTS):
        try:
            connection = psycopg2.connect(**options)
            connection.autocommit = True
            cache[key] = connection
            return connection
        except psycopg2.OperationalError as exc:
            last_error = exc
            if attempt + 1 < _CONNECT_ATTEMPTS:
                time.sleep(_RETRY_DELAY_S)
    raise last_error


def _json_value(value: Any, default: Any):
    """兼容 psycopg2 已解码 JSONB 和测试/其他驱动返回的 JSON 文本。"""
    if value is None:
        return default
    if isinstance(value, str):
        import json

        try:
            return json.loads(value)
        except ValueError:
            return default
    return value


def _fetchall(config: Mapping[str, Any], query: str, args=()):
    import psycopg2

    for attempt in range(_CONNECT_ATTEMPTS):
        connection = _connect(config)
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(args))
                return cursor.fetchall()
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            _discard(config)
            if attempt + 1 < _CONNECT_ATTEMPTS:
                time.sleep(_RETRY_DELAY_S)
                continue
            raise


def query_dataset_list(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _fetchall(
        config,
        '''SELECT "id", "name", "categoryId", "categoryName", "pid",
                  "datatype", "datarange", "normalizedLayers"
           FROM "datasetList"
           ORDER BY "id"''',
    )
    return [
        {
            "id": row[0],
            "name": row[1],
            "categoryId": row[2],
            "categoryName": row[3],
            "pid": row[4],
            "datatype": _json_value(row[5], []),
            "datarange": _json_value(row[6], []),
            # 接口文档字段名没有 d；数据库列名保留现有 normalizedLayers。
            "normalizeLayers": _json_value(row[7], []),
        }
        for row in rows
    ]


def query_dataset_time_list(
    config: Mapping[str, Any], data_source: str, is_env: bool, query_month: int
) -> dict[str, Any]:
    rows = _fetchall(
        config,
        '''SELECT "dataSource", "isEnv", "displayList", "standardTimeList"
           FROM "datasetTimeList"
           WHERE "dataSource" = %s AND "isEnv" = %s AND "queryDataInMonth" = %s
           ORDER BY "id"''',
        (data_source, is_env, query_month),
    )
    display_list: list[Any] = []
    for row in rows:
        display_list.extend(_json_value(row[2], []))

    # 多行数据合并时保持原顺序并去重。
    display_list = list(dict.fromkeys(display_list))
    return {
        "dataSource": data_source,
        "isEnv": is_env,
        "displayList": display_list,
        # 协议将该字段定义为预留字段，并要求当前结果与 displayList 相同。
        "standardTimeList": list(display_list),
    }


def query_data_type_meta_list(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _fetchall(
        config,
        '''SELECT "id", "name", "unit", "minVal", "maxVal"
           FROM "dataTypeMeta"
           ORDER BY "id"''',
    )
    return [
        {"id": row[0], "name": row[1], "unit": row[2], "minVal": row[3], "maxVal": row[4]}
        for row in rows
    ]


def query_dataset_file(
    config: Mapping[str, Any],
    data_source: str,
    data_type: str,
    time_value: str,
    is_env: bool = True,
) -> dict[str, Any] | None:
    rows = _fetchall(
        config,
        '''SELECT "id", "dataSource", "dataType", "time", "isEnv",
                  "fileName", "filePath", "fieldSchema", "normalizedLayers", "datarange"
           FROM "datasetFileCatalog"
           WHERE "dataSource" = %s
             AND "dataType" = %s
             AND ("time" = %s OR "time" IS NULL)
             AND "isEnv" = %s
           -- 有时次的数据优先；海底地声等静态数据以 NULL time 作为兜底。
           ORDER BY ("time" IS NULL), "id"
           LIMIT 1''',
        (data_source, data_type, time_value, is_env),
    )
    if not rows:
        return None
    row = rows[0]
    return {
        "id": row[0],
        "dataSource": row[1],
        "dataType": row[2],
        "time": row[3],
        "isEnv": row[4],
        "fileName": row[5],
        "filePath": row[6],
        "fieldSchema": _json_value(row[7], {}),
        "normalizedLayers": _json_value(row[8], []),
        "datarange": _json_value(row[9], {}),
    }
