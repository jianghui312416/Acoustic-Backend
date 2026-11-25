# -*- coding: utf-8 -*-
"""环境数据统一读取入口。

配置只使用两组新字段：

* ``DATA_SOURCES`` 描述每类数据从哪里读取，以及各服务使用的变量标识；
* ``DATA_SERVICES`` 描述 api0、api1、postgres 等服务的连接参数。

读取器最终都返回与 NetCDF 文件读取一致的 cube/field 结构，因此声场计算层
不需要关心数据来自本地文件、文件服务、目录库还是 PostgreSQL 数值表。
psycopg2 采用惰性导入，纯文件模式不安装数据库驱动也可以运行。
"""
from __future__ import annotations

import os
import ctypes
import hashlib
import math
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import requests
from netCDF4 import Dataset
from scipy.spatial import cKDTree

from backworker import catalog_queries, hycom_service

_NETCDF_READ_CACHE_LOCK = threading.Lock()


def data_source_config(config: Mapping[str, Any], kind: str) -> Dict[str, Any]:
    """返回 ``DATA_SOURCES.<kind>``，配置缺失时直接指出新字段路径。"""
    sources = config.get("DATA_SOURCES")
    if not isinstance(sources, Mapping):
        raise ValueError("data_config.json 缺少 DATA_SOURCES 配置。")
    source = sources.get(kind)
    if not isinstance(source, Mapping):
        raise ValueError(f"data_config.json 缺少 DATA_SOURCES.{kind} 配置。")
    return dict(source)


def data_service_config(config: Mapping[str, Any], name: str) -> Dict[str, Any]:
    """返回 ``DATA_SERVICES.<name>``，不读取任何旧配置字段。"""
    services = config.get("DATA_SERVICES")
    if not isinstance(services, Mapping):
        raise ValueError("data_config.json 缺少 DATA_SERVICES 配置。")
    service = services.get(name)
    if not isinstance(service, Mapping):
        raise ValueError(f"data_config.json 缺少 DATA_SERVICES.{name} 配置。")
    return dict(service)


def default_environment_time(config: Mapping[str, Any]) -> str:
    """返回当前水体数据源的默认环境时次。"""
    try:
        mode = data_source_mode(config, "temperature", "file")
        if mode == "hycom":
            return hycom_service.default_time(config)
        if mode == "file":
            time_sets = []
            for kind in ("temperature", "salinity", "ice"):
                if data_source_mode(config, kind, "file") != "file":
                    continue
                files = data_source_config(config, kind).get("files")
                if isinstance(files, Mapping) and files:
                    time_sets.append({_normalise_time_key(value) for value in files})
            common = sorted(set.intersection(*time_sets)) if time_sets else []
            catalog = config.get("DATA_CATALOG")
            configured_default = (
                _normalise_time_key(catalog.get("default_time"))
                if isinstance(catalog, Mapping)
                else ""
            )
            if configured_default in common:
                return configured_default
            if common:
                return common[-1]
    except (KeyError, ValueError, hycom_service.HycomServiceError):
        pass
    return _text(data_service_config(config, "api1").get("default_time"))


def _normalise_time_key(value: Any) -> str:
    text = _text(value).replace("T", " ").removesuffix("Z")
    return f"{text}:00" if len(text) == 16 else text


def data_source_file(config: Mapping[str, Any], kind: str, time_value: str = "") -> str:
    """返回某类数据本地文件的绝对路径，可按时次从 ``files`` 映射选择。"""
    source = data_source_config(config, kind)
    value = None
    files = source.get("files")
    if isinstance(files, Mapping) and files:
        normalised = {_normalise_time_key(key): item for key, item in files.items()}
        requested = _normalise_time_key(time_value)
        if requested:
            value = normalised.get(requested)
            if value in (None, ""):
                raise ValueError(
                    f"DATA_SOURCES.{kind}.files 没有时次 {requested}；"
                    f"可用时次：{', '.join(sorted(normalised))}"
                )
        else:
            value = source.get("file") or next(iter(files.values()))
    else:
        value = source.get("file")
    if value in (None, ""):
        raise ValueError(f"data_config.json 缺少 DATA_SOURCES.{kind}.file 配置。")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        # data_config.json 中的相对路径以项目根目录（backworker 的上一级）为基准，
        # 不能按各调用模块自身目录解析。
        path = Path(__file__).resolve().parent.parent / path
    return str(path.resolve())


def nearest_data_source_file(config: Mapping[str, Any], kind: str, time_value: str = "") -> str:
    """返回时次最接近的本地数据文件；用于覆盖时次不完整的慢变数据（如海冰）。

    优先取不晚于请求时次的最近文件；没有更早的文件时再取绝对差最小的文件。
    没有 ``files`` 映射或请求时次本身可用时，行为与 ``data_source_file`` 一致。
    """
    try:
        return data_source_file(config, kind, time_value)
    except ValueError:
        source = data_source_config(config, kind)
        files = source.get("files")
        if not isinstance(files, Mapping) or not files:
            raise
        from datetime import datetime

        def parse(text: Any):
            value = _normalise_time_key(text)
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            return None

        target = parse(time_value)
        if target is None:
            return data_source_file(config, kind, "")
        normalised_keys = {_normalise_time_key(key): str(key) for key in files}
        candidates = [
            (key, parse(key))
            for key in files
            if parse(key) is not None
        ]
        if not candidates:
            raise
        preceding = [item for item in candidates if item[1] <= target]
        chosen = (
            max(preceding, key=lambda item: item[1])
            if preceding
            else min(candidates, key=lambda item: abs((item[1] - target).total_seconds()))
        )
        original_key = normalised_keys.get(_normalise_time_key(chosen[0]), chosen[0])
        return data_source_file(config, kind, original_key)


def data_source_tag(config: Mapping[str, Any], kind: str, mode: str, field: str = "") -> str:
    """读取数据变量标识；海浪的三个分量通过 ``field`` 区分。"""
    source = data_source_config(config, kind)
    if mode == "api0":
        key = "api0_tags" if field else "api0_tag"
    elif mode == "api1":
        key = "api1_data_types" if field else "api1_data_type"
    elif mode == "hycom":
        key = "hycom_variables" if field else "hycom_variable"
    elif mode == "postgres":
        key = "postgres_tags" if field else "postgres_tag"
    else:
        raise ValueError(f"{kind} 的 mode={mode!r} 不使用远程变量标识。")
    values = source.get(key)
    value = values.get(field) if field and isinstance(values, Mapping) else values
    if value in (None, ""):
        suffix = f".{field}" if field else ""
        raise ValueError(f"data_config.json 缺少 DATA_SOURCES.{kind}.{key}{suffix} 配置。")
    return str(value).strip()


def optional_data_source_tag(
    config: Mapping[str, Any],
    kind: str,
    mode: str,
    field: str = "",
    default: str = "",
) -> str:
    """读取非当前请求必需的变量标识，未配置时返回空值。"""
    try:
        return data_source_tag(config, kind, mode, field)
    except ValueError:
        return _text(default)


def data_source_mode(config: Mapping[str, Any], kind: str, default: str = "file") -> str:
    """返回某类数据的统一来源模式：file、hycom、api0、api1 或 postgres。

    模式只从 ``DATA_SOURCES.<kind>.mode`` 读取。
    """
    raw = _text(data_source_config(config, kind).get("mode"), default).lower()
    if raw not in {"file", "hycom", "api0", "api1", "postgres"}:
        raise ValueError(
            f"DATA_SOURCES.{kind}.mode={raw!r} 无效，仅支持 file、hycom、api0、api1、postgres。"
        )
    return raw


# =============================================================================
# 通用数组和配置工具
# =============================================================================
def _normalise_lon_array(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _nearest_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    order = np.argsort(ref)
    sorted_ref = ref[order]
    pos = np.searchsorted(sorted_ref, np.asarray(query, dtype=np.float64))
    pos = np.clip(pos, 1, len(sorted_ref) - 1)
    left, right = sorted_ref[pos - 1], sorted_ref[pos]
    choose_right = np.abs(query - right) < np.abs(query - left)
    return order[pos - 1 + choose_right.astype(np.int64)]


def _text(value: Any, default: str = "") -> str:
    """把配置值整理成字符串；空值使用默认值。"""
    if value is None or value == "":
        value = default
    return str(value).strip()


def _int_value(value: Any, default: int) -> int:
    if value is None or value == "":
        value = default
    return int(value)


def _windows_short_path(path: str) -> str:
    if os.name != "nt" or str(path).isascii():
        return str(path)
    try:
        get_short_path = ctypes.windll.kernel32.GetShortPathNameW
        needed = get_short_path(str(path), None, 0)
        if needed <= 0:
            return str(path)
        buffer = ctypes.create_unicode_buffer(needed)
        get_short_path(str(path), buffer, needed)
        short = buffer.value
        return short if short and short.isascii() and os.path.exists(short) else str(path)
    except Exception:
        return str(path)


def _ascii_netcdf_read_path(path: str) -> str:
    source = os.path.abspath(str(path))
    if not os.path.exists(source):
        raise FileNotFoundError(source)
    if os.name != "nt" or source.isascii():
        return source
    short = _windows_short_path(source)
    if short.isascii() and os.path.exists(short):
        return short

    stat = os.stat(source)
    cache_key = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "bellhop_netcdf_read_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{cache_key}_{stat.st_size}_{int(stat.st_mtime)}.nc"
    with _NETCDF_READ_CACHE_LOCK:
        if not target.exists() or target.stat().st_size != stat.st_size:
            temporary = target.with_suffix(f".{os.getpid()}.tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
    return str(target)


# =============================================================================
# 数据库数据读取
# =============================================================================
class PostgresDataReader:
    """从 PostgreSQL 表中读取温度、盐度和海浪数据。"""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        time_value: str = "",
    ) -> None:
        service = data_service_config(config, "postgres")
        self._table = _text(service.get("table"), "env_data")
        columns = service.get("columns") or {}
        self._filename_col = _text(columns.get("filename"), "filename")
        self._time_col = _text(columns.get("time"), "time")
        self._depth_col = _text(columns.get("depth"), "depth")
        self._lat_col = _text(columns.get("lat"), "lat")
        self._lon_col = _text(columns.get("lon"), "lon")
        self._value_col = _text(columns.get("value"), "value")

        self.temperature_tag = data_source_tag(config, "temperature", "postgres")
        self.salinity_tag = data_source_tag(config, "salinity", "postgres")
        self.hs_tag = data_source_tag(config, "wave", "postgres", "height")
        self.tp_tag = data_source_tag(config, "wave", "postgres", "period")
        self.direction_tag = data_source_tag(config, "wave", "postgres", "direction")
        self.topography_tag = _text(service.get("topography_tag"), "topography")
        self.ice_tag = _text(service.get("ice_tag"), "ice_thickness")
        self.bottom_geoacoustics_tag = _text(
            service.get("bottom_geoacoustics_tag"), "bottom_geoacoustics"
        )

        self._connect_kwargs = {
            "host": _text(service.get("host"), "127.0.0.1"),
            "port": _int_value(service.get("port"), 5432),
            "dbname": _text(service.get("dbname"), "ocean"),
            "user": _text(service.get("user"), "postgres"),
            "password": _text(service.get("password")),
            "connect_timeout": _int_value(service.get("connect_timeout"), 5),
        }
        configured_time = time_value or service.get("default_time")
        self._time_value = None if configured_time in (None, "") else configured_time

        # 分表模式：tag -> (表名, 数值列名)
        self._per_var: Dict[str, Tuple[str, str]] = {}
        tables = service.get("tables") or {}
        if isinstance(tables, dict):
            for key, spec in tables.items():
                if not isinstance(spec, dict):
                    continue
                table = _text(spec.get("table") or spec.get("name"))
                value = _text(spec.get("value") or spec.get("value_col"))
                if table and value:
                    self._per_var[str(key)] = (table, value)
        self._per_var_enabled = bool(self._per_var)

        self._conn = None
        self._axes_cache: Dict[Tuple[str, str, str], np.ndarray] = {}
        self._time_cache: Dict[Tuple[str, str], Any] = {}

    # ------------------------------------------------------------------ #
    # 数据库连接工具
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _connect(self):
        if self._conn is None:
            import psycopg2

            self._conn = psycopg2.connect(**self._connect_kwargs)
        return self._conn

    def _execute(self, query: str, args: Sequence[Any]) -> List[Tuple[Any, ...]]:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(query, tuple(args))
            return cur.fetchall()

    def _ident(self, name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    @staticmethod
    def _qualified(table: str) -> str:
        parts = str(table).split(".", 1)
        if len(parts) == 2:
            schema, name = parts
            return '"' + schema.replace('"', '""') + '"."' + name.replace('"', '""') + '"'
        return '"' + str(table).replace('"', '""') + '"'

    # ------------------------------------------------------------------ #
    # 表名和数值列解析
    # ------------------------------------------------------------------ #
    def _table_for(self, tag: str) -> Tuple[str, str]:
        """根据变量标识返回规范化表名和数值列名。"""
        key = None
        if tag == self.temperature_tag:
            key = "temperature"
        elif tag == self.salinity_tag:
            key = "salinity"
        elif tag == self.hs_tag:
            key = "hs"
        elif tag == self.tp_tag:
            key = "tp"
        elif tag == self.direction_tag:
            key = "direction"
        if key and key in self._per_var:
            table, value = self._per_var[key]
            return self._qualified(table), value
        return self._qualified(self._table), self._value_col

    def _single_table_mode(self, tag: str) -> bool:
        key = None
        if tag == self.temperature_tag:
            key = "temperature"
        elif tag == self.salinity_tag:
            key = "salinity"
        elif tag == self.hs_tag:
            key = "hs"
        elif tag == self.tp_tag:
            key = "tp"
        elif tag == self.direction_tag:
            key = "direction"
        return not (key and key in self._per_var)

    # ------------------------------------------------------------------ #
    # 坐标轴和时间工具（带时间过滤，避免在大表上全表 DISTINCT）
    # ------------------------------------------------------------------ #
    def _axis(self, table: str, column: str, time_value: Any) -> np.ndarray:
        key = (table, column, "" if time_value is None else str(time_value))
        if key not in self._axes_cache:
            query = f"SELECT DISTINCT {self._ident(column)} FROM {table}"
            args: List[Any] = []
            if time_value is not None:
                query += f" WHERE {self._ident(self._time_col)} = %s"
                args.append(time_value)
            query += f" ORDER BY {self._ident(column)}"
            rows = self._execute(query, args)
            self._axes_cache[key] = np.asarray(
                [float(row[0]) for row in rows if row[0] is not None], dtype=np.float64
            )
        return self._axes_cache[key]

    def _time_filter(self, table: str, tag: str, single_table: bool) -> Any:
        key = (table, tag)
        if key not in self._time_cache:
            if self._time_value is None:
                query = f"SELECT MIN({self._ident(self._time_col)}) FROM {table}"
                args: List[Any] = []
                if single_table:
                    query += f" WHERE {self._ident(self._filename_col)} = %s"
                    args.append(tag)
                rows = self._execute(query, args)
                self._time_cache[key] = rows[0][0] if rows else None
            else:
                self._time_cache[key] = self._time_value
        return self._time_cache[key]

    # ------------------------------------------------------------------ #
    # 三维数据块和二维场读取（保持与 NetCDF 读取器一致的返回结构）
    # ------------------------------------------------------------------ #
    def read_profile_cube(
        self, tag: str, query_lon: np.ndarray, query_lat: np.ndarray
    ) -> Dict[str, Any]:
        """返回包含 depth、lat、lon、values 等字段的三维数据块。

        返回结构与 bellhop_model.runner._read_hycom_cube 保持一致。
        """
        table, value_col = self._table_for(tag)
        single_table = self._single_table_mode(tag)
        time_value = self._time_filter(table, tag, single_table)

        depth = self._axis(table, self._depth_col, time_value)
        lat_full = self._axis(table, self._lat_col, time_value)
        lon_full = self._axis(table, self._lon_col, time_value)

        query_lon = np.asarray(query_lon, dtype=np.float64)
        query_lat = np.asarray(query_lat, dtype=np.float64)

        centre = float(np.nanmean(query_lon))
        lon_norm = _normalise_lon_array(lon_full)
        lon_unwrapped = centre + ((lon_norm - centre + 180.0) % 360.0) - 180.0
        query_unwrapped = centre + ((_normalise_lon_array(query_lon) - centre + 180.0) % 360.0) - 180.0

        lat_idx_all = _nearest_indices(lat_full, query_lat.ravel())
        lon_idx_all = _nearest_indices(lon_unwrapped, query_unwrapped.ravel())
        lat_indices = np.arange(
            max(0, int(lat_idx_all.min()) - 1), min(len(lat_full), int(lat_idx_all.max()) + 2)
        )
        wanted_lon = np.unique(lon_idx_all)
        if np.ptp(wanted_lon) > len(lon_full) // 2:
            lon_indices = wanted_lon
        else:
            lon_indices = np.arange(
                max(0, int(wanted_lon.min()) - 1), min(len(lon_full), int(wanted_lon.max()) + 2)
            )

        sub_lat_raw = lat_full[lat_indices]
        sub_lon_raw = lon_full[lon_indices]
        lat_order = np.argsort(sub_lat_raw)
        lon_order = np.argsort(lon_unwrapped[lon_indices])
        sub_lat = sub_lat_raw[lat_order]
        sub_lon = lon_unwrapped[lon_indices][lon_order]
        sub_lon_raw_sorted = sub_lon_raw[lon_order]

        depth_to_idx = {round(float(v), 3): i for i, v in enumerate(depth)}
        lat_to_idx = {round(float(v), 6): i for i, v in enumerate(sub_lat)}
        lon_to_idx = {round(float(v), 6): i for i, v in enumerate(sub_lon_raw_sorted)}

        if single_table:
            query = (
                f"SELECT {self._ident(self._depth_col)}, {self._ident(self._lat_col)}, "
                f"{self._ident(self._lon_col)}, {self._ident(value_col)} "
                f"FROM {table} "
                f"WHERE {self._ident(self._filename_col)} = %s "
                f"AND {self._ident(self._time_col)} = %s "
                f"AND {self._ident(self._lat_col)}::float8 = ANY(%s) "
                f"AND {self._ident(self._lon_col)}::float8 = ANY(%s)"
            )
            args = [tag, time_value, [float(v) for v in sub_lat_raw], [float(v) for v in sub_lon_raw]]
        else:
            query = (
                f"SELECT {self._ident(self._depth_col)}, {self._ident(self._lat_col)}, "
                f"{self._ident(self._lon_col)}, {self._ident(value_col)} "
                f"FROM {table} "
                f"WHERE {self._ident(self._time_col)} = %s "
                f"AND {self._ident(self._lat_col)}::float8 = ANY(%s) "
                f"AND {self._ident(self._lon_col)}::float8 = ANY(%s)"
            )
            args = [time_value, [float(v) for v in sub_lat_raw], [float(v) for v in sub_lon_raw]]
        rows = self._execute(query, args)

        values = np.full((len(depth), len(sub_lat), len(sub_lon)), np.nan, dtype=np.float64)
        for dval, lval, loval, vval in rows:
            if dval is None or lval is None or loval is None or vval is None:
                continue
            idp = depth_to_idx.get(round(float(dval), 3))
            ila = lat_to_idx.get(round(float(lval), 6))
            ilo = lon_to_idx.get(round(float(loval), 6))
            if idp is None or ila is None or ilo is None:
                continue
            values[idp, ila, ilo] = float(vval)

        return {
            "depth": depth,
            "lat": sub_lat,
            "lon": sub_lon,
            "values": values,
            "query_lon": query_unwrapped,
            "centre_lon": centre,
        }

    def read_scalar_field(
        self, tag: str, query_lon: np.ndarray, query_lat: np.ndarray
    ) -> np.ndarray:
        """在查询点上返回最近邻采样得到的二维场。

        返回结构与 bellhop_model.runner._read_wave_field 保持一致。
        """
        table, value_col = self._table_for(tag)
        single_table = self._single_table_mode(tag)
        time_value = self._time_filter(table, tag, single_table)

        lat_full = self._axis(table, self._lat_col, time_value)
        lon_full = self._axis(table, self._lon_col, time_value)

        query_lon = np.asarray(query_lon, dtype=np.float64)
        query_lat = np.asarray(query_lat, dtype=np.float64)
        qlon = query_lon
        if float(np.nanmax(lon_full)) > 180.0:
            qlon = query_lon % 360.0

        ilat = _nearest_indices(lat_full, query_lat.ravel())
        ilon = _nearest_indices(lon_full, qlon.ravel())

        needed_lats = sorted({float(lat_full[i]) for i in ilat})
        needed_lons = sorted({float(lon_full[i]) for i in ilon})
        lat_to_idx = {round(float(v), 6): i for i, v in enumerate(lat_full)}
        lon_to_idx = {round(float(v), 6): i for i, v in enumerate(lon_full)}

        if single_table:
            query = (
                f"SELECT {self._ident(self._lat_col)}, {self._ident(self._lon_col)}, "
                f"{self._ident(value_col)} "
                f"FROM {table} "
                f"WHERE {self._ident(self._filename_col)} = %s "
                f"AND {self._ident(self._time_col)} = %s "
                f"AND {self._ident(self._lat_col)}::float8 = ANY(%s) "
                f"AND {self._ident(self._lon_col)}::float8 = ANY(%s)"
            )
            rows = self._execute(query, (tag, time_value, needed_lats, needed_lons))
        else:
            query = (
                f"SELECT {self._ident(self._lat_col)}, {self._ident(self._lon_col)}, "
                f"{self._ident(value_col)} "
                f"FROM {table} "
                f"WHERE {self._ident(self._time_col)} = %s "
                f"AND {self._ident(self._lat_col)}::float8 = ANY(%s) "
                f"AND {self._ident(self._lon_col)}::float8 = ANY(%s)"
            )
            rows = self._execute(query, (time_value, needed_lats, needed_lons))

        grid: Dict[Tuple[int, int], float] = {}
        for lval, loval, vval in rows:
            if lval is None or loval is None or vval is None:
                continue
            ila = lat_to_idx.get(round(float(lval), 6))
            ilo = lon_to_idx.get(round(float(loval), 6))
            if ila is None or ilo is None:
                continue
            grid[(ila, ilo)] = float(vval)

        result = np.full(query_lat.size, np.nan, dtype=np.float64)
        for q in range(query_lat.size):
            result[q] = grid.get((int(ilat[q]), int(ilon[q])), np.nan)
        return result.reshape(query_lat.shape)


class PostgresBottomGeoacousticsReader:
    """从结构化 PostgreSQL 表批量读取二维海底声学参数。

    与温盐长表不同，海底表的一行同时保存经纬度和全部声学字段。一次任务只做
    一次区域查询，再用最近邻把记录映射到公共环境网格，避免按方位或采样点反复
    访问数据库。
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        source_config: Mapping[str, Any],
        *,
        data_source: str = "",
        time_value: str = "",
    ) -> None:
        service_name = _text(source_config.get("service"), "postgres")
        service = data_service_config(config, service_name)
        self._table = _text(source_config.get("table"))
        if not self._table:
            raise ValueError("DATA_SOURCES.bottom_geoacoustics.table 不能为空。")

        columns = source_config.get("columns") or {}
        if not isinstance(columns, Mapping):
            raise ValueError("DATA_SOURCES.bottom_geoacoustics.columns 必须是对象。")
        self._lon_col = _text(columns.get("longitude") or columns.get("lon"), "longitude")
        self._lat_col = _text(columns.get("latitude") or columns.get("lat"), "latitude")
        self._field_columns = {
            str(name): _text(column)
            for name, column in columns.items()
            if str(name) not in {"longitude", "lon", "latitude", "lat"} and _text(column)
        }
        self._data_source_col = _text(source_config.get("data_source_column"))
        self._time_col = _text(source_config.get("time_column"))
        self._quality_col = _text(source_config.get("quality_column"))
        self._quality_max = source_config.get("quality_max")
        self._data_source = _text(data_source)
        self._time_value = _text(time_value)
        self._margin_deg = max(0.0, float(source_config.get("query_margin_deg", 0.25)))
        self._max_distance_deg = float(source_config.get("max_nearest_distance_deg", 0.5))
        raw_empty_fallback = source_config.get("fallback_on_empty", True)
        self._fallback_on_empty = (
            raw_empty_fallback.strip().lower() in {"1", "true", "yes", "on"}
            if isinstance(raw_empty_fallback, str)
            else bool(raw_empty_fallback)
        )
        self._longitude_domain = _text(source_config.get("longitude_domain"), "-180_180").lower()
        if self._longitude_domain not in {"-180_180", "0_360"}:
            raise ValueError(
                "DATA_SOURCES.bottom_geoacoustics.longitude_domain "
                "只能是 -180_180 或 0_360。"
            )
        self.interpolation = _text(source_config.get("interpolation"), "nearest").lower()
        if self.interpolation not in {"linear", "nearest"}:
            raise ValueError(
                "DATA_SOURCES.bottom_geoacoustics.interpolation 只能是 linear 或 nearest。"
            )
        self.source = f"postgres:{self._table}"

        self._connect_kwargs = {
            "host": _text(service.get("host"), "127.0.0.1"),
            "port": _int_value(service.get("port"), 5432),
            "dbname": _text(service.get("dbname"), "ocean"),
            "user": _text(service.get("user"), "postgres"),
            "password": _text(service.get("password")),
            "connect_timeout": _int_value(service.get("connect_timeout"), 5),
        }
        self._conn = None

    @staticmethod
    def _ident(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    @staticmethod
    def _qualified(table: str) -> str:
        return ".".join(
            '"' + part.replace('"', '""') + '"'
            for part in str(table).split(".")
        )

    def _connect(self):
        if self._conn is None:
            import psycopg2

            self._conn = psycopg2.connect(**self._connect_kwargs)
        return self._conn

    def _execute(self, query: str, args: Sequence[Any]) -> List[Tuple[Any, ...]]:
        conn = self._connect()
        with conn.cursor() as cursor:
            cursor.execute(query, tuple(args))
            return cursor.fetchall()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _database_longitudes(self, values: np.ndarray) -> np.ndarray:
        normalised = _normalise_lon_array(values)
        return normalised % 360.0 if self._longitude_domain == "0_360" else normalised

    def read_fields(
        self,
        query_lon: np.ndarray,
        query_lat: np.ndarray,
        field_names: Sequence[str],
    ) -> Dict[str, np.ndarray]:
        """一次区域查询并返回与 ``query_lon/query_lat`` 同形状的全部字段。"""
        lon = np.asarray(query_lon, dtype=np.float64)
        lat = np.asarray(query_lat, dtype=np.float64)
        if lon.shape != lat.shape:
            raise ValueError("海底声学查询经纬度形状必须一致。")
        if lon.size == 0 or not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
            raise ValueError("海底声学查询经纬度必须是非空有限数组。")

        selected = [name for name in field_names if name in self._field_columns]
        if not selected:
            return {}
        qlon = self._database_longitudes(lon)
        lon_min = float(np.min(qlon)) - self._margin_deg
        lon_max = float(np.max(qlon)) + self._margin_deg
        lat_min = float(np.min(lat)) - self._margin_deg
        lat_max = float(np.max(lat)) + self._margin_deg

        select_columns = [self._lat_col, self._lon_col] + [
            self._field_columns[name] for name in selected
        ]
        query = (
            "SELECT " + ", ".join(self._ident(name) for name in select_columns)
            + f" FROM {self._qualified(self._table)}"
            + f" WHERE {self._ident(self._lat_col)}::float8 BETWEEN %s AND %s"
        )
        args: List[Any] = [lat_min, lat_max]
        # 跨日期变更线时不能用一个 BETWEEN；区域表通常不跨线，此处退化为只按纬度取数。
        if lon_max - lon_min <= 180.0:
            query += f" AND {self._ident(self._lon_col)}::float8 BETWEEN %s AND %s"
            args.extend([lon_min, lon_max])
        if self._data_source_col and self._data_source:
            query += f" AND {self._ident(self._data_source_col)} = %s"
            args.append(self._data_source)
        if self._time_col and self._time_value:
            query += f" AND {self._ident(self._time_col)} = %s"
            args.append(self._time_value)
        if self._quality_col and self._quality_max not in (None, ""):
            query += f" AND {self._ident(self._quality_col)} <= %s"
            args.append(float(self._quality_max))

        rows = self._execute(query, args)
        if not rows:
            if self._fallback_on_empty:
                return {
                    name: np.full(lat.shape, np.nan, dtype=np.float64)
                    for name in selected
                }
            raise ValueError(
                f"海底声学表 {self._table} 在请求区域内没有记录："
                f"lon=[{lon_min:g}, {lon_max:g}], lat=[{lat_min:g}, {lat_max:g}]"
            )

        row_lat = np.asarray([float(row[0]) for row in rows], dtype=np.float64)
        row_lon = self._database_longitudes(
            np.asarray([float(row[1]) for row in rows], dtype=np.float64)
        )
        mean_lat = float(np.mean(lat))
        longitude_weight = max(1.0e-6, abs(math.cos(math.radians(mean_lat))))
        row_points = np.column_stack((row_lat, row_lon * longitude_weight))
        query_points = np.column_stack((lat.ravel(), qlon.ravel() * longitude_weight))
        distance, nearest = cKDTree(row_points).query(query_points, k=1)
        too_far = (
            distance > self._max_distance_deg
            if self._max_distance_deg > 0.0
            else np.zeros(distance.shape, dtype=bool)
        )

        result: Dict[str, np.ndarray] = {}
        for offset, name in enumerate(selected, start=2):
            source_values = np.asarray(
                [np.nan if row[offset] is None else float(row[offset]) for row in rows],
                dtype=np.float64,
            )
            values = source_values[np.asarray(nearest, dtype=np.int64)]
            values[too_far] = np.nan
            result[name] = values.reshape(lat.shape)
        return result


# =============================================================================
# 本地 NC 文件接口数据读取
# =============================================================================
class NcFileReader:
    """
    服务器部署模式 NC 文件直接读取器。

    流程：
    /api/nc/list -> 获取 NC 文件 id
                 -> /api/nc/{id} 获取 filePath
                 -> netCDF4.Dataset(filePath) 直接读取变量

    要求 Bellhop 服务与 NC 文件存储在同一服务器。
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        mode: str = "api0",
        time_value: str = "",
    ) -> None:
        service = data_service_config(config, mode)
        path_mapping = service.get("path_mapping") or {}
        self._base = _text(service.get("base_url")).rstrip("/")
        self._file_type = _text(service.get("file_type"), "global_3D")
        # 请求时间优先；未传入时使用配置中的统一默认环境时次。
        self._time_value = _text(
            time_value or service.get("default_time") or default_environment_time(config)
        )
        self._container_root = _text(path_mapping.get("source_root")).rstrip("/")
        self._local_root = _text(path_mapping.get("local_root")).rstrip("/")
        self.temperature_tag = optional_data_source_tag(config, "temperature", mode)
        self.salinity_tag = optional_data_source_tag(config, "salinity", mode)
        self.hs_tag = optional_data_source_tag(config, "wave", mode, "height")
        self.tp_tag = optional_data_source_tag(config, "wave", mode, "period")
        self.direction_tag = optional_data_source_tag(config, "wave", mode, "direction")
        self.topography_tag = optional_data_source_tag(config, "topography", mode)
        self.ice_tag = optional_data_source_tag(config, "ice", mode)
        self.bottom_geoacoustics_tag = optional_data_source_tag(
            config, "bottom_geoacoustics", mode
        )
        self._meta_cache = {}
        self._ds_cache = {}

    def _json(self, url, params=None):
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _find_nc(self, field):
        key=(field,self._time_value)
        if key in self._meta_cache:
            return self._meta_cache[key]
        params={
            "fileType": self._file_type,
            "collectionStartTime": self._time_value,
            "collectionEndTime": self._time_value,
            "current":1,
            "size":100
        }
        data=self._json(self._base+"/api/nc/list", params)
        records=(data.get("data") or {}).get("records") or []
        candidates=[field]
        if field==self.temperature_tag:
            candidates += ["temperature","thetao"]
        if field==self.salinity_tag:
            candidates += ["so"]
        chosen=None
        for f in records:
            schema=f.get("fieldSchema") or {}
            if any(v in schema for v in candidates):
                chosen=f
                break
        if chosen is None:
            raise FileNotFoundError(f"没有找到包含变量 {field} 的NC文件")
        detail=self._json(self._base+f"/api/nc/{chosen['id']}")
        meta=(detail.get("data") or {})
        self._meta_cache[key]=meta
        return meta

    def _remap_path(self, path: str) -> str:
        """把容器内部文件路径映射成本机挂载路径。

        NC 接口返回的 filePath 通常是容器里的路径，例如 /app/uploads/nc/...。
        Bellhop 所在主机可能无法直接读取它，因此这里按配置映射到本机挂载目录；
        如果没有配置，则使用已知的部署目录作为兜底。
        """
        if os.path.exists(path):
            return path
        mappings = []
        if self._container_root and self._local_root:
            mappings.append((self._container_root, self._local_root))
        mappings.append(("/app/uploads", "/mnt/ocean-data/ocean-cloud/uploads"))
        for container_root, local_root in mappings:
            prefix = container_root.rstrip("/") + "/"
            if path.startswith(prefix):
                candidate = local_root.rstrip("/") + "/" + path[len(prefix):]
                if os.path.exists(candidate):
                    return candidate
        return path

    def _dataset_meta(self, field):
        return self._find_nc(field)

    def _dataset_from_meta(self, meta):
        path=meta.get("filePath") or meta.get("filePath")
        if not path:
            raise FileNotFoundError("NC文件没有返回filePath")
        path = self._remap_path(path)
        if path not in self._ds_cache:
            self._ds_cache[path]=Dataset(_ascii_netcdf_read_path(path),"r")
        return self._ds_cache[path]

    def _dataset(self, field):
        return self._dataset_from_meta(self._dataset_meta(field))

    def _schema_candidates(self, meta, keys, fallback):
        schema = meta.get("fieldSchema") if isinstance(meta, Mapping) else {}
        if isinstance(schema, str):
            import json

            try:
                schema = json.loads(schema)
            except ValueError:
                schema = {}
        names = []
        if isinstance(schema, Mapping):
            for key in keys:
                value = schema.get(key)
                if isinstance(value, str):
                    names.append(value)
                elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    names.extend(str(item) for item in value if item not in (None, ""))
        names.extend(str(item) for item in fallback if item not in (None, ""))
        return list(dict.fromkeys(names))

    def read_profile_cube(self, tag, query_lon, query_lat):
        meta=self._dataset_meta(tag)
        ds=self._dataset_from_meta(meta)
        names=[tag]
        if tag==self.temperature_tag:
            names = self._schema_candidates(meta, ("temperature", "temp"), names + ["water_temp","temperature","thetao"])
        elif tag==self.salinity_tag:
            names = self._schema_candidates(meta, ("salinity", "salt"), names + ["salinity","so"])
        else:
            names = self._schema_candidates(
                meta,
                (tag, "variable", "variableName", "field", "value", "data"),
                names,
            )
            schema = meta.get("fieldSchema") if isinstance(meta, Mapping) else {}
            if isinstance(schema, str):
                import json

                try:
                    schema = json.loads(schema)
                except ValueError:
                    schema = {}

            def schema_names(value):
                if isinstance(value, str):
                    return [value]
                if isinstance(value, Mapping):
                    result = []
                    for child in value.values():
                        result.extend(schema_names(child))
                    return result
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    result = []
                    for child in value:
                        result.extend(schema_names(child))
                    return result
                return []

            names.extend(schema_names(schema))
            names = list(dict.fromkeys(names))
        def has_horizontal_axes(candidate):
            dimensions = [str(value).lower() for value in candidate.dimensions]
            has_lat = any(dim in {"lat", "latitude", "y"} or "lat" in dim for dim in dimensions)
            has_lon = any(dim in {"lon", "longitude", "x"} or "lon" in dim for dim in dimensions)
            return has_lat and has_lon

        var=None
        for n in names:
            if n in ds.variables and has_horizontal_axes(ds.variables[n]):
                var=ds.variables[n]
                break
        if var is None:
            coordinate_names = {
                "time", "lat", "latitude", "y", "lon", "longitude", "x",
                "depth", "depth_m", "lev", "level", "layer", "z",
            }
            candidates = []
            for name, candidate in ds.variables.items():
                if name.lower() not in coordinate_names and has_horizontal_axes(candidate):
                    candidates.append(candidate)
            if len(candidates) == 1:
                var = candidates[0]
        if var is None:
            raise KeyError(f"NC中不存在变量 {names}")

        # 自动识别 depth/lat/lon 轴
        dims=list(var.dimensions)
        def find_axis(keys, exact=()):
            for i,d in enumerate(dims):
                lower = d.lower()
                if lower in exact or any(k in lower for k in keys):
                    return i
            return None
        iz=find_axis(["depth","lev","layer"], {"z"})
        iy=find_axis(["lat"], {"y"})
        ix=find_axis(["lon"], {"x"})
        if iy is None or ix is None:
            raise ValueError(f"无法识别NC维度:{dims}")

        def coord(name_list, n):
            for k in name_list:
                if k in ds.variables:
                    return np.asarray(ds.variables[k][:],dtype=float)
            return np.arange(n,dtype=float)

        depth=(
            coord(["depth", "depth_m", "lev", "level", "layer", "z"], var.shape[iz])
            if iz is not None
            else np.asarray([0.0], dtype=np.float64)
        )
        lat=coord(["lat", "latitude", "y"],var.shape[iy])
        lon=coord(["lon", "longitude", "x"],var.shape[ix])

        query_lon=np.asarray(query_lon,dtype=np.float64)
        query_lat=np.asarray(query_lat,dtype=np.float64)
        centre=float(np.mean(query_lon))
        lon_norm=_normalise_lon_array(lon)
        lon_unwrapped=centre+((lon_norm-centre+180.0)%360.0)-180.0
        query_unwrapped=centre+((_normalise_lon_array(query_lon)-centre+180.0)%360.0)-180.0

        lat_idx_all=_nearest_indices(lat,query_lat.ravel())
        lon_idx_all=_nearest_indices(lon_unwrapped,query_unwrapped.ravel())
        lat_indices=np.arange(max(0,int(lat_idx_all.min())-1),min(len(lat),int(lat_idx_all.max())+2))
        wanted_lon=np.unique(lon_idx_all)
        if np.ptp(wanted_lon)>len(lon)//2:
            lon_indices=wanted_lon
        else:
            lon_indices=np.arange(max(0,int(wanted_lon.min())-1),min(len(lon),int(wanted_lon.max())+2))

        key=[0]*len(dims)
        kept=[]
        for i,dim in enumerate(dims):
            if iz is not None and i==iz:
                key[i]=slice(None); kept.append("depth")
            elif i==iy:
                key[i]=lat_indices; kept.append("lat")
            elif i==ix:
                key[i]=lon_indices; kept.append("lon")
            else:
                key[i]=0
        raw=var[tuple(key)]
        # 先转为浮点掩码数组，否则 classId 等整型变量无法用 NaN 填充。
        values=np.ma.asarray(raw,dtype=np.float64).filled(np.nan)
        if iz is None:
            values=np.transpose(values,[kept.index("lat"),kept.index("lon")])[None, :, :]
        else:
            values=np.transpose(values,[kept.index("depth"),kept.index("lat"),kept.index("lon")])
        sub_lon=lon_unwrapped[lon_indices]
        sub_lat=lat[lat_indices]
        lon_order=np.argsort(sub_lon)
        lat_order=np.argsort(sub_lat)
        values=values[:,lat_order,:][:,:,lon_order]

        return {
            "depth":depth,
            "lat":sub_lat[lat_order],
            "lon":sub_lon[lon_order],
            "values":values,
            "query_lon":query_unwrapped,
            "centre_lon":centre,
        }

    def read_scalar_field(self, tag, query_lon, query_lat):
        cube=self.read_profile_cube(tag,query_lon,query_lat)
        layer=cube["values"][0]
        ilat=_nearest_indices(cube["lat"],query_lat)
        ilon=_nearest_indices(cube["lon"],query_lon)
        return layer[ilat,ilon]

    def close(self):
        for ds in self._ds_cache.values():
            try:
                ds.close()
            except Exception:
                pass
        self._ds_cache.clear()
        self._meta_cache.clear()


class CatalogNcFileReader(NcFileReader):
    """通过本地目录库定位 NC 文件，再复用 NC 读取逻辑。"""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        data_source: str = "HYCOM",
        time_value: str = "",
    ) -> None:
        # 调用父类NcFileReader构造函数，设置 mode="api1" 以使用目录库定位逻辑
        super().__init__(config, mode="api1", time_value=time_value)
        self._config = config
        self._data_source = _text(data_source, "HYCOM")

    def _find_nc(self, field):
        key = (self._data_source, field, self._time_value)
        if key in self._meta_cache:
            return self._meta_cache[key]
        if not self._time_value:
            raise FileNotFoundError("目录库查询缺少 time。")
        meta = catalog_queries.query_dataset_file(
            self._config,
            self._data_source,
            field,
            self._time_value,
            True,
        )
        if meta is None:
            raise FileNotFoundError(
                f"目录库未找到 dataSource={self._data_source}, dataType={field}, time={self._time_value} 的NC文件"
            )
        self._meta_cache[key] = meta
        return meta


# =============================================================================
# 读取器工厂入口
# =============================================================================
def create_data_reader(
    config: Mapping[str, Any],
    mode: str,
    *,
    data_source: str = "HYCOM",
    time_value: str = "",
):
    """按新配置模式创建读取器，并显式传入本次请求上下文。"""
    reader = _text(mode).lower()
    if reader == "api1":
        return CatalogNcFileReader(
            config,
            data_source=data_source,
            time_value=time_value,
        )
    if reader == "api0":
        return NcFileReader(config, mode="api0", time_value=time_value)
    if reader == "hycom":
        return hycom_service.HycomNcssReader(config, time_value=time_value)
    if reader == "postgres":
        return PostgresDataReader(config, time_value=time_value)
    raise ValueError(f"数据读取模式 {reader!r} 无效，仅支持 hycom、api0、api1、postgres。")


def resolve_data_file(
    config: Mapping[str, Any],
    mode: str,
    kind: str,
    *,
    data_source: str = "HYCOM",
    time_value: str = "",
) -> str:
    """通过 api0/api1 定位需要由现有文件解析器打开的数据文件。"""
    selected_mode = _text(mode).lower()
    if selected_mode not in {"file", "hycom", "api0", "api1", "postgres"}:
        raise ValueError(f"DATA_SOURCES.{kind}.mode={selected_mode!r} 无效。")
    if selected_mode == "file":
        raise ValueError(f"{kind} 使用 file 模式时应直接读取对应的本地文件配置。")
    if selected_mode == "postgres":
        raise ValueError(f"{kind} 是文件型数据，不能使用 postgres 长表模式。")
    if selected_mode == "hycom":
        raise ValueError(f"{kind} 使用 HYCOM NCSS 流式读取，不对应本地文件路径。")

    reader = create_data_reader(
        config,
        selected_mode,
        data_source=data_source,
        time_value=time_value,
    )
    try:
        tag_name = {
            "temperature": "temperature_tag",
            "salinity": "salinity_tag",
            "topography": "topography_tag",
            "ice": "ice_tag",
            "bottom_geoacoustics": "bottom_geoacoustics_tag",
        }.get(kind)
        if tag_name:
            tag = getattr(reader, tag_name)
        else:
            raise ValueError(f"不支持为 {kind!r} 定位文件。")
        meta = reader._dataset_meta(tag)
        path = str(meta.get("filePath") or "").strip()
        if not path:
            raise FileNotFoundError(f"{kind} 数据记录缺少 filePath。")
        path = reader._remap_path(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{kind} 数据文件不存在：{path}")
        return path
    finally:
        reader.close()
