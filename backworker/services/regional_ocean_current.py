"""HYCOM 原生网格区域三维海流服务。"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .. import hycom_service


_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_CACHE_LIMIT = 32
_KEY_LOCKS_LOCK = threading.Lock()
_KEY_LOCKS: dict[tuple[Any, ...], threading.Lock] = {}


def _key_lock(key: tuple[Any, ...]) -> threading.Lock:
    """同一区域、同一HYCOM时次只允许一次上游下载。"""
    with _KEY_LOCKS_LOCK:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _KEY_LOCKS[key] = lock
        return lock


def _parse_time(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("forecastTime 不能为空。")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ValueError("forecastTime 必须是 ISO 8601 时间。") from exc
    if parsed.tzinfo is None:
        raise ValueError("forecastTime 必须包含时区，例如 Z 或 +08:00。")
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _nearest_hycom_time(config: Mapping[str, Any], requested: datetime) -> datetime:
    available = hycom_service.latest_times(config)
    if not available:
        raise ValueError("HYCOM 当前没有可用的海流分析或预报时次。")
    selected = min(available, key=lambda value: abs((value - requested).total_seconds()))
    step_hours = max(1.0, float(
        hycom_service._service_config(config).get("time_step_hours", 3)
    ))
    offset = abs((selected - requested).total_seconds()) / 3600.0
    if offset > step_hours / 2.0 + 0.1:
        raise ValueError(
            f"HYCOM 没有接近 {_iso(requested)} 的海流预报，最近时次为 {_iso(selected)}。"
        )
    return selected


def _json_array(values: np.ndarray, decimals: int = 6) -> list[Any]:
    rounded = np.round(np.asarray(values, dtype=np.float64), decimals)
    return np.where(np.isfinite(rounded), rounded, None).tolist()


def _depth_mean(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(u) & np.isfinite(v)
    count = np.sum(valid, axis=0)
    mean_u = np.divide(
        np.sum(np.where(valid, u, 0.0), axis=0), count,
        out=np.full(count.shape, np.nan, dtype=np.float64), where=count > 0,
    )
    mean_v = np.divide(
        np.sum(np.where(valid, v, 0.0), axis=0), count,
        out=np.full(count.shape, np.nan, dtype=np.float64), where=count > 0,
    )
    return mean_u, mean_v


def fetch_region_current(
    config: Mapping[str, Any], *, west: float, south: float,
    east: float, north: float, forecast_time: datetime | str,
    _single_flight: bool = False,
) -> dict[str, Any]:
    """下载一个HYCOM时次的原生u/v三维区域子集，并返回全深度矢量平均。"""
    requested = forecast_time if isinstance(forecast_time, datetime) else _parse_time(forecast_time)
    if requested.tzinfo is None:
        raise ValueError("forecastTime 必须包含时区。")
    requested = requested.astimezone(timezone.utc)
    selected = _nearest_hycom_time(config, requested)
    key = (
        round(float(west), 6), round(float(south), 6),
        round(float(east), 6), round(float(north), 6), _iso(selected),
    )
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached

    if not _single_flight:
        # 滑杆相邻小时常映射到同一个3小时HYCOM时次；首个请求未完成时，
        # 其余请求等待并复用结果，避免并发生成相同的全深度NetCDF子集。
        with _key_lock(key):
            with _CACHE_LOCK:
                cached = _CACHE.get(key)
            if cached is not None:
                return cached
            return fetch_region_current(
                config,
                west=west, south=south, east=east, north=north,
                forecast_time=selected, _single_flight=True,
            )

    reader = hycom_service.HycomNcssReader(
        config, time_value=selected.strftime("%Y-%m-%d %H:%M:%S")
    )
    try:
        query_lon = np.asarray([west, east], dtype=np.float64)
        query_lat = np.asarray([south, north], dtype=np.float64)
        u_cube = reader.read_profile_cube(reader.current_u_tag, query_lon, query_lat)
        v_cube = reader.read_profile_cube(reader.current_v_tag, query_lon, query_lat)
    finally:
        reader.close()

    depth = np.asarray(u_cube["depth"], dtype=np.float64)
    lat = np.asarray(u_cube["lat"], dtype=np.float64)
    lon = np.asarray(u_cube["lon"], dtype=np.float64)
    u = np.asarray(u_cube["values"], dtype=np.float64)
    v = np.asarray(v_cube["values"], dtype=np.float64)
    if u.shape != v.shape or u.shape != (len(depth), len(lat), len(lon)):
        raise ValueError("HYCOM u/v 海流分量的网格或深度轴不一致。")
    if not np.allclose(depth, np.asarray(v_cube["depth"], dtype=np.float64), equal_nan=True):
        raise ValueError("HYCOM u/v 海流分量的深度轴不一致。")
    # 前端统一使用北到南的图像行顺序；HYCOM reader 返回南到北，因此翻转。
    u = u[:, ::-1, :]
    v = v[:, ::-1, :]
    lat = lat[::-1]
    mean_u, mean_v = _depth_mean(u, v)
    result = {
        "provider": "HYCOM THREDDS/NCSS",
        "product": u_cube.get("product"),
        "requested_time": _iso(requested),
        "forecast_time": _iso(selected),
        "time_offset_hours": round((selected - requested).total_seconds() / 3600.0, 3),
        "requested_bounds": {
            "west": float(west), "south": float(south),
            "east": float(east), "north": float(north),
        },
        "grid": {
            "rows": len(lat), "columns": len(lon),
            "row_order": "north_to_south",
            "latitudes": _json_array(lat),
            "longitudes": _json_array(lon),
        },
        "depths_m": _json_array(depth, 3),
        "units": {"u_mps": "m/s", "v_mps": "m/s", "depth": "m"},
        "data": {
            "u_mps": _json_array(u),
            "v_mps": _json_array(v),
            "depth_mean_u_mps": _json_array(mean_u),
            "depth_mean_v_mps": _json_array(mean_v),
        },
        "depth_mean_note": "全深度按每个水平网格的有效层对u、v分量分别做算术平均。",
    }
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = result
    return result
