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


def _axis_tiles(start: float, end: float, maximum_span: float) -> list[tuple[float, float]]:
    """把一个坐标轴均分为不超过上游安全跨度的连续区间。"""
    span = float(end) - float(start)
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("HYCOM 区域边界必须是递增的有限数。")
    count = max(1, int(math.ceil(span / maximum_span)))
    step = span / count
    return [
        (float(start) + index * step, float(end) if index + 1 == count else float(start) + (index + 1) * step)
        for index in range(count)
    ]


def _tile_bounds(config: Mapping[str, Any], west: float, south: float, east: float, north: float) -> list[tuple[float, float, float, float]]:
    """按 HYCOM NCSS 的单次限制拆分区域，并预留读取器添加的请求边距。"""
    options = hycom_service._service_config(config)
    request_span = max(0.5, float(options.get("max_request_span_degrees", 12.0)))
    margin = max(0.01, float(options.get("request_margin_degrees", 0.10)))
    # HycomNcssReader 会在每片四周再扩 margin；减去双侧边距和微小浮点余量，
    # 保证传给上游的实际 NCSS 子集始终不超过 request_span。
    core_span = request_span - 2.0 * margin - 1.0e-6
    if core_span <= 0.0:
        raise ValueError("HYCOM 单次区域上限必须大于两倍请求边距。")
    longitude_tiles = _axis_tiles(west, east, core_span)
    latitude_tiles = _axis_tiles(south, north, core_span)
    return [
        (tile_west, tile_south, tile_east, tile_north)
        for tile_south, tile_north in latitude_tiles
        for tile_west, tile_east in longitude_tiles
    ]


def _validate_uv_cubes(u_cube: Mapping[str, Any], v_cube: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(u_cube["depth"], dtype=np.float64)
    lat = np.asarray(u_cube["lat"], dtype=np.float64)
    lon = np.asarray(u_cube["lon"], dtype=np.float64)
    u = np.asarray(u_cube["values"], dtype=np.float64)
    v = np.asarray(v_cube["values"], dtype=np.float64)
    if u.shape != v.shape or u.shape != (len(depth), len(lat), len(lon)):
        raise ValueError("HYCOM u/v 海流分量的网格或深度轴不一致。")
    if not np.allclose(depth, np.asarray(v_cube["depth"], dtype=np.float64), equal_nan=True):
        raise ValueError("HYCOM u/v 海流分量的深度轴不一致。")
    if not np.allclose(lat, np.asarray(v_cube["lat"], dtype=np.float64), equal_nan=True) or not np.allclose(
        lon, np.asarray(v_cube["lon"], dtype=np.float64), equal_nan=True
    ):
        raise ValueError("HYCOM u/v 海流分量的水平网格不一致。")
    return depth, lat, lon, u, v


def _merge_tiles(tiles: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把带边缘重叠的 HYCOM 原生网格片合并，并去除重复坐标。"""
    if not tiles:
        raise ValueError("HYCOM 区域分片为空。")
    if len(tiles) == 1:
        return tiles[0]
    depth = tiles[0][0]
    for tile_depth, *_rest in tiles[1:]:
        if depth.shape != tile_depth.shape or not np.allclose(depth, tile_depth, equal_nan=True):
            raise ValueError("HYCOM 区域分片的深度轴不一致。")

    # HYCOM 原生坐标精度远低于 1e-8 度；量化键仅用于消除相邻分片边距产生的重复行列。
    latitudes = np.asarray(sorted({round(float(value), 8) for tile in tiles for value in tile[1]}), dtype=np.float64)
    longitudes = np.asarray(sorted({round(float(value), 8) for tile in tiles for value in tile[2]}), dtype=np.float64)
    lat_index = {round(float(value), 8): index for index, value in enumerate(latitudes)}
    lon_index = {round(float(value), 8): index for index, value in enumerate(longitudes)}
    shape = (len(depth), len(latitudes), len(longitudes))
    merged_u = np.full(shape, np.nan, dtype=np.float64)
    merged_v = np.full(shape, np.nan, dtype=np.float64)
    depth_indices = np.arange(len(depth))
    for _tile_depth, tile_lat, tile_lon, tile_u, tile_v in tiles:
        row_indices = np.asarray([lat_index[round(float(value), 8)] for value in tile_lat])
        column_indices = np.asarray([lon_index[round(float(value), 8)] for value in tile_lon])
        target = np.ix_(depth_indices, row_indices, column_indices)
        merged_u[target] = tile_u
        merged_v[target] = tile_v
    return depth, latitudes, longitudes, merged_u, merged_v


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

    tile_bounds = _tile_bounds(config, west, south, east, north)
    reader = hycom_service.HycomNcssReader(
        config, time_value=selected.strftime("%Y-%m-%d %H:%M:%S")
    )
    cubes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    product: Any = None
    try:
        for tile_west, tile_south, tile_east, tile_north in tile_bounds:
            query_lon = np.asarray([tile_west, tile_east], dtype=np.float64)
            query_lat = np.asarray([tile_south, tile_north], dtype=np.float64)
            u_cube = reader.read_profile_cube(reader.current_u_tag, query_lon, query_lat)
            v_cube = reader.read_profile_cube(reader.current_v_tag, query_lon, query_lat)
            product = product or u_cube.get("product")
            cubes.append(_validate_uv_cubes(u_cube, v_cube))
    finally:
        reader.close()

    depth, lat, lon, u, v = _merge_tiles(cubes)
    # 前端统一使用北到南的图像行顺序；HYCOM reader 返回南到北，因此翻转。
    u = u[:, ::-1, :]
    v = v[:, ::-1, :]
    lat = lat[::-1]
    mean_u, mean_v = _depth_mean(u, v)
    result = {
        "provider": "HYCOM THREDDS/NCSS",
        "product": product,
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
            "tile_count": len(tile_bounds),
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
