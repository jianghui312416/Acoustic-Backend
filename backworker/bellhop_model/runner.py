# -*- coding: utf-8 -*-
"""三维网页场景使用的多方位 Bellhop 二维执行引擎。

每个方位角都会作为一个独立的 Bellhop 二维剖面计算，可以使用固定、平均或
随距离变化的声速剖面。计算得到的 R/Z 声线和深度/距离传播损失
会旋转到以声源为中心的 X/Y 坐标中，再拼接成后续 NetCDF 导出和前端显示
需要的极坐标三维场。
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import threading
import time
import ctypes
import hashlib
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import netCDF4
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation

from . import environment as bellhop_environment
from . import results as bellhop_results
from .. import ice_reflection_trc
from .. import pm_wave_spectrum
from ..database_source import (
    PostgresBottomGeoacousticsReader,
    create_data_reader,
    data_source_config,
    data_source_file,
    data_source_mode,
    resolve_data_file,
)


# =============================================================================
# 全局常量和通用工具
# =============================================================================

# Bellhop 实现迁入模型子包后，配置中的相对路径仍以 backworker 目录为基准，
# 从而与迁移前的路径解析行为保持一致。
BASE_DIR = Path(__file__).resolve().parent.parent
_NETCDF_READ_CACHE_LOCK = threading.Lock()


class ParameterValidationError(ValueError):
    """可以直接展示给用户的参数校验错误。"""


def _number(value: Any, default: float = 0.0) -> float:
    """把输入转换为浮点数，转换失败时返回默认值。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _finite(value: Any) -> bool:
    """判断输入是否可以转换为有限浮点数。"""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _text(value: Any, default: str = "") -> str:
    """把配置或请求里的值整理成字符串；空值使用默认值。"""
    if value is None or value == "":
        value = default
    return str(value).strip()


def _lower_text(value: Any, default: str = "") -> str:
    """把可选输入整理为小写字符串。"""
    return _text(value, default).lower()


def _upper_text(value: Any, default: str = "") -> str:
    """把可选输入整理为大写字符串。"""
    return _text(value, default).upper()


def _first_value(values: Sequence[Any], default: Any = None) -> Any:
    """返回第一个不是 None 且不是空字符串的值。"""
    for value in values:
        if value is not None and value != "":
            return value
    return default


def _is_bellhopcxx_solver(executable: Any) -> bool:
    """根据可执行文件名称判断是否为支持内部并行的 BellhopCXX。"""
    name = str(executable).lower()
    return "bellhopcxx" in name or "bellhopcuda" in name or "bhc_" in name


def _requested_sea_state(params: Mapping[str, Any]) -> Any:
    """读取门户 seaState；只有请求中显式给出海况时才启用 PM 海面。"""
    containers = []
    portal_request = params.get("portal_request")
    if isinstance(portal_request, Mapping):
        containers.append(portal_request)
    containers.append(params)
    for container in containers:
        for key in ("seaState", "sea_state"):
            if key not in container or container.get(key) in (None, ""):
                continue
            try:
                level = int(round(float(container[key])))
            except (TypeError, ValueError) as exc:
                raise ValueError("海况 seaState 必须是 0 到 9 的整数。") from exc
            if level < 0 or level > 9:
                raise ValueError("海况 seaState 必须位于 0 到 9。")
            return level
    return None


def _wave_random_seed(params: Mapping[str, Any]) -> Any:
    """为同一任务内的随机海面生成稳定种子。"""
    explicit = params.get("wave_random_seed")
    if explicit not in (None, ""):
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
    token = params.get("job_id")
    if token not in (None, ""):
        digest = hashlib.sha256(str(token).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")
    return None


# =============================================================================
# 请求参数校验
# =============================================================================

def validate_params(params: Mapping[str, Any]) -> None:
    """执行与原 MATLAB validateParams 一致的用户输入校验。"""
    lat = _number(params.get("center_lat"), math.nan)
    lon = _number(params.get("center_lon"), math.nan)
    if not math.isfinite(lat) or lat < -90.0 or lat > 90.0:
        raise ParameterValidationError("纬度输入不合法，必须位于 -90 到 90°。")
    if not math.isfinite(lon) or lon < -360.0 or lon > 360.0:
        raise ParameterValidationError("经度输入不合法，必须位于 -360 到 360°。")
    positive = [
        ("freq_hz", "频率"),
        ("source_depth_m", "声源深度"),
        ("receiver_depth_max_m", "最大接收深度"),
        ("radius_km", "半径范围"),
    ]
    for key, label in positive:
        value = _number(params.get(key), math.nan)
        if not math.isfinite(value) or value <= 0.0:
            raise ParameterValidationError(f"{label}必须为正数。")
    n_grid = _number(params.get("n_grid"), math.nan)
    if not math.isfinite(n_grid) or round(n_grid) < 3:
        raise ParameterValidationError("局部网格点数至少为 3。")
    sections = _number(params.get("section_count"), math.nan)
    calculation_mode = _lower_text(params.get("calculation_mode"), "3d")
    minimum_sections = 1 if calculation_mode == "2d" else 3
    if not math.isfinite(sections) or round(sections) < minimum_sections:
        if calculation_mode == "2d":
            raise ParameterValidationError("二维声场必须包含 1 个方位切面。")
        raise ParameterValidationError("三维声场切面数量必须是不小于 3 的整数。")
    for key, label in [
        ("alpha_min", "仰角下限"), ("alpha_max", "仰角上限"),
        ("beta_min", "方位角下限"), ("beta_max", "方位角上限"),
    ]:
        if not _finite(params.get(key)):
            raise ParameterValidationError(f"{label}必须是有限数值。")
    alpha_min = _number(params.get("alpha_min"), math.nan)
    alpha_max = _number(params.get("alpha_max"), math.nan)
    if alpha_min >= alpha_max:
        raise ParameterValidationError("波束角范围必须满足下限小于上限。")
    points = params.get("selection_points", [])
    if not isinstance(points, list) or not points:
        raise ParameterValidationError("当前多方位 2D 拼接版本仅支持一个声源点。")
    for index, point in enumerate(points, 1):
        if not isinstance(point, Mapping):
            raise ParameterValidationError(f"第 {index} 个声源点格式无效。")
        plat = _number(point.get("lat"), math.nan)
        plon = _number(point.get("lon"), math.nan)
        if not math.isfinite(plat) or plat < -90.0 or plat > 90.0:
            raise ParameterValidationError(f"第 {index} 个声源点纬度不合法。")
        if not math.isfinite(plon) or plon < -360.0 or plon > 360.0:
            raise ParameterValidationError(f"第 {index} 个声源点经度不合法。")


# =============================================================================
# 显示参数整理
# =============================================================================

def _safe_visual_range_from_config(config: Mapping[str, Any]) -> Tuple[float, float]:
    """读取有效递增的 TL 显示范围；这里只影响显示，不参与物理参数校验。"""
    visual = config.get("VISUALIZATION") if isinstance(config, Mapping) else None
    visual = visual if isinstance(visual, Mapping) else {}
    task_defaults = config.get("TASK_DEFAULTS") if isinstance(config, Mapping) else None
    task_defaults = task_defaults if isinstance(task_defaults, Mapping) else {}
    fallback_min = _number(task_defaults.get("tl_caxis_min_db"), 60.0)
    fallback_max = _number(task_defaults.get("tl_caxis_max_db"), 120.0)
    if not math.isfinite(fallback_min) or not math.isfinite(fallback_max) or fallback_min >= fallback_max:
        fallback_min, fallback_max = 60.0, 120.0
    cmin = _number(visual.get("MAIN_TL_CAXIS_MIN_DB"), fallback_min)
    cmax = _number(visual.get("MAIN_TL_CAXIS_MAX_DB"), fallback_max)
    if not math.isfinite(cmin) or not math.isfinite(cmax) or cmin >= cmax:
        cmin, cmax = fallback_min, fallback_max
    return float(cmin), float(cmax)


def _normalise_visual_params(params: Dict[str, Any], config: Mapping[str, Any]) -> List[str]:
    """整理可选显示参数；显示参数无效时给出警告，不中断 Bellhop 计算。"""
    warnings: List[str] = []
    default_min, default_max = _safe_visual_range_from_config(config)
    raw_min = params.get("tl_caxis_min_db")
    raw_max = params.get("tl_caxis_max_db")
    cmin = _number(raw_min, math.nan)
    cmax = _number(raw_max, math.nan)
    if not math.isfinite(cmin) or not math.isfinite(cmax) or cmin >= cmax:
        warnings.append(
            f"TL 色标参数无效（min={raw_min!r}, max={raw_max!r}），"
            f"已回退为 {default_min:g} ~ {default_max:g} dB；该参数仅影响显示，不影响 Bellhop 计算。"
        )
        cmin, cmax = default_min, default_max
    params["tl_caxis_min_db"] = float(cmin)
    params["tl_caxis_max_db"] = float(cmax)
    if warnings:
        existing = params.get("_parameter_warnings")
        merged = list(existing) if isinstance(existing, list) else []
        merged.extend(warnings)
        params["_parameter_warnings"] = merged
    return warnings


def _cfg(config: Mapping[str, Any], name: str, default: Any) -> Any:
    """从配置中获取值，支持大小写不敏感。"""
    if name in config and config[name] not in (None, ""):
        return config[name]
    lower = name.lower()
    if lower in config and config[lower] not in (None, ""):
        return config[lower]
    return default


def _required_cfg(config: Mapping[str, Any], name: str) -> Any:
    """从配置中获取必填值，支持大小写不敏感。"""
    value = _cfg(config, name, None)
    if value in (None, ""):
        raise ValueError(f"分类配置文件缺少必填配置：{name}")
    return value


def _task_defaults(config: Mapping[str, Any]) -> Dict[str, Any]:
    """读取并复制后端公共任务默认参数。"""
    defaults = _required_cfg(config, "TASK_DEFAULTS")
    if not isinstance(defaults, Mapping):
        raise ValueError("TASK_DEFAULTS 必须是 JSON 对象")
    return dict(defaults)


# =============================================================================
# 路径和文件兼容处理
# =============================================================================

def _resolve_project_path(value: Any) -> str:
    """把配置或输入路径解析成相对于项目目录的绝对路径。"""
    path = Path(str(value)).expanduser()   # .expanduser()兼容Linux/MacOS的~，Windows下不会有影响
    return str((path if path.is_absolute() else BASE_DIR / path).resolve())


def resolve_configured_data_file(
    config: Mapping[str, Any],
    params: Mapping[str, Any],
    kind: str,
) -> str:
    """按 ``DATA_SOURCES`` 为文件型资源选择本地、api0 或 api1 路径。"""
    mode = data_source_mode(config, kind, "file")
    if mode == "file":
        return _resolve_project_path(
            data_source_file(config, kind, str(params.get("environment_time") or ""))
        )
    return resolve_data_file(
        config,
        mode,
        kind,
        data_source=str(params.get("data_source") or "EnvReanalysis"),
        time_value=str(params.get("environment_time") or ""),
    )


def _windows_short_path(path: str) -> str:
    """在 Windows 可用时返回 ASCII 8.3 短路径。"""
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
    """项目路径含中文时，为 netCDF4/HDF5 准备 ASCII 可读路径。"""
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


@contextmanager
def _open_netcdf_read(path: str):
    """使用兼容中文路径的方式只读打开 NetCDF 文件。"""
    with netCDF4.Dataset(_ascii_netcdf_read_path(path), "r") as dataset:
        yield dataset


def _normalise_lon(lon: float) -> float:
    """把经度规范到 [-180, 180) 范围。"""
    return (float(lon) + 180.0) % 360.0 - 180.0


# =============================================================================
# 坐标、局部网格和海水声速
# =============================================================================

def _source_offsets(points: Sequence[Mapping[str, Any]]) -> List[Tuple[float, float]]:
    """计算多个声源相对中心位置的东向和北向偏移。"""
    lat0, lon0 = float(points[0]["lat"]), _normalise_lon(float(points[0]["lon"]))
    result = []
    for point in points:
        lat = float(point["lat"])
        lon = _normalise_lon(float(point["lon"]))
        x, y = bellhop_results.lonlat_to_local_offset(
            np.asarray(lon), np.asarray(lat), lon0, lat0
        )
        result.append((float(x), float(y)))
    return result


def _core_padded_axis(radius_km: float, core_count: int, padding_km: float) -> Tuple[np.ndarray, int]:
    """生成“传播区域细网格 + 外围稀疏保护区”的声源相对坐标轴。

    ``core_count`` 只定义 ``[-radius_km, radius_km]`` 有效传播区域中的节点数。
    padding 仍完整保留，但使用旧版整幅网格的近似间距追加节点，避免为了短距离
    传播把温盐/声速三维环境场扩张到数百平方节点。
    """
    radius = float(radius_km)
    padding = max(0.0, float(padding_km))
    count = max(3, int(core_count))
    core = np.linspace(-radius, radius, count, dtype=np.float64)
    if padding <= 0.0:
        return core, 0

    # 旧语义中 n_grid 个节点覆盖整个 R + padding 区域；外围继续采用不粗于
    # 该间距的采样，而有效传播区域固定保留 core_count 个节点。
    legacy_spacing = 2.0 * (radius + padding) / float(count - 1)
    padding_count = max(1, int(math.ceil(padding / legacy_spacing)))
    left = np.linspace(-(radius + padding), -radius, padding_count + 1, dtype=np.float64)[:-1]
    right = np.linspace(radius, radius + padding, padding_count + 1, dtype=np.float64)[1:]
    return np.concatenate((left, core, right)), padding_count


def _local_grid(lat: float, lon: float, radius_km: float, n_grid: int, padding_km: float) -> Dict[str, Any]:
    """围绕声源建立局部环境网格；n_grid 表示有效传播区域的节点数。"""
    half = radius_km + padding_km
    relative_axis, padding_count = _core_padded_axis(radius_km, n_grid, padding_km)
    origin_lon_value, origin_lat_value = bellhop_results.local_offset_to_lonlat(
        np.asarray(-half), np.asarray(-half), lon, lat
    )
    origin_lon, origin_lat = float(origin_lon_value), float(origin_lat_value)
    source_x_value, source_y_value = bellhop_results.lonlat_to_local_offset(
        np.asarray(lon), np.asarray(lat), origin_lon, origin_lat
    )
    source_x, source_y = float(source_x_value), float(source_y_value)
    # 以反算得到的声源坐标为中心铺设相对轴，确保 51 等奇数核心网格的中心
    # 节点严格落在声源，而不是受局部投影往返误差影响。
    x_axis = source_x + relative_axis
    y_axis = source_y + relative_axis
    xx, yy = np.meshgrid(x_axis, y_axis)
    grid_lon, grid_lat = bellhop_results.local_offset_to_lonlat(xx, yy, origin_lon, origin_lat)
    x_rel = xx - source_x
    y_rel = yy - source_y
    return {
        "origin_lat": origin_lat, "origin_lon": origin_lon,
        "source_lat": lat, "source_lon": lon,
        "x_axis_km": x_axis, "y_axis_km": y_axis,
        "x_rel_axis_km": x_axis - source_x, "y_rel_axis_km": y_axis - source_y,
        "x_km": xx, "y_km": yy, "x_rel_km": x_rel, "y_rel_km": y_rel,
        "lon": grid_lon, "lat": grid_lat,
        "source_x_km": source_x, "source_y_km": source_y,
        "padding_km": padding_km, "half_km": half,
        "core_grid_count": int(n_grid),
        "padding_grid_count_per_side": padding_count,
        "grid_layout": "core_plus_sparse_padding",
    }


def _mackenzie_sound_speed(temp_c: np.ndarray, salinity: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """使用 Mackenzie 经验公式计算海水声速。"""
    t, s, d = temp_c, salinity, depth_m
    return (
        1448.96 + 4.591 * t - 5.304e-2 * t**2 + 2.374e-4 * t**3
        + 1.340 * (s - 35.0) + 1.630e-2 * d + 1.675e-7 * d**2
        - 1.025e-2 * t * (s - 35.0) - 7.139e-13 * t * d**3
    )


def _find_name(container: Mapping[str, Any], candidates: Sequence[str], label: str) -> str:
    """从候选名称中查找数据容器实际包含的字段名。"""
    names = {}
    for name in container.keys():
        names[name.lower()] = name
    for candidate in candidates:
        if candidate.lower() in names:
            return names[candidate.lower()]
    raise ValueError(f"NetCDF 中找不到{label}变量（候选：{', '.join(candidates)}）")


def _nearest_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """返回参考坐标轴中最接近各查询值的索引。"""
    ref = np.asarray(reference, dtype=np.float64)
    order = np.argsort(ref)
    sorted_ref = ref[order]
    pos = np.searchsorted(sorted_ref, np.asarray(query, dtype=np.float64))
    pos = np.clip(pos, 1, len(sorted_ref) - 1)
    left, right = sorted_ref[pos - 1], sorted_ref[pos]
    choose_right = np.abs(query - right) < np.abs(query - left)
    return order[pos - 1 + choose_right.astype(np.int64)]


def _read_hycom_cube(path: str, variable_names: Sequence[str], query_lon: np.ndarray, query_lat: np.ndarray) -> Dict[str, Any]:
    """按查询范围读取 HYCOM 温盐三维数据子集。"""
    with _open_netcdf_read(path) as ds:
        lon_name = _find_name(ds.variables, ["lon", "longitude"], "经度")
        lat_name = _find_name(ds.variables, ["lat", "latitude"], "纬度")
        depth_name = _find_name(ds.variables, ["depth", "depth_m", "lev"], "深度")
        var_name = _find_name(ds.variables, variable_names, variable_names[0])
        lon = np.asarray(ds.variables[lon_name][:], dtype=np.float64)
        lat = np.asarray(ds.variables[lat_name][:], dtype=np.float64)
        depth = np.asarray(ds.variables[depth_name][:], dtype=np.float64)

        centre = float(np.nanmean(query_lon))
        lon_unwrapped = centre + ((_normalise_lon_array(lon) - centre + 180.0) % 360.0) - 180.0
        query_unwrapped = centre + ((_normalise_lon_array(query_lon) - centre + 180.0) % 360.0) - 180.0
        lat_idx_all = _nearest_indices(lat, query_lat.ravel())
        lon_idx_all = _nearest_indices(lon_unwrapped, query_unwrapped.ravel())
        lat_indices = np.arange(max(0, int(lat_idx_all.min()) - 1), min(len(lat), int(lat_idx_all.max()) + 2))
        # 经度可能跨越 180°，因此保留排序后的紧凑索引集合。
        wanted_lon = np.unique(lon_idx_all)
        if np.ptp(wanted_lon) > len(lon) // 2:
            lon_indices = wanted_lon
        else:
            lon_indices = np.arange(max(0, int(wanted_lon.min()) - 1), min(len(lon), int(wanted_lon.max()) + 2))

        variable = ds.variables[var_name]
        key: List[Any] = []
        kept: List[str] = []
        for dim_name in variable.dimensions:
            low = dim_name.lower()
            if low in {"time", "valid_time"}:
                key.append(0)
            elif low == depth_name.lower() or low in {"depth", "lev"}:
                key.append(slice(None)); kept.append("depth")
            elif low == lat_name.lower() or low in {"lat", "latitude"}:
                key.append(lat_indices); kept.append("lat")
            elif low == lon_name.lower() or low in {"lon", "longitude"}:
                key.append(lon_indices); kept.append("lon")
            else:
                key.append(0)
        raw = variable[tuple(key)]
        values = np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64)
        if set(kept) != {"depth", "lat", "lon"}:
            raise ValueError(f"变量 {var_name} 无法整理为 depth/lat/lon 三维数组")
        values = np.transpose(values, [kept.index("depth"), kept.index("lat"), kept.index("lon")])
        sub_lon = lon_unwrapped[lon_indices]
        sub_lat = lat[lat_indices]
        lon_order = np.argsort(sub_lon)
        lat_order = np.argsort(sub_lat)
        values = values[:, lat_order, :][:, :, lon_order]
        return {
            "depth": depth,
            "lat": sub_lat[lat_order],
            "lon": sub_lon[lon_order],
            "values": values,
            "query_lon": query_unwrapped,
            "centre_lon": centre,
        }


def _normalise_lon_array(values: np.ndarray) -> np.ndarray:
    """把经度数组逐元素规范到 [-180, 180) 范围。"""
    return (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _unwrap_lon_for_cube(cube: Mapping[str, Any], query_lon: np.ndarray) -> np.ndarray:
    """将查询经度展开到环境数据立方体使用的连续经度区间。"""
    centre = float(cube["centre_lon"])
    values = _normalise_lon_array(query_lon)
    return centre + ((values - centre + 180.0) % 360.0) - 180.0


def _profile_cell_cache(
    cube: Dict[str, Any],
    target_depth: np.ndarray,
    query_lon: np.ndarray,
    query_lat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回去重后的垂向插值剖面，以及每个查询点对应的剖面索引。"""
    query_lat = np.asarray(query_lat, dtype=np.float64).ravel()
    query_lon = _unwrap_lon_for_cube(cube, query_lon).ravel()
    lat_idx = _nearest_indices(cube["lat"], query_lat)
    lon_idx = _nearest_indices(cube["lon"], query_lon)
    n_lon = len(cube["lon"])
    keys = lat_idx.astype(np.int64) * n_lon + lon_idx.astype(np.int64)
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    unique_lat_idx = unique_keys // n_lon
    unique_lon_idx = unique_keys % n_lon
    values = cube["values"]
    unique = np.full((len(target_depth), len(unique_keys)), np.nan, dtype=np.float64)
    depth = np.asarray(cube["depth"], dtype=np.float64)
    for point, (iy, ix) in enumerate(zip(unique_lat_idx, unique_lon_idx)):
        profile = values[:, int(iy), int(ix)]
        good = np.isfinite(profile) & np.isfinite(depth)
        if np.count_nonzero(good) >= 4:
            z = depth[good]
            v = profile[good]
            order = np.argsort(z)
            unique[:, point] = np.interp(
                target_depth, z[order], v[order], left=v[order][0], right=v[order][-1]
            )
    return unique, inverse


def _profile_cube_to_queries(
    cube: Dict[str, Any],
    target_depth: np.ndarray,
    query_lon: np.ndarray,
    query_lat: np.ndarray,
) -> np.ndarray:
    """采样剖面时，每个被选中的 HYCOM 网格只计算一次。"""
    profiles, inverse = _profile_cell_cache(cube, target_depth, query_lon, query_lat)
    return profiles[:, inverse]


def _profile_cube_to_grid(
    cube: Dict[str, Any], target_depth: np.ndarray, grid: Mapping[str, Any]
) -> np.ndarray:
    """把剖面数据立方体采样到目标二维经纬度网格。"""
    return _profile_cube_to_queries(cube, target_depth, grid["lon"], grid["lat"])


# =============================================================================
# 三维环境场构建
# =============================================================================
# 从HYCOM数据构建环境场
def _environment_from_profiles(
    temp_profiles: np.ndarray,
    salt_profiles: np.ndarray,
    depth: np.ndarray,
    grid_shape: Tuple[int, int],
) -> Dict[str, Any]:
    """根据已采样的温度和盐度剖面构建求解器环境。"""
    nz, ny, nx = len(depth), *grid_shape
    temp = np.asarray(temp_profiles, dtype=np.float64).reshape(nz, ny, nx).copy()
    salt = np.asarray(salt_profiles, dtype=np.float64).reshape(nz, ny, nx).copy()
    depth_cube = depth[:, None, None]
    sound_speed = _mackenzie_sound_speed(temp, salt, depth_cube)
    for iz in range(nz):
        layer = sound_speed[iz]
        good = np.isfinite(layer) & (layer >= 1400.0) & (layer <= 1700.0)
        fill = float(np.mean(layer[good])) if np.any(good) else 1500.0
        layer[~good] = fill
        sound_speed[iz] = layer
        for field in (temp, salt):
            values = field[iz]
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                values[~np.isfinite(values)] = float(np.mean(finite_values))
    if not np.any(np.isfinite(temp)):
        raise ValueError("HYCOM 温度场在目标区域全部缺测")
    gradient = np.abs(np.diff(temp, axis=0) / np.diff(depth)[:, None, None])
    peak_index = np.nanargmax(np.where(np.isfinite(gradient), gradient, -np.inf), axis=0)
    peak = np.take_along_axis(gradient, peak_index[None, :, :], axis=0)[0]
    thermo_depth = (depth[:-1][peak_index] + depth[1:][peak_index]) * 0.5
    thermo_gradient = peak.astype(np.float64, copy=True)
    weak = peak < 0.02
    thermo_depth[weak] = np.nan
    thermo_gradient[weak] = np.nan
    return {
        "depth_m": depth,
        "sound_speed_mps": sound_speed,
        "temperature_c": temp,
        "salinity": salt,
        "thermocline_depth_m": thermo_depth,
        "thermocline_gradient_c_per_m": thermo_gradient,
    }


def _extend_sound_speed_to_bathymetry(
    environment: Mapping[str, Any],
    bathymetry: Mapping[str, Any],
    fill_speed_mps: float = 1500.0,
) -> Dict[str, Any]:
    """当真实海深超过 SSP 深度轴时，按末端梯度外插声速到海底。"""
    depth = np.asarray(environment["depth_m"], dtype=np.float64).ravel()
    sound_speed = np.asarray(environment["sound_speed_mps"], dtype=np.float64)
    if depth.ndim != 1 or depth.size < 2 or sound_speed.shape[0] != depth.size:
        raise ValueError("声速场深度轴或数据尺寸无效。")

    bottom = np.asarray(bathymetry["depth_m"], dtype=np.float64)
    valid_bottom = np.isfinite(bottom) & (bottom > 0.0)
    land_mask = bathymetry.get("land_mask")
    if land_mask is not None:
        land = np.asarray(land_mask, dtype=bool)
        if land.shape == bottom.shape:
            valid_bottom &= ~land
    if not np.any(valid_bottom):
        return dict(environment)

    deepest_bottom = float(np.max(bottom[valid_bottom]))
    profile_bottom = float(depth[-1])
    tolerance = max(1.0, abs(deepest_bottom)) * 1.0e-12
    if deepest_bottom <= profile_bottom + tolerance:
        return dict(environment)

    # 沿用原深度轴末段的采样间隔，并确保真实最大海深本身成为最后一个节点。
    step = float(depth[-1] - depth[-2])
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("声速场深度轴必须严格递增。")
    extension = np.arange(profile_bottom + step, deepest_bottom, step, dtype=np.float64)
    extension = np.concatenate((extension, np.asarray([deepest_bottom], dtype=np.float64)))

    extended = dict(environment)
    extended["depth_m"] = np.concatenate((depth, extension))
    # 每个水平网格点分别使用最后两个有效深度层的线性梯度外插。HYCOM
    # 最深层以下仍保留压力导致的声速变化，避免整段水体被硬填为 1500 m/s。
    # 1400--1700 m/s 与上游环境场的有效范围一致，可抑制异常末端梯度在
    # 数千米外插距离上被无限放大。仅当末端值或梯度无效时才使用兜底值。
    terminal_speed = np.asarray(sound_speed[-1], dtype=np.float64)
    terminal_gradient = (sound_speed[-1] - sound_speed[-2]) / step
    terminal_speed = np.where(
        np.isfinite(terminal_speed), terminal_speed, float(fill_speed_mps)
    )
    terminal_gradient = np.where(np.isfinite(terminal_gradient), terminal_gradient, 0.0)
    delta_depth = (extension - profile_bottom).reshape(
        (len(extension),) + (1,) * terminal_speed.ndim
    )
    extrapolated_speed = terminal_speed[None, ...] + delta_depth * terminal_gradient[None, ...]
    extrapolated_speed = np.clip(extrapolated_speed, 1400.0, 1700.0)
    extended["sound_speed_mps"] = np.concatenate(
        (sound_speed, extrapolated_speed),
        axis=0,
    )
    # 温盐没有观测依据，仅保持数组维度一致；声学求解使用上面的外插声速层。
    for name in ("temperature_c", "salinity"):
        values = environment.get(name)
        if values is None:
            continue
        array = np.asarray(values, dtype=np.float64)
        if array.shape[0] == depth.size:
            extended[name] = np.concatenate(
                (array, np.full((len(extension),) + array.shape[1:], np.nan, dtype=np.float64)),
                axis=0,
            )
    return extended


def _load_sound_speed(
    data: "_DataAccess",
    grid: Dict[str, Any],
    depth_grid_m: Sequence[float],
) -> Dict[str, Any]:
    """读取温盐数据并计算目标深度轴上的三维声速场。"""
    temp_cube = data.read_hycom_cube("temperature", ["water_temp", "temperature", "thetao"], grid["lon"], grid["lat"])
    salt_cube = data.read_hycom_cube("salinity", ["salinity", "so"], grid["lon"], grid["lat"])
    configured_depth = np.asarray(depth_grid_m, dtype=np.float64)
    if configured_depth.ndim != 1 or len(configured_depth) < 4 or np.any(np.diff(configured_depth) <= 0.0):
        raise ValueError("DEPTH_GRID_M 必须是至少含 4 个严格递增深度的数组")
    max_depth = min(
        float(np.nanmax(temp_cube["depth"])),
        float(np.nanmax(salt_cube["depth"])),
        float(configured_depth[-1]),
    )
    depth = configured_depth[configured_depth <= max_depth]
    if len(depth) < 4:
        raise ValueError("HYCOM 有效深度层不足，无法构建三维声速场")
    temp = _profile_cube_to_grid(temp_cube, depth, grid)
    salt = _profile_cube_to_grid(salt_cube, depth, grid)
    return _environment_from_profiles(temp, salt, depth, grid["lat"].shape)


class TopographyReader:
    """统一地形读取器，按扩展名自动选择 Sandwell .img 或 GEBCO NetCDF。"""

    NX = 21600
    NY = 17280

    def __init__(self, path: str):
        """打开地形文件：.nc 走 GEBCO NetCDF，其余按 Sandwell .img 处理。"""
        self.path = str(path)
        self._gebco = self.path.lower().endswith(".nc")
        if self._gebco:
            self._init_gebco()
        else:
            self._init_img()

    def _init_img(self):
        """打开 Sandwell 二进制地形文件并建立内存映射。"""
        expected = self.NX * self.NY * 2
        actual = os.path.getsize(self.path)
        if actual != expected:
            raise ValueError(f"地形 IMG 尺寸应为 {expected} 字节，实际为 {actual} 字节")
        self.data = np.memmap(self.path, dtype=">i2", mode="r", shape=(self.NY, self.NX))

    def _init_gebco(self):
        """惰性打开 GEBCO NetCDF，只读取坐标轴，高程按需切片读取。"""
        try:
            import h5py
        except ImportError as error:  # pragma: no cover - 依赖缺失时给出明确提示
            raise ValueError("读取 GEBCO NetCDF 需要 h5py，请先安装 h5py。") from error
        self._h5 = h5py.File(self.path, "r")
        try:
            self._lon = np.asarray(self._h5["lon"][:], dtype=np.float64)
            self._lat = np.asarray(self._h5["lat"][:], dtype=np.float64)
            self._elevation = self._h5["elevation"]
            fill = getattr(self._elevation, "fillvalue", None)
            self._fill = float(fill) if fill is not None else -32767.0
        except Exception:
            self._h5.close()
            raise
        if self._lon.size < 2 or self._lat.size < 2:
            self._h5.close()
            raise ValueError("GEBCO NetCDF 缺少有效的 lon/lat 坐标轴。")

    def close(self):
        """释放 GEBCO 文件句柄；.img 使用内存映射，无需显式释放。"""
        handle = getattr(self, "_h5", None)
        if handle is not None:
            handle.close()
            self._h5 = None

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """按经纬度数组双线性采样地形高程。"""
        if getattr(self, "_gebco", False):
            return self._sample_gebco(lon, lat)
        return self._sample_img(lon, lat)

    def _sample_img(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """按 Sandwell 1 分钟 Mercator 网格双线性采样。"""
        lon_values = np.asarray(lon, dtype=np.float64)
        # 当前限制纬度范围 [-80.72, 80.72]，有新的数据来源记得修改
        lat_values = np.clip(np.asarray(lat, dtype=np.float64), -80.72, 80.72)
        # 因为这个地形文件横向有：21600 = 360° × 60个网格点，此步骤把“度”转换成“网格编号”
        x = (lon_values % 360.0) * 60.0
        # 把纬度转换成 Mercator 投影下的纵向网格坐标。
        # topo.img 是 Mercator 投影，纬度方向不是线性排列，需要使用 Mercator 投影的公式
        # Mercator 的核心公式是Y = ln[tan(π/4 + φ/2)]，φ = 纬度（弧度）
        y = 8640.0 - (10800.0 / math.pi) * np.log(np.tan(math.pi / 4.0 + np.deg2rad(lat_values) / 2.0))
        x0 = np.floor(x).astype(np.int64) % self.NX
        y0 = np.clip(np.floor(y).astype(np.int64), 0, self.NY - 2)
        x1 = (x0 + 1) % self.NX
        y1 = y0 + 1
        wx, wy = x - np.floor(x), y - np.floor(y)
        z00 = np.asarray(self.data[y0, x0], dtype=np.float64)
        z10 = np.asarray(self.data[y0, x1], dtype=np.float64)
        z01 = np.asarray(self.data[y1, x0], dtype=np.float64)
        z11 = np.asarray(self.data[y1, x1], dtype=np.float64)
        result = (1 - wy) * ((1 - wx) * z00 + wx * z10) + wy * ((1 - wx) * z01 + wx * z11)
        result[np.abs(result) >= 32700] = np.nan
        return result

    def _sample_gebco(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """按 GEBCO 等经纬度网格双线性采样，只读取查询点的包围盒。"""
        lon_values = np.asarray(lon, dtype=np.float64)
        lat_values = np.asarray(lat, dtype=np.float64)
        shape = lon_values.shape
        nx = self._lon.size
        ny = self._lat.size
        lon_norm = np.mod(lon_values, 360.0)
        lon_norm = np.where(lon_norm > 180.0, lon_norm - 360.0, lon_norm)
        x = (lon_norm + 180.0) / 360.0 * nx - 0.5
        y = (lat_values + 90.0) / 180.0 * ny - 0.5
        x = np.clip(x, 0.0, nx - 1.0)
        y = np.clip(y, 0.0, ny - 1.0)
        x0 = np.clip(np.floor(x).astype(np.int64), 0, nx - 1)
        y0 = np.clip(np.floor(y).astype(np.int64), 0, ny - 1)
        x1 = np.clip(x0 + 1, 0, nx - 1)
        y1 = np.clip(y0 + 1, 0, ny - 1)
        wx = x - np.floor(x)
        wy = y - np.floor(y)

        iy_min, iy_max = int(y0.min()), int(y1.max())
        ix_min, ix_max = int(x0.min()), int(x1.max())
        block = np.asarray(
            self._elevation[iy_min:iy_max + 1, ix_min:ix_max + 1], dtype=np.float64
        )
        block[block == self._fill] = np.nan

        z00 = block[y0 - iy_min, x0 - ix_min]
        z10 = block[y0 - iy_min, x1 - ix_min]
        z01 = block[y1 - iy_min, x0 - ix_min]
        z11 = block[y1 - iy_min, x1 - ix_min]
        result = (1.0 - wy) * ((1.0 - wx) * z00 + wx * z10) + wy * ((1.0 - wx) * z01 + wx * z11)
        return result.reshape(shape)


# =============================================================================
# 地形、海浪和海底声学参数
# =============================================================================

def _bathymetry_from_elevation(
    elevation: np.ndarray, centre_elevation: float, source_depth_m: float
) -> Dict[str, Any]:
    """把已采样地形值整理成 Bellhop 可用的海底水深。"""
    elevation = np.asarray(elevation, dtype=np.float64)
    ocean = np.isfinite(elevation) & (elevation < 0.0)
    land = np.isfinite(elevation) & (elevation >= 0.0)
    missing = ~np.isfinite(elevation)
    if not np.any(ocean):
        raise ValueError("当前地形区域没有有效海洋网格，无法构建多方位 2D 海底剖面")
    depth = np.full_like(elevation, np.nan)
    # 海底水深定义为负值，所以需要取负
    depth[ocean] = -elevation[ocean]
    median = float(np.median(depth[ocean]))
    depth[missing] = median
    depth[land] = 1.0

    if not math.isfinite(centre_elevation) or centre_elevation >= 0.0:
        raise ValueError("声源位置落在陆地区域，不能进行水下声传播计算。请调整中心经纬度。")
    centre_depth = -centre_elevation
    if centre_depth <= source_depth_m:
        raise ValueError(
            f"声源深度 {source_depth_m:.1f} m 不小于当地水深 {centre_depth:.1f} m，请减小声源深度或调整位置。"
        )
    return {
        "elevation_m": elevation, "depth_m": depth,
        "ocean_mask": ocean, "land_mask": land, "missing_mask": missing,
        "median_ocean_depth_m": median, "source_bottom_depth_m": centre_depth,
        "land_depth_m": 1.0,
    }


def _build_bathymetry(reader: TopographyReader, grid: Dict[str, Any], source_depth_m: float) -> Dict[str, Any]:
    """根据声学网格和声源位置，构建 Bellhop 可用的海底水深。"""
    elevation = reader.sample(grid["lon"], grid["lat"])
    centre_elevation = float(reader.sample(
        np.asarray([grid["source_lon"]]), np.asarray([grid["source_lat"]])
    )[0])
    return _bathymetry_from_elevation(elevation, centre_elevation, source_depth_m)


def _read_scalar_grid_field(
    path: str,
    candidates: Sequence[str],
    query_lon: np.ndarray,
    query_lat: np.ndarray,
) -> np.ndarray:
    """从二维或带时间维的 NetCDF 网格中按最近格点读取一个标量字段。"""
    with _open_netcdf_read(path) as ds:
        lon_name = _find_name(ds.variables, ["longitude", "lon"], "网格经度")
        lat_name = _find_name(ds.variables, ["latitude", "lat"], "网格纬度")
        var_name = _find_name(ds.variables, candidates, candidates[0])
        lon = np.asarray(ds.variables[lon_name][:], dtype=np.float64)
        lat = np.asarray(ds.variables[lat_name][:], dtype=np.float64)
        qlon = np.asarray(query_lon, dtype=np.float64)
        if np.nanmax(lon) > 180.0:
            qlon = qlon % 360.0
        ilat = _nearest_indices(lat, query_lat.ravel())
        ilon = _nearest_indices(lon, qlon.ravel())
        var = ds.variables[var_name]
        dims = [name.lower() for name in var.dimensions]
        keys = ilat.astype(np.int64) * len(lon) + ilon.astype(np.int64)
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        result_unique = np.empty(len(unique_keys), dtype=np.float64)
        # 保持标量 NetCDF 读取语义，同时避免相邻查询点反复读取同一网格单元。
        for i, key_value in enumerate(unique_keys):
            iy, ix = divmod(int(key_value), len(lon))
            key = []
            for dim in dims:
                if dim in {"time", "valid_time"}:
                    key.append(0)
                elif dim in {lat_name.lower(), "lat", "latitude"}:
                    key.append(int(iy))
                elif dim in {lon_name.lower(), "lon", "longitude"}:
                    key.append(int(ix))
                else:
                    key.append(0)
            value = var[tuple(key)]
            result_unique[i] = float(np.ma.filled(value, np.nan))
        return result_unique[inverse].reshape(query_lat.shape)


def _read_wave_field(path: str, candidates: Sequence[str], query_lon: np.ndarray, query_lat: np.ndarray) -> np.ndarray:
    """兼容旧调用名；波浪与底质标量网格共用最近邻读取实现。"""
    return _read_scalar_grid_field(path, candidates, query_lon, query_lat)


BOTTOM_GEO_FIELDS = (
    "compressional_speed_mps",
    "shear_speed_mps",
    "density_g_cm3",
    "compressional_attenuation",
    "shear_attenuation",
)


def _load_bottom_geoacoustics(
    data: "_DataAccess",
    grid: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> Dict[str, Any]:
    """读取或构造长格式 BTY 需要的五个半空间海底声学参数。"""
    defaults = settings["bottom_fixed"]
    shape = np.asarray(grid["lat"]).shape
    if settings["bottom_geo_mode"] == "fixed":
        return {
            "mode": "fixed",
            "source": str(settings.get("bottom_fixed_source", "BELLHOP_ENV.BOTTOM fixed values")),
            "interpolation": "linear",
            "fields": {
                name: np.full(shape, float(defaults[name]), dtype=np.float64)
                for name in BOTTOM_GEO_FIELDS
            },
        }

    required = set(settings["bottom_geo_required_fields"])
    candidates_by_field = settings["bottom_geo_variables"]
    scale_by_field = settings["bottom_geo_scale"]
    offset_by_field = settings["bottom_geo_offset"]
    ranges_by_field = settings["bottom_geo_valid_ranges"]
    raw_fields = data.read_bottom_geoacoustic_fields(
        candidates_by_field,
        grid["lon"],
        grid["lat"],
    )
    fields: Dict[str, np.ndarray] = {}
    loaded_fields: List[str] = []
    fallback_fields: List[str] = []
    for name in BOTTOM_GEO_FIELDS:
        values = raw_fields.get(name)
        if values is None:
            if name in required:
                raise ValueError(
                    f"海底声学数据源缺少必填字段 {name!r}"
                )
            values = np.full(shape, float(defaults[name]), dtype=np.float64)
            fallback_fields.append(name)
        else:
            values = (
                np.asarray(values, dtype=np.float64) * float(scale_by_field[name])
                + float(offset_by_field[name])
            )
        if values.shape != shape:
            raise ValueError(f"海底声学字段 {name!r} 的网格形状无效")
        low, high = (float(value) for value in ranges_by_field[name])
        finite = np.isfinite(values)
        if np.any(finite):
            loaded_fields.append(name)
        if np.any(~finite) and name not in fallback_fields:
            fallback_fields.append(name)
        outside = finite & ((values < low) | (values > high))
        if np.any(outside):
            sample = float(values[outside][0])
            raise ValueError(
                f"海底声学字段 {name!r} 包含超出配置范围 [{low:g}, {high:g}] 的值 {sample:g}。"
                "请检查 SCALE/OFFSET 和单位。"
            )
        values = values.copy()
        values[~finite] = float(defaults[name])
        fields[name] = values
    return {
        "mode": str(settings["bottom_geo_mode"]),
        "source": data.bottom_geo_source,
        "interpolation": str(settings.get("bottom_geo_interpolation", "linear")),
        "loaded_fields": loaded_fields,
        "fallback_fields": fallback_fields,
        "fields": fields,
    }


def _altimetry_from_wave_fields(
    hs: np.ndarray,
    tp: np.ndarray,
    direction: np.ndarray,
    grid: Dict[str, Any],
    bathymetry: Dict[str, Any],
) -> Dict[str, Any]:
    """把已有波面模型应用到已采样的海浪场。"""

    def fill(values: np.ndarray, default: float) -> np.ndarray:
        """用有限值中位数或默认值填补波浪字段空值。"""
        finite_values = values[np.isfinite(values)]
        value = float(np.median(finite_values)) if finite_values.size else default
        return np.where(np.isfinite(values), values, value)

    hs = np.maximum(fill(hs, 0.0), 0.0)
    tp = fill(tp, 8.0)
    tp[tp <= 0.0] = 8.0
    direction = fill(direction, 0.0)
    amplitude = hs * 0.5
    wavelength = 9.81 * tp**2 / (2.0 * math.pi)
    wave_number = 2.0 * math.pi / wavelength
    phase_distance = grid["x_km"] * 1000.0 * np.cos(np.deg2rad(direction)) + grid["y_km"] * 1000.0 * np.sin(np.deg2rad(direction))
    elevation = amplitude * np.cos(wave_number * phase_distance)
    elevation -= float(np.mean(elevation[np.isfinite(elevation)]))

    land = bathymetry["land_mask"]
    calm = binary_dilation(land, iterations=1) if np.any(land) else land.copy()
    elevation[calm] = 0.0
    limit = np.maximum(0.0, bathymetry["depth_m"] - 0.5)
    elevation = np.clip(elevation, -limit, limit)
    elevation[land] = 0.0
    return {
        "model": "file",
        "hs_m": hs,
        "tp_s": tp,
        "direction_deg": direction,
        "elevation_m": elevation,
    }


def _build_altimetry(data: "_DataAccess", grid: Dict[str, Any], bathymetry: Dict[str, Any]) -> Dict[str, Any]:
    """按门户海况生成 PM 海面，或读取波浪场构建海面起伏数据。"""
    sea_state = getattr(data, "requested_sea_state", None)
    if sea_state is not None:
        return pm_wave_spectrum.build_pm_altimetry(
            sea_state,
            grid,
            bathymetry,
            seed=getattr(data, "wave_seed", None),
        )
    hs = data.read_wave_field("hs", ["VHM0", "HTSGW", "swh"], grid["lon"], grid["lat"])
    tp = data.read_wave_field("tp", ["VTPK", "PERPW", "peak_wave_period"], grid["lon"], grid["lat"])
    direction = data.read_wave_field("direction", ["VMDR", "DIRPW", "mean_wave_direction"], grid["lon"], grid["lat"])
    return _altimetry_from_wave_fields(hs, tp, direction, grid, bathymetry)


# =============================================================================
# 多声源共享环境缓存
# =============================================================================

class SharedSourceEnvironment:
    """多声源共用一次环境读取，同时保留每个声源自己的精确局部网格视图。"""

    def __init__(
        self,
        depth: np.ndarray,
        temp_profiles: np.ndarray,
        temp_inverse: np.ndarray,
        salt_profiles: np.ndarray,
        salt_inverse: np.ndarray,
        elevation: np.ndarray,
        centre_elevation: np.ndarray,
        hs: np.ndarray,
        tp: np.ndarray,
        direction: np.ndarray,
        bottom_geoacoustics: Mapping[str, Any],
        slices: Sequence[slice],
        *,
        wave_model: str = "file",
        sea_state: Any = None,
        wave_seed: Any = None,
    ) -> None:
        """保存多声源共享的环境数组和数据访问统计信息。"""
        self.depth = depth
        self.temp_profiles = temp_profiles
        self.temp_inverse = temp_inverse
        self.salt_profiles = salt_profiles
        self.salt_inverse = salt_inverse
        self.elevation = elevation
        self.centre_elevation = centre_elevation
        self.hs = hs
        self.tp = tp
        self.direction = direction
        self.wave_model = str(wave_model)
        self.sea_state = sea_state
        self.wave_seed = wave_seed
        self.bottom_geo_mode = str(bottom_geoacoustics["mode"])
        self.bottom_geo_source = str(bottom_geoacoustics["source"])
        self.bottom_geo_interpolation = str(bottom_geoacoustics.get("interpolation", "linear"))
        self.bottom_geo_loaded_fields = list(bottom_geoacoustics.get("loaded_fields", []))
        self.bottom_geo_fallback_fields = list(bottom_geoacoustics.get("fallback_fields", []))
        self.bottom_geo_fields = {
            name: np.asarray(values, dtype=np.float64).ravel()
            for name, values in bottom_geoacoustics["fields"].items()
        }
        self.slices = list(slices)
        self.point_count = int(len(elevation))
        self.temperature_cell_count = int(temp_profiles.shape[1])
        self.salinity_cell_count = int(salt_profiles.shape[1])

    def for_source(
        self, source_index: int, grid: Dict[str, Any], source_depth_m: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """提取单个声源对应的环境、地形、海面和海底数据。"""
        point_slice = self.slices[source_index]
        shape = grid["lat"].shape
        temperature = self.temp_profiles[:, self.temp_inverse[point_slice]]
        salinity = self.salt_profiles[:, self.salt_inverse[point_slice]]
        environment = _environment_from_profiles(temperature, salinity, self.depth, shape)
        elevation = self.elevation[point_slice].reshape(shape)
        bathymetry = _bathymetry_from_elevation(
            elevation, float(self.centre_elevation[source_index]), source_depth_m
        )
        environment = _extend_sound_speed_to_bathymetry(environment, bathymetry)
        if self.wave_model == "pierson_moskowitz":
            seed = (
                None
                if self.wave_seed is None
                else (int(self.wave_seed) + 0x9E3779B1 * int(source_index)) % (2**32)
            )
            altimetry = pm_wave_spectrum.build_pm_altimetry(
                self.sea_state,
                grid,
                bathymetry,
                seed=seed,
            )
        else:
            altimetry = _altimetry_from_wave_fields(
                self.hs[point_slice].reshape(shape),
                self.tp[point_slice].reshape(shape),
                self.direction[point_slice].reshape(shape),
                grid,
                bathymetry,
            )
        bottom_geoacoustics = {
            "mode": self.bottom_geo_mode,
            "source": self.bottom_geo_source,
            "interpolation": self.bottom_geo_interpolation,
            "loaded_fields": list(self.bottom_geo_loaded_fields),
            "fallback_fields": list(self.bottom_geo_fallback_fields),
            "fields": {
                name: values[point_slice].reshape(shape)
                for name, values in self.bottom_geo_fields.items()
            },
        }
        return environment, bathymetry, altimetry, bottom_geoacoustics


def _prepare_shared_source_environment(
    data: "_DataAccess",
    topo: TopographyReader,
    grids: Sequence[Dict[str, Any]],
    depth_grid_m: Sequence[float],
    settings: Mapping[str, Any],
) -> SharedSourceEnvironment:
    """一次性读取所有局部网格的并集，并保存每个网格对应的精确切片。"""
    if not grids:
        raise ValueError("至少需要一个声源局部网格")
    slices: List[slice] = []
    offset = 0
    all_lon_parts: List[np.ndarray] = []
    all_lat_parts: List[np.ndarray] = []
    for grid in grids:
        count = int(np.asarray(grid["lat"]).size)
        slices.append(slice(offset, offset + count))
        offset += count
        all_lon_parts.append(np.asarray(grid["lon"], dtype=np.float64).ravel())
        all_lat_parts.append(np.asarray(grid["lat"], dtype=np.float64).ravel())
    all_lon = np.concatenate(all_lon_parts)
    all_lat = np.concatenate(all_lat_parts)

    temp_cube = data.read_hycom_cube("temperature", ["water_temp", "temperature", "thetao"], all_lon, all_lat)
    salt_cube = data.read_hycom_cube("salinity", ["salinity", "so"], all_lon, all_lat)
    configured_depth = np.asarray(depth_grid_m, dtype=np.float64)
    if configured_depth.ndim != 1 or len(configured_depth) < 4 or np.any(np.diff(configured_depth) <= 0.0):
        raise ValueError("DEPTH_GRID_M 必须是至少含 4 个严格递增深度的数组")
    max_depth = min(
        float(np.nanmax(temp_cube["depth"])),
        float(np.nanmax(salt_cube["depth"])),
        float(configured_depth[-1]),
    )
    depth = configured_depth[configured_depth <= max_depth]
    if len(depth) < 4:
        raise ValueError("HYCOM 有效深度层不足，无法构建三维声速场")
    temp_profiles, temp_inverse = _profile_cell_cache(temp_cube, depth, all_lon, all_lat)
    salt_profiles, salt_inverse = _profile_cell_cache(salt_cube, depth, all_lon, all_lat)

    elevation = topo.sample(all_lon, all_lat).ravel()
    centre_lon = np.asarray([grid["source_lon"] for grid in grids], dtype=np.float64)
    centre_lat = np.asarray([grid["source_lat"] for grid in grids], dtype=np.float64)
    centre_elevation = topo.sample(centre_lon, centre_lat).ravel()
    sea_state = getattr(data, "requested_sea_state", None)
    if sea_state is not None:
        hs_value, tp_value, _ = pm_wave_spectrum.sea_state_parameters(sea_state)
        hs = np.full(all_lon.shape, hs_value, dtype=np.float64)
        tp = np.full(all_lon.shape, tp_value, dtype=np.float64)
        direction = np.zeros(all_lon.shape, dtype=np.float64)
        wave_model = "pierson_moskowitz"
        wave_seed = getattr(data, "wave_seed", None)
    else:
        hs = data.read_wave_field("hs", ["VHM0", "HTSGW", "swh"], all_lon, all_lat).ravel()
        tp = data.read_wave_field("tp", ["VTPK", "PERPW", "peak_wave_period"], all_lon, all_lat).ravel()
        direction = data.read_wave_field("direction", ["VMDR", "DIRPW", "mean_wave_direction"], all_lon, all_lat).ravel()
        wave_model = "file"
        wave_seed = None
    bottom_geoacoustics = _load_bottom_geoacoustics(
        data, {"lon": all_lon, "lat": all_lat}, settings
    )
    return SharedSourceEnvironment(
        depth, temp_profiles, temp_inverse, salt_profiles, salt_inverse,
        elevation, centre_elevation, hs, tp, direction, bottom_geoacoustics, slices,
        wave_model=wave_model,
        sea_state=sea_state,
        wave_seed=wave_seed,
    )


# =============================================================================
# 二维输入文件生成（Bellhop）
# =============================================================================

def _write_vector(handle, values: np.ndarray, slash: bool = False) -> None:
    """向 Bellhop 文本输入文件写入一行数值向量。"""
    arr = np.asarray(values).ravel()
    if slash:
        starts = list(range(0, len(arr), 8)) or [0]
        for chunk_index, start in enumerate(starts):
            handle.write(" ".join(f"{value:.6f}" for value in arr[start:start + 8]))
            handle.write(" /\n" if chunk_index == len(starts) - 1 else "\n")
    else:
        handle.write(" ".join(f"{value:.6f}" for value in arr) + "\n")


def _bearing_angles(params: Mapping[str, Any]) -> np.ndarray:
    """返回网页表单请求的去重二维剖面方位角。"""
    calculation_mode = _lower_text(params.get("calculation_mode"), "3d")
    minimum_count = 1 if calculation_mode == "2d" else 3
    count = max(minimum_count, int(round(float(params["section_count"]))))
    start = float(params["beta_min"])
    stop = float(params["beta_max"])
    if count == 1:
        return np.asarray([start % 360.0], dtype=np.float64)
    span = stop - start
    # 旧版网页用 0..359 表示完整一圈。这里按完整一圈处理，
    # 避免在几乎相同的 0°/359° 剖面上重复计算。
    if abs(span) >= 359.0 - 1.0e-9:
        direction = -1.0 if span < 0.0 else 1.0
        values = start + direction * np.linspace(0.0, 360.0, count, endpoint=False)
    else:
        values = np.linspace(start, stop, count)
    return np.mod(values, 360.0).astype(np.float64)

# 从3D环境场采样2D切片
def _sample_2d_slice(
    grid: Dict[str, Any],
    environment: Dict[str, Any],
    bathymetry: Dict[str, Any],
    altimetry: Dict[str, Any],
    bottom_geoacoustics: Mapping[str, Any],
    bearing_deg: float,
    calculation_range_km: float,
    profile_count: int,
    extra_range_km: float = 0.0,
) -> Dict[str, Any]:
    """沿声源中心的某个方位，从三维环境场采样出二维剖面。"""
    from ..acoustic_environment import range_samples

    ranges = range_samples(calculation_range_km, profile_count, extra_range_km)
    angle = math.radians(float(bearing_deg))
    # 参数 bearing_deg 是导航方位角：0° 指北，顺时针增大。
    # 环境网格中 x 表示东、y 表示北，所以东向分量是 sin(bearing)，不是 cos(bearing)。
    x = ranges * math.sin(angle)
    y = ranges * math.cos(angle)
    points = np.column_stack([y, x])
    axes = (
        np.asarray(grid["y_rel_axis_km"], dtype=np.float64),
        np.asarray(grid["x_rel_axis_km"], dtype=np.float64),
    )

    def sample(values: np.ndarray, method: str = "linear") -> np.ndarray:
        """沿当前方位射线对二维环境字段进行插值采样。"""
        interpolator = RegularGridInterpolator(
            axes, np.asarray(values, dtype=np.float64), method=method,
            bounds_error=False, fill_value=np.nan,
        )
        return interpolator(points)

    speed = np.vstack([sample(layer) for layer in environment["sound_speed_mps"]])
    for row in speed:
        finite = row[np.isfinite(row)]
        row[~np.isfinite(row)] = float(np.mean(finite)) if finite.size else 1500.0
    bottom = sample(bathymetry["depth_m"])
    surface_depth = -sample(altimetry["elevation_m"])
    land = sample(np.asarray(bathymetry["land_mask"], dtype=np.float64), method="nearest") >= 0.5
    model_depth = float(environment["depth_m"][-1])
    bottom = np.clip(bottom, 1.0, model_depth)
    surface_depth = np.where(np.isfinite(surface_depth), surface_depth, 0.0)
    # 距离相关 Q-SSP 从 0 m 深度开始，Bellhop 2D 不能接受高于该深度的波峰。
    # 因此保留波谷，并把波峰裁剪到声速剖面的上边界。
    surface_depth = np.clip(surface_depth, 0.0, bottom - 0.5)
    bottom_interpolation = str(bottom_geoacoustics.get("interpolation", "linear"))
    bottom_fields = {
        name: sample(np.asarray(values, dtype=np.float64), method=bottom_interpolation)
        for name, values in bottom_geoacoustics["fields"].items()
    }
    for name, values in bottom_fields.items():
        fallback = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else 0.0
        values[~np.isfinite(values)] = fallback
    return {
        "bearing_deg": float(bearing_deg),
        "range_km": ranges,
        "x_km": x,
        "y_km": y,
        "sound_speed_mps": speed,
        "bottom_depth_m": bottom,
        "surface_depth_m": surface_depth,
        "land_mask": land,
        "bottom_geoacoustics": {
            "mode": str(bottom_geoacoustics["mode"]),
            "source": str(bottom_geoacoustics["source"]),
            "interpolation": bottom_interpolation,
            "loaded_fields": list(bottom_geoacoustics.get("loaded_fields", [])),
            "fallback_fields": list(bottom_geoacoustics.get("fallback_fields", [])),
            "fields": bottom_fields,
        },
    }


def _apply_ssp_mode_to_section(section: Dict[str, Any], ssp_mode: str) -> Dict[str, Any]:
    """按有效 SSP 模式整理切面；固定模式复用距离 0 km 处的声速剖面。"""
    if ssp_mode != "fixed":
        return section
    ranges = np.asarray(section["range_km"], dtype=np.float64)
    sound_speed = np.asarray(section["sound_speed_mps"], dtype=np.float64)
    if ranges.ndim != 1 or ranges.size == 0:
        raise ValueError("固定 SSP 的距离轴必须是一维非空数组")
    if sound_speed.ndim != 2 or sound_speed.shape[1] != ranges.size:
        raise ValueError("固定 SSP 声速网格必须是 (depth, range) 形状")
    fixed_section = dict(section)
    fixed_section["sound_speed_mps"] = np.repeat(sound_speed[:, :1], ranges.size, axis=1)
    return fixed_section


def _water_column_mode_name(ssp_mode: str) -> str:
    """把内部 SSP 模式转换为结果摘要使用的水体模式名称。"""
    return {
        "range_dependent": "range_dependent",
        "fixed": "fixed_source_profile",
        "mean": "mean_profile",
    }.get(ssp_mode, str(ssp_mode))

# 写入2D SSP文件
def _write_2d_ssp(path: Path, section: Dict[str, Any]) -> None:
    """通过环境文件模块写出 Bellhop 距离相关 SSP。"""
    bellhop_environment.write_ssp_file(path, section)


def _section_with_left_guard(section: Dict[str, Any], guard_km: float = 0.5) -> Dict[str, Any]:
    """通过环境文件模块给剖面增加 Q 插值所需的左侧保护点。"""
    return bellhop_environment.add_left_guard(section, guard_km)


def _write_2d_boundary(
    path: Path,
    ranges_km: np.ndarray,
    depths_m: np.ndarray,
    interpolation: str = "L",
    geoacoustics=None,
) -> None:
    """通过环境文件模块写出海面或海底二维边界文件。"""
    bellhop_environment.write_boundary_file(
        path,
        ranges_km,
        depths_m,
        interpolation=interpolation,
        geoacoustics=geoacoustics,
    )


def _write_2d_env(
    path: Path,
    params: Mapping[str, Any],
    settings: Mapping[str, Any],
    environment: Dict[str, Any],
    section: Dict[str, Any],
    calculation_radius_km: float,
    receiver_depth_max_m: float,
    run_type: str,
) -> None:
    """通过环境文件模块写出 Bellhop 二维 ENV 文件。"""
    bellhop_environment.write_environment_file(
        path,
        params,
        settings,
        environment,
        section,
        calculation_radius_km,
        receiver_depth_max_m,
        run_type,
    )


def _stage_ice_reflection_for_env(
    env_path: Path,
    params: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> None:
    """为单个 ENV 文件生成并布置冰面反射系数表。"""
    if not settings.get("ice_reflection_enabled"):
        return
    trc_path = env_path.with_suffix(".trc")
    kwargs: Dict[str, Any] = {
        "source_lat": float(params["center_lat"]),
        "source_lon": float(params["center_lon"]),
        "domain_radius_km": float(settings.get("ice_reflection_domain_radius_km") or params["radius_km"]),
        "aggregate": str(settings.get("ice_reflection_domain_aggregate") or "median"),
        "min_domain_cells": int(settings.get("ice_reflection_min_domain_cells") or 3),
        "force_model": str(settings.get("ice_reflection_force_model") or "auto"),
        "enforce_passive": bool(settings.get("ice_reflection_enforce_passive", True)),
        "data_source": str(settings.get("ice_reflection_data_source") or "HYCOM"),
    }
    bumden = settings.get("ice_reflection_bumden_domain")
    if bumden is None:
        ice_data = ice_reflection_trc.selected_ice_data_path(
            settings.get("ice_reflection_ice_data")
        )
        if not ice_data:
            raise ValueError(
                "Ice 接口需要配置 DATA_SOURCES.ice 或 ICE_REFLECTION.BUMDEN_DOMAIN"
            )
        kwargs["ice_data"] = str(ice_data)
    else:
        kwargs["bumden_domain_ridges_per_km"] = float(bumden)
    result = ice_reflection_trc.generate_ice_reflection_trc(
        trc_path,
        float(params["freq_hz"]),
        **kwargs,
    )
    manifest = {
        "mode": "ice_reflection_trc",
        "env_file": env_path.name,
        "trc_file": trc_path.name,
        "frequency_hz": result.frequency_hz,
        "ice_data_source": kwargs["data_source"],
        "ice_data_file": str(settings.get("ice_reflection_ice_data") or ""),
        "BUMDEN_domain_ridges_per_km": result.bumden_domain_ridges_per_km,
        "domain_selection": (
            settings.get("ice_reflection_diagnostics", {}).get("selection", {})
        ),
        "selected_model": result.selected_model,
        "angle_count": result.angle_count,
        "passivity_clipped_count": result.passivity_clipped_count,
        "integration": {
            "SSPOPT_change": "replace second character by F",
            "staging_rule": "TRC file uses the same basename as the ENV file",
        },
    }
    (env_path.with_suffix(".ice_trc.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =============================================================================
# 从分类配置文件生成运行参数
# =============================================================================

def _effective_settings(params: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    """从分类配置文件生成运行参数。"""

    def mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        """从配置parent中获取key的映射值，支持大小写不敏感。
           取值并确认它 真的是字典 ，不是就立刻抛出清晰报错
        """
        value = parent.get(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"分类配置字段 {key!r} 必须是 JSON 对象")
        return value

    def boolean(value: Any) -> bool:
        """将值转换为布尔值，支持大小写不敏感。"""
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    env_config = mapping(config, "BELLHOP_ENV")
    ssp_config = mapping(env_config, "SSP")
    surface_config = mapping(env_config, "SURFACE")
    bottom_config = mapping(env_config, "BOTTOM")
    common_environment = config.get("ACOUSTIC_ENVIRONMENT")
    common_bottom_geo = (
        common_environment.get("BOTTOM_GEOACOUSTICS")
        if isinstance(common_environment, Mapping)
        else None
    )
    bottom_geo_config = (
        common_bottom_geo
        if isinstance(common_bottom_geo, Mapping)
        else mapping(config, "BOTTOM_GEOACOUSTICS")
    )

    mode_aliases = {
        "mean": "mean",
        "average": "mean",
        "fixed": "fixed",
        "fixed_source": "fixed",
        "source": "fixed",
        "range_dependent": "range_dependent",
        "distance_dependent": "range_dependent",
        "q": "range_dependent",
    }
    raw_ssp_mode = _lower_text(ssp_config.get("MODE"), "range_dependent")
    try:
        ssp_mode = mode_aliases[raw_ssp_mode]
    except KeyError as exc:
        raise ValueError("BELLHOP_ENV.SSP.MODE 必须是 'fixed'、'mean' 或 'range_dependent'") from exc
    # 门户字段优先于全局 SSP.MODE。水体不变时，整条切面固定使用声源位置
    # （距离 0 km）的剖面；水体变化时，沿传播距离使用不同剖面。
    if params.get("is_water_changed") is not None:
        ssp_mode = "range_dependent" if boolean(params.get("is_water_changed")) else "fixed"
    if ssp_mode == "range_dependent":
        ssp_interpolation = _upper_text(ssp_config.get("RANGE_DEPENDENT_INTERPOLATION"), "Q")
        if ssp_interpolation != "Q":
            raise ValueError("距离相关 Bellhop 2D SSP 插值方式必须是 'Q'")
    else:
        ssp_interpolation = _upper_text(ssp_config.get("MEAN_INTERPOLATION"), "C")
        if ssp_interpolation not in {"C", "N", "P", "S"}:
            raise ValueError("平均 SSP 插值方式必须是 C、N、P 或 S")

    surface_boundary = _upper_text(surface_config.get("BOUNDARY_CONDITION"), "V")
    if surface_boundary not in {"V", "R"}:
        raise ValueError("SURFACE.BOUNDARY_CONDITION 必须是 'V' 或 'R'")
    attenuation_units = _upper_text(surface_config.get("ATTENUATION_UNITS"), "W")
    if attenuation_units not in {"N", "F", "M", "W", "Q", "L"}:
        raise ValueError("SURFACE.ATTENUATION_UNITS 必须是 N、F、M、W、Q 或 L")
    volume_attenuation = _upper_text(surface_config.get("VOLUME_ATTENUATION"), " ") or " "
    if volume_attenuation not in {" ", "T"}:
        raise ValueError("SURFACE.VOLUME_ATTENUATION 当前只支持空格或 'T'（Thorp）")

    bottom_boundary = _upper_text(bottom_config.get("BOUNDARY_CONDITION"), "A")
    if bottom_boundary != "A":
        raise ValueError("本项目当前要求 BOTTOM.BOUNDARY_CONDITION='A'")
    use_bathymetry_file = boolean(bottom_config.get("USE_BATHYMETRY_FILE", True))
    if not use_bathymetry_file:
        raise ValueError("本项目依赖地形文件，要求 USE_BATHYMETRY_FILE=true")
    boundary_interpolation = _upper_text(bottom_config.get("BATHYMETRY_INTERPOLATION"), "L")
    if boundary_interpolation not in {"L", "C"}:
        raise ValueError("BOTTOM.BATHYMETRY_INTERPOLATION 必须是 'L' 或 'C'")

    bottom_fixed = {
        "compressional_speed_mps": float(bottom_config.get("COMPRESSIONAL_SPEED_MPS", 1600.0)),
        "shear_speed_mps": float(bottom_config.get("SHEAR_SPEED_MPS", 0.0)),
        "density_g_cm3": float(bottom_config.get("DENSITY_G_CM3", 1.8)),
        "compressional_attenuation": float(bottom_config.get("COMPRESSIONAL_ATTENUATION", 0.8)),
        "shear_attenuation": float(bottom_config.get("SHEAR_ATTENUATION", 0.0)),
    }
    if bottom_fixed["compressional_speed_mps"] <= 0.0 or bottom_fixed["density_g_cm3"] <= 0.0:
        raise ValueError("固定海底的纵波声速和密度必须为正数")
    if any(bottom_fixed[name] < 0.0 for name in (
        "shear_speed_mps", "compressional_attenuation", "shear_attenuation"
    )):
        raise ValueError("固定海底的剪切波声速和衰减参数不能为负数")

    bottom_geo_enabled = boolean(bottom_geo_config.get("ENABLED", False))
    source_mode = "file"
    source_interpolation = "linear"
    sources = config.get("DATA_SOURCES")
    if isinstance(sources, Mapping) and isinstance(sources.get("bottom_geoacoustics"), Mapping):
        source_config = sources["bottom_geoacoustics"]
        source_mode = _lower_text(source_config.get("mode"), "file")
        source_interpolation = _lower_text(
            source_config.get("interpolation"),
            "nearest" if source_mode == "postgres" else "linear",
        )
    if source_interpolation not in {"linear", "nearest"}:
        raise ValueError("海底声学 interpolation 必须是 linear 或 nearest")
    required_fields = []
    for value in bottom_geo_config.get("REQUIRED_FIELDS", []):
        required_fields.append(str(value))
    unknown_required = set(required_fields) - set(BOTTOM_GEO_FIELDS)
    if unknown_required:
        raise ValueError(f"未知的必填海底字段：{sorted(unknown_required)}")
    raw_variables = mapping(bottom_geo_config, "NETCDF_VARIABLES")
    raw_scale = mapping(bottom_geo_config, "SCALE")
    raw_offset = mapping(bottom_geo_config, "OFFSET")
    raw_ranges = mapping(bottom_geo_config, "VALID_RANGES")
    variables: Dict[str, List[str]] = {}
    scale: Dict[str, float] = {}
    offset: Dict[str, float] = {}
    valid_ranges: Dict[str, List[float]] = {}
    for name in BOTTOM_GEO_FIELDS:
        names = raw_variables.get(name, [])
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, Sequence):
            raise ValueError(f"NETCDF_VARIABLES.{name} 必须是字符串或列表")
        variables[name] = []
        for value in names:
            text = _text(value)
            if text:
                variables[name].append(text)
        scale[name] = float(raw_scale.get(name, 1.0))
        offset[name] = float(raw_offset.get(name, 0.0))
        bounds = raw_ranges.get(name)
        if not isinstance(bounds, Sequence) or isinstance(bounds, str) or len(bounds) != 2:
            raise ValueError(f"VALID_RANGES.{name} 必须包含 [minimum, maximum]")
        valid_ranges[name] = [float(bounds[0]), float(bounds[1])]
        if valid_ranges[name][0] >= valid_ranges[name][1]:
            raise ValueError(f"VALID_RANGES.{name} 的最小值必须小于最大值")

    raw_parallel_mode = _lower_text(_required_cfg(config, "BELLHOP_PARALLEL_MODE"))
    parallel_mode_aliases = {
        "internal": "internal",
        "bellhop_internal": "internal",
        "solver_internal": "internal",
        "outer": "outer",
        "outer_bearings": "outer",
        "multi_bearing": "outer",
        "hybrid": "hybrid",
        "internal_outer": "hybrid",
        "outer_internal": "hybrid",
        "internal_external": "hybrid",
    }
    try:
        parallel_mode = parallel_mode_aliases[raw_parallel_mode]
    except KeyError as exc:
        raise ValueError(
            "BELLHOP_PARALLEL_MODE 必须是 'internal'、'outer' 或 'hybrid'"
        ) from exc
    solver_supports_internal = _is_bellhopcxx_solver(_required_cfg(config, "BELLHOP_CXX"))
    if parallel_mode in {"internal", "hybrid"} and not solver_supports_internal:
        raise ValueError(
            f"BELLHOP_PARALLEL_MODE={parallel_mode!r} 需要 BellhopCXX；"
            "请把 BELLHOP_CXX 指向 bellhopcxx.exe/bellhopcuda.exe，或选择 'outer'。二选一！"
        )
    padding_km = float(_required_cfg(config, "PADDING_KM"))
    box_guard_km = float(_required_cfg(config, "BOX_GUARD_KM"))
    trace_extra_km = float(_required_cfg(config, "TRACE_EXTRA_KM"))
    n_2d_profiles = int(_required_cfg(config, "N_2D_PROFILES"))
    if box_guard_km <= 0.0:
        raise ValueError("BOX_GUARD_KM 必须为正数，才能覆盖环境边界")
    if n_2d_profiles < 3:
        raise ValueError("N_2D_PROFILES 至少为 3")
    if padding_km < trace_extra_km + box_guard_km:
        raise ValueError(
            "PADDING_KM 必须覆盖 TRACE_EXTRA_KM + BOX_GUARD_KM，"
            "确保 SSP/BTY/ATI 剖面仍在环境网格内部"
        )
    n_media = int(env_config.get("N_MEDIA", 1))
    if n_media != 1:
        raise ValueError("Bellhop 当前只支持一个水体介质层（N_MEDIA=1）")
    beam_type = _text(env_config.get("BEAM_TYPE"), "G") or "G"
    if beam_type not in {"G", "g", "B", "b", "S"}:
        raise ValueError("BEAM_TYPE 必须是 G、g、B、b 或 S")
    ray_run_type = _upper_text(_first_value([
        params.get("ray_run_type"),
        env_config.get("RAY_RUN_TYPE"),
    ], "R"))
    tl_run_type = _upper_text(_first_value([
        params.get("tl_run_type"),
        env_config.get("TL_RUN_TYPE"),
    ], "I"))
    if ray_run_type != "R" or tl_run_type not in {"I", "C", "S"}:
        raise ValueError("RAY_RUN_TYPE 必须是 R，TL_RUN_TYPE 必须是 I、C 或 S")
    receiver_depth_min_m = float(params.get("receiver_depth_min_m", env_config.get("RECEIVER_DEPTH_MIN_M", 0.0)))
    receiver_range_min_km = float(params.get("receiver_range_min_km", env_config.get("RECEIVER_RANGE_MIN_KM", 0.0)))
    depth_guard_m = float(env_config.get("DEPTH_GUARD_M", 10.0))
    zbox_margin_m = float(env_config.get("ZBOX_MARGIN_M", 10.0))
    bottom_roughness_m = float(bottom_config.get("ROUGHNESS_M", 0.0))
    if receiver_depth_min_m < 0.0 or receiver_range_min_km < 0.0:
        raise ValueError("接收深度/距离下限不能为负数")
    if depth_guard_m <= 0.0 or zbox_margin_m < 0.0 or bottom_roughness_m < 0.0:
        raise ValueError("深度保护量必须为正数；边界余量和粗糙度不能为负数")

    ice_config = config.get("ICE_REFLECTION") if isinstance(config.get("ICE_REFLECTION"), Mapping) else {}
    ice_reflection_enabled = boolean(params.get("ice_reflection_enabled", False))
    reflection_source = ice_reflection_trc._normalise_ice_data_source()
    ice_data = _first_value([
        params.get("ice_data"),
        params.get("ice_reflection_ice_data"),
    ], "")
    # 门户传入的 ice_data 是 HYCOM 冰厚文件；切换反射宏时必须重新路由。
    if ice_reflection_enabled and reflection_source != "HYCOM":
        ice_data = ""
    macro_ice_data = ice_reflection_trc.selected_ice_data_path(None, reflection_source)
    if macro_ice_data is not None:
        ice_data = str(macro_ice_data)
    # 海冰文件属于数据路由配置；反射模型参数仍保留在 ICE_REFLECTION 中。
    if ice_reflection_enabled and not ice_data:
        ice_kind = ice_reflection_trc.reflection_data_kind(reflection_source)
        if ice_kind is not None:
            ice_mode = data_source_mode(config, ice_kind, "file")
            if ice_mode == "file":
                ice_data = data_source_file(
                    config, ice_kind, str(params.get("environment_time") or "")
                )
            else:
                ice_data = resolve_data_file(
                    config,
                    ice_mode,
                    ice_kind,
                    data_source=str(params.get("data_source") or "EnvReanalysis"),
                    time_value=str(params.get("environment_time") or ""),
                )
    if ice_data:
        ice_data = _resolve_project_path(ice_data)
    bumden_domain = _first_value([
        params.get("bumden_domain_ridges_per_km"),
        params.get("ice_reflection_bumden_domain"),
        ice_config.get("BUMDEN_DOMAIN"),
        ice_config.get("BUMDEN_DOMAIN_RIDGES_PER_KM"),
    ], None)
    if bumden_domain is not None:
        bumden_domain = float(bumden_domain)
        if bumden_domain < 0.0:
            raise ValueError("ICE_REFLECTION.BUMDEN_DOMAIN 必须大于等于 0")
    if ice_reflection_enabled:
        if bumden_domain is None and not ice_data:
            if reflection_source == "ICESAT2":
                raise ValueError(
                    "ICESAT2 反射入口需要在 ice_reflection_trc.py 设置 ICESAT2_ICE_DATA_FILE"
                )
            raise ValueError(
                f"Ice 接口需要配置 {reflection_source} 海冰文件或 ICE_REFLECTION.BUMDEN_DOMAIN"
            )
        if ice_data and not os.path.isfile(str(ice_data)):
            raise FileNotFoundError(f"海冰数据文件不存在：{ice_data}")
    ice_domain_radius = _first_value([
        params.get("ice_reflection_domain_radius_km"),
        ice_config.get("DOMAIN_RADIUS_KM"),
    ], None)
    if ice_domain_radius is not None:
        ice_domain_radius = float(ice_domain_radius)
        if ice_domain_radius <= 0.0:
            raise ValueError("ICE_REFLECTION.DOMAIN_RADIUS_KM 必须为正数")
    ice_force_model = str(_first_value([
        params.get("ice_reflection_force_model"),
        ice_config.get("FORCE_MODEL"),
    ], "auto")).strip().lower()
    if ice_force_model not in {"auto", "mathieu", "high_frequency"}:
        raise ValueError("ICE_REFLECTION.FORCE_MODEL 必须是 auto、mathieu 或 high_frequency")
    ice_domain_aggregate = str(_first_value([
        params.get("ice_reflection_domain_aggregate"),
        ice_config.get("DOMAIN_AGGREGATE"),
    ], "median")).strip().lower()
    if ice_domain_aggregate not in {"median", "mean", "p25", "p75"}:
        raise ValueError("ICE_REFLECTION.DOMAIN_AGGREGATE 必须是 median、mean、p25 或 p75")
    ice_min_domain_cells = max(1, int(_first_value([
        params.get("ice_reflection_min_domain_cells"),
        ice_config.get("MIN_DOMAIN_CELLS"),
    ], 3)))
    ice_enforce_passive = boolean(_first_value([
        params.get("ice_reflection_enforce_passive"),
        ice_config.get("ENFORCE_PASSIVE"),
    ], True))

    return {
        "config_version": str(_required_cfg(config, "CONFIG_VERSION")),
        "title_prefix": str(env_config.get("TITLE_PREFIX") or "Bellhop 2D"),
        "n_media": n_media,
        "ssp_mode": ssp_mode,
        "ssp_interpolation": ssp_interpolation,
        "surface_boundary": surface_boundary,
        "attenuation_units": attenuation_units,
        "volume_attenuation": volume_attenuation,
        # 海冰接口设置为不支持 ati 文件
        "use_altimetry_file": boolean(surface_config.get("USE_ALTIMETRY_FILE", True)) and not ice_reflection_enabled,
        "bottom_boundary": bottom_boundary,
        "use_bathymetry_file": use_bathymetry_file,
        "bottom_roughness_m": bottom_roughness_m,
        "boundary_interpolation": boundary_interpolation,
        "bottom_fixed": bottom_fixed,
        "bottom_geo_mode": f"{source_mode}_long" if bottom_geo_enabled else "fixed",
        "bottom_geo_long": bottom_geo_enabled,
        "bottom_geo_interpolation": source_interpolation,
        "bottom_geo_required_fields": required_fields,
        "bottom_geo_variables": variables,
        "bottom_geo_scale": scale,
        "bottom_geo_offset": offset,
        "bottom_geo_valid_ranges": valid_ranges,
        "beam_type": beam_type,
        "ray_run_type": ray_run_type,
        "tl_run_type": tl_run_type,
        "receiver_depth_min_m": receiver_depth_min_m,
        "receiver_range_min_km": receiver_range_min_km,
        "depth_guard_m": depth_guard_m,
        "zbox_margin_m": zbox_margin_m,
        "padding_km": padding_km,
        "box_guard_km": box_guard_km,
        "trace_extra_km": trace_extra_km,
        "n_alpha": max(0, int(params.get("n_alpha") if params.get("n_alpha") is not None else _required_cfg(config, "N_ALPHA"))),
        "n_range": max(2, int(params.get("n_range") or _required_cfg(config, "N_RANGE"))),
        "n_rd": max(2, int(params.get("n_rd") or _required_cfg(config, "N_RD"))),
        "tl_n_range": max(2, int(params.get("tl_n_range") or _required_cfg(config, "TL_N_RANGE"))),
        "tl_n_rd": max(2, int(params.get("tl_n_rd") or _required_cfg(config, "TL_N_RD"))),
        "step_m": float(_required_cfg(config, "STEP_M")),
        "n_2d_profiles": n_2d_profiles,
        "depth_grid_m": list(_required_cfg(config, "DEPTH_GRID_M")),
        # 模式 internal：一次只运行一个多线程 BellhopCXX 进程。
        # 模式 outer：同时运行多个单线程 BellhopCXX 进程。
        # 模式 hybrid：同时运行多个多线程 BellhopCXX 进程；当前 BellhopCXX 版本没有
        # 控制内部线程数的命令行参数，所以 MAX_2D_WORKERS 需要保守设置。
        "bellhop_parallel_mode": parallel_mode,
        "solver_singlethread": parallel_mode == "outer",
        "internal_parallel_enabled": parallel_mode in {"internal", "hybrid"},
        "outer_parallel_enabled": parallel_mode in {"outer", "hybrid"},
        # 分层并行：声源级进程池 + 每个声源内部的方位线程池。
        # 配置 MAX_2D_WORKERS 仍然是同时运行 Bellhop 任务的全局硬上限。
        "max_2d_workers": max(1, int(_required_cfg(config, "MAX_2D_WORKERS"))),
        "max_source_processes": max(1, int(_cfg(config, "MAX_SOURCE_PROCESSES", 4))),
        "bearing_threads_per_source": max(1, int(_cfg(config, "BEARING_THREADS_PER_SOURCE", 3))),
        "timeout_sec": float(_required_cfg(config, "MAX_WAIT_TIME")),
        "ice_reflection_enabled": ice_reflection_enabled,
        "ice_reflection_data_source": reflection_source,
        "ice_reflection_ice_data": ice_data,
        "ice_reflection_environment_time": params.get("environment_time"),
        "ice_reflection_bumden_domain": bumden_domain,
        "ice_reflection_domain_radius_km": ice_domain_radius,
        "ice_reflection_domain_aggregate": ice_domain_aggregate,
        "ice_reflection_min_domain_cells": ice_min_domain_cells,
        "ice_reflection_force_model": ice_force_model,
        "ice_reflection_enforce_passive": ice_enforce_passive,
    }


# =============================================================================
# 子进程执行和单方位计算（Bellhop）
# =============================================================================

def _prt_error(path: Path) -> str:
    """从 Bellhop 打印文件中提取最后的致命错误信息。"""
    if not path.exists():
        return ""
    text = path.read_text(encoding="ascii", errors="replace")
    low = text.lower()
    keys = ["fatal error", "outside the box", "outside the ocean", "bad real number", "error termination"]
    if not any(key in low for key in keys):
        return ""
    lines = text.splitlines()
    hits = [line.strip() for line in lines if any(key in line.lower() for key in keys)]
    return "；".join(hits[-6:]) or "Bellhop .prt 报告致命错误"


def _run_solver(
    executable: str,
    root_name: str,
    work_dir: Path,
    timeout_sec: float,
    singlethread: bool,
) -> None:
    """执行一次 Bellhop 子进程并校验返回码和打印文件。"""
    is_cxx = _is_bellhopcxx_solver(executable)
    flags = ["--2D"] if is_cxx else []
    if is_cxx and singlethread:
        flags.insert(0, "--singlethread")
    command = [executable] + flags + [root_name]
    process_env = os.environ.copy()
    if singlethread:
        # 每个方位已经由外层线程池并行；禁止求解器或其数值库再次扩展线程，
        # 否则 16 个方位任务会产生远超 16 个 CPU 工作线程。
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            process_env[name] = "1"
    (work_dir / f"{root_name}_command.txt").write_text(subprocess.list2cmdline(command), encoding="utf-8")
    try:
        completed = subprocess.run(
            command, cwd=str(work_dir), timeout=timeout_sec,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", check=False,
            env=process_env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Bellhop 运行超过 {timeout_sec:.0f} 秒") from exc
    (work_dir / f"{root_name}_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (work_dir / f"{root_name}_stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    prt_problem = _prt_error(work_dir / f"{root_name}.prt")
    if completed.returncode != 0 or prt_problem:
        detail = prt_problem or completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Bellhop {root_name} 失败（返回码 {completed.returncode}）：{detail[-1500:]}")


def _ray_is_usable(path: Path) -> bool:
    """判断声线结果文件是否存在且包含可解析内容。"""
    if not path.exists() or path.stat().st_size < 64:
        return False
    try:
        bellhop_results.parse_ray_file(path)
        return True
    except Exception:
        return False


def _bearing_directory_name(index: int, bearing_deg: float) -> str:
    """生成稳定且适合文件系统的方位任务目录名。"""
    angle = float(bearing_deg) % 360.0
    return f"bearing_{index:03d}_{angle:07.3f}deg".replace(".", "p")


def _parse_2d_bearing_outputs(
    bearing_index: int,
    bearing_deg: float,
    work_dir: Path,
    params: Dict[str, Any],
    section: Dict[str, Any],
    calculation_radius_km: float,
    tl_depth_max_m: float,
) -> Dict[str, Any]:
    """解析一个已完成方位的结果，方便导出失败后复用求解器输出。"""
    header, raw_rays = bellhop_results.parse_ray_file(work_dir / "Point2D.ray")
    for ray in raw_rays:
        ray["bearing_deg"] = float(bearing_deg)
        ray["bearing_index"] = int(bearing_index)
    rays = bellhop_results.normalise_rays(
        raw_rays, header, float(params["center_lon"]), float(params["center_lat"]),
        0.0, 0.0, calculation_radius_km,
    )
    shd = bellhop_results.read_shd_bin(work_dir / "Point2D_TL.shd")
    shd = bellhop_results.align_shd_axes_to_environment(
        shd, bearing_deg, bearing_deg, calculation_radius_km, tl_depth_max_m,
        receiver_range_min_km=float(params.get("receiver_range_min_km", 0.0)),
        receiver_depth_min_m=float(params.get("receiver_depth_min_m", 0.0)),
    )
    if np.asarray(shd["tl"]).shape[:2] != (1, 1):
        raise ValueError(
            f"方位 {bearing_deg:.3f}° 的 2D SHD 维度异常：{np.asarray(shd['tl']).shape}"
        )
    header["bearing_deg"] = float(bearing_deg)
    header["bearing_index"] = int(bearing_index)
    return {
        "bearing_deg": float(bearing_deg),
        "header": header,
        "rays": rays,
        "depth_m": np.asarray(shd["receiver_depth_m"], dtype=np.float64),
        "range_km": np.asarray(shd["range_km"], dtype=np.float64),
        "layers": np.asarray(shd["tl"], dtype=np.float64)[0, 0, :, :],
        "profile_range_km": np.asarray(section["range_km"], dtype=np.float64),
        "profile_bottom_depth_m": np.asarray(section["bottom_depth_m"], dtype=np.float64),
        "profile_surface_depth_m": np.asarray(section["surface_depth_m"], dtype=np.float64),
        "profile_sound_speed_mps": np.asarray(section["sound_speed_mps"], dtype=np.float64),
        "directory": work_dir.name,
    }


def _run_2d_bearing(
    bearing_index: int,
    bearing_deg: float,
    work_dir: Path,
    params: Dict[str, Any],
    settings: Dict[str, Any],
    executable: str,
    grid: Dict[str, Any],
    environment: Dict[str, Any],
    bathymetry: Dict[str, Any],
    altimetry: Dict[str, Any],
    bottom_geoacoustics: Dict[str, Any],
    calculation_radius_km: float,
    tl_depth_max_m: float,
    progress: Callable[[str, str], None],
) -> Dict[str, Any]:
    """运行并解析一个二维方位的声线任务和非相干 TL 任务。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    # 环境剖面需要超出 RBOX。Q-SSP 的试探步需要这个保护距离，
    # 同时使用同一套距离轴可以保持 BTY/ATI/SSP 输入对齐。
    profile_extra = float(settings["trace_extra_km"]) + float(settings["box_guard_km"])
    section = _sample_2d_slice(
        grid, environment, bathymetry, altimetry, bottom_geoacoustics, bearing_deg,
        calculation_radius_km, int(settings["n_2d_profiles"]),
        extra_range_km=profile_extra,
    )
    section = _apply_ssp_mode_to_section(section, settings["ssp_mode"])
    if (
        bool(params.get("receiver_depth_auto_max"))
        and _lower_text(params.get("receiver_depth_auto_scope")) == "path"
    ):
        within_receiver_range = np.asarray(section["range_km"], dtype=np.float64) <= float(calculation_radius_km) + 1.0e-9
        max_bottom = _max_positive_depth(
            section["bottom_depth_m"],
            within_receiver_range,
            fallback=float(params["source_depth_m"]),
        )
        tl_depth_max_m = _receiver_depth_limit(
            max_bottom,
            float(params["source_depth_m"]),
            environment,
            settings,
        )
        params["receiver_depth_max_m"] = tl_depth_max_m
    solver_section = _section_with_left_guard(section) if settings["ssp_mode"] == "range_dependent" else section
    root = "Point2D"
    if _ray_is_usable(work_dir / f"{root}.ray") and (work_dir / "Point2D_TL.shd").is_file():
        try:
            progress("ray", f"方位 {bearing_deg:.3f}°：复用已完成的 Bellhop 2D 结果")
            result = _parse_2d_bearing_outputs(
                bearing_index, bearing_deg, work_dir, params, section,
                calculation_radius_km, tl_depth_max_m,
            )
            result["reused"] = True
            return result
        except Exception:
            # 缓存不完整或版本不兼容时，下面会重新计算。
            pass

    if settings["ssp_mode"] == "range_dependent":
        _write_2d_ssp(work_dir / f"{root}.ssp", solver_section)
    long_bottom = (
        solver_section["bottom_geoacoustics"]["fields"]
        if settings["bottom_geo_long"]
        else None
    )
    _write_2d_boundary(
        work_dir / f"{root}.bty",
        solver_section["range_km"],
        solver_section["bottom_depth_m"],
        settings["boundary_interpolation"],
        long_bottom,
    )
    if settings["use_altimetry_file"]:
        _write_2d_boundary(
            work_dir / f"{root}.ati",
            solver_section["range_km"],
            solver_section["surface_depth_m"],
            "L",
        )
    _write_2d_env(
        work_dir / f"{root}.env", params, settings, environment, solver_section,
        calculation_radius_km, float(params["receiver_depth_max_m"]), settings["ray_run_type"],
    )
    _stage_ice_reflection_for_env(work_dir / f"{root}.env", params, settings)

    progress("ray", f"方位 {bearing_deg:.3f}°：Bellhop 2D 声线")
    ray_path = work_dir / f"{root}.ray"
    first_error = ""
    try:
        _run_solver(
            executable, root, work_dir, settings["timeout_sec"],
            bool(settings["solver_singlethread"]),
        )
        if not _ray_is_usable(ray_path):
            first_error = "未生成有效的 2D .ray 文件"
    except Exception as exc:
        first_error = str(exc)
    if first_error:
        outside_ssp_box = "outside the box" in first_error.lower() and "soundspeed" in first_error.lower()
        if settings["ssp_mode"] == "range_dependent" and outside_ssp_box:
            progress(
                "ray",
                f"方位 {bearing_deg:.3f}°：Q-SSP 声线触及左边界，改用平均声速重算声线",
            )
            ray_settings = dict(settings)
            ray_settings["ssp_mode"] = "mean"
            ray_settings["ssp_interpolation"] = "C"
            ray_long_bottom = (
                section["bottom_geoacoustics"]["fields"]
                if settings["bottom_geo_long"]
                else None
            )
            _write_2d_boundary(
                work_dir / f"{root}.bty",
                section["range_km"],
                section["bottom_depth_m"],
                settings["boundary_interpolation"],
                ray_long_bottom,
            )
            if settings["use_altimetry_file"]:
                _write_2d_boundary(
                    work_dir / f"{root}.ati",
                    section["range_km"],
                    section["surface_depth_m"],
                    "L",
                )
            _write_2d_env(
                work_dir / f"{root}.env", params, ray_settings, environment, section,
                calculation_radius_km, float(params["receiver_depth_max_m"]), settings["ray_run_type"],
            )
            _stage_ice_reflection_for_env(work_dir / f"{root}.env", params, ray_settings)
            for stale in (ray_path, work_dir / f"{root}.prt"):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
            try:
                _run_solver(
                    executable, root, work_dir, settings["timeout_sec"],
                    bool(settings["solver_singlethread"]),
                )
                first_error = "" if _ray_is_usable(ray_path) else "平均声速重试后仍未生成有效的 2D .ray 文件"
            except Exception as exc:
                first_error = f"{first_error}\n平均声速声线重试仍失败：{exc}"
    if first_error:
        if not np.any(section["land_mask"]):
            raise RuntimeError(first_error)
        wave_abs = float(np.nanmax(np.abs(section["surface_depth_m"])))
        retry_depth = min(max(5.0, math.ceil(wave_abs + 1.5)), 20.0)
        section["bottom_depth_m"][section["land_mask"]] = retry_depth
        section["surface_depth_m"][section["land_mask"]] = 0.0
        solver_section = _section_with_left_guard(section) if settings["ssp_mode"] == "range_dependent" else section
        long_bottom = (
            solver_section["bottom_geoacoustics"]["fields"]
            if settings["bottom_geo_long"]
            else None
        )
        _write_2d_boundary(
            work_dir / f"{root}.bty",
            solver_section["range_km"],
            solver_section["bottom_depth_m"],
            settings["boundary_interpolation"],
            long_bottom,
        )
        if settings["use_altimetry_file"]:
            _write_2d_boundary(
                work_dir / f"{root}.ati", solver_section["range_km"], solver_section["surface_depth_m"], "L"
            )
        for stale in (ray_path, work_dir / f"{root}.prt"):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        try:
            _run_solver(
                executable, root, work_dir, settings["timeout_sec"],
                bool(settings["solver_singlethread"]),
            )
        except Exception as exc:
            raise RuntimeError(
                f"方位 {bearing_deg:.3f}° 的 1 m 陆地计算失败：{first_error}\n"
                f"{retry_depth:.1f} m 陆地重试仍失败：{exc}"
            ) from exc
        if not _ray_is_usable(ray_path):
            raise RuntimeError(f"方位 {bearing_deg:.3f}° 陆地重试后仍无有效 2D .ray：{first_error}")

    progress("tl", f"方位 {bearing_deg:.3f}°：Bellhop 2D 精确 TL")
    tl_root = "Point2D_TL"
    for extension in (".ssp", ".bty", ".ati"):
        source_input = work_dir / f"{root}{extension}"
        if source_input.is_file():
            shutil.copy2(source_input, work_dir / f"{tl_root}{extension}")
    _write_2d_env(
        work_dir / f"{tl_root}.env", params, settings, environment, solver_section,
        calculation_radius_km, tl_depth_max_m, settings["tl_run_type"],
    )
    _stage_ice_reflection_for_env(work_dir / f"{tl_root}.env", params, settings)
    try:
        _run_solver(
            executable, tl_root, work_dir, settings["timeout_sec"],
            bool(settings["solver_singlethread"]),
        )
    except Exception as exc:
        tl_error = str(exc)
        outside_ssp_box = "outside the box" in tl_error.lower() and "soundspeed" in tl_error.lower()
        if not (settings["ssp_mode"] == "range_dependent" and outside_ssp_box):
            raise
        progress(
            "tl",
            f"方位 {bearing_deg:.3f}°：Q-SSP TL 触及左边界，改用平均声速重算 TL",
        )
        tl_settings = dict(settings)
        tl_settings["ssp_mode"] = "mean"
        tl_settings["ssp_interpolation"] = "C"
        tl_long_bottom = (
            section["bottom_geoacoustics"]["fields"]
            if settings["bottom_geo_long"]
            else None
        )
        _write_2d_boundary(
            work_dir / f"{tl_root}.bty",
            section["range_km"],
            section["bottom_depth_m"],
            settings["boundary_interpolation"],
            tl_long_bottom,
        )
        if settings["use_altimetry_file"]:
            _write_2d_boundary(
                work_dir / f"{tl_root}.ati",
                section["range_km"],
                section["surface_depth_m"],
                "L",
            )
        try:
            (work_dir / f"{tl_root}.ssp").unlink()
        except FileNotFoundError:
            pass
        _write_2d_env(
            work_dir / f"{tl_root}.env", params, tl_settings, environment, section,
            calculation_radius_km, tl_depth_max_m, settings["tl_run_type"],
        )
        _stage_ice_reflection_for_env(work_dir / f"{tl_root}.env", params, tl_settings)
        for stale in (work_dir / f"{tl_root}.shd", work_dir / f"{tl_root}.prt"):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        try:
            _run_solver(
                executable, tl_root, work_dir, settings["timeout_sec"],
                bool(settings["solver_singlethread"]),
            )
        except Exception as retry_exc:
            raise RuntimeError(f"{tl_error}\n平均声速 TL 重试仍失败：{retry_exc}") from retry_exc

    result = _parse_2d_bearing_outputs(
        bearing_index, bearing_deg, work_dir, params, section,
        calculation_radius_km, tl_depth_max_m,
    )
    result["reused"] = False
    return result



# =============================================================================
# 数据源统一访问
# =============================================================================
# 读取HYCOM数据
class _DataAccess:
    """统一调度水体、海浪和可选海底声学数据读取。

    每类变量的数据源模式来自 data_config.json 的 DATA_SOURCES，可选
    file、api0、api1 或 postgres。本地文件统一读取
    ``DATA_SOURCES.<kind>.file``，请求参数仍可临时覆盖水文和海浪文件。
    """

    requested_sea_state = None
    wave_seed = None

    def __init__(
        self,
        params: Mapping[str, Any],
        config: Mapping[str, Any],
        *,
        bottom_geo_config: Mapping[str, Any],
    ) -> None:
        """按数据源配置建立本地文件或远程数据读取器。"""
        self.requested_sea_state = _requested_sea_state(params)
        self.wave_seed = _wave_random_seed(params)
        self._modes: Dict[str, str] = {}
        for var in ("temperature", "salinity"):
            self._modes[var] = data_source_mode(config, var, "file")
        data_source = str(params.get("data_source") or "EnvReanalysis")
        environment_time = str(params.get("environment_time") or "")
        self._readers = {
            mode: create_data_reader(
                config,
                mode,
                data_source=data_source,
                time_value=environment_time,
            )
            for mode in set(self._modes.values())
            if mode != "file"
        }
        self._temp_file = None
        self._salt_file = None
        if self._modes["temperature"] != "file":
            self._temp_file = None
        else:
            self._temp_file = _resolve_project_path(
                _first_value([
                    params.get("hycom_temp_file"),
                    params.get("hycom_file"),
                    data_source_file(config, "temperature", environment_time),
                ])
            )
        if self._modes["salinity"] != "file":
            self._salt_file = None
        else:
            self._salt_file = _resolve_project_path(
                _first_value([
                    params.get("hycom_sal_file"),
                    params.get("hycom_file"),
                    data_source_file(config, "salinity", environment_time),
                ])
            )
        if not isinstance(bottom_geo_config, Mapping):
            raise ValueError("bottom_geo_config 必须是海底声学配置对象。")
        bottom_geo = bottom_geo_config
        self._bottom_geo_enabled = bool(bottom_geo.get("ENABLED", False))
        self._bottom_geo_file = ""
        self._bottom_geo_reader = None
        self._bottom_geo_mode = "fixed"
        self._bottom_geo_source = str(
            bottom_geo.get("SOURCE") or "ACOUSTIC_ENVIRONMENT.BOTTOM fixed values"
        )
        if self._bottom_geo_enabled:
            bottom_mode = data_source_mode(config, "bottom_geoacoustics", "file")
            self._bottom_geo_mode = bottom_mode
            if bottom_mode == "file":
                self._bottom_geo_file = _resolve_project_path(
                    data_source_file(config, "bottom_geoacoustics")
                )
                self._bottom_geo_source = self._bottom_geo_file
            elif bottom_mode == "postgres":
                self._bottom_geo_reader = PostgresBottomGeoacousticsReader(
                    config,
                    data_source_config(config, "bottom_geoacoustics"),
                    data_source=data_source,
                    time_value=environment_time,
                )
                self._bottom_geo_source = self._bottom_geo_reader.source
            else:
                self._bottom_geo_file = resolve_data_file(
                    config,
                    bottom_mode,
                    "bottom_geoacoustics",
                    data_source=str(params.get("data_source") or "EnvReanalysis"),
                    time_value=str(params.get("environment_time") or ""),
                )
                self._bottom_geo_source = self._bottom_geo_file

    @property
    def bottom_geo_file(self) -> str:
        """返回当前海底声学数据文件路径。"""
        return self._bottom_geo_file

    @property
    def bottom_geo_source(self) -> str:
        """返回任务清单中使用的海底声学数据来源。"""
        return self._bottom_geo_source

    def validate(self) -> None:
        """检查所有启用的本地环境数据文件是否存在。"""
        bottom_geo_file = (
            self._bottom_geo_file
            if self._bottom_geo_enabled and self._bottom_geo_mode != "postgres"
            else None
        )
        for path, label in (
            (self._temp_file if self._modes["temperature"] == "file" else None, "HYCOM temperature"),
            (self._salt_file if self._modes["salinity"] == "file" else None, "HYCOM salinity"),
            (bottom_geo_file, "bottom geoacoustics"),
        ):
            if path is not None and not os.path.isfile(path):
                raise FileNotFoundError(f"{label} file does not exist: {path}")

    def close(self) -> None:
        """关闭该任务创建的全部远程或数据库读取器。"""
        for reader in self._readers.values():
            reader.close()
        if self._bottom_geo_reader is not None:
            self._bottom_geo_reader.close()

    def read_hycom_cube(
        self, kind: str, variable_names: Sequence[str], query_lon: np.ndarray, query_lat: np.ndarray
    ) -> Dict[str, Any]:
        """从已配置的数据源读取温度或盐度剖面立方体。"""
        if self._modes[kind] != "file":
            reader = self._readers[self._modes[kind]]
            tag = reader.temperature_tag if kind == "temperature" else reader.salinity_tag
            return reader.read_profile_cube(tag, query_lon, query_lat)
        path = self._temp_file if kind == "temperature" else self._salt_file
        return _read_hycom_cube(path, variable_names, query_lon, query_lat)

    def read_wave_field(
        self, field: str, candidates: Sequence[str], query_lon: np.ndarray, query_lat: np.ndarray
    ) -> np.ndarray:
        """当前前端只使用 seaState 生成海面，不再读取外部波浪文件。"""
        raise ValueError("当前后端未配置外部波浪场；请在请求中提供 seaState。")

    def read_bottom_geoacoustic_field(
        self,
        field: str,
        candidates: Sequence[str],
        query_lon: np.ndarray,
        query_lat: np.ndarray,
    ) -> np.ndarray:
        """从海底声学数据文件读取指定标量字段。"""
        values = self.read_bottom_geoacoustic_fields(
            {field: candidates}, query_lon, query_lat
        )
        if field not in values:
            raise ValueError(f"海底声学数据源缺少字段 {field!r}")
        return values[field]

    def read_bottom_geoacoustic_fields(
        self,
        candidates_by_field: Mapping[str, Sequence[str]],
        query_lon: np.ndarray,
        query_lat: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """批量读取全部海底声学字段；数据库模式只执行一次区域查询。"""
        if not self._bottom_geo_enabled:
            raise ValueError("海底声学读取未启用")
        if self._bottom_geo_mode == "postgres":
            if self._bottom_geo_reader is None:
                raise ValueError("海底声学数据库读取器未初始化")
            return self._bottom_geo_reader.read_fields(
                query_lon,
                query_lat,
                list(candidates_by_field),
            )
        if not self._bottom_geo_file:
            raise ValueError("海底声学数据文件未解析")
        fields: Dict[str, np.ndarray] = {}
        for field, candidates in candidates_by_field.items():
            if not candidates:
                continue
            try:
                fields[field] = _read_scalar_grid_field(
                    self._bottom_geo_file, candidates, query_lon, query_lat
                )
            except ValueError:
                continue
        return fields


# =============================================================================
# 单声源任务准备
# =============================================================================

def _max_positive_depth(values: Any, mask: Any = None, fallback: float = 1000.0) -> float:
    """返回数组中的最大正深度，无有效值时使用回退值。"""
    depth = np.asarray(values, dtype=np.float64)
    good = np.isfinite(depth) & (depth > 0.0)
    if mask is not None:
        good &= np.asarray(mask, dtype=bool)
    finite = depth[good]
    return float(np.max(finite)) if finite.size else float(fallback)


def _receiver_depth_limit(
    max_water_depth_m: float,
    source_depth_m: float,
    environment: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> float:
    """根据水深、声源和保护量确定接收深度上限。"""
    model_depth = float(np.asarray(environment["depth_m"], dtype=np.float64)[-1])
    depth_guard = float(settings["depth_guard_m"])
    lower = max(float(settings.get("receiver_depth_min_m", 0.0)) + 1.0, float(source_depth_m))
    wanted = max(float(max_water_depth_m), lower)
    limit = model_depth - depth_guard
    if limit <= lower:
        raise ValueError("HYCOM 最大深度无法覆盖声源深度或自动接收深度")
    return min(wanted, limit)


def _prepare_source_context(
    source_index: int,
    source_lat: float,
    source_lon: float,
    work_dir: Path,
    params: Dict[str, Any],
    config: Mapping[str, Any],
    settings: Dict[str, Any],
    topo: TopographyReader,
    calculation_radius_km: float,
    progress: Callable[[str, str], None],
    prepared=None,
    data=None,
) -> Dict[str, Any]:
    """准备单个声源上下文，然后交给专用声源进程执行。"""
    source_params = dict(params)
    source_params["center_lat"] = float(source_lat)
    source_params["center_lon"] = _normalise_lon(float(source_lon))

    work_dir.mkdir(parents=True, exist_ok=True)
    progress("1/7", f"声源 {source_index}：构建局部网格")
    grid = prepared[0] if prepared is not None else _local_grid(
        source_lat,
        source_lon,
        calculation_radius_km,
        max(3, int(round(float(source_params["n_grid"]))),),
        float(settings["padding_km"]),
    )
    progress("2/7", f"声源 {source_index}：读取温盐并计算声速")
    environment = prepared[1] if prepared is not None else _load_sound_speed(
        data, grid, settings["depth_grid_m"]
    )
    progress("3/7", f"声源 {source_index}：读取地形并按 seaState 生成海面")
    bathymetry = prepared[2] if prepared is not None else _build_bathymetry(
        topo, grid, float(source_params["source_depth_m"])
    )
    environment = _extend_sound_speed_to_bathymetry(environment, bathymetry)
    auto_receiver_depth = bool(source_params.get("receiver_depth_auto_max"))
    pre_bathymetry_depth_check = float(source_params["source_depth_m"])
    if not auto_receiver_depth:
        pre_bathymetry_depth_check = max(
            pre_bathymetry_depth_check,
            float(source_params["receiver_depth_max_m"]),
        )
    if float(environment["depth_m"][-1]) <= max(
        float(source_params["source_depth_m"]),
        pre_bathymetry_depth_check,
    ) + float(settings["depth_guard_m"]):
        raise ValueError("声速剖面填补后的最大深度无法覆盖声源深度或最大接收深度")
    altimetry = prepared[3] if prepared is not None else _build_altimetry(
        data, grid, bathymetry
    )
    bottom_geoacoustics = prepared[4] if prepared is not None else _load_bottom_geoacoustics(
        data, grid, settings
    )
    local_settings = dict(settings)
    if np.any(bathymetry["land_mask"]):
        local_settings["step_m"] = min(float(local_settings["step_m"]), 20.0)
    if local_settings.get("ice_reflection_enabled") and local_settings.get("ice_reflection_bumden_domain") is None:
        progress("3/7", f"声源 {source_index}：计算海冰反射域 BUMDEN")
        bumden_info = ice_reflection_trc.resolve_bumden_domain(
            str(local_settings["ice_reflection_ice_data"]),
            float(source_lat),
            _normalise_lon(float(source_lon)),
            float(local_settings.get("ice_reflection_domain_radius_km") or calculation_radius_km),
            aggregate=str(local_settings.get("ice_reflection_domain_aggregate") or "median"),
            min_domain_cells=int(local_settings.get("ice_reflection_min_domain_cells") or 3),
            data_source=str(local_settings["ice_reflection_data_source"]),
        )
        ice_reflection_trc.validate_reflection_data_time(
            bumden_info, local_settings.get("ice_reflection_environment_time")
        )
        local_settings["ice_reflection_bumden_domain"] = float(bumden_info["BUMDEN_domain"])
        local_settings["ice_reflection_diagnostics"] = {
            "domain_stats": bumden_info.get("domain_stats", {}),
            "ice_stats": bumden_info.get("ice_stats", {}),
            "selection": bumden_info.get("selection", {}),
        }
        selection = bumden_info.get("selection", {})
        if selection.get("degraded"):
            progress(
                "3/7",
                f"声源 {source_index}："
                f"{selection.get('message', 'BUMDEN 数据稀疏，已降级继续计算。')}",
            )

    bearings = _bearing_angles(source_params)
    executable = _resolve_project_path(_required_cfg(config, "BELLHOP_CXX"))
    model_depth = float(environment["depth_m"][-1])
    if auto_receiver_depth and _lower_text(source_params.get("receiver_depth_auto_scope")) == "area":
        source_x = float(grid["source_x_km"])
        source_y = float(grid["source_y_km"])
        distance = np.hypot(
            np.asarray(grid["x_km"], dtype=np.float64) - source_x,
            np.asarray(grid["y_km"], dtype=np.float64) - source_y,
        )
        max_bottom = _max_positive_depth(
            bathymetry["depth_m"],
            distance <= float(calculation_radius_km) + 1.0e-9,
            fallback=float(source_params["source_depth_m"]),
        )
        tl_depth_max = _receiver_depth_limit(
            max_bottom,
            float(source_params["source_depth_m"]),
            environment,
            local_settings,
        )
        source_params["receiver_depth_max_m"] = tl_depth_max
    else:
        tl_depth_max = min(
            float(source_params["receiver_depth_max_m"]),
            model_depth - float(settings["depth_guard_m"]),
        )
    parallel_mode = str(local_settings["bellhop_parallel_mode"])
    manifest: Dict[str, Any] = {
        "engine": "Bellhop multi-source, multi-bearing 2D",
        "config_version": local_settings["config_version"],
        "ssp_mode": local_settings["ssp_mode"],
        "ssp_interpolation": local_settings["ssp_interpolation"],
        "ssp_range_dependent": local_settings["ssp_mode"] == "range_dependent",
        "is_water_changed": local_settings["ssp_mode"] == "range_dependent",
        "water_column_mode": _water_column_mode_name(local_settings["ssp_mode"]),
        "ssp_profile_count": int(local_settings["n_2d_profiles"]),
        "ssp_box_guard_km": float(local_settings["box_guard_km"]),
        "bottom_geoacoustics_mode": local_settings["bottom_geo_mode"],
        "bty_format": "long" if local_settings["bottom_geo_long"] else "short",
        "bottom_geoacoustics_source": bottom_geoacoustics["source"],
        "bottom_geoacoustics_loaded_fields": bottom_geoacoustics.get("loaded_fields", []),
        "bottom_geoacoustics_fallback_fields": bottom_geoacoustics.get("fallback_fields", []),
        "surface_wave_model": str(altimetry.get("model", "file")),
        "surface_sea_state": altimetry.get("sea_state"),
        "surface_significant_wave_height_m": altimetry.get("significant_wave_height_m"),
        "surface_peak_period_s": altimetry.get("peak_period_s"),
        "source_index": source_index,
        "source_lat": source_lat,
        "source_lon": source_lon,
        "parallel_mode": parallel_mode,
        "solver_singlethread": bool(local_settings["solver_singlethread"]),
        "internal_parallel_enabled": bool(local_settings["internal_parallel_enabled"]),
        "outer_parallel_enabled": bool(local_settings["outer_parallel_enabled"]),
        "task_policy": "one process per source task; bearing thread pool inside each source process",
        "bearing_count": len(bearings),
        "bearings": [],
    }
    if local_settings.get("ice_reflection_enabled"):
        manifest["ice_reflection"] = {
            "BUMDEN_domain_ridges_per_km": float(
                local_settings.get("ice_reflection_bumden_domain") or 0.0
            ),
            **local_settings.get("ice_reflection_diagnostics", {}),
        }
    _write_flag(work_dir / "multibearing_2d_manifest.json", manifest)
    return {
        "source_index": source_index,
        "source_lat": source_lat,
        "source_lon": source_lon,
        "work_dir": work_dir,
        "params": source_params,
        "settings": local_settings,
        "grid": grid,
        "environment": environment,
        "bathymetry": bathymetry,
        "altimetry": altimetry,
        "bottom_geoacoustics": bottom_geoacoustics,
        "calculation_radius_km": calculation_radius_km,
        "tl_depth_max_m": tl_depth_max,
        "bearings": bearings,
        "executable": executable,
        "manifest": manifest,
        "sections_by_index": {},
    }


# =============================================================================
# 声源结果拼接和导出辅助
# =============================================================================

def _run_prepared_bearing(
    context: Dict[str, Any],
    bearing_index: int,
    bearing_deg: float,
    progress: Callable[[str, str], None],
    progress_lock: threading.Lock,
) -> Dict[str, Any]:
    """执行一个已经准备完成的方位剖面任务。"""
    directory = context["work_dir"] / "bearings_2d" / _bearing_directory_name(
        bearing_index, bearing_deg
    )

    def bearing_progress(kind: str, message: str) -> None:
        """把单方位进度转发为声源级进度消息。"""
        stage = "5/7" if kind == "ray" else "6/7"
        with progress_lock:
            progress(
                stage,
                f"声源 {context['source_index']} [{bearing_index}/{len(context['bearings'])}] {message}",
            )

    return _run_2d_bearing(
        bearing_index,
        bearing_deg,
        directory,
        context["params"],
        context["settings"],
        context["executable"],
        context["grid"],
        context["environment"],
        context["bathymetry"],
        context["altimetry"],
        context["bottom_geoacoustics"],
        context["calculation_radius_km"],
        context["tl_depth_max_m"],
        bearing_progress,
    )


def _ordered_bearing_sections(context):
    """按方位索引返回已完成的剖面结果。"""
    bearings = context["bearings"]
    sections_by_index = context["sections_by_index"]
    sections = []
    for index in range(1, len(bearings) + 1):
        sections.append(sections_by_index[index])
    return sections


def _merge_section_rays(sections):
    """合并所有方位剖面的声线记录。"""
    rays = []
    for section in sections:
        for ray in section["rays"]:
            rays.append(ray)
    for ray_id, ray in enumerate(rays, 1):
        ray["id"] = ray_id
    return rays


def _bearing_float_list(bearings):
    """把方位角数组转换为可序列化的浮点数列表。"""
    values = []
    for bearing in bearings:
        values.append(float(bearing))
    return values


def _section_headers(sections):
    """收集各方位声线文件头信息。"""
    headers = []
    for section in sections:
        headers.append(section["header"])
    return headers


def _index_value(item):
    """返回带索引结果项的整数索引，用于稳定排序。"""
    return int(item["index"])


def _finalise_source_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """拼接单声源全部方位结果并导出声源级文件。"""
    bearings = context["bearings"]
    sections = _ordered_bearing_sections(context)
    first_depth = sections[0]["depth_m"]
    first_range = sections[0]["range_km"]
    for item in sections[1:]:
        if (
            item["depth_m"].shape != first_depth.shape
            or item["range_km"].shape != first_range.shape
            or not np.allclose(item["depth_m"], first_depth, rtol=0.0, atol=1.0e-5)
            or not np.allclose(item["range_km"], first_range, rtol=0.0, atol=1.0e-7)
        ):
            raise ValueError("同一声源的 Bellhop 2D TL 网格不一致，无法拼接")
    layer_parts = []
    for item in sections:
        layer_parts.append(item["layers"])
    layers = np.stack(layer_parts, axis=1)
    grid = context["grid"]
    bathymetry = context["bathymetry"]
    rays = _merge_section_rays(sections)
    rays = bellhop_results.reclassify_ray_reflections(
        rays, grid["x_rel_axis_km"], grid["y_rel_axis_km"], bathymetry["depth_m"]
    )
    bottom_polar = bellhop_results.sample_bottom_polar(
        grid["x_rel_axis_km"], grid["y_rel_axis_km"], bathymetry["depth_m"],
        bearings, first_range,
    )
    centre_y = int(np.abs(grid["y_rel_axis_km"]).argmin())
    centre_x = int(np.abs(grid["x_rel_axis_km"]).argmin())
    result = {
        "index": context["source_index"],
        "lat": context["source_lat"],
        "lon": context["source_lon"],
        "x_km": 0.0,
        "y_km": 0.0,
        "display_radius_km": float(context["params"]["radius_km"]),
        "calculation_radius_km": float(context["calculation_radius_km"]),
        "header": {
            "simulation_dimension": "multi-bearing-2D",
            "ssp_mode": context["settings"]["ssp_mode"],
            "ssp_interpolation": context["settings"]["ssp_interpolation"],
            "ssp_range_dependent": context["settings"]["ssp_mode"] == "range_dependent",
            "is_water_changed": context["settings"]["ssp_mode"] == "range_dependent",
            "water_column_mode": _water_column_mode_name(context["settings"]["ssp_mode"]),
            "bottom_geoacoustics_mode": context["settings"]["bottom_geo_mode"],
            "bty_format": "long" if context["settings"]["bottom_geo_long"] else "short",
            "bearing_count": len(bearings),
            "bearings_deg": _bearing_float_list(bearings),
            "section_headers": _section_headers(sections),
            "actual_ray_count": len(rays),
        },
        "rays": rays,
        "source_bottom_depth_m": float(bathymetry.get("source_bottom_depth_m", np.nan)),
        "ssp_depth_m": context["environment"]["depth_m"],
        "source_depth_m": float(context["params"]["source_depth_m"]),
        "reverberation_boundaries": [{
            "bearing_deg": item["bearing_deg"],
            "range_km": item["profile_range_km"],
            "bottom_depth_m": item["profile_bottom_depth_m"],
            "surface_depth_m": item["profile_surface_depth_m"],
            "sound_speed_mps": item["profile_sound_speed_mps"],
        } for item in sections],
        "ssp_sound_speed_mps": context["environment"]["sound_speed_mps"][:, centre_y, centre_x],
        "thermocline_depth_m": context["environment"]["thermocline_depth_m"],
        "thermocline_gradient_c_per_m": context["environment"]["thermocline_gradient_c_per_m"],
        "tl": {
            "theta_deg": bearings,
            "depth_m": first_depth,
            "range_km": first_range,
            "layers": layers,
            "bottom_depth_m": bottom_polar,
        },
        "grid": {
            "x_rel_km": grid["x_rel_axis_km"],
            "y_rel_km": grid["y_rel_axis_km"],
            "bottom_depth_m": bathymetry["depth_m"],
            "thermocline_depth_m": context["environment"]["thermocline_depth_m"],
            "thermocline_gradient_c_per_m": context["environment"]["thermocline_gradient_c_per_m"],
        },
    }
    bellhop_results.attach_tl_to_rays(result)
    return result


def _build_master_grid(
    params: Dict[str, Any],
    all_points: Sequence[Mapping[str, Any]],
    all_offsets: Sequence[Tuple[float, float]],
    sources: Sequence[Dict[str, Any]],
    topo: TopographyReader,
) -> Dict[str, Any]:
    """建立覆盖全部声源局部网格的公共笛卡尔主网格。"""
    radius = float(params["radius_km"])
    x_points = np.asarray([p[0] for p in all_offsets])
    y_points = np.asarray([p[1] for p in all_offsets])
    xmin, xmax = float(x_points.min() - radius), float(x_points.max() + radius)
    ymin, ymax = float(y_points.min() - radius), float(y_points.max() + radius)
    base_n = max(41, min(121, int(round(float(params["n_grid"]))) * 2 + 1))
    span = max(xmax - xmin, ymax - ymin, 1.0)
    nx = max(31, int(round(base_n * (xmax - xmin) / span)))
    ny = max(31, int(round(base_n * (ymax - ymin) / span)))
    xs, ys = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = bellhop_results.local_offset_to_lonlat(xx, yy, float(all_points[0]["lon"]), float(all_points[0]["lat"]))
    elevation = topo.sample(lon, lat)
    ocean = np.isfinite(elevation) & (elevation < 0.0)
    finite_ocean = -elevation[ocean]
    fill = float(np.median(finite_ocean)) if finite_ocean.size else 3000.0
    bottom = np.where(ocean, -elevation, np.where(np.isfinite(elevation), 1.0, fill))

    calculation_mask = np.hypot(xx, yy) <= radius

    thermo = np.full_like(xx, np.nan, dtype=np.float64)
    thermo_gradient = np.full_like(xx, np.nan, dtype=np.float64)
    best_distance = np.full_like(xx, np.inf, dtype=np.float64)
    query_points = None
    for source in sources:
        local_x = xx - source["x_km"]
        local_y = yy - source["y_km"]
        grid = source["grid"]
        query_points = np.column_stack([local_y.ravel(), local_x.ravel()])
        interp_depth = RegularGridInterpolator(
            (grid["y_rel_km"], grid["x_rel_km"]), grid["thermocline_depth_m"],
            bounds_error=False, fill_value=np.nan,
        )
        interp_gradient = RegularGridInterpolator(
            (grid["y_rel_km"], grid["x_rel_km"]), grid["thermocline_gradient_c_per_m"],
            bounds_error=False, fill_value=np.nan,
        )
        depth_values = interp_depth(query_points).reshape(xx.shape)
        gradient_values = interp_gradient(query_points).reshape(xx.shape)
        distance = np.hypot(local_x, local_y)
        allowed = distance <= source["calculation_radius_km"]
        take = allowed & np.isfinite(depth_values) & np.isfinite(gradient_values) & (distance < best_distance)
        thermo[take] = depth_values[take]
        thermo_gradient[take] = gradient_values[take]
        best_distance[take] = distance[take]
    thermo[~calculation_mask] = np.nan
    thermo_gradient[~calculation_mask] = np.nan

    point = all_points[0]
    x, y = all_offsets[0]
    selection = [{
        "index": 1, "x_km": float(x), "y_km": float(y),
        "is_source": True,
        "lat": float(point["lat"]), "lon": _normalise_lon(float(point["lon"])),
        "color": bellhop_results.SOURCE_COLORS[0],
    }]
    return {
        "x_km": xs, "y_km": ys, "bottom_depth_m": bottom,
        "terrain_mask": np.ones_like(bottom, dtype=np.int8),
        "calculation_mask": calculation_mask.astype(np.int8),
        "thermocline_depth_m": thermo,
        "thermocline_gradient_c_per_m": thermo_gradient,
        "selection_points": selection,
    }


def _write_flag(path: Path, value: Dict[str, Any]) -> None:
    """以 UTF-8 JSON 格式写入任务状态或完成标志。"""
    # 将字典写入 JSON 文件，确保 UTF-8 编码和可读缩进
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# 单声源执行和并行计划
# =============================================================================

def _run_source_process(
    context: Dict[str, Any],
    out_dir: str,
    topo_file: str,
    bearing_threads: int,
) -> Dict[str, Any]:
    """在一个工作进程内完整执行一个声源。

    每个声源进程内部有一个受限线程池。每个方位线程都会在自己的
    bearings_2d/bearing_* 目录中启动独立 Bellhop 子进程，
    因此 .env/.ray/.shd/.prt 文件不会互相覆盖。
    """
    started = time.time()
    bearings = context["bearings"]
    thread_count = max(1, min(int(bearing_threads), len(bearings)))
    progress_lock = threading.Lock()

    def quiet_progress(stage: str, message: str) -> None:
        """在子进程中忽略网页回调，仅保留函数签名。"""
        # 子进程不能安全调用父进程的网页进度回调。
        # 进度写入每个声源的 manifest，再由父进程轮询读取。
        return None

    with ThreadPoolExecutor(
        max_workers=thread_count,
        thread_name_prefix=f"source-{int(context['source_index']):03d}-bearing",
    ) as pool:
        futures = {}
        for bearing_index, bearing_deg in enumerate(bearings, 1):
            future = pool.submit(
                _run_prepared_bearing,
                context,
                bearing_index,
                float(bearing_deg),
                quiet_progress,
                progress_lock,
            )
            futures[future] = (bearing_index, float(bearing_deg))
        try:
            for future in as_completed(futures):
                bearing_index, bearing_deg = futures[future]
                section_result = future.result()
                context["sections_by_index"][bearing_index] = section_result
                context["manifest"]["bearings"].append({
                    "index": bearing_index,
                    "bearing_deg": bearing_deg,
                    "directory": str(Path("bearings_2d") / section_result["directory"]),
                    "ray_count": len(section_result["rays"]),
                    "reused": bool(section_result.get("reused", False)),
                    "status": "ready",
                })
                context["manifest"]["bearings"].sort(key=_index_value)
                context["manifest"]["source_process_pid"] = os.getpid()
                context["manifest"]["bearing_thread_count"] = thread_count
                _write_flag(
                    context["work_dir"] / "multibearing_2d_manifest.json",
                    context["manifest"],
                )
        except Exception:
            for future in futures:
                future.cancel()
            raise

    source = _finalise_source_context(context)
    out = Path(out_dir)

    # 仅用于显示的参数警告由父进程负责汇总和上报。
    # 派生出的声源进程不能引用父进程本地变量，例如 visual_warnings 或 notify 回调。
    topo = TopographyReader(topo_file)
    source_params = context["params"]
    source_point = {"lat": source["lat"], "lon": source["lon"]}
    source_master = _build_master_grid(
        source_params, [source_point], [(0.0, 0.0)], [source], topo
    )
    output_dir = out if source["index"] == 1 else context["work_dir"]
    source_export = bellhop_results.export_job_results(
        [source], source_params, output_dir, source_master
    )
    prefix = Path(".") if source["index"] == 1 else context["work_dir"].relative_to(out)
    files = []
    for filename in source_export["files"]:
        files.append(str((prefix / filename).as_posix()))
    return {
        "index": source["index"],
        "lat": source["lat"],
        "lon": source["lon"],
        "result_directory": str(prefix.as_posix()),
        "ray_count": source_export["ray_count"],
        "files": files,
        "bearing_count": len(bearings),
        "bearing_thread_count": thread_count,
        "source_process_pid": os.getpid(),
        "elapsed_sec": time.time() - started,
    }


def _hierarchical_parallel_plan(
    source_count: int,
    bearing_count: int,
    settings: Mapping[str, Any],
) -> Tuple[int, int, int]:
    """返回声源进程数、每个声源的方位线程数和 Bellhop 全局并发上限。"""
    if not settings["outer_parallel_enabled"]:
        return 1, 1, 1
    global_cap = max(1, int(settings["max_2d_workers"]))
    source_processes = min(
        max(1, int(source_count)),
        max(1, int(settings["max_source_processes"])),
        global_cap,
    )
    requested_threads = max(1, int(settings["bearing_threads_per_source"]))
    threads_per_source = min(
        max(1, int(bearing_count)),
        requested_threads,
        max(1, global_cap // source_processes),
    )
    effective_cap = source_processes * threads_per_source
    return source_processes, threads_per_source, effective_cap


def _normalise_source_points(params, max_sources):
    """校验声源点列表，并同步更新主声源字段。"""
    requested_points = params.get("selection_points")
    if isinstance(requested_points, list) and requested_points:
        raw_points = []
        for point in requested_points:
            if isinstance(point, Mapping):
                raw_points.append(dict(point))
            else:
                raw_points.append(point)
    else:
        raw_points = [{
            "lat": float(params["center_lat"]),
            "lon": _normalise_lon(float(params["center_lon"])),
        }]

    if len(raw_points) > max_sources:
        raise ParameterValidationError(f"声源数 {len(raw_points)} 超过 MAX_SOURCES={max_sources}")

    points = []
    for point in raw_points:
        points.append({
            "lat": float(point["lat"]),
            "lon": _normalise_lon(float(point["lon"])),
        })

    params["selection_points"] = points
    params["selection_point_count"] = len(points)
    params["source_mode"] = "independent_multi_source" if len(points) > 1 else "single_source"
    params["center_lat"] = points[0]["lat"]
    params["center_lon"] = points[0]["lon"]
    return points


def _build_source_grids(points, params, settings):
    """为所有声源建立环境采样局部网格。"""
    grid_size = max(3, int(round(float(params["n_grid"]))))
    source_grids = []
    for point in points:
        grid = _local_grid(
            float(point["lat"]),
            _normalise_lon(float(point["lon"])),
            float(params["radius_km"]),
            grid_size,
            float(settings["padding_km"]),
        )
        source_grids.append(grid)
    return source_grids


def _source_manifest_rows(source_contexts, out_dir):
    """生成多声源任务清单中的声源摘要行。"""
    rows = []
    for context in source_contexts:
        rows.append({
            "index": context["source_index"],
            "lat": context["source_lat"],
            "lon": context["source_lon"],
            "directory": str(context["work_dir"].relative_to(out_dir)),
            "bearing_count": len(context["bearings"]),
        })
    return rows


def _build_global_manifest(
    settings,
    source_contexts,
    shared_cache_info,
    task_count,
    source_process_count,
    bearing_threads,
    effective_bellhop_cap,
    out_dir,
):
    """生成本次声场任务的全局运行清单，记录“准备怎么计算、用了什么配置、启动多少任务和进程、有哪些声源和方位、每个声源的计算结果目录”等信息。"""
    return {
        "engine": "Bellhop multi-source, multi-bearing 2D",
        "config_version": settings["config_version"],
        "ssp_mode": settings["ssp_mode"],
        "ssp_interpolation": settings["ssp_interpolation"],
        "ssp_range_dependent": settings["ssp_mode"] == "range_dependent",
        "is_water_changed": settings["ssp_mode"] == "range_dependent",
        "water_column_mode": _water_column_mode_name(settings["ssp_mode"]),
        "ssp_profile_count": int(settings["n_2d_profiles"]),
        "ssp_box_guard_km": float(settings["box_guard_km"]),
        "bottom_geoacoustics_mode": settings["bottom_geo_mode"],
        "bty_format": "long" if settings["bottom_geo_long"] else "short",
        "source_count": len(source_contexts),
        "task_count": task_count,
        "parallel_mode": settings["bellhop_parallel_mode"],
        "solver_singlethread": settings["solver_singlethread"],
        "internal_parallel_enabled": settings["internal_parallel_enabled"],
        "outer_parallel_enabled": settings["outer_parallel_enabled"],
        "max_parallel_workers": effective_bellhop_cap,
        "source_process_count": source_process_count,
        "bearing_threads_per_source": bearing_threads,
        "global_bellhop_process_cap": effective_bellhop_cap,
        "task_policy": "source process pool + bounded bearing thread pool per source",
        "shared_environment_cache": shared_cache_info,
        "sources": _source_manifest_rows(source_contexts, out_dir),
    }


def _progress_manifest_paths(source_contexts):
    """返回每个声源的方位进度清单路径。"""
    # 生成每个声源的进度清单文件路径列表，供父进程轮询读取。
    paths = []
    for context in source_contexts:
        paths.append(context["work_dir"] / "multibearing_2d_manifest.json")
    return paths


def _bearing_task_count(source_contexts):
    """统计所有声源需要执行的方位任务总数。"""
    total = 0
    for context in source_contexts:
        total += len(context["bearings"])
    return total


def _max_bearing_count(source_contexts):
    """返回单个声源包含的最大方位数量。"""
    largest = 0
    for context in source_contexts:
        largest = max(largest, len(context["bearings"]))
    return largest


def _ready_bearing_count(payload):
    """从进度清单中统计已经完成的方位数量。"""
    completed = 0
    for item in payload.get("bearings", []):
        if _lower_text(item.get("status")) == "ready":
            completed += 1
    return completed


def _completed_bearing_count_from_exports(source_exports):
    """从声源导出结果中统计已经完成的方位数量。"""
    completed = 0
    for item in source_exports:
        completed += int(item.get("bearing_count", 0))
    return completed


def _sort_source_exports(source_exports):
    """按照声源索引稳定排序导出结果。"""
    source_exports.sort(key=_index_value)


def _collect_export_files(source_exports):
    """汇总所有声源生成的相对结果文件路径。"""
    files = ["multisource_manifest.json"]
    for item in source_exports:
        files.extend(item["files"])
    files.append("multisource_results.json")
    return files


def _total_ray_count(source_exports):
    """统计全部声源导出的声线数量。"""
    total = 0
    for item in source_exports:
        total += int(item["ray_count"])
    return total


# =============================================================================
# 对外主入口
# =============================================================================

def run(
    job_dir: str,
    params: Dict[str, Any],
    config: Mapping[str, Any],
    progress_callback=None,
) -> Dict[str, Any]:
    """执行支持可配置 SSP 和多方位的 Bellhop 二维任务。"""
    from .. import acoustic_environment
    from ..ocean_environment_source import DEFAULT_SOURCE as environment_source

    started = time.time()
    merged_params = _task_defaults(config)
    # 传统的 params 优先级高于默认值，默认值打底 + 实际值覆盖。
    merged_params.update(params)
    params = merged_params
    visual_warnings = _normalise_visual_params(params, config)
    validate_params(params)
    out = Path(job_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = _effective_settings(params, config)
    max_sources = max(1, int(_required_cfg(config, "MAX_SOURCES")))
    points = _normalise_source_points(params, max_sources)
    executable = _resolve_project_path(_required_cfg(config, "BELLHOP_CXX"))
    if not executable or not os.path.isfile(executable):
        raise FileNotFoundError(f"Bellhop 可执行文件不存在：{executable or '(空路径)'}")

    def notify(stage: str, message: str) -> None:
        """向外部回调发送经过全局阶段过滤的进度消息。"""
        if stage == "7/7" and not params.get("_global_tasks_complete"):
            return
        if progress_callback:
            progress_callback(stage, message)
    total_sources = len(points)
    notify(
        "1/7",
        f"准备 {total_sources} 个局部网格；一次性读取共享的温度、盐度和地形范围",
    )
    prepared_environment = acoustic_environment.acquire(
        params,
        config,
        settings,
        points,
        environment_source,
    )
    topo_file = prepared_environment.topography_file
    shared_cache_info = prepared_environment.cache_info()
    notify(
        "3/7",
        f"共享环境已准备：{shared_cache_info['local_grid_sample_count']} 个局部网格采样点，"
        f"复用 {shared_cache_info['temperature_cell_count']} 个温度格点和 "
        f"{shared_cache_info['salinity_cell_count']} 个盐度格点",
    )
    # 为每个声源准备独立的计算上下文，包括局部网格、环境、地形、参数和方位列表
    source_contexts: List[Dict[str, Any]] = []
    try:
        for index, _point in enumerate(points, 1):
            source_dir = out / "sources" / f"source_{index:03d}"

            def source_progress(stage: str, message: str, idx=index) -> None:
                """给进度消息增加当前声源序号前缀。"""
                notify(stage, f"[{idx}/{total_sources}] {message}")

            source_contexts.append(
                prepared_environment.build_bellhop_source_context(
                    index - 1,
                    source_dir,
                    source_progress,
                    _prepare_source_context,
                )
            )
    finally:
        prepared_environment.close()
    del prepared_environment

    task_count = _bearing_task_count(source_contexts)
    max_bearing_count = _max_bearing_count(source_contexts)
    source_process_count, bearing_threads, effective_bellhop_cap = _hierarchical_parallel_plan(
        len(source_contexts), max_bearing_count, settings
    )
    azimuths_snapshot = []
    for value in source_contexts[0]["bearings"]:
        azimuths_snapshot.append(float(value))
    global_manifest = _build_global_manifest(
        settings,
        source_contexts,
        shared_cache_info,
        task_count,
        source_process_count,
        bearing_threads,
        effective_bellhop_cap,
        out,
    )
    _write_flag(out / "multisource_manifest.json", global_manifest)
    solver_threading = "内部多线程" if settings["internal_parallel_enabled"] else "内部单线程"
    notify(
        "4/7",
        f"提交 {task_count} 个声源×方位任务；声源进程池 {source_process_count} 个，"
        f"每个声源最多 {bearing_threads} 个方位线程，全局最多 {effective_bellhop_cap} 个 "
        f"Bellhop 进程，每进程{solver_threading}",
    )

    source_exports: List[Dict[str, Any]] = []
    progress_stop = threading.Event()
    progress_paths = _progress_manifest_paths(source_contexts)
    progress_state = {"completed": -1}

    def _poll_child_progress() -> None:
        """轮询子进程进度清单并汇总为父任务进度。"""
        while not progress_stop.wait(0.35):
            completed = 0
            for manifest_path in progress_paths:
                try:
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    completed += _ready_bearing_count(payload)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
            if completed != progress_state["completed"]:
                progress_state["completed"] = completed
                notify("6/7", f"已完成 {completed}/{task_count} 个声源×方位任务")

    progress_thread = threading.Thread(
        target=_poll_child_progress, name="bellhop-process-progress", daemon=True
    )
    progress_thread.start()

    try:
        # 门户的一个三维声场请求只有一个声源。该请求已经位于 dispatcher 的
        # 专用声场进程中，直接在此进程建立方位线程池，避免再 spawn 一层进程并
        # 序列化完整环境数据。只有真正的多声源请求才使用下一级声源进程池。
        if len(source_contexts) == 1:
            context = source_contexts[0]
            source_export = _run_source_process(
                context,
                str(out),
                str(topo_file),
                bearing_threads,
            )
            source_exports.append(source_export)
            progress_state["completed"] = task_count
            notify("6/7", f"已完成 {task_count}/{task_count} 个方位任务；正在导出结果")
        else:
            with ProcessPoolExecutor(
                max_workers=source_process_count,
                mp_context=multiprocessing.get_context("spawn"),
            ) as process_pool:
                futures = {}
                for context in source_contexts:
                    future = process_pool.submit(
                        _run_source_process,
                        context,
                        str(out),
                        str(topo_file),
                        bearing_threads,
                    )
                    futures[future] = int(context["source_index"])
                try:
                    completed_sources = 0
                    for future in as_completed(futures):
                        source_index = futures[future]
                        source_export = future.result()
                        source_exports.append(source_export)
                        completed_sources += 1
                        completed_by_sources = _completed_bearing_count_from_exports(source_exports)
                        if completed_by_sources > progress_state["completed"]:
                            progress_state["completed"] = completed_by_sources
                            notify(
                                "6/7",
                                f"已完成 {completed_by_sources}/{task_count} 个声源×方位任务；"
                                f"声源 {source_index} 已拼接并导出 "
                                f"({completed_sources}/{len(source_contexts)})",
                            )
                except Exception:
                    for future in futures:
                        future.cancel()
                    raise
    finally:
        progress_stop.set()
        progress_thread.join(timeout=1.0)

    if progress_state["completed"] < task_count:
        notify("6/7", f"已完成 {task_count}/{task_count} 个声源×方位任务")

    _sort_source_exports(source_exports)
    params["azimuths_deg"] = azimuths_snapshot
    params["azimuth_count"] = len(params["azimuths_deg"])
    _write_flag(out / "multisource_results.json", {
        "source_mode": "independent_multi_source",
        "result_policy": "each source has an independent field; no merged field",
        "sources": source_exports,
    })
    all_files = _collect_export_files(source_exports)
    export_info = {
        "ray_count": _total_ray_count(source_exports),
        "source_count": len(source_exports),
        "files": all_files,
        "source_results": source_exports,
    }
    params["_global_tasks_complete"] = True
    notify("7/7", "已导出各声源独立三维声场与 NetCDF 结果")
    params.pop("_global_tasks_complete", None)
    elapsed = time.time() - started
    summary = {
        "status": "ready", "elapsed_sec": elapsed,
        "source_count": len(source_exports), "ray_count": export_info["ray_count"],
        "azimuth_count": int(params.get("azimuth_count", 0)),
        "azimuths_deg": params.get("azimuths_deg", []),
        "parallel_mode": settings["bellhop_parallel_mode"],
        "solver_singlethread": settings["solver_singlethread"],
        "config_version": settings["config_version"],
        "effective_settings": settings,
        "tl_source": "stitched Bellhop 2D transmission-loss sections",
        "compute_engine": (
            f"Python + Bellhop {settings['ssp_mode']} SSP + "
            f"{settings['bottom_geo_mode']} bottom, multi-bearing 2D stitching"
        ),
        "result_policy": "independent per-source fields; no merged field",
        "source_results": export_info["source_results"],
        "files": export_info["files"], "params": params,
        "visual_ready": True, "backend_export_complete": True,
    }
    _write_flag(out / "result_summary.json", summary)
    flag = {"status": "ready", "elapsed": elapsed, "source_count": len(source_exports)}
    _write_flag(out / "visual_ready.flag", flag)
    _write_flag(out / "backend_export_done.flag", flag)
    _write_flag(out / "result_ready.flag", flag)
    return {"status": "ready", "elapsed": elapsed, **export_info}
