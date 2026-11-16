# -*- coding: utf-8 -*-
"""把统一声学后端的公共任务参数和一维 SSP 转换为 RAM 配置。"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


def _finite(value: Any, name: str) -> float:
    """读取有限浮点数并生成清晰的中文错误。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数值。") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值。")
    return number


def _profile(depth_m: Sequence[float], speed_mps: Sequence[float], water_depth_m: float) -> list[list[float]]:
    """清理一维 SSP，并将最后一个点插值或延伸到海底。"""
    depth = np.asarray(depth_m, dtype=np.float64)
    speed = np.asarray(speed_mps, dtype=np.float64)
    if depth.ndim != 1 or speed.ndim != 1 or depth.size != speed.size or depth.size < 2:
        raise ValueError("RAM 适配器要求等长的一维深度和声速数组。")
    valid = np.isfinite(depth) & np.isfinite(speed) & (depth >= 0.0) & (speed > 0.0)
    depth, speed = depth[valid], speed[valid]
    if depth.size < 2:
        raise ValueError("有效 SSP 点少于两个。")
    order = np.argsort(depth)
    depth, speed = depth[order], speed[order]
    depth, unique_index = np.unique(depth, return_index=True)
    speed = speed[unique_index]
    if depth[0] > 0.0:
        depth = np.insert(depth, 0, 0.0)
        speed = np.insert(speed, 0, speed[0])
    inside = depth < water_depth_m
    bottom_speed = float(np.interp(water_depth_m, depth, speed, left=speed[0], right=speed[-1]))
    return [[float(z), float(c)] for z, c in zip(np.append(depth[inside], water_depth_m), np.append(speed[inside], bottom_speed))]


def build_overrides_from_backend(
    params: Mapping[str, Any],
    depth_m: Sequence[float],
    sound_speed_mps: Sequence[float],
    *,
    water_depth_m: float,
    bathymetry: Optional[Sequence[Sequence[float]]] = None,
    bottom: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """将公共任务参数及单方位环境转换为 RAM runner 的覆盖配置。"""
    frequency = _finite(params.get("freq_hz"), "freq_hz")
    source_depth = _finite(params.get("source_depth_m"), "source_depth_m")
    receiver_depth = _finite(params.get("horizontal_tl_depth_m", source_depth), "horizontal_tl_depth_m")
    maximum_range = 1000.0 * _finite(params.get("radius_km"), "radius_km")
    water_depth = _finite(water_depth_m, "water_depth_m")
    if min(frequency, maximum_range, water_depth) <= 0.0:
        raise ValueError("频率、传播距离和水深必须为正数。")
    if not 0.0 < source_depth < water_depth or not 0.0 <= receiver_depth < water_depth:
        raise ValueError("声源和接收深度必须位于水体内部。")
    ssp = _profile(depth_m, sound_speed_mps, water_depth)
    bottom_values = {
        "sound_speed_mps": 1700.0,
        "density_g_cm3": 1.8,
        "attenuation_db_per_wavelength": 0.5,
    }
    if bottom:
        bottom_values.update({key: float(value) for key, value in bottom.items()})
    maximum_depth = max(water_depth + 300.0, water_depth * 1.5)
    bathy = list(bathymetry) if bathymetry is not None else [[0.0, water_depth], [maximum_range, water_depth]]
    return {
        "MODEL": {
            "FREQUENCY_HZ": frequency,
            "SOURCE_DEPTH_M": source_depth,
            "RECEIVER_DEPTH_M": receiver_depth,
            "MAX_RANGE_M": maximum_range,
            "OUTPUT_DEPTH_MAX_M": water_depth,
            "MAX_DEPTH_M": maximum_depth,
        },
        "ENVIRONMENT": {
            "BATHYMETRY": bathy,
            "PROFILES": [{
                "START_RANGE_M": 0.0,
                "WATER_SOUND_SPEED_MPS": ssp,
                "BOTTOM_SOUND_SPEED_MPS": [[0.0, bottom_values["sound_speed_mps"]]],
                "BOTTOM_DENSITY_G_CM3": [[0.0, bottom_values["density_g_cm3"]]],
                "BOTTOM_ATTENUATION_DB_PER_WAVELENGTH": [
                    [0.0, bottom_values["attenuation_db_per_wavelength"]],
                    [maximum_depth - 100.0, bottom_values["attenuation_db_per_wavelength"]],
                    [maximum_depth, 10.0],
                ],
            }],
        },
    }
