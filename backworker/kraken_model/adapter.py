# -*- coding: utf-8 -*-
"""把统一声学后端的公共任务参数转换为 Kraken 独立配置。"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


def _finite_float(value: Any, name: str) -> float:
    """读取有限浮点数，并把字段名写入错误信息。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数值。") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值。")
    return number


def _prepare_profile(
    depth_m: Sequence[float],
    sound_speed_mps: Sequence[float],
    water_depth_m: float,
) -> list[list[float]]:
    """整理一维 SSP，并把剖面可靠地延伸或裁剪到实际水深。"""
    depth = np.asarray(depth_m, dtype=np.float64)
    speed = np.asarray(sound_speed_mps, dtype=np.float64)
    if depth.ndim != 1 or speed.ndim != 1 or depth.size != speed.size or depth.size < 2:
        raise ValueError("Kraken 适配器要求等长的一维 depth_m 和 sound_speed_mps,且长度必须大于等于2。")
    valid = np.isfinite(depth) & np.isfinite(speed) & (depth >= 0.0) & (speed > 0.0)
    depth, speed = depth[valid], speed[valid]
    if depth.size < 2:
        raise ValueError("有效声速剖面点少于两个。")
    order = np.argsort(depth)
    depth, speed = depth[order], speed[order]
    unique_depth, unique_index = np.unique(depth, return_index=True)
    depth, speed = unique_depth, speed[unique_index]
    if depth[0] > 0.0:
        depth = np.insert(depth, 0, 0.0)
        speed = np.insert(speed, 0, speed[0])
    inside = depth < water_depth_m
    result_depth = depth[inside]
    result_speed = speed[inside]
    bottom_speed = float(np.interp(water_depth_m, depth, speed, left=speed[0], right=speed[-1]))
    result_depth = np.append(result_depth, water_depth_m)
    result_speed = np.append(result_speed, bottom_speed)
    return [[float(z), float(c)] for z, c in zip(result_depth, result_speed)]


def build_overrides_from_backend(
    params: Mapping[str, Any],
    depth_m: Sequence[float],
    sound_speed_mps: Sequence[float],
    *,
    water_depth_m: Optional[float] = None,
    mode_sample_depth_m: Optional[float] = None,
    receiver_depths_m: Optional[Sequence[float]] = None,
    bottom: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """把公共任务参数和一维 SSP 转换为 Kraken runner 的覆盖配置。"""
    frequency = _finite_float(params.get("freq_hz"), "freq_hz")
    physical_source_depth = _finite_float(params.get("source_depth_m"), "source_depth_m")
    mode_sample_depth = _finite_float(
        physical_source_depth if mode_sample_depth_m is None else mode_sample_depth_m,
        "mode_sample_depth_m",
    )
    maximum_range = _finite_float(params.get("radius_km"), "radius_km")
    inferred_depth = float(np.nanmax(np.asarray(depth_m, dtype=np.float64)))
    water_depth = _finite_float(
        inferred_depth if water_depth_m is None else water_depth_m,
        "water_depth_m",
    )
    if min(frequency, maximum_range, water_depth) <= 0.0:
        raise ValueError("频率、最大距离和水深必须为正数。")
    if mode_sample_depth < 0.0 or mode_sample_depth >= water_depth:
        raise ValueError("Kraken 模态采样深度必须位于当前局部水体内部。")
    profile = _prepare_profile(depth_m, sound_speed_mps, water_depth)
    if receiver_depths_m is None:
        requested_max = _finite_float(params.get("receiver_depth_max_m", water_depth * 0.95), "receiver_depth_max_m")
        receiver_max = min(max(requested_max, 1.0), water_depth - 1.0e-3)
        receiver_count = int(params.get("tl_n_rd", 121))
        receiver_count = min(max(receiver_count, 2), 200)
        receivers = np.linspace(0.0, receiver_max, receiver_count).tolist()
    else:
        receivers = [float(value) for value in receiver_depths_m]
    range_count = int(params.get("tl_n_range", 1001))
    range_count = min(max(range_count, 2), 4093)
    environment: Dict[str, Any] = {
        "TITLE": "Kraken backend profile",
        "FREQUENCY_HZ": frequency,
        "WATER_DEPTH_M": water_depth,
        "SOUND_SPEED_PROFILE": profile,
        "MAX_RANGE_KM": maximum_range,
    }
    if bottom:
        environment["BOTTOM"] = dict(bottom)
    return {
        "ENVIRONMENT": environment,
        "FIELD": {
            "SOURCE_DEPTHS_M": [mode_sample_depth],
            "RECEIVER_DEPTHS_M": receivers,
            "RECEIVER_RANGES_KM": {
                "MIN": max(0.001, maximum_range / range_count),
                "MAX": maximum_range,
                "COUNT": range_count,
            },
        },
    }
