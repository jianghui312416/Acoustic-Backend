# -*- coding: utf-8 -*-
"""Pierson-Moskowitz（PM）海浪谱随机海面生成。

门户接口提供 ``seaState`` 时，不再依赖本地海浪 NetCDF 文件中的
有效波高、峰值周期和波向，而是按海况等级换算代表波高，并用 PM
谱叠加随机相位生成随机起伏的二维/三维海面高程场。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import binary_dilation


GRAVITY_MPS2 = 9.81
PIERSON_MOSKOWITZ_ALPHA = 0.0081
PIERSON_MOSKOWITZ_BETA = 1.25

# WMO/Douglas 海况等级对应的代表有效波高（m）。取各等级范围的中值，
# 0 级按无有效波浪处理，9 级取 14 m 以上的代表值。
SEA_STATE_SIGNIFICANT_WAVE_HEIGHT_M = {
    0: 0.0,
    1: 0.05,
    2: 0.30,
    3: 0.875,
    4: 1.875,
    5: 3.25,
    6: 5.00,
    7: 7.50,
    8: 11.50,
    9: 15.00,
}


def sea_state_parameters(sea_state: Any) -> tuple[float, float, float]:
    """由海况等级返回目标有效波高、峰值周期和充分成长风速。"""
    try:
        level = int(round(float(sea_state)))
    except (TypeError, ValueError) as exc:
        raise ValueError("海况 seaState 必须是 0 到 9 的整数。") from exc
    if level < 0 or level > 9:
        raise ValueError("海况 seaState 必须位于 0 到 9。")
    hs_m = float(SEA_STATE_SIGNIFICANT_WAVE_HEIGHT_M[level])
    if hs_m <= 0.0:
        return 0.0, 0.0, 0.0
    # 充分成长 PM 海况：Hs = 0.21 U^2 / g，Tp = 0.751 U。
    wind_speed_mps = math.sqrt(hs_m * GRAVITY_MPS2 / 0.21)
    peak_period_s = 0.751 * wind_speed_mps
    return hs_m, peak_period_s, wind_speed_mps


def pierson_moskowitz_spectrum(
    frequency_hz: np.ndarray,
    peak_period_s: float,
) -> np.ndarray:
    """返回单边 Pierson-Moskowitz 频率谱 S(f)，单位 m^2/Hz。"""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    peak_period = float(peak_period_s)
    if peak_period <= 0.0:
        return np.zeros_like(frequency)
    peak_frequency = 1.0 / peak_period
    safe_frequency = np.maximum(frequency, 1.0e-9)
    spectrum = (
        PIERSON_MOSKOWITZ_ALPHA
        * GRAVITY_MPS2**2
        / ((2.0 * math.pi) ** 4 * safe_frequency**5)
        * np.exp(-PIERSON_MOSKOWITZ_BETA * (peak_frequency / safe_frequency) ** 4)
    )
    return np.where(np.isfinite(spectrum), spectrum, 0.0)


def _grid_coordinates_m(grid: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """读取相对声源中心的二维局部网格坐标，并换算为米。"""
    x_km = grid.get("x_rel_km")
    y_km = grid.get("y_rel_km")
    if x_km is not None and y_km is not None:
        x = np.asarray(x_km, dtype=np.float64) * 1000.0
        y = np.asarray(y_km, dtype=np.float64) * 1000.0
        if x.shape == y.shape:
            return x, y
    x_axis = np.asarray(grid["x_rel_axis_km"], dtype=np.float64) * 1000.0
    y_axis = np.asarray(grid["y_rel_axis_km"], dtype=np.float64) * 1000.0
    return np.meshgrid(x_axis, y_axis)


def _directional_weights(count: int) -> tuple[np.ndarray, np.ndarray]:
    """返回以主波向为中心的 cos^2 方向分布角度和积分权重。"""
    if count < 1:
        raise ValueError("方向分量数量必须为正整数。")
    edges = np.linspace(-0.5 * math.pi, 0.5 * math.pi, count + 1)
    angles = 0.5 * (edges[:-1] + edges[1:])
    delta = math.pi / count
    weights = (2.0 / math.pi) * np.cos(angles) ** 2 * delta
    return angles, weights


def synthesize_pm_elevation(
    x_m: np.ndarray,
    y_m: np.ndarray,
    significant_wave_height_m: float,
    peak_period_s: float,
    direction_deg: float,
    rng: np.random.Generator,
    *,
    frequency_count: int = 64,
    direction_count: int = 16,
) -> np.ndarray:
    """用 PM 谱叠加随机相位，在给定网格上生成随机海面高程。"""
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    hs_m = float(significant_wave_height_m)
    if x.shape != y.shape:
        raise ValueError("海面网格的 x/y 坐标形状必须一致。")
    if not math.isfinite(hs_m) or hs_m <= 0.0:
        return np.zeros(x.shape, dtype=np.float64)
    peak_period = float(peak_period_s)
    if not math.isfinite(peak_period) or peak_period <= 0.0:
        return np.zeros(x.shape, dtype=np.float64)

    peak_frequency = 1.0 / peak_period
    minimum_frequency = max(0.02, 0.35 * peak_frequency)
    maximum_frequency = max(4.0 * peak_frequency, 0.5)
    frequencies = np.linspace(
        minimum_frequency,
        maximum_frequency,
        max(8, int(frequency_count)),
        dtype=np.float64,
    )
    delta_frequency = float(frequencies[1] - frequencies[0])
    spectrum = pierson_moskowitz_spectrum(frequencies, peak_period)

    relative_angles, direction_weights = _directional_weights(
        max(4, int(direction_count))
    )
    angles = math.radians(float(direction_deg)) + relative_angles
    amplitudes = np.sqrt(
        2.0
        * spectrum[:, None]
        * direction_weights[None, :]
        * delta_frequency
    )

    omega = 2.0 * math.pi * frequencies
    # 深水色散关系；PM 海浪能量集中在长周期重力波，深水近似足够稳定。
    wavenumber = omega**2 / GRAVITY_MPS2
    kx = wavenumber[:, None] * np.cos(angles)[None, :]
    ky = wavenumber[:, None] * np.sin(angles)[None, :]
    phases = rng.uniform(0.0, 2.0 * math.pi, amplitudes.shape)

    elevation = np.zeros(x.shape, dtype=np.float64)
    for index in range(frequencies.size):
        phase = (
            kx[index, :, None, None] * x[None, :, :]
            + ky[index, :, None, None] * y[None, :, :]
            + phases[index, :, None, None]
        )
        elevation += np.sum(
            amplitudes[index, :, None, None] * np.cos(phase),
            axis=0,
        )

    finite = np.isfinite(elevation)
    if not np.any(finite):
        return np.zeros_like(elevation)
    elevation[finite] -= float(np.mean(elevation[finite]))
    standard_deviation = float(np.std(elevation[finite]))
    if standard_deviation > 0.0:
        # 随机相位叠加后按目标有效波高归一化，消除离散积分误差。
        elevation *= (hs_m / 4.0) / standard_deviation
    elevation[~np.isfinite(elevation)] = 0.0
    return elevation


def build_pm_altimetry(
    sea_state: Any,
    grid: Mapping[str, Any],
    bathymetry: Mapping[str, Any],
    *,
    seed: Any = None,
) -> dict[str, Any]:
    """按海况等级生成 Bellhop 使用的随机 PM 海面高程场。"""
    hs_m, peak_period_s, wind_speed_mps = sea_state_parameters(sea_state)
    rng = np.random.default_rng(seed)
    direction_deg = float(rng.uniform(0.0, 360.0)) if hs_m > 0.0 else 0.0
    x_m, y_m = _grid_coordinates_m(grid)
    elevation = synthesize_pm_elevation(
        x_m,
        y_m,
        hs_m,
        peak_period_s,
        direction_deg,
        rng,
    )

    land = np.asarray(bathymetry.get("land_mask"), dtype=bool)
    if land.shape != elevation.shape:
        raise ValueError("海面高程与陆地掩膜网格形状不一致。")
    calm = binary_dilation(land, iterations=1) if np.any(land) else land.copy()
    elevation[calm] = 0.0
    depth = np.asarray(bathymetry.get("depth_m"), dtype=np.float64)
    if depth.shape != elevation.shape:
        raise ValueError("海面高程与海底水深网格形状不一致。")
    limit = np.maximum(0.0, depth - 0.5)
    elevation = np.clip(elevation, -limit, limit)
    elevation[land] = 0.0

    water = (~calm) & np.isfinite(elevation)
    if np.any(water):
        elevation[water] -= float(np.mean(elevation[water]))

    shape = elevation.shape
    level = int(round(float(sea_state)))
    return {
        "model": "pierson_moskowitz",
        "sea_state": level,
        "hs_m": np.full(shape, hs_m, dtype=np.float64),
        "tp_s": np.full(shape, peak_period_s, dtype=np.float64),
        "direction_deg": np.full(shape, direction_deg, dtype=np.float64),
        "elevation_m": elevation,
        "significant_wave_height_m": hs_m,
        "peak_period_s": peak_period_s,
        "wind_speed_mps": wind_speed_mps,
        "random_seed": None if seed is None else int(seed),
    }
