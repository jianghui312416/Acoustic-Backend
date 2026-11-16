"""指定时次和频带的区域环境噪声网格。"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .environment_noise import (
    coates_stojanovic_shipping_psd_db,
    coates_stojanovic_wind_psd_db,
    rain_noise_psd_db,
    total_from_component_levels,
)
from .forecast_environment_noise import shipping_activity_from_ais


def _finite_nonnegative(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _band_level_db(frequency_hz: np.ndarray, spectrum_db: Any) -> float:
    linear_psd = np.power(10.0, np.asarray(spectrum_db, dtype=float) / 10.0)
    # 频带噪声级 = 10lg(频带内 PSD 功率积分)，单位 dB re 1 μPa²。
    power = float(np.trapezoid(linear_psd, frequency_hz))
    return round(10.0 * math.log10(max(power, np.finfo(float).tiny)), 6)


def build_region_noise(
    *,
    regional_weather: Mapping[str, Any],
    ais: Mapping[str, Any],
    min_frequency_hz: float,
    max_frequency_hz: float,
    frequency_points: int = 128,
) -> dict[str, Any]:
    """把区域天气网格与固定 AIS 快照合成为频带环境噪声伪彩图数据。"""
    vessels = ais.get("vessels")
    if not isinstance(vessels, list):
        raise ValueError("ais.vessels 必须是数组。")

    grid = regional_weather.get("grid") or {}
    latitudes = grid.get("latitudes")
    longitudes = grid.get("longitudes")
    weather_data = regional_weather.get("data") or {}
    winds = weather_data.get("wind_speed_mps")
    rains = weather_data.get("rain_rate_mmh")
    if not isinstance(latitudes, list) or not isinstance(longitudes, list):
        raise ValueError("区域天气响应缺少经纬度网格。")
    if not isinstance(winds, list) or not isinstance(rains, list):
        raise ValueError("区域天气响应缺少风速或雨强网格。")

    frequency = np.geomspace(
        float(min_frequency_hz), float(max_frequency_hz), int(frequency_points)
    )
    frequency_khz = frequency / 1000.0
    turbulence = 17.0 - 30.0 * np.log10(frequency_khz)
    thermal = -15.0 + 20.0 * np.log10(frequency_khz)

    field_names = (
        "noise_level_db",
        "wind_noise_level_db",
        "rain_noise_level_db",
        "shipping_noise_level_db",
        "shipping_activity",
    )
    output = {
        name: [[None for _ in longitudes] for _ in latitudes]
        for name in field_names
    }

    for row, latitude in enumerate(latitudes):
        for column, longitude in enumerate(longitudes):
            wind = _finite_nonnegative(winds[row][column])
            rain = _finite_nonnegative(rains[row][column])
            if wind is None or rain is None:
                continue
            shipping_meta = shipping_activity_from_ais(
                vessels, float(longitude), float(latitude)
            )
            shipping_activity = float(shipping_meta["shipping_activity"])
            shipping = np.asarray(
                coates_stojanovic_shipping_psd_db(frequency, shipping_activity),
                dtype=float,
            )
            wind_curve = np.asarray(
                coates_stojanovic_wind_psd_db(frequency, wind), dtype=float
            )
            rain_curve = (
                np.asarray(rain_noise_psd_db(frequency, rain), dtype=float)
                if rain > 0.0 else None
            )
            components = {
                "turbulence": turbulence,
                "shipping": shipping,
                "thermal": thermal,
                "wind": wind_curve,
            }
            if rain_curve is not None:
                components["rain"] = rain_curve
            total = total_from_component_levels(components)

            output["noise_level_db"][row][column] = _band_level_db(frequency, total)
            output["wind_noise_level_db"][row][column] = _band_level_db(frequency, wind_curve)
            output["rain_noise_level_db"][row][column] = (
                _band_level_db(frequency, rain_curve) if rain_curve is not None else None
            )
            output["shipping_noise_level_db"][row][column] = _band_level_db(frequency, shipping)
            output["shipping_activity"][row][column] = round(shipping_activity, 6)

    return {
        "model": "Continuous Wenz Coates/Stojanovic regional composite",
        "provider": regional_weather.get("provider"),
        "forecast_time": regional_weather.get("forecast_time"),
        "fetched_at": regional_weather.get("fetched_at"),
        "requested_bounds": regional_weather.get("requested_bounds"),
        "grid": grid,
        "band": {
            "low_hz": float(min_frequency_hz),
            "high_hz": float(max_frequency_hz),
            "frequency_points": int(frequency_points),
            "quantity": "band_sound_pressure_level",
            "unit": "dB re 1 uPa^2",
        },
        "units": {
            "noise_level_db": "dB re 1 uPa^2",
            "wind_noise_level_db": "dB re 1 uPa^2",
            "rain_noise_level_db": "dB re 1 uPa^2",
            "shipping_noise_level_db": "dB re 1 uPa^2",
            "shipping_activity": "1",
        },
        "data": output,
        "fixed_ais": {
            "source": ais.get("source", "AISStream"),
            "collection_started_at_utc": ais.get("collection_started_at_utc"),
            "collection_ended_at_utc": ais.get("collection_ended_at_utc"),
            "vessel_count": len(vessels),
        },
    }
