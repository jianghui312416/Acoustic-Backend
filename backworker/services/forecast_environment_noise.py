"""用固定 AIS 快照和 48 小时天气预报生成 Wenz 型噪声谱帧。"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .environment_noise import (
    coates_stojanovic_shipping_psd_db,
    coates_stojanovic_wind_psd_db,
    rain_noise_psd_db,
    total_from_component_levels,
)


FORECAST_HOURS = 48
AIS_ACTIVITY_SCALE = 6.0
AIS_DISTANCE_SCALE_KM = 100.0
RAIN_LEVEL_RATES_MM_H = np.asarray(
    [0.0, 0.25, 1.0, 2.5, 7.5, 20.0, 50.0], dtype=float
)
BEAUFORT_UPPER_MPS = np.asarray(
    [0.2, 1.5, 3.3, 5.4, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6],
    dtype=float,
)

# 由接口下发的交互频带。前端据此切换频带值和频谱图选区，避免前后端各自
# 维护一套范围；频带值按“线性功率平均 PSD + 10lg(带宽)”在前端实时计算，
# 因为航运活动度滑杆会在不重新请求接口的情况下改变船舶与总噪声曲线。
BAND_PRESETS = (
    {"id": "100hz-10khz", "label": "100 Hz–10 kHz", "low_hz": 100.0, "high_hz": 10_000.0},
    {"id": "1khz-10khz", "label": "1–10 kHz", "low_hz": 1_000.0, "high_hz": 10_000.0},
    {"id": "10hz-100hz", "label": "10–100 Hz", "low_hz": 10.0, "high_hz": 100.0},
    {"id": "50hz-1khz", "label": "50 Hz–1 kHz", "low_hz": 50.0, "high_hz": 1_000.0},
)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def beaufort_level(wind_speed_mps: Any) -> int:
    """按 WMO 10 米风速边界映射为 0—12 级。"""
    speed = _finite_float(wind_speed_mps)
    if speed is None or speed < 0.0:
        raise ValueError("天气预报中的 wind10 必须是非负有限数。")
    return int(np.searchsorted(BEAUFORT_UPPER_MPS, speed, side="left"))


def rain_level(rain_rate_mmh: Any) -> int:
    """把连续雨强映射到旧噪声模型的 0—6 级。"""
    rate = _finite_float(rain_rate_mmh)
    if rate is None or rate < 0.0:
        raise ValueError("天气预报中的 precip_mmh 必须是非负有限数。")
    if rate == 0.0:
        return 0
    return max(1, int(np.argmin(np.abs(RAIN_LEVEL_RATES_MM_H - rate))))


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_m = 6_371_008.8
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius_m * math.asin(min(1.0, math.sqrt(value)))


def shipping_activity_from_ais(
    vessels: Sequence[Mapping[str, Any]], receiver_lon: float, receiver_lat: float
) -> dict[str, Any]:
    """从 AIS 加权交通量估计 Coates/Stojanovic 航运活动因子。"""
    effective_activity = 0.0
    modeled_count = 0
    skipped_count = 0
    within_100km = 0
    moving_count = 0

    for vessel in vessels:
        if not isinstance(vessel, Mapping):
            skipped_count += 1
            continue
        longitude = _finite_float(vessel.get("longitude"))
        latitude = _finite_float(vessel.get("latitude"))
        if (
            longitude is None
            or latitude is None
            or not -180.0 <= longitude <= 180.0
            or not -90.0 <= latitude <= 90.0
        ):
            skipped_count += 1
            continue
        speed = max(0.0, _finite_float(vessel.get("sog_kn")) or 0.0)
        length = _finite_float(vessel.get("length_m"))
        length = min(400.0, max(20.0, length if length is not None else 100.0))
        distance_km = _haversine_m(
            receiver_lon, receiver_lat, longitude, latitude
        ) / 1000.0
        if distance_km <= 100.0:
            within_100km += 1
        if speed >= 0.5:
            moving_count += 1
        distance_weight = math.exp(-distance_km / AIS_DISTANCE_SCALE_KM)
        speed_weight = 0.2 + 0.8 * min(speed / 15.0, 1.0)
        length_weight = math.sqrt(length / 100.0)
        effective_activity += distance_weight * speed_weight * length_weight
        modeled_count += 1

    activity = 1.0 - math.exp(-effective_activity / AIS_ACTIVITY_SCALE)
    return {
        "shipping_activity": round(activity, 6),
        "effective_ais_activity": round(effective_activity, 6),
        "input_vessel_count": len(vessels),
        "modeled_vessel_count": modeled_count,
        "skipped_vessel_count": skipped_count,
        "vessels_within_100km": within_100km,
        "moving_vessel_count": moving_count,
        "distance_scale_km": AIS_DISTANCE_SCALE_KM,
        "activity_scale": AIS_ACTIVITY_SCALE,
        "model": "Wenz Coates/Stojanovic shipping PSD with AIS-weighted activity",
        "ais_mapping": (
            "sum(exp(-range_km/100) * (0.2+0.8*min(SOG_kn/15,1)) * "
            "sqrt(clamp(length_m,20,400)/100)); s=1-exp(-sum/6)"
        ),
        "calibration_note": (
            "AIS 到 s 的映射是可解释的工程参数化，并非 Coates 原式的一部分；"
            "前端滑杆直接控制连续活动因子 s∈[0,1]。"
        ),
    }


def _hourly_payload(forecast: Mapping[str, Any]) -> Mapping[str, Any]:
    hourly = forecast.get("hourly")
    return hourly if isinstance(hourly, Mapping) else forecast


def _series(hourly: Mapping[str, Any], name: str, count: int) -> list[Any]:
    values = hourly.get(name)
    if not isinstance(values, list) or len(values) < count:
        raise ValueError(f"48 小时天气响应缺少完整字段 {name}。")
    return values[:count]


def _optional_at(values: list[Any] | None, index: int) -> float | None:
    if not values or index >= len(values):
        return None
    return _finite_float(values[index])


def _rounded(values: Any) -> list[float]:
    return np.round(np.asarray(values, dtype=float), 6).tolist()


def build_forecast_noise(
    *,
    longitude: float,
    latitude: float,
    ais: Mapping[str, Any],
    weather: Mapping[str, Any],
    min_frequency_hz: float = 10.0,
    max_frequency_hz: float = 100_000.0,
    frequency_points: int = 512,
) -> dict[str, Any]:
    """构造可供前端时间滑杆直接切换的 48 帧噪声谱。"""
    vessels = ais.get("vessels")
    if not isinstance(vessels, list):
        raise ValueError("ais.vessels 必须是数组。")
    shipping = shipping_activity_from_ais(vessels, longitude, latitude)
    shipping_activity = float(shipping["shipping_activity"])
    frequency = np.geomspace(
        float(min_frequency_hz), float(max_frequency_hz), int(frequency_points)
    )
    shipping_curve = np.asarray(
        coates_stojanovic_shipping_psd_db(
            frequency, shipping_activity
        ),
        dtype=float,
    )
    frequency_khz = frequency / 1000.0
    turbulence_curve = 17.0 - 30.0 * np.log10(frequency_khz)
    thermal_curve = -15.0 + 20.0 * np.log10(frequency_khz)

    hourly = _hourly_payload(weather)
    times = _series(hourly, "time", FORECAST_HOURS)
    winds = _series(hourly, "wind10", FORECAST_HOURS)
    rains = _series(hourly, "precip_mmh", FORECAST_HOURS)
    sst = hourly.get("sst_c") if isinstance(hourly.get("sst_c"), list) else None
    wave_height = hourly.get("wave_hs_m") if isinstance(hourly.get("wave_hs_m"), list) else None
    wave_period = hourly.get("wave_tp_s") if isinstance(hourly.get("wave_tp_s"), list) else None
    wave_direction = hourly.get("wave_dir") if isinstance(hourly.get("wave_dir"), list) else None
    swell_height = hourly.get("swell_hs_m") if isinstance(hourly.get("swell_hs_m"), list) else None
    sea_level_pressure = hourly.get("mslp_hpa") if isinstance(hourly.get("mslp_hpa"), list) else None
    gust = hourly.get("gust") if isinstance(hourly.get("gust"), list) else None
    wind_direction = hourly.get("wind_dir") if isinstance(hourly.get("wind_dir"), list) else None

    frames = []
    for index, (time_value, wind_value, rain_value) in enumerate(
        zip(times, winds, rains)
    ):
        wind = _finite_float(wind_value)
        rain = _finite_float(rain_value)
        wind_curve = coates_stojanovic_wind_psd_db(frequency, wind)
        rain_curve = rain_noise_psd_db(frequency, rain) if rain > 0.0 else None
        components = {
            "turbulence": turbulence_curve,
            "shipping": shipping_curve,
            "thermal": thermal_curve,
            "wind": wind_curve,
        }
        if rain_curve is not None:
            components["rain"] = rain_curve
        total = total_from_component_levels(components)
        frames.append({
            "index": index,
            "time": str(time_value),
            "weather": {
                "wind10_mps": wind,
                "wind_direction_deg": _optional_at(wind_direction, index),
                "precipitation_mmh": rain,
                "sst_c": _optional_at(sst, index),
                "wave_hs_m": _optional_at(wave_height, index),
                "wave_period_s": _optional_at(wave_period, index),
                "wave_direction_deg": _optional_at(wave_direction, index),
                "swell_hs_m": _optional_at(swell_height, index),
                "mslp_hpa": _optional_at(sea_level_pressure, index),
                "gust_mps": _optional_at(gust, index),
            },
            "drivers": {
                "wind_speed_mps": wind,
                "rain_rate_mmh": rain,
                "shipping_activity": shipping_activity,
            },
            "varying_components_db": {
                "wind": _rounded(wind_curve),
                "rain": _rounded(rain_curve) if rain_curve is not None else None,
                "total": _rounded(total),
            },
        })

    return {
        "model": "Continuous Wenz Coates/Stojanovic composite driven by fixed AIS and 48-hour weather forecast",
        "receiver": {"longitude": longitude, "latitude": latitude},
        "frequency_hz": _rounded(frequency),
        "frequency_spacing": "logarithmic",
        "spectrum": {
            "quantity": "power_spectral_density",
            "unit": "dB re 1 uPa^2/Hz",
            "band_level_unit": "dB re 1 uPa^2",
            "band_level_formula": (
                "10*log10(integral(10^(PSD_dB/10),df)/(high_hz-low_hz)) "
                "+ 10*log10(high_hz-low_hz)"
            ),
            "default_band_id": BAND_PRESETS[0]["id"],
            "band_presets": [
                {**band, "bandwidth_hz": band["high_hz"] - band["low_hz"]}
                for band in BAND_PRESETS
            ],
            "components": [
                {"id": "wind", "label": "风速噪声", "variability": "hourly"},
                {"id": "rain", "label": "降雨噪声", "variability": "hourly"},
                {"id": "shipping", "label": "船舶噪声", "variability": "interactive_fixed_ais"},
                {"id": "turbulence", "label": "湍流噪声", "variability": "fixed"},
                {"id": "thermal", "label": "热噪声", "variability": "fixed"},
                {"id": "total", "label": "总噪声", "variability": "derived"},
            ],
        },
        "fixed_ais": {
            "source": ais.get("source", "AISStream"),
            "collection_started_at_utc": ais.get("collection_started_at_utc"),
            "collection_ended_at_utc": ais.get("collection_ended_at_utc"),
            **shipping,
        },
        "fixed_components_db": {
            "shipping": _rounded(shipping_curve),
            "turbulence": _rounded(turbulence_curve),
            "thermal": _rounded(thermal_curve),
        },
        "forecast": {
            "provider": weather.get("provider"),
            "ensemble_members": weather.get("ensemble_members"),
            "fetched_at": weather.get("fetched_at"),
            "grid_lat": weather.get("grid_lat", weather.get("latitude")),
            "grid_lon": weather.get("grid_lon", weather.get("longitude")),
            "hours": FORECAST_HOURS,
        },
        "slider": {
            "min": 0,
            "max": FORECAST_HOURS - 1,
            "step": 1,
            "initial": 0,
            "time_field": "frames[index].time",
        },
        "frames": frames,
        "frontend_note": (
            "AIS 航运活动 s 与 fixed_components_db 在 48 帧内保持不变；风噪声直接由 "
            "wind10_mps、雨噪声直接由 precipitation_mmh 计算，不再先量化后估算。"
        ),
    }
