"""单个预报时次的区域风、雨与海况网格。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Callable

from . import open_meteo_weather


# WeatherNext 2 与 DWD GWAM 当前都使用 0.25°原生经纬度网格。区域采样点
# 对齐到该网格，不再人为固定行列数，避免大区域欠采样和小区域重复采样。
NATIVE_GRID_DEGREES = 0.25
# 前端预报时间轴为当前整点起48小时；一次缓存完整时间轴，滑杆变化只读取
# 已有网格，不再为每个时次重复请求上游服务。
REGIONAL_CACHE_FORECAST_HOURS = open_meteo_weather.FORECAST_HOURS
MAX_FORECAST_LEAD_HOURS = REGIONAL_CACHE_FORECAST_HOURS - 1

FIELD_MAP = {
    "wind_speed_mps": "wind10",
    "wind_direction_deg": "wind_dir",
    "wind_speed_p10_mps": "wind10_p10",
    "wind_speed_p90_mps": "wind10_p90",
    "rain_rate_mmh": "precip_mmh",
    "rain_rate_p90_mmh": "precip_p90",
    "wave_height_m": "wave_hs_m",
    "wave_direction_deg": "wave_dir",
    "wave_period_s": "wave_tp_s",
    "swell_height_m": "swell_hs_m",
    "swell_direction_deg": "swell_dir",
    "sea_surface_temperature_c": "sst_c",
    "mean_sea_level_pressure_hpa": "mslp_hpa",
}

UNITS = {
    "wind_speed_mps": "m/s",
    "wind_direction_deg": "degree",
    "wind_speed_p10_mps": "m/s",
    "wind_speed_p90_mps": "m/s",
    "rain_rate_mmh": "mm/h",
    "rain_rate_p90_mmh": "mm/h",
    "wave_height_m": "m",
    "wave_direction_deg": "degree",
    "wave_period_s": "s",
    "swell_height_m": "m",
    "swell_direction_deg": "degree",
    "sea_surface_temperature_c": "degree_Celsius",
    "mean_sea_level_pressure_hpa": "hPa",
}


def parse_forecast_time(value: Any) -> datetime:
    """解析带时区的 ISO 8601 整点，并统一转换到 UTC。"""
    text = str(value or "").strip()
    if not text:
        raise ValueError("forecastTime 不能为空。")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError as exc:
        raise ValueError("forecastTime 必须是 ISO 8601 时间。") from exc
    if parsed.tzinfo is None:
        raise ValueError("forecastTime 必须包含时区，例如 Z 或 +08:00。")
    parsed = parsed.astimezone(timezone.utc)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError("forecastTime 必须是整点时间。")
    return parsed


def _native_axis(start: float, end: float) -> list[float]:
    """向外覆盖请求边界，并严格落在0.25°原生网格线上。"""
    first = math.floor(float(start) / NATIVE_GRID_DEGREES + 1e-9)
    last = math.ceil(float(end) / NATIVE_GRID_DEGREES - 1e-9)
    return [
        round(index * NATIVE_GRID_DEGREES, 6)
        for index in range(first, last + 1)
    ]


def _value_at(hourly: dict[str, Any], name: str, index: int) -> float | None:
    values = hourly.get(name)
    if not isinstance(values, list) or index >= len(values):
        return None
    value = values[index]
    return float(value) if value is not None else None


def fetch_region_forecast(
    west: float,
    south: float,
    east: float,
    north: float,
    forecast_time: datetime | str,
    *,
    now_utc: datetime | None = None,
    point_fetcher: Callable[..., dict[str, Any]] = open_meteo_weather.fetch_forecast,
) -> dict[str, Any]:
    """按天气模型0.25°原生网格获取指定整点的一小时区域预报。"""
    if isinstance(forecast_time, datetime):
        if forecast_time.tzinfo is None:
            raise ValueError("forecastTime 必须包含时区，例如 Z 或 +08:00。")
        valid_time = forecast_time.astimezone(timezone.utc)
    else:
        valid_time = parse_forecast_time(forecast_time)
    if valid_time.minute or valid_time.second or valid_time.microsecond:
        raise ValueError("forecastTime 必须是整点时间。")

    fetched_at = now_utc or datetime.now(timezone.utc)
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    fetched_at = fetched_at.astimezone(timezone.utc)
    current_hour = fetched_at.replace(minute=0, second=0, microsecond=0)
    lead_hours = int((valid_time - current_hour).total_seconds() // 3600)
    if valid_time < current_hour or lead_hours > MAX_FORECAST_LEAD_HOURS:
        raise ValueError(
            f"forecastTime 必须位于 UTC 当前整点至未来 {MAX_FORECAST_LEAD_HOURS} 小时。"
        )

    longitudes = _native_axis(west, east)
    # 地图栅格按北到南排列，和图像行方向一致。
    latitudes = list(reversed(_native_axis(south, north)))
    jobs = [
        (row, column, longitude, latitude)
        for row, latitude in enumerate(latitudes)
        for column, longitude in enumerate(longitudes)
    ]
    results: dict[tuple[int, int], dict[str, Any]] = {}

    if point_fetcher is open_meteo_weather.fetch_forecast:
        forecasts = open_meteo_weather.fetch_forecasts(
            [(longitude, latitude) for _, _, longitude, latitude in jobs],
            forecast_hours=REGIONAL_CACHE_FORECAST_HOURS,
            now_utc=current_hour,
        )
        for (row, column, _longitude, _latitude), forecast in zip(jobs, forecasts):
            results[(row, column)] = forecast
    else:
        # 保留可注入的单点获取器，供离线测试与替代数据源使用。
        def fetch(job):
            row, column, longitude, latitude = job
            forecast = point_fetcher(
                longitude,
                latitude,
                forecast_hours=REGIONAL_CACHE_FORECAST_HOURS,
                now_utc=current_hour,
            )
            return row, column, forecast

        with ThreadPoolExecutor(max_workers=min(6, len(jobs))) as executor:
            futures = [executor.submit(fetch, job) for job in jobs]
            for future in as_completed(futures):
                row, column, forecast = future.result()
                results[(row, column)] = forecast

    fields = {
        output_name: [[None for _ in longitudes] for _ in latitudes]
        for output_name in FIELD_MAP
    }
    source_latitudes = [[None for _ in longitudes] for _ in latitudes]
    source_longitudes = [[None for _ in longitudes] for _ in latitudes]
    ensemble_members: list[int] = []

    for row, _latitude in enumerate(latitudes):
        for column, _longitude in enumerate(longitudes):
            forecast = results[(row, column)]
            hourly = forecast.get("hourly") or {}
            for output_name, source_name in FIELD_MAP.items():
                fields[output_name][row][column] = _value_at(
                    hourly, source_name, lead_hours
                )
            source_latitudes[row][column] = forecast.get("grid_lat")
            source_longitudes[row][column] = forecast.get("grid_lon")
            if isinstance(forecast.get("ensemble_members"), int):
                ensemble_members.append(forecast["ensemble_members"])
    return {
        "provider": "open-meteo/weathernext2-ensemble+marine/dwd-gwam",
        "forecast_time": valid_time.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "forecast_hours": 1,
        "fetched_at": fetched_at.isoformat(timespec="seconds"),
        "requested_bounds": {
            "west": float(west),
            "south": float(south),
            "east": float(east),
            "north": float(north),
        },
        "grid": {
            "rows": len(latitudes),
            "columns": len(longitudes),
            "row_order": "north_to_south",
            "native_resolution_degrees": NATIVE_GRID_DEGREES,
            "sampling_bounds": {
                "west": longitudes[0],
                "south": latitudes[-1],
                "east": longitudes[-1],
                "north": latitudes[0],
            },
            "latitudes": latitudes,
            "longitudes": longitudes,
            "source_grid_latitudes": source_latitudes,
            "source_grid_longitudes": source_longitudes,
        },
        "ensemble_members": min(ensemble_members) if ensemble_members else None,
        "units": UNITS,
        "data": fields,
    }

