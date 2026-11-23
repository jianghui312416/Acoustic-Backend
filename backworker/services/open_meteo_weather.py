"""Open-Meteo WeatherNext 2 与 Marine API 的逐小时海洋天气服务。"""

from __future__ import annotations

import math
import re
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
STANDARD_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHERNEXT_MODEL = "google_weathernext2_ensemble"
# 区域覆盖层只消费集合均值。直接使用官方预计算均值，避免为每个0.25°网格点
# 下载全部集合成员；单点天气/噪声预报仍继续使用上面的完整集合模型。
WEATHERNEXT_MEAN_MODEL = "google_weathernext2_ensemble_mean"
MARINE_MODEL = "dwd_gwam"
FORECAST_HOURS = 48
MIN_FORECAST_HOURS = 1
MAX_FORECAST_HOURS = 168

_CACHE_GRID_DEG = 0.25
# 区域图层会连续读取同一片0.25°原生网格的多个预报时次。缓存必须能容纳
# 完整区域，否则超过旧的225点规模后会边请求边清空，并在噪声接口中重复下载。
_CACHE_LIMIT = 8192
_BATCH_SIZE = 24
_BATCH_WORKERS = 4
_FORECAST_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_FORECAST_CACHE_LOCK = threading.Lock()
_BATCH_FETCH_LOCK = threading.Lock()

WEATHER_VARIABLES = (
    "pressure_msl",
    "sea_surface_temperature",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
)
# 区域伪彩图和区域噪声只需要这三个天气量；其余字段不从公网下载。
REGIONAL_WEATHER_VARIABLES = (
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
)
MARINE_VARIABLES = (
    "wave_height",
    "wave_direction",
    "wave_period",
    "wind_wave_height",
    "wind_wave_peak_period",
    "swell_wave_height",
    "swell_wave_direction",
    "swell_wave_peak_period",
)
OUTPUT_UNITS = {
    "gust": "m/s",
    "mslp_hpa": "hPa",
    "precip_p90": "mm/h",
    "sst_c": "°C",
    "swell_dir": "°",
    "swell_hs_m": "m",
    "time": "UTC",
    "wind10": "m/s",
    "wind10_p10": "m/s",
    "wind10_p90": "m/s",
    "wind_dir": "°",
    "wave_dir": "°",
    "wave_hs_m": "m",
    "wave_tp_s": "s",
    "precip_mmh": "mm/h",
}
_MEMBER_SUFFIX = re.compile(r"_member(\d+)$")

_BEAUFORT_UPPER_MPS = np.asarray(
    [0.2, 1.5, 3.3, 5.4, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6, np.inf],
    dtype=float,
)
_RAIN_RATE_MM_H = np.asarray([0.0, 0.25, 1.0, 2.5, 7.5, 20.0, 50.0], dtype=float)
_DOUGLAS_UPPER_HS_M = np.asarray(
    [0.0, 0.1, 0.5, 1.25, 2.5, 4.0, 6.0, 9.0, 14.0, np.inf], dtype=float
)


class OpenMeteoError(RuntimeError):
    """Open-Meteo 请求失败或返回结构不符合约定。"""


def _new_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        other=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    # Open-Meteo 的 TLS 长连接在部分网络/代理下会返回 SSLEOFError；
    # 强制每次请求重新握手，并允许 urllib3 对 SSL/协议错误重试。
    session.headers.update({
        "User-Agent": "XWOcean-weather-service/1.0",
        "Accept": "application/json",
        "Connection": "close",
    })
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_json(session: Any, url: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        response = session.get(url, params=dict(params), timeout=(15, 60))
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise OpenMeteoError(f"Open-Meteo 请求失败：{exc}") from exc
    except ValueError as exc:
        raise OpenMeteoError("Open-Meteo 返回了无法解析的 JSON。") from exc
    if not isinstance(payload, Mapping):
        raise OpenMeteoError("Open-Meteo 返回值不是 JSON 对象。")
    if payload.get("error"):
        raise OpenMeteoError(
            f"Open-Meteo 拒绝请求：{payload.get('reason') or '未知原因'}"
        )
    return payload


def _get_weather_json(
    session: Any, params: Mapping[str, Any]
) -> tuple[Mapping[str, Any], bool]:
    """优先使用 WeatherNext2 集合接口；该域名不可达时回退到标准预报接口。"""
    try:
        return _get_json(session, ENSEMBLE_URL, params), False
    except OpenMeteoError as ensemble_error:
        fallback = dict(params)
        fallback.pop("models", None)
        try:
            return _get_json(session, STANDARD_URL, fallback), True
        except OpenMeteoError as fallback_error:
            raise OpenMeteoError(
                f"{ensemble_error}；标准预报回退也失败：{fallback_error}"
            ) from fallback_error


def _get_json_many(
    session: Any, url: str, params: Mapping[str, Any], expected: int
) -> list[Mapping[str, Any]]:
    """请求多个坐标；Open-Meteo 对多坐标返回对象数组。"""
    try:
        response = session.get(url, params=dict(params), timeout=(15, 90))
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise OpenMeteoError(f"Open-Meteo 批量请求失败：{exc}") from exc
    except ValueError as exc:
        raise OpenMeteoError("Open-Meteo 批量请求返回了无法解析的 JSON。") from exc
    values = [payload] if isinstance(payload, Mapping) else payload
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise OpenMeteoError("Open-Meteo 批量返回值不是 JSON 对象数组。")
    results = list(values)
    if len(results) != expected or not all(isinstance(item, Mapping) for item in results):
        raise OpenMeteoError(
            f"Open-Meteo 批量返回坐标数不匹配：请求 {expected}，返回 {len(results)}。"
        )
    for item in results:
        if item.get("error"):
            raise OpenMeteoError(
                f"Open-Meteo 拒绝批量请求：{item.get('reason') or '未知原因'}"
            )
    return results


def _get_weather_json_many(
    session: Any, params: Mapping[str, Any], expected: int
) -> tuple[list[Mapping[str, Any]], bool]:
    try:
        return _get_json_many(session, ENSEMBLE_URL, params, expected), False
    except OpenMeteoError as ensemble_error:
        fallback = dict(params)
        fallback.pop("models", None)
        try:
            return _get_json_many(session, STANDARD_URL, fallback, expected), True
        except OpenMeteoError as fallback_error:
            raise OpenMeteoError(
                f"{ensemble_error}；标准预报回退也失败：{fallback_error}"
            ) from fallback_error


class _MemoryResponse:
    def __init__(self, payload: Mapping[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Mapping[str, Any]:
        return self._payload


class _PointPayloadSession:
    """把批量响应中的一个坐标交给现有单点解析逻辑。"""

    def __init__(self, weather: Mapping[str, Any], marine: Mapping[str, Any]):
        self.weather = weather
        self.marine = marine

    def get(self, url: str, **_kwargs: Any) -> _MemoryResponse:
        return _MemoryResponse(self.marine if url == MARINE_URL else self.weather)

    def close(self) -> None:
        return None


def _cache_key(
    longitude: float, latitude: float, forecast_hours: int, start_utc: datetime
) -> tuple[Any, ...]:
    return (
        round(float(latitude) / _CACHE_GRID_DEG),
        round(float(longitude) / _CACHE_GRID_DEG),
        int(forecast_hours),
        start_utc.isoformat(),
    )


def _get_cached(key: tuple[Any, ...]) -> dict[str, Any] | None:
    with _FORECAST_CACHE_LOCK:
        value = _FORECAST_CACHE.get(key)
        if value is not None:
            _FORECAST_CACHE.move_to_end(key)
        return value


def _put_cached(key: tuple[Any, ...], value: dict[str, Any]) -> None:
    with _FORECAST_CACHE_LOCK:
        _FORECAST_CACHE[key] = value
        _FORECAST_CACHE.move_to_end(key)
        while len(_FORECAST_CACHE) > _CACHE_LIMIT:
            _FORECAST_CACHE.popitem(last=False)


def _hourly(payload: Mapping[str, Any], source: str) -> Mapping[str, Any]:
    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping):
        raise OpenMeteoError(f"{source} 响应缺少 hourly 对象。")
    return hourly


def _times(hourly: Mapping[str, Any], source: str) -> list[str]:
    values = hourly.get("time")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise OpenMeteoError(f"{source} 响应缺少 hourly.time 数组。")
    return [str(value) for value in values]


def _member_keys(hourly: Mapping[str, Any], variable: str) -> list[str]:
    keys: list[tuple[int, str]] = []
    for key in hourly:
        text = str(key)
        if text == variable:
            keys.append((0, text))
            continue
        if not text.startswith(f"{variable}_member"):
            continue
        match = _MEMBER_SUFFIX.search(text)
        if match:
            keys.append((int(match.group(1)), text))
    keys.sort(key=lambda item: item[0])
    return [key for _, key in keys]


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _ensemble_mean(
    hourly: Mapping[str, Any], variable: str, indices: Sequence[int]
) -> tuple[list[float | None], int]:
    keys = _member_keys(hourly, variable)
    if not keys:
        raise OpenMeteoError(f"WeatherNext 2 响应缺少 {variable}。")
    series: list[Sequence[Any]] = []
    for key in keys:
        values = hourly.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise OpenMeteoError(f"WeatherNext 2 字段 {key} 不是数组。")
        if indices and len(values) <= max(indices):
            raise OpenMeteoError(f"WeatherNext 2 字段 {key} 不足请求的预报时段。")
        series.append(values)
    result: list[float | None] = []
    for index in indices:
        numbers = [number for values in series if (number := _number(values[index])) is not None]
        result.append(sum(numbers) / len(numbers) if numbers else None)
    return result, len(series)


def _linear_percentile(numbers: Sequence[float], percentile: float) -> float | None:
    values = sorted(float(value) for value in numbers if math.isfinite(float(value)))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * float(percentile)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return values[left]
    fraction = position - left
    return values[left] * (1.0 - fraction) + values[right] * fraction


def _ensemble_percentile(
    hourly: Mapping[str, Any],
    variable: str,
    indices: Sequence[int],
    percentile: float,
) -> tuple[list[float | None], int]:
    keys = _member_keys(hourly, variable)
    if not keys:
        raise OpenMeteoError(f"WeatherNext 2 响应缺少 {variable}。")
    series: list[Sequence[Any]] = []
    for key in keys:
        values = hourly.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise OpenMeteoError(f"WeatherNext 2 字段 {key} 不是数组。")
        if indices and len(values) <= max(indices):
            raise OpenMeteoError(f"WeatherNext 2 字段 {key} 不足请求的预报时段。")
        series.append(values)
    result: list[float | None] = []
    for index in indices:
        numbers = [
            number
            for values in series
            if (number := _number(values[index])) is not None
        ]
        result.append(_linear_percentile(numbers, percentile))
    return result, len(series)


def _ensemble_direction_mean(
    hourly: Mapping[str, Any], variable: str, indices: Sequence[int]
) -> tuple[list[float | None], int]:
    keys = _member_keys(hourly, variable)
    if not keys:
        raise OpenMeteoError(f"WeatherNext 2 响应缺少 {variable}。")
    series: list[Sequence[Any]] = []
    for key in keys:
        values = hourly.get(key)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise OpenMeteoError(f"WeatherNext 2 字段 {key} 不是数组。")
        if indices and len(values) <= max(indices):
            raise OpenMeteoError(f"WeatherNext 2 字段 {key} 不足请求的预报时段。")
        series.append(values)
    result: list[float | None] = []
    for index in indices:
        angles = [
            math.radians(number)
            for values in series
            if (number := _number(values[index])) is not None
        ]
        if not angles:
            result.append(None)
            continue
        sin_mean = sum(math.sin(angle) for angle in angles) / len(angles)
        cos_mean = sum(math.cos(angle) for angle in angles) / len(angles)
        if abs(sin_mean) < 1.0e-12 and abs(cos_mean) < 1.0e-12:
            result.append(None)
        else:
            direction = math.degrees(math.atan2(sin_mean, cos_mean)) % 360.0
            # 浮点舍入可能把正北表示成 360；协议统一使用 [0, 360)。
            result.append(0.0 if math.isclose(direction, 360.0, abs_tol=1.0e-10) else direction)
    return result, len(series)


def _aligned_series(
    hourly: Mapping[str, Any], variable: str, target_times: Sequence[str], source: str
) -> list[float | None]:
    source_times = _times(hourly, source)
    values = hourly.get(variable)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise OpenMeteoError(f"{source} 响应缺少 {variable} 数组。")
    if len(values) != len(source_times):
        raise OpenMeteoError(f"{source} 的 time 与 {variable} 长度不一致。")
    by_time = {time_value: _number(value) for time_value, value in zip(source_times, values)}
    missing = [time_value for time_value in target_times if time_value not in by_time]
    if missing:
        raise OpenMeteoError(f"{source} 与 WeatherNext 2 的逐小时时间轴无法对齐。")
    return [by_time[time_value] for time_value in target_times]


def _dominant_peak_period(
    hourly: Mapping[str, Any], target_times: Sequence[str]
) -> list[float | None]:
    """用占优的风浪/涌浪分量构造总浪峰值周期。

    Open-Meteo Marine 暂未提供可用的总浪 ``wave_peak_period`` 数组，
    DWD GWAM 则提供风浪和涌浪各自的峰值周期。这里按两种分量中较高的
    有效波高选择对应峰值周期；分量峰值缺测时退回总浪平均周期。
    """
    wind_height = _aligned_series(
        hourly, "wind_wave_height", target_times, "Open-Meteo Marine"
    )
    wind_peak = _aligned_series(
        hourly, "wind_wave_peak_period", target_times, "Open-Meteo Marine"
    )
    swell_height = _aligned_series(
        hourly, "swell_wave_height", target_times, "Open-Meteo Marine"
    )
    swell_peak = _aligned_series(
        hourly, "swell_wave_peak_period", target_times, "Open-Meteo Marine"
    )
    mean_period = _aligned_series(
        hourly, "wave_period", target_times, "Open-Meteo Marine"
    )
    result: list[float | None] = []
    for wh, wp, sh, sp, fallback in zip(
        wind_height, wind_peak, swell_height, swell_peak, mean_period
    ):
        choices = [
            (height, peak)
            for height, peak in ((wh, wp), (sh, sp))
            if height is not None and peak is not None
        ]
        result.append(max(choices, key=lambda item: item[0])[1] if choices else fallback)
    return result


def _utc_hour_start(now_utc: datetime | None = None) -> datetime:
    value = now_utc or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _target_times(start_utc: datetime, count: int) -> list[str]:
    return [
        (start_utc + timedelta(hours=index)).strftime("%Y-%m-%dT%H:00")
        for index in range(count)
    ]


def _time_indices(source_times: Sequence[str], target_times: Sequence[str], source: str) -> list[int]:
    by_time = {str(value): index for index, value in enumerate(source_times)}
    missing = [value for value in target_times if value not in by_time]
    if missing:
        raise OpenMeteoError(
            f"{source} 未覆盖UTC当前整点起的完整预报时段；首个缺失时次为 {missing[0]}。"
        )
    return [by_time[value] for value in target_times]


def _beaufort_force(wind_speed_mps: float | None) -> int | None:
    if wind_speed_mps is None or not math.isfinite(float(wind_speed_mps)):
        return None
    return int(np.searchsorted(_BEAUFORT_UPPER_MPS, max(0.0, float(wind_speed_mps)), side="left"))


def _douglas_sea_state(wave_height_m: float | None) -> int | None:
    if wave_height_m is None or not math.isfinite(float(wave_height_m)):
        return None
    height = max(0.0, float(wave_height_m))
    if height == 0.0:
        return 0
    return int(np.searchsorted(_DOUGLAS_UPPER_HS_M, height, side="left"))


def _ambient_broadband_db(
    wind_speed_mps: float | None,
    rain_rate_mmh: float | None,
) -> float | None:
    """Integrate forecastable natural-noise terms from 100 Hz to 10 kHz.

    Shipping is intentionally excluded: it is a real-time AIS-driven term and
    must not be represented by a fixed activity level in weather forecasts.
    """
    if wind_speed_mps is None or not math.isfinite(float(wind_speed_mps)):
        return None
    frequency = np.geomspace(100.0, 10_000.0, 2048)
    frequency_khz = frequency / 1000.0
    levels = [
        17.0 - 30.0 * np.log10(frequency_khz),
        (
            50.0
            + 7.5 * math.sqrt(max(0.0, float(wind_speed_mps)))
            + 20.0 * np.log10(frequency_khz)
            - 40.0 * np.log10(frequency_khz + 0.4)
        ),
        -15.0 + 20.0 * np.log10(frequency_khz),
    ]
    if rain_rate_mmh is not None and math.isfinite(float(rain_rate_mmh)) and rain_rate_mmh > 0.0:
        rain = float(rain_rate_mmh)
        level_at_5khz = 50.0 + 17.0 * math.log10(rain)
        peak_frequency = 13_500.0
        peak_level = level_at_5khz + 15.0 * math.log2(peak_frequency / 5000.0)
        levels.append(
            np.where(
                frequency < peak_frequency,
                level_at_5khz + 15.0 * np.log2(frequency / 5000.0),
                peak_level - 7.0 * np.log2(frequency / peak_frequency),
            )
        )
    total_psd = np.sum([np.power(10.0, level / 10.0) for level in levels], axis=0)
    broadband_power = float(np.trapezoid(total_psd, frequency))
    if not math.isfinite(broadband_power) or broadband_power <= 0.0:
        return None
    return round(10.0 * math.log10(broadband_power), 1)


def fetch_forecast(
    longitude: float,
    latitude: float,
    *,
    forecast_hours: int = FORECAST_HOURS,
    session: Any | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """获取UTC当前整点起指定小时数的集合天气与海浪预报。"""
    if not MIN_FORECAST_HOURS <= int(forecast_hours) <= MAX_FORECAST_HOURS:
        raise ValueError(
            f"forecast_hours 必须位于 {MIN_FORECAST_HOURS} 到 {MAX_FORECAST_HOURS}。"
        )
    forecast_hours = int(forecast_hours)
    fetched_at = now_utc or datetime.now(timezone.utc)
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    fetched_at = fetched_at.astimezone(timezone.utc)
    start_utc = _utc_hour_start(fetched_at)
    target_times = _target_times(start_utc, forecast_hours)

    cache_key = None
    if session is None:
        cache_key = _cache_key(longitude, latitude, forecast_hours, start_utc)
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

    owns_session = session is None
    client = session or _new_session()
    try:
        common = {
            "latitude": latitude,
            "longitude": longitude,
            "forecast_hours": forecast_hours,
            "timezone": "GMT",
        }
        weather, weather_fallback = _get_weather_json(
            client,
            {
                **common,
                "hourly": ",".join(WEATHER_VARIABLES),
                "models": WEATHERNEXT_MODEL,
                "wind_speed_unit": "ms",
                "precipitation_unit": "mm",
                "cell_selection": "sea",
            },
        )
        marine = _get_json(
            client,
            MARINE_URL,
            {
                **common,
                "hourly": ",".join(MARINE_VARIABLES),
                "models": MARINE_MODEL,
                "cell_selection": "sea",
            },
        )
    finally:
        if owns_session:
            client.close()

    weather_hourly = _hourly(weather, "WeatherNext 2")
    marine_hourly = _hourly(marine, "Open-Meteo Marine")
    weather_times = _times(weather_hourly, "WeatherNext 2")
    weather_indices = _time_indices(weather_times, target_times, "WeatherNext 2")

    mslp, mslp_members = _ensemble_mean(weather_hourly, "pressure_msl", weather_indices)
    sst, sst_members = _ensemble_mean(
        weather_hourly, "sea_surface_temperature", weather_indices
    )
    wind, wind_members = _ensemble_mean(
        weather_hourly, "wind_speed_10m", weather_indices
    )
    wind_direction, direction_members = _ensemble_direction_mean(
        weather_hourly, "wind_direction_10m", weather_indices
    )
    precipitation, precipitation_members = _ensemble_mean(
        weather_hourly, "precipitation", weather_indices
    )
    wind_p10, wind_p10_members = _ensemble_percentile(
        weather_hourly, "wind_speed_10m", weather_indices, 0.10
    )
    wind_p90, wind_p90_members = _ensemble_percentile(
        weather_hourly, "wind_speed_10m", weather_indices, 0.90
    )
    precipitation_p90, precipitation_p90_members = _ensemble_percentile(
        weather_hourly, "precipitation", weather_indices, 0.90
    )
    member_count = min(
        mslp_members,
        sst_members,
        wind_members,
        direction_members,
        precipitation_members,
        wind_p10_members,
        wind_p90_members,
        precipitation_p90_members,
    )

    wave_height = _aligned_series(
        marine_hourly, "wave_height", target_times, "Open-Meteo Marine"
    )
    wave_period = _dominant_peak_period(marine_hourly, target_times)
    swell_height = _aligned_series(
        marine_hourly, "swell_wave_height", target_times, "Open-Meteo Marine"
    )
    wave_direction = _aligned_series(
        marine_hourly, "wave_direction", target_times, "Open-Meteo Marine"
    )
    swell_direction = _aligned_series(
        marine_hourly, "swell_wave_direction", target_times, "Open-Meteo Marine"
    )

    grid_latitude = _number(weather.get("latitude"))
    grid_longitude = _number(weather.get("longitude"))
    grid_latitude = float(latitude) if grid_latitude is None else grid_latitude
    grid_longitude = float(longitude) if grid_longitude is None else grid_longitude
    current_wind = wind[0] if wind else None
    current_rain = precipitation[0] if precipitation else None
    current_wave = wave_height[0] if wave_height else None

    result = {
        "ambient_now": {
            "beaufort": _beaufort_force(current_wind),
            "broadband_100hz_10khz_db": _ambient_broadband_db(current_wind, current_rain),
            "model": (
                "Wenz natural-noise forecast: turbulence + wind + rain + thermal; shipping excluded"
            ),
            "sea_state_douglas": _douglas_sea_state(current_wave),
            "time": target_times[0],
        },
        "cached": False,
        "ensemble_members": member_count,
        "fetched_at": fetched_at.isoformat(timespec="seconds"),
        "grid_lat": grid_latitude,
        "grid_lon": grid_longitude,
        "hourly": {
            "gust": [None] * forecast_hours,
            "mslp_hpa": mslp,
            "precip_mmh": precipitation,
            "precip_p90": precipitation_p90,
            "sst_c": sst,
            "swell_dir": swell_direction,
            "swell_hs_m": swell_height,
            "time": target_times,
            "wave_dir": wave_direction,
            "wave_hs_m": wave_height,
            "wave_tp_s": wave_period,
            "wind10": wind,
            "wind10_p10": wind_p10,
            "wind10_p90": wind_p90,
            "wind_dir": wind_direction,
        },
        "hs_derived_from_wind": False,
        "lat": float(latitude),
        "lon": float(longitude),
        "model_run": weather.get("model_run"),
        "now_index": 0,
        "provider": (
            "open-meteo/best-match (weathernext2 fallback)"
            if weather_fallback else "open-meteo/weathernext2-ensemble"
        ),
    }
    if cache_key is not None:
        _put_cached(cache_key, result)
    return result


def fetch_forecasts(
    points: Sequence[tuple[float, float]],
    *,
    forecast_hours: int = FORECAST_HOURS,
    now_utc: datetime | None = None,
) -> list[dict[str, Any]]:
    """批量获取多个原生网格点，并按输入顺序返回单点预报。

    同一批区域天气与区域噪声请求通过缓存和批量锁复用结果，避免同时向
    Open-Meteo 重复提交整片区域。
    """
    if not MIN_FORECAST_HOURS <= int(forecast_hours) <= MAX_FORECAST_HOURS:
        raise ValueError(
            f"forecast_hours 必须位于 {MIN_FORECAST_HOURS} 到 {MAX_FORECAST_HOURS}。"
        )
    forecast_hours = int(forecast_hours)
    fetched_at = now_utc or datetime.now(timezone.utc)
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    fetched_at = fetched_at.astimezone(timezone.utc)
    start_utc = _utc_hour_start(fetched_at)
    normalized = [(float(lon), float(lat)) for lon, lat in points]
    if not normalized:
        return []

    def collect_cached() -> tuple[list[dict[str, Any] | None], list[int]]:
        values: list[dict[str, Any] | None] = []
        missing: list[int] = []
        for index, (longitude, latitude) in enumerate(normalized):
            value = _get_cached(
                _cache_key(longitude, latitude, forecast_hours, start_utc)
            )
            values.append(value)
            if value is None:
                missing.append(index)
        return values, missing

    results, missing_indices = collect_cached()
    if not missing_indices:
        return [value for value in results if value is not None]

    # 防止天气覆盖层与噪声覆盖层首次并发打开时下载两遍相同区域。
    with _BATCH_FETCH_LOCK:
        results, missing_indices = collect_cached()
        if not missing_indices:
            return [value for value in results if value is not None]

        chunks = [
            missing_indices[offset : offset + _BATCH_SIZE]
            for offset in range(0, len(missing_indices), _BATCH_SIZE)
        ]

        def fetch_chunk(indices: list[int]) -> list[tuple[int, dict[str, Any]]]:
            chunk_points = [normalized[index] for index in indices]
            client = _new_session()
            try:
                common = {
                    "latitude": ",".join(str(lat) for _, lat in chunk_points),
                    "longitude": ",".join(str(lon) for lon, _ in chunk_points),
                    "forecast_hours": forecast_hours,
                    "timezone": "GMT",
                }
                weather_items, _weather_fallback = _get_weather_json_many(
                    client,
                    {
                        **common,
                        "hourly": ",".join(REGIONAL_WEATHER_VARIABLES),
                        "models": WEATHERNEXT_MEAN_MODEL,
                        "wind_speed_unit": "ms",
                        "precipitation_unit": "mm",
                        "cell_selection": "sea",
                    },
                    len(indices),
                )
                marine_items = _get_json_many(
                    client,
                    MARINE_URL,
                    {
                        **common,
                        "hourly": ",".join(MARINE_VARIABLES),
                        "models": MARINE_MODEL,
                        "cell_selection": "sea",
                    },
                    len(indices),
                )
            finally:
                client.close()

            chunk_results: list[tuple[int, dict[str, Any]]] = []
            for index, weather, marine in zip(indices, weather_items, marine_items):
                longitude, latitude = normalized[index]
                # 复用单点结果组装逻辑；区域未请求的温度/气压字段显式填空。
                weather_hourly = weather.get("hourly")
                if isinstance(weather_hourly, dict):
                    count = len(weather_hourly.get("time") or [])
                    weather_hourly.setdefault("pressure_msl", [None] * count)
                    weather_hourly.setdefault("sea_surface_temperature", [None] * count)
                value = fetch_forecast(
                    longitude,
                    latitude,
                    forecast_hours=forecast_hours,
                    session=_PointPayloadSession(weather, marine),
                    now_utc=fetched_at,
                )
                _put_cached(
                    _cache_key(longitude, latitude, forecast_hours, start_utc), value
                )
                chunk_results.append((index, value))
            return chunk_results

        with ThreadPoolExecutor(max_workers=min(_BATCH_WORKERS, len(chunks))) as executor:
            futures = [executor.submit(fetch_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                for index, value in future.result():
                    results[index] = value

    if any(value is None for value in results):
        raise OpenMeteoError("Open-Meteo 批量请求存在缺失网格点。")
    return [value for value in results if value is not None]
