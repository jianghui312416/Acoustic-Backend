"""固定 AIS 快照与 48 小时天气驱动的环境噪声谱接口。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..services import forecast_environment_noise as noise_service
from . import get_weather_forecast as weather_endpoint
from .common import create_portal_router, handle_portal_payload
from .registry import PORTAL_PLUGIN_SN_MAP


PLUGIN_NAME = "AnalyseForecastEnvironmentNoise"
ROUTE_PATH = "/AnalyseForecastEnvironmentNoise"
SN = PORTAL_PLUGIN_SN_MAP[PLUGIN_NAME]
REQUEST_FIELDS = {
    "coordX",
    "coordY",
    "ais",
    "forecastHours",
    "minFreq",
    "maxFreq",
    "numPoints",
}
REQUIRED_FIELDS = {"coordX", "coordY", "ais"}


def _request_example() -> dict[str, Any]:
    return {
        "coordX": 114.125,
        "coordY": 22.4,
        "forecastHours": 48,
        "minFreq": 10.0,
        "maxFreq": 100000.0,
        "numPoints": 512,
        "ais": {
            "source": "AISStream",
            "collection_started_at_utc": "2026-10-30T00:00:00Z",
            "collection_ended_at_utc": "2026-10-30T00:00:05Z",
            "vessel_count": 1,
            "vessels": [{
                "mmsi": 413000001,
                "longitude": 114.12,
                "latitude": 22.31,
                "sog_kn": 12.4,
                "ship_type": 70,
                "length_m": 180.0,
            }],
        },
    }


def _openapi_extra() -> dict[str, Any]:
    request = _request_example()
    wrapped = {
        "Data": {"SN": SN, "Units": [{"Plugin": PLUGIN_NAME, "Request": request}]}
    }
    return {
        "requestBody": {"required": True, "content": {"application/json": {
            "schema": {"type": "object"},
            "examples": {
                "flat": {"summary": "前端直接请求", "value": request},
                "wrapped": {"summary": "统一门户请求", "value": wrapped},
            },
        }}},
        "responses": {"200": {"description": "48 小时逐小时环境噪声谱帧"}},
    }


OPENAPI_EXTRA = _openapi_extra()


def _extract_ais(value: Any) -> Mapping[str, Any]:
    """接受 GET 原始响应，也兼容旧门户包装响应。"""
    if not isinstance(value, Mapping):
        raise ValueError("ais 必须是前端 GET /GetRealtimeAIS 得到的 JSON 对象。")
    if isinstance(value.get("vessels"), list):
        return value
    data = value.get("Data")
    if isinstance(data, Mapping):
        units = data.get("Units")
        if isinstance(units, list) and units and isinstance(units[0], Mapping):
            response = units[0].get("Response")
            if isinstance(response, Mapping) and isinstance(response.get("vessels"), list):
                return response
    raise ValueError("ais 响应中缺少 vessels 数组。")


def _normalise_request(
    data: Mapping[str, Any], core: Any, plugin_name: str = PLUGIN_NAME
) -> dict[str, Any]:
    core._reject_unsupported_request_fields(data, REQUEST_FIELDS, plugin_name)
    missing = [
        name for name in sorted(REQUIRED_FIELDS)
        if name not in data or data.get(name) is None
    ]
    if missing:
        raise core.PortalRequestError(
            f"{plugin_name}.Request 缺少必填字段：{', '.join(missing)}。"
        )
    longitude = core._api_float(data, ("coordX",), None, "接收点经度 coordX")
    latitude = core._api_float(data, ("coordY",), None, "接收点纬度 coordY")
    if not math.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
        raise core.PortalRequestError("coordX 必须位于 -180 到 180 度。")
    if not math.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise core.PortalRequestError("coordY 必须位于 -90 到 90 度。")
    forecast_hours = core._api_int(
        data, ("forecastHours",), 48, "预报时长 forecastHours"
    )
    if forecast_hours != noise_service.FORECAST_HOURS:
        raise core.PortalRequestError("forecastHours 当前固定为 48。")
    min_frequency = core._api_float(data, ("minFreq",), 10.0, "最小频率")
    max_frequency = core._api_float(data, ("maxFreq",), 100000.0, "最大频率")
    frequency_points = core._api_int(data, ("numPoints",), 512, "频率采样点数")
    if min_frequency <= 0.0 or max_frequency <= min_frequency:
        raise core.PortalRequestError("频率必须满足 0 < minFreq < maxFreq。")
    if max_frequency > 100000.0:
        raise core.PortalRequestError("maxFreq 不能超过 100000 Hz。")
    if not 64 <= frequency_points <= 2000:
        raise core.PortalRequestError("numPoints 必须位于 64 到 2000。")
    try:
        ais = _extract_ais(data.get("ais"))
    except ValueError as exc:
        raise core.PortalRequestError(str(exc)) from exc
    return {
        "longitude": longitude,
        "latitude": latitude,
        "forecast_hours": forecast_hours,
        "min_frequency_hz": min_frequency,
        "max_frequency_hz": max_frequency,
        "frequency_points": frequency_points,
        "ais": ais,
    }


def calculate(
    request_payload: Mapping[str, Any], core: Any, plugin_name: str = PLUGIN_NAME
) -> dict[str, Any]:
    data = _normalise_request(request_payload, core, plugin_name)
    weather = weather_endpoint.calculate({
        "coordX": data["longitude"],
        "coordY": data["latitude"],
        "forecastHours": data["forecast_hours"],
    }, core, weather_endpoint.PLUGIN_NAME)
    try:
        return noise_service.build_forecast_noise(
            longitude=data["longitude"],
            latitude=data["latitude"],
            ais=data["ais"],
            weather=weather,
            min_frequency_hz=data["min_frequency_hz"],
            max_frequency_hz=data["max_frequency_hz"],
            frequency_points=data["frequency_points"],
        )
    except ValueError as exc:
        raise core.PortalRequestError(str(exc)) from exc


def handle(payload: Any):
    def build_response(request_payload, core, meta):
        return calculate(request_payload, core, meta.get("plugin", PLUGIN_NAME)), None

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router():
    return create_portal_router(ROUTE_PATH, handle, OPENAPI_EXTRA)
