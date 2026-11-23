"""指定时次和频带的区域环境噪声预报接口。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..services import open_meteo_weather, regional_forecast_noise, regional_weather
from .analyse_forecast_environment_noise import _extract_ais
from .common import create_portal_router, handle_portal_payload
from .registry import PORTAL_PLUGIN_SN_MAP


PLUGIN_NAME = "AnalyseRegionalForecastEnvironmentNoise"
ROUTE_PATH = "/AnalyseRegionalForecastEnvironmentNoise"
SN = PORTAL_PLUGIN_SN_MAP[PLUGIN_NAME]
REQUEST_FIELDS = {
    "bottomLeftX", "bottomLeftY", "topRightX", "topRightY", "forecastTime",
    "ais", "minFreq", "maxFreq", "numPoints",
}
REQUIRED_FIELDS = {
    "bottomLeftX", "bottomLeftY", "topRightX", "topRightY", "forecastTime", "ais",
}


def _openapi_extra() -> dict[str, Any]:
    request = {
        "bottomLeftX": 113.75, "bottomLeftY": 22.10,
        "topRightX": 114.50, "topRightY": 22.70,
        "forecastTime": "2026-10-30T18:00:00+08:00",
        "minFreq": 100.0, "maxFreq": 10000.0, "numPoints": 128,
        "ais": {"source": "AISStream", "vessels": []},
    }
    wrapped = {"Data": {"SN": SN, "Units": [{"Plugin": PLUGIN_NAME, "Request": request}]}}
    return {
        "requestBody": {"required": True, "content": {"application/json": {
            "schema": {"type": "object"},
            "examples": {
                "flat": {"summary": "前端直接请求", "value": request},
                "wrapped": {"summary": "统一门户请求", "value": wrapped},
            },
        }}},
        "responses": {"200": {"description": "指定频带、按0.25°原生分辨率对齐的区域环境噪声网格"}},
    }


OPENAPI_EXTRA = _openapi_extra()


def _normalise_request(
    data: Mapping[str, Any], core: Any, plugin_name: str = PLUGIN_NAME
) -> dict[str, Any]:
    core._reject_unsupported_request_fields(data, REQUEST_FIELDS, plugin_name)
    missing = [
        name for name in sorted(REQUIRED_FIELDS)
        if name not in data or data.get(name) is None
        or (isinstance(data.get(name), str) and not data.get(name).strip())
    ]
    if missing:
        raise core.PortalRequestError(
            f"{plugin_name}.Request 缺少必填字段：{', '.join(missing)}。"
        )

    west = core._api_float(data, ("bottomLeftX",), None, "西边界 bottomLeftX")
    south = core._api_float(data, ("bottomLeftY",), None, "南边界 bottomLeftY")
    east = core._api_float(data, ("topRightX",), None, "东边界 topRightX")
    north = core._api_float(data, ("topRightY",), None, "北边界 topRightY")
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise core.PortalRequestError("经纬度范围必须是有限数。")
    if not -180.0 <= west <= 180.0 or east <= west or east - west > 360.0:
        raise core.PortalRequestError("经度必须满足 -180 ≤ bottomLeftX < topRightX ≤ 180。")
    if not -90.0 <= south < north <= 90.0:
        raise core.PortalRequestError("纬度必须满足 -90 ≤ bottomLeftY < topRightY ≤ 90。")
    try:
        forecast_time = regional_weather.parse_forecast_time(data.get("forecastTime"))
        ais = _extract_ais(data.get("ais"))
    except ValueError as exc:
        raise core.PortalRequestError(str(exc)) from exc

    min_frequency = core._api_float(data, ("minFreq",), 100.0, "最小频率")
    max_frequency = core._api_float(data, ("maxFreq",), 10000.0, "最大频率")
    frequency_points = core._api_int(data, ("numPoints",), 128, "频率采样点数")
    if min_frequency <= 0.0 or max_frequency <= min_frequency:
        raise core.PortalRequestError("频率必须满足 0 < minFreq < maxFreq。")
    if max_frequency > 100000.0:
        raise core.PortalRequestError("maxFreq 不能超过 100000 Hz。")
    if not 64 <= frequency_points <= 512:
        raise core.PortalRequestError("numPoints 必须位于 64 到 512。")
    return {
        "west": west, "south": south, "east": east, "north": north,
        "forecast_time": forecast_time, "ais": ais,
        "min_frequency_hz": min_frequency,
        "max_frequency_hz": max_frequency,
        "frequency_points": frequency_points,
    }


def calculate(
    request_payload: Mapping[str, Any], core: Any, plugin_name: str = PLUGIN_NAME
) -> dict[str, Any]:
    data = _normalise_request(request_payload, core, plugin_name)
    try:
        weather = regional_weather.fetch_region_forecast(
            west=data["west"], south=data["south"],
            east=data["east"], north=data["north"],
            forecast_time=data["forecast_time"],
        )
        return regional_forecast_noise.build_region_noise(
            regional_weather=weather,
            ais=data["ais"],
            min_frequency_hz=data["min_frequency_hz"],
            max_frequency_hz=data["max_frequency_hz"],
            frequency_points=data["frequency_points"],
        )
    except (open_meteo_weather.OpenMeteoError, ValueError) as exc:
        raise core.PortalRequestError(str(exc)) from exc


def handle(payload: Any):
    def build_response(request_payload, core, meta):
        return calculate(request_payload, core, meta.get("plugin", PLUGIN_NAME)), None

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router():
    return create_portal_router(ROUTE_PATH, handle, OPENAPI_EXTRA)
