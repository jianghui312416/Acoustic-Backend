"""指定时次的区域风、雨与海况预报接口。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..services import open_meteo_weather, regional_weather as service
from .common import create_portal_router, handle_portal_payload
from .registry import PORTAL_PLUGIN_SN_MAP


PLUGIN_NAME = "GetRegionalWeatherForecast"
ROUTE_PATH = "/GetRegionalWeatherForecast"
SN = PORTAL_PLUGIN_SN_MAP[PLUGIN_NAME]
REQUEST_FIELDS = {
    "bottomLeftX", "bottomLeftY", "topRightX", "topRightY", "forecastTime"
}
REQUIRED_FIELDS = REQUEST_FIELDS


def _request_example() -> dict[str, Any]:
    return {
        "bottomLeftX": 113.75,
        "bottomLeftY": 22.10,
        "topRightX": 114.50,
        "topRightY": 22.70,
        "forecastTime": "2026-10-30T18:00:00+08:00",
    }


def _openapi_extra() -> dict[str, Any]:
    request = _request_example()
    wrapped = {"Data": {"SN": SN, "Units": [{"Plugin": PLUGIN_NAME, "Request": request}]}}
    response = {
        "forecast_time": "2026-10-30T10:00:00Z",
        "forecast_hours": 1,
        "grid": {
            "rows": 4,
            "columns": 4,
            "row_order": "north_to_south",
            "native_resolution_degrees": 0.25,
            "latitudes": [22.75, 22.5, 22.25, 22.0],
            "longitudes": [113.75, 114.0, 114.25, 114.5],
        },
        "data": {
            "wind_speed_mps": [[5.1, 5.2, 5.3, 5.4], [5.0, 5.1, 5.2, 5.3], [4.9, 5.0, 5.1, 5.2], [4.8, 4.9, 5.0, 5.1]],
            "rain_rate_mmh": [[0.0, 0.1, 0.1, 0.2], [0.0, 0.1, 0.1, 0.1], [0.0, 0.0, 0.1, 0.1], [0.0, 0.0, 0.0, 0.1]],
            "wave_height_m": [[1.1, 1.2, 1.2, 1.3], [1.0, 1.1, 1.1, 1.2], [0.9, 1.0, 1.0, 1.1], [0.8, 0.9, 0.9, 1.0]],
        },
    }
    return {
        "requestBody": {"required": True, "content": {"application/json": {
            "schema": {"type": "object"},
            "examples": {
                "flat": {"summary": "平铺请求", "value": request},
                "wrapped": {"summary": "统一门户请求", "value": wrapped},
            },
        }}},
        "responses": {"200": {"description": "指定整点、按0.25°原生分辨率对齐的区域天气与海况网格", "content": {
            "application/json": {"example": {**response, "ResultCode": 0}}
        }}},
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
        raise core.PortalRequestError(
            "经度必须满足 -180 ≤ bottomLeftX ≤ 180、topRightX > bottomLeftX，且跨度不超过 360°"
        )
    if not -90.0 <= south < north <= 90.0:
        raise core.PortalRequestError(
            "纬度必须满足 -90 ≤ bottomLeftY < topRightY ≤ 90。"
        )
    try:
        forecast_time = service.parse_forecast_time(data.get("forecastTime"))
    except ValueError as exc:
        raise core.PortalRequestError(str(exc)) from exc
    return {
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "forecast_time": forecast_time,
    }


def calculate(
    request_payload: Mapping[str, Any], core: Any, plugin_name: str = PLUGIN_NAME
) -> dict[str, Any]:
    data = _normalise_request(request_payload, core, plugin_name)
    try:
        return service.fetch_region_forecast(**data)
    except (open_meteo_weather.OpenMeteoError, ValueError) as exc:
        raise core.PortalRequestError(str(exc)) from exc


def handle(payload: Any):
    def build_response(request_payload, core, meta):
        return calculate(request_payload, core, meta.get("plugin", PLUGIN_NAME)), None

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router():
    return create_portal_router(ROUTE_PATH, handle, OPENAPI_EXTRA)

