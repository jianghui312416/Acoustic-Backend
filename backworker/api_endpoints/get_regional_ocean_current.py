"""HYCOM 区域三维海流 u/v 预报接口。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .. import hycom_service
from ..services import regional_ocean_current as service
from .common import create_portal_router, handle_portal_payload
from .registry import PORTAL_PLUGIN_SN_MAP


PLUGIN_NAME = "GetRegionalOceanCurrent"
ROUTE_PATH = "/GetRegionalOceanCurrent"
SN = PORTAL_PLUGIN_SN_MAP[PLUGIN_NAME]
REQUEST_FIELDS = {"bottomLeftX", "bottomLeftY", "topRightX", "topRightY", "forecastTime"}
REQUIRED_FIELDS = REQUEST_FIELDS


def _openapi_extra() -> dict[str, Any]:
    request = {
        "bottomLeftX": 115.4, "bottomLeftY": 15.4,
        "topRightX": 116.6, "topRightY": 16.6,
        "forecastTime": "2026-11-16T06:00:00Z",
    }
    return {
        "requestBody": {"required": True, "content": {"application/json": {
            "schema": {"type": "object"}, "example": request,
        }}},
        "responses": {"200": {"description": "HYCOM原生网格、全部深度层及全深度平均u/v海流"}},
    }


OPENAPI_EXTRA = _openapi_extra()


def _normalise_request(data: Mapping[str, Any], core: Any, plugin_name: str = PLUGIN_NAME):
    core._reject_unsupported_request_fields(data, REQUEST_FIELDS, plugin_name)
    missing = [name for name in sorted(REQUIRED_FIELDS) if data.get(name) in (None, "")]
    if missing:
        raise core.PortalRequestError(f"{plugin_name}.Request 缺少必填字段：{', '.join(missing)}。")
    west = core._api_float(data, ("bottomLeftX",), None, "西边界 bottomLeftX")
    south = core._api_float(data, ("bottomLeftY",), None, "南边界 bottomLeftY")
    east = core._api_float(data, ("topRightX",), None, "东边界 topRightX")
    north = core._api_float(data, ("topRightY",), None, "北边界 topRightY")
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise core.PortalRequestError("经纬度范围必须是有限数。")
    if not -180.0 <= west <= 180.0 or east <= west or east - west > 360.0:
        raise core.PortalRequestError("经度必须满足 -180 ≤ bottomLeftX < topRightX ≤ 180。")
    if not -80.0 <= south < north <= 90.0:
        raise core.PortalRequestError("HYCOM纬度必须满足 -80 ≤ bottomLeftY < topRightY ≤ 90。")
    try:
        forecast_time = service._parse_time(data.get("forecastTime"))
    except ValueError as exc:
        raise core.PortalRequestError(str(exc)) from exc
    return dict(west=west, south=south, east=east, north=north, forecast_time=forecast_time)


def calculate(request_payload: Mapping[str, Any], core: Any, plugin_name: str = PLUGIN_NAME):
    data = _normalise_request(request_payload, core, plugin_name)
    try:
        return service.fetch_region_current(core.CONFIG, **data)
    except (hycom_service.HycomServiceError, ValueError) as exc:
        raise core.PortalRequestError(str(exc)) from exc


def handle(payload: Any):
    def build_response(request_payload, core, meta):
        return calculate(request_payload, core, meta.get("plugin", PLUGIN_NAME)), None
    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router():
    return create_portal_router(ROUTE_PATH, handle, OPENAPI_EXTRA)
