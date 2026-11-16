"""门户海洋天气预报接口。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..services import open_meteo_weather as service
from .common import create_portal_router, handle_portal_payload
from .registry import PORTAL_PLUGIN_SN_MAP


PLUGIN_NAME = "GetWeatherForecast"
ROUTE_PATH = "/GetWeatherForecast"
SN = PORTAL_PLUGIN_SN_MAP[PLUGIN_NAME]
REQUEST_FIELDS = {"coordX", "coordY", "forecastHours"}
REQUIRED_FIELDS = {"coordX", "coordY", "forecastHours"}


def _openapi_extra() -> dict[str, Any]:
    request = {
        "coordX": 116.0,
        "coordY": 16.0,
        "forecastHours": service.FORECAST_HOURS,
    }
    response = {
        "ambient_now": {
            "beaufort": 3,
            "broadband_100hz_10khz_db": 94.2,
            "model": "Wenz natural-noise forecast: turbulence + wind + rain + thermal; shipping excluded",
            "sea_state_douglas": 3,
            "time": "2026-10-30T09:00",
        },
        "cached": False,
        "ensemble_members": 63,
        "fetched_at": "2026-10-30T09:14:37+00:00",
        "grid_lat": 16.0,
        "grid_lon": 116.0,
        "hourly": {
            "gust": [None, "..."],
            "mslp_hpa": [1010.2, "..."],
            "precip_mmh": [0.0, "..."],
            "precip_p90": [0.2, "..."],
            "sst_c": [28.1, "..."],
            "swell_dir": [145.0, "..."],
            "swell_hs_m": [0.8, "..."],
            "time": ["2026-10-30T09:00", "..."],
            "wave_dir": [170.0, "..."],
            "wave_hs_m": [1.2, "..."],
            "wave_tp_s": [8.5, "..."],
            "wind10": [6.4, "..."],
            "wind10_p10": [4.8, "..."],
            "wind10_p90": [8.1, "..."],
            "wind_dir": [84.0, "..."],
        },
        "hs_derived_from_wind": False,
        "lat": 16.0,
        "lon": 116.0,
        "model_run": None,
        "now_index": 0,
        "provider": "open-meteo/weathernext2-ensemble",
    }
    wrapped = {"Data": {"SN": SN, "Units": [{"Plugin": PLUGIN_NAME, "Request": request}]}}
    return {
        "requestBody": {"required": True, "content": {"application/json": {
            "schema": {"type": "object"},
            "examples": {
                "flat": {"summary": "平铺请求", "value": request},
                "wrapped": {"summary": "统一门户请求", "value": wrapped},
            },
        }}},
        "responses": {"200": {"description": "UTC当前整点起指定小时数的逐小时海洋天气", "content": {
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
    longitude = core._api_float(data, ("coordX",), None, "经度 coordX")
    latitude = core._api_float(data, ("coordY",), None, "纬度 coordY")
    if not math.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
        raise core.PortalRequestError("coordX 必须位于 -180 到 180 度。")
    if not math.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise core.PortalRequestError("coordY 必须位于 -90 到 90 度。")
    forecast_hours = core._api_int(
        data, ("forecastHours",), None, "预报时长 forecastHours"
    )
    if not service.MIN_FORECAST_HOURS <= forecast_hours <= service.MAX_FORECAST_HOURS:
        raise core.PortalRequestError(
            f"forecastHours 必须位于 {service.MIN_FORECAST_HOURS} 到 "
            f"{service.MAX_FORECAST_HOURS}。"
        )
    return {
        "coordX": longitude,
        "coordY": latitude,
        "forecastHours": forecast_hours,
    }


def calculate(
    request_payload: Mapping[str, Any], core: Any, plugin_name: str = PLUGIN_NAME
) -> dict[str, Any]:
    data = _normalise_request(request_payload, core, plugin_name)
    try:
        return service.fetch_forecast(
            data["coordX"],
            data["coordY"],
            forecast_hours=data["forecastHours"],
        )
    except (service.OpenMeteoError, ValueError) as exc:
        raise core.PortalRequestError(str(exc)) from exc


def handle(payload: Any):
    def build_response(request_payload, core, meta):
        return calculate(request_payload, core, meta.get("plugin", PLUGIN_NAME)), None

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router():
    return create_portal_router(ROUTE_PATH, handle, OPENAPI_EXTRA)
