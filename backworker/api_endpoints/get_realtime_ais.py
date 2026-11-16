"""实时 AIS 船舶快照接口。

保留门户 POST 协议，同时提供适合前端直接调用的 GET 接口。GET 在未指定
区域时订阅全球范围；AISStream 密钥仍只保存在服务端。
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException, Query

from ..services import ais_realtime as service
from .common import create_portal_router, handle_portal_payload
from .registry import PORTAL_PLUGIN_SN_MAP


PLUGIN_NAME = "GetRealtimeAIS"
ROUTE_PATH = "/GetRealtimeAIS"
SN = PORTAL_PLUGIN_SN_MAP[PLUGIN_NAME]
REQUEST_FIELDS = {
    "bottomLeftX",
    "bottomLeftY",
    "topRightX",
    "topRightY",
    "durationSeconds",
    "maxVessels",
}
REQUIRED_FIELDS = {"bottomLeftX", "bottomLeftY", "topRightX", "topRightY"}
DEFAULT_DURATION_SECONDS = 5 * 60
MAX_DURATION_SECONDS = 5 * 60
GLOBAL_BOTTOM_LEFT_LON = -180.0
GLOBAL_BOTTOM_LEFT_LAT = -90.0
GLOBAL_TOP_RIGHT_LON = 180.0
GLOBAL_TOP_RIGHT_LAT = 90.0


def _request_example() -> dict[str, Any]:
    return {
        "bottomLeftX": 113.75,
        "bottomLeftY": 22.10,
        "topRightX": 114.50,
        "topRightY": 22.70,
        "durationSeconds": DEFAULT_DURATION_SECONDS,
    }


def _openapi_extra() -> dict[str, Any]:
    request = _request_example()
    response = {
        "source": "AISStream",
        "snapshot_time_utc": "2026-11-05T06:30:00.000Z",
        "history_window_seconds": 300,
        "history_complete": True,
        "vessel_count": 1,
        "vessels": [{
            "longitude": 114.1234,
            "latitude": 22.4567,
            "sog_kn": 12.4,
            "cog_deg": 86.7,
            "heading_deg": 87,
            "mmsi": 413000001,
            "vessel_name": "EXAMPLE VESSEL",
            "ship_type": 70,
            "ship_type_name": "货船",
            "length_m": 180,
            "width_m": 28,
            "draught_m": 8.5,
            "data_time_utc": "2026-10-29T12:00:00Z",
        }],
    }
    wrapped = {
        "Data": {"SN": SN, "Units": [{"Plugin": PLUGIN_NAME, "Request": request}]}
    }
    return {
        "requestBody": {"required": True, "content": {"application/json": {
            "schema": {"type": "object"},
            "examples": {
                "flat": {"summary": "平铺请求", "value": request},
                "wrapped": {"summary": "统一门户请求", "value": wrapped},
            },
        }}},
        "responses": {"200": {"description": "指定区域实时 AIS 船舶快照", "content": {
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

    bottom_left_lon = core._api_float(
        data, ("bottomLeftX",), None, "区域左下角经度 bottomLeftX"
    )
    bottom_left_lat = core._api_float(
        data, ("bottomLeftY",), None, "区域左下角纬度 bottomLeftY"
    )
    top_right_lon = core._api_float(
        data, ("topRightX",), None, "区域右上角经度 topRightX"
    )
    top_right_lat = core._api_float(
        data, ("topRightY",), None, "区域右上角纬度 topRightY"
    )
    values = (bottom_left_lon, bottom_left_lat, top_right_lon, top_right_lat)
    if not all(math.isfinite(value) for value in values):
        raise core.PortalRequestError("区域经纬度必须是有限数值。")
    if not (-180 <= bottom_left_lon < top_right_lon <= 180):
        raise core.PortalRequestError(
            "区域经度必须满足 -180 <= bottomLeftX < topRightX <= 180。"
        )
    if not (-90 <= bottom_left_lat < top_right_lat <= 90):
        raise core.PortalRequestError(
            "区域纬度必须满足 -90 <= bottomLeftY < topRightY <= 90。"
        )

    duration_seconds = core._api_int(
        data,
        ("durationSeconds",),
        DEFAULT_DURATION_SECONDS,
        "实时采集时长 durationSeconds",
    )
    if not 1 <= duration_seconds <= MAX_DURATION_SECONDS:
        raise core.PortalRequestError(
            f"durationSeconds 必须位于 1 到 {MAX_DURATION_SECONDS} 秒。"
        )
    return {
        "bottom_left_lon": bottom_left_lon,
        "bottom_left_lat": bottom_left_lat,
        "top_right_lon": top_right_lon,
        "top_right_lat": top_right_lat,
        "duration_seconds": duration_seconds,
        # 兼容旧客户端传入 maxVessels，但服务端不再限制返回船舶数量。
        "max_vessels": None,
    }


def calculate(
    request_payload: Mapping[str, Any], core: Any, plugin_name: str = PLUGIN_NAME
) -> dict[str, Any]:
    data = _normalise_request(request_payload, core, plugin_name)
    try:
        return service.fetch_vessels(**data)
    except (service.AISStreamError, ValueError) as exc:
        raise core.PortalRequestError(str(exc)) from exc


def handle(payload: Any):
    def build_response(request_payload, core, meta):
        return calculate(request_payload, core, meta.get("plugin", PLUGIN_NAME)), None

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router():
    router = create_portal_router(ROUTE_PATH, handle, OPENAPI_EXTRA)

    @router.get(
        ROUTE_PATH,
        summary="获取实时 AIS 船舶快照",
        description=(
            "供前端直接 GET。默认返回当前时刻向前五分钟内出现过的全部唯一船舶，"
            "按 MMSI 保留最新位置；区域参数均可省略，全部省略时查询全球范围。"
            "AISStream API Key 由服务端保管，不会返回给浏览器。"
        ),
    )
    async def get_realtime_ais(
        bottom_left_x: float = Query(
            GLOBAL_BOTTOM_LEFT_LON, alias="bottomLeftX", description="左下角经度"
        ),
        bottom_left_y: float = Query(
            GLOBAL_BOTTOM_LEFT_LAT, alias="bottomLeftY", description="左下角纬度"
        ),
        top_right_x: float = Query(
            GLOBAL_TOP_RIGHT_LON, alias="topRightX", description="右上角经度"
        ),
        top_right_y: float = Query(
            GLOBAL_TOP_RIGHT_LAT, alias="topRightY", description="右上角纬度"
        ),
        duration_seconds: int = Query(
            DEFAULT_DURATION_SECONDS,
            alias="durationSeconds",
            description="向当前时刻回看的历史窗口秒数，默认和最大均为 300 秒",
        ),
        max_vessels: int | None = Query(
            None,
            alias="maxVessels",
            description="兼容旧客户端的废弃参数；服务端不再限制船舶数量",
            deprecated=True,
        ),
    ):
        from .. import backend_worker as core

        request_payload = {
            "bottomLeftX": bottom_left_x,
            "bottomLeftY": bottom_left_y,
            "topRightX": top_right_x,
            "topRightY": top_right_y,
            "durationSeconds": duration_seconds,
        }
        if max_vessels is not None:
            request_payload["maxVessels"] = max_vessels
        try:
            return await asyncio.to_thread(calculate, request_payload, core)
        except core.PortalRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return router
