"""GEBCO SCUFN 海底地名视口查询接口。"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from ..services import seafloor_features as service
from .common import handle_portal_payload
from .registry import PORTAL_PLUGIN_SN_MAP


PLUGIN_NAME = "GetSeafloorFeatures"
ROUTE_PATH = "/GetSeafloorFeatures"
SN = PORTAL_PLUGIN_SN_MAP[PLUGIN_NAME]
REQUEST_FIELDS = {
    "bottomLeftX", "bottomLeftY", "topRightX", "topRightY",
    "categories", "minExtentM", "limit",
}
REQUIRED_FIELDS = {"bottomLeftX", "bottomLeftY", "topRightX", "topRightY"}
DEFAULT_LIMIT = 500
MAX_LIMIT = 2000


def _request_example() -> dict[str, Any]:
    return {
        "bottomLeftX": 113.75,
        "bottomLeftY": 20.10,
        "topRightX": 116.50,
        "topRightY": 22.70,
        "categories": ["seamount", "trench", "strait"],
        "minExtentM": 0,
        "limit": DEFAULT_LIMIT,
    }


def _openapi_extra() -> dict[str, Any]:
    request = _request_example()
    response = {
        "source": "GEBCO SCUFN",
        "source_file": "features.xlsx",
        "total": 12,
        "count": 12,
        "truncated": False,
        "categories": service.category_definitions(),
        "features": [{
            "id": "scufn-00001",
            "name": "Sultanah",
            "generic_term": "seamount",
            "generic_term_zh": "海山",
            "category": "seamount",
            "category_label": "海山/海丘",
            "category_color": "#ffc65c",
            "longitude": 59.01832,
            "latitude": 15.24649,
            "min_depth_m": 1857,
            "max_depth_m": 4187,
            "total_relief_m": 2330,
            "dimension": "14 km x 28 km",
            "extent_m": 7000,
            "origin": "Named from the first Omani ship \"Sultanah\" ...",
        }],
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
        "responses": {"200": {"description": "当前视口范围内的海底地名标注数据", "content": {
            "application/json": {"example": {**response, "ResultCode": 0}}
        }}},
    }


OPENAPI_EXTRA = _openapi_extra()


def _coerce_categories(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value]
    else:
        items = [str(value).strip()]
    return [item for item in items if item] or None


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
            f"{plugin_name}.Request 缺少必填字段：{', '.join(missing)}"
        )

    west = core._api_float(data, ("bottomLeftX",), None, "西边界 bottomLeftX")
    south = core._api_float(data, ("bottomLeftY",), None, "南边界 bottomLeftY")
    east = core._api_float(data, ("topRightX",), None, "东边界 topRightX")
    north = core._api_float(data, ("topRightY",), None, "北边界 topRightY")
    if not -180.0 <= west <= 180.0 or east <= west or east - west > 360.0:
        raise core.PortalRequestError(
            "经度必须满足 -180 ≤ bottomLeftX ≤ 180、topRightX > bottomLeftX，且跨度不超过 360°"
        )
    if not -90.0 <= south < north <= 90.0:
        raise core.PortalRequestError("纬度必须满足 -90 ≤ bottomLeftY < topRightY ≤ 90")
    if west > 180.0:
        west -= 360.0
        east -= 360.0
    if east > 180.0:
        east -= 360.0

    categories = _coerce_categories(data.get("categories"))
    if categories:
        unknown = sorted(set(categories) - service.category_keys())
        if unknown:
            raise core.PortalRequestError(f"未知的地貌类型：{', '.join(unknown)}")

    min_extent = data.get("minExtentM")
    min_extent_m = None
    if min_extent not in (None, ""):
        min_extent_m = core._api_float(data, ("minExtentM",), None, "最小尺度 minExtentM")
        if min_extent_m < 0:
            raise core.PortalRequestError("minExtentM 不能为负数")

    limit = DEFAULT_LIMIT
    if data.get("limit") not in (None, ""):
        limit = core._api_int(data, ("limit",), DEFAULT_LIMIT, "返回数量 limit")
    if not 1 <= limit <= MAX_LIMIT:
        raise core.PortalRequestError(f"limit 必须位于 1 到 {MAX_LIMIT} 之间")

    return {
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "categories": categories,
        "min_extent_m": min_extent_m,
        "limit": limit,
    }


def _features_path(core: Any) -> str:
    config = core.CONFIG.get("SEAFLOOR_FEATURES") or {}
    configured = config.get("file") if isinstance(config, Mapping) else None
    if configured:
        return core._project_path(configured)
    topography = ((core.CONFIG.get("DATA_SOURCES") or {}).get("topography") or {})
    topo_file = topography.get("file") if isinstance(topography, Mapping) else None
    if topo_file:
        return os.path.join(os.path.dirname(core._project_path(topo_file)), "features.xlsx")
    raise service.SeafloorFeatureError("未配置海底地名数据文件 SEAFLOOR_FEATURES.file")

def _origin_max_chars(core: Any) -> int:
    config = core.CONFIG.get("SEAFLOOR_FEATURES") or {}
    if not isinstance(config, Mapping):
        return 600
    try:
        return max(0, int(config.get("origin_max_chars", 600)))
    except (TypeError, ValueError):
        return 600


def _name_map_path(core: Any) -> str | None:
    config = core.CONFIG.get("SEAFLOOR_FEATURES") or {}
    configured = config.get("name_map_file") if isinstance(config, Mapping) else None
    if configured:
        path = core._project_path(configured)
        if os.path.isfile(path):
            return path
    fallback = os.path.join(
        os.path.dirname(_features_path(core)), "gebco_scufn_zh_mapping.json"
    )
    return fallback if os.path.isfile(fallback) else None


def calculate(
    request_payload: Mapping[str, Any], core: Any, plugin_name: str = PLUGIN_NAME
) -> dict[str, Any]:
    data = _normalise_request(request_payload, core, plugin_name)
    return service.query_features(
        _features_path(core),
        west=data["west"],
        south=data["south"],
        east=data["east"],
        north=data["north"],
        categories=data["categories"],
        min_extent_m=data["min_extent_m"],
        limit=data["limit"],
        origin_max_chars=_origin_max_chars(core),
        name_map_path=_name_map_path(core),
    )


def handle(payload: Any):
    def build_response(request_payload, core, meta):
        return calculate(request_payload, core, meta.get("plugin", PLUGIN_NAME)), None

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router():
    router = APIRouter()

    @router.post(ROUTE_PATH, openapi_extra=OPENAPI_EXTRA)
    async def post_seafloor_features(request: Request):
        from .. import backend_worker as core

        payload = await core._request_payload(request)
        return await asyncio.to_thread(handle, payload)

    @router.get(
        ROUTE_PATH,
        summary="按视口范围获取海底地名标注",
        description=(
            "供前端直接使用 GET。按当前地形视口经纬度范围返回 GEBCO SCUFN "
            "国际海底地名，可按地貌类型、最小尺度过滤并按重要度截断。"
        ),
    )
    async def get_seafloor_features(
        west: float = Query(..., description="视口西边界经度"),
        south: float = Query(..., description="视口南边界纬度"),
        east: float = Query(..., description="视口东边界经度"),
        north: float = Query(..., description="视口北边界纬度"),
        categories: str | None = Query(
            None, description="逗号分隔的地貌类型，如 seamount,trench,strait"
        ),
        min_extent_m: float | None = Query(
            None, alias="minExtentM", description="最小地貌尺度（米）"
        ),
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="最大返回数量"),
    ):
        from .. import backend_worker as core

        request_payload: dict[str, Any] = {
            "bottomLeftX": west,
            "bottomLeftY": south,
            "topRightX": east,
            "topRightY": north,
            "limit": limit,
        }
        if categories:
            request_payload["categories"] = categories
        if min_extent_m is not None:
            request_payload["minExtentM"] = min_extent_m
        try:
            return await asyncio.to_thread(calculate, request_payload, core)
        except core.PortalRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except service.SeafloorFeatureError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return router
