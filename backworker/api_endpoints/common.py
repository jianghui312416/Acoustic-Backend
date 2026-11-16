"""门户接口共用工具。

这里放门户接口重复使用的协议适配代码：创建 FastAPI 路由、统一捕获错误、
统一包装门户响应。模型计算和任务编排放在 ``backworker.services``。
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)


PortalBusiness = Callable[[dict[str, Any], Any, dict[str, Any]], tuple[dict[str, Any], str | None]]


def apply_fixed_acoustic_model(params: dict[str, Any], model: str) -> None:
    
    fixed_model = str(model).strip().lower()
    if fixed_model not in {"bellhop", "kraken", "ram"}:
        raise ValueError(f"未知的固定声学模型：{model}")
    params["acoustic_model"] = fixed_model


def handle_portal_payload(
    payload: Any,
    plugin_name: str,
    build_response: PortalBusiness,
):
    """按统一门户格式处理一次请求。

    build_response 只负责本接口自己的业务逻辑，并返回：
    - response_payload：放入门户响应 Response 字段的业务结果；
    - job_id：如果该接口会启动计算任务，就返回任务号；否则返回 None。
    """
    from .. import backend_worker as core

    meta = None
    try:
        core._reload_config_if_changed()
        request_payload, meta = core._portal_unit_request(payload, plugin_name)
        response_payload, job_id = build_response(request_payload, core, meta)
        # 每个计算接口的响应都留存在对应任务目录，工作区可直接复用，
        # 不必再依赖所有 NetCDF 中间产品。
        if job_id:
            try:
                job_dir = os.path.join(core.JOBS_DIR, str(job_id))
                if os.path.isdir(job_dir):
                    core._write_json(
                        os.path.join(job_dir, f"response_{plugin_name}.json"),
                        response_payload,
                    )
            except Exception as exc:
                logger.warning("保存接口响应失败 job=%s plugin=%s: %s", job_id, plugin_name, exc)
        wrapped = core._wrap_portal_response(response_payload, meta, plugin_name, job_id)
        return core._api_json_response(wrapped)
    except core.PortalRequestError as exc:
        return core._portal_failure(exc, 400, meta, plugin_name)
    except Exception as exc:
        return core._portal_failure(exc, 500, meta, plugin_name)


def create_portal_router(route_path: str, handle: Callable[[Any], Any], openapi_extra):
    """创建只有一个 POST 接口的路由。

    每个接口文件只需要提供路径、OpenAPI 示例和 handle 函数即可。
    """
    router = APIRouter()

    @router.post(route_path, openapi_extra=openapi_extra)
    async def api_endpoint(request: Request):
        from .. import backend_worker as core

        payload = await core._request_payload(request)
        return await asyncio.to_thread(handle, payload)

    return router
