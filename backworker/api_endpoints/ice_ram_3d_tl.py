"""RAM 三维冰区传播损失接口。"""

from .common import apply_fixed_acoustic_model, create_portal_router, handle_portal_payload


PLUGIN_NAME = "AnalyseIceRam3DTLRD"
ROUTE_PATH = "/AnalyseIceRam3DTLRD"


def handle(payload):
    """处理 RAM 三维冰区传播损失门户请求。"""
    def build_response(request_payload, core, meta):
        """校验冰区三维参数、执行 RAM 多方位计算并构造响应。"""
        core._reject_unsupported_request_fields(
            request_payload,
            core.PORTAL_ICE_3D_TL_REQUEST_FIELDS,
            meta.get("plugin", PLUGIN_NAME),
        )
        ice_data_path = core._portal_ice_data_path(request_payload)

        params, row, col, sector = core._portal_3d_params(request_payload, "ram")
        apply_fixed_acoustic_model(params, "ram")
        params["ice_reflection_enabled"] = True
        params["ice_data"] = ice_data_path
        job_id, job_dir, _ = core._run_portal_job(params)
        response_payload = core._build_ice_3d_transmission_loss_response(
            job_dir, row, col, sector, params.get("freq_hz"), params
        )
        if isinstance(response_payload, dict):
            response_payload["modelNote"] = (
                "当前 RAM 1.5 计算仅输出冰厚辅助数据，尚不包含冰盖边界反射物理。"
            )
        return response_payload, job_id

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router(openapi_extra):
    """创建 RAM 三维冰区传播损失独立路由。"""
    return create_portal_router(ROUTE_PATH, handle, openapi_extra)
