"""RAM 二维冰区传播损失接口。"""

from .common import apply_fixed_acoustic_model, create_portal_router, handle_portal_payload


PLUGIN_NAME = "AnalyseIce2DTLRDPE"
ROUTE_PATH = "/AnalyseIce2DTLRDPE"


def handle(payload):
    """处理 RAM 二维冰区传播损失门户请求。"""
    def build_response(request_payload, core, meta):
        """校验冰区二维参数、执行 RAM 并构造含冰厚的响应。"""
        core._reject_unsupported_request_fields(
            request_payload,
            core.PORTAL_ICE_2D_TL_REQUEST_FIELDS,
            meta.get("plugin", PLUGIN_NAME),
        )
        ice_data_path = core._portal_ice_data_path(request_payload)

        params, row, col = core._portal_2d_tl_params(request_payload, "ram")
        apply_fixed_acoustic_model(params, "ram")
        params["ice_reflection_enabled"] = True
        params["ice_data"] = ice_data_path
        job_id, job_dir, _ = core._run_portal_job(params)
        response_payload = core._build_ice_2d_transmission_loss_response(
            job_dir, row, col, params.get("freq_hz"), params
        )
        if isinstance(response_payload, dict):
            response_payload["modelNote"] = (
                "当前 RAM 1.5 计算仅输出冰厚辅助数据，尚不包含冰盖边界反射物理。"
            )
        return response_payload, job_id

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router(openapi_extra):
    """创建 RAM 二维冰区传播损失独立路由。"""
    return create_portal_router(ROUTE_PATH, handle, openapi_extra)
