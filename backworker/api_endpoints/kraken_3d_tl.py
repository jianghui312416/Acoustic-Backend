"""Kraken 三维传播损失接口。"""

from .common import apply_fixed_acoustic_model, create_portal_router, handle_portal_payload


PLUGIN_NAME = "AnalyseKraken3DTLRD"
ROUTE_PATH = "/AnalyseKraken3DTLRD"


def handle(payload):
    """处理 Kraken 三维传播损失门户请求。"""
    def build_response(request_payload, core, meta):
        """校验三维参数、执行 Kraken 多方位计算并构造响应。"""
        core._reject_unsupported_request_fields(
            request_payload,
            core.PORTAL_3D_TL_REQUEST_FIELDS,
            meta.get("plugin", PLUGIN_NAME),
        )

        params, row, col, sector = core._portal_3d_params(request_payload, "kraken")
        apply_fixed_acoustic_model(params, "kraken")
        job_id, job_dir, _ = core._run_portal_job(params)
        response_payload = core._build_3d_transmission_loss_response(
            job_dir, row, col, sector, params.get("freq_hz")
        )
        return response_payload, job_id

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router(openapi_extra):
    """创建 Kraken 三维传播损失独立路由。"""
    return create_portal_router(ROUTE_PATH, handle, openapi_extra)
