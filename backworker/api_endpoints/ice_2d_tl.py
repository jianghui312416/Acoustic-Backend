"""二维冰下传播损失接口。"""

from .common import apply_fixed_acoustic_model, create_portal_router, handle_portal_payload


PLUGIN_NAME = "AnalyseIce2DTLRD"
ROUTE_PATH = "/AnalyseIce2DTLRD"


def handle(payload):
    def build_response(request_payload, core, meta):
        core._reject_unsupported_request_fields(
            request_payload,
            core.PORTAL_ICE_2D_TL_REQUEST_FIELDS,
            meta.get("plugin", PLUGIN_NAME),
        )
        ice_data_path = core._portal_ice_data_path(request_payload)

        params, row, col = core._portal_2d_tl_params(request_payload)
        apply_fixed_acoustic_model(params, "bellhop")
        params["ice_reflection_enabled"] = True
        params["ice_data"] = ice_data_path
        job_id, job_dir, _ = core._run_portal_job(params)
        response_payload = core._build_ice_2d_transmission_loss_response(
            job_dir, row, col, params.get("freq_hz"), params
        )
        return response_payload, job_id

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router(openapi_extra):
    return create_portal_router(ROUTE_PATH, handle, openapi_extra)
