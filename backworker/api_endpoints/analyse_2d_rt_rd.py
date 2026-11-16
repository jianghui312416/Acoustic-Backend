"""二维声线轨迹接口。"""

from .common import apply_fixed_acoustic_model, create_portal_router, handle_portal_payload


PLUGIN_NAME = "Analyse2DRTRD"
ROUTE_PATH = "/Analyse2DRTRD"


def handle(payload):
    def build_response(request_payload, core, meta):
        core._reject_unsupported_request_fields(
            request_payload,
            core.PORTAL_2D_RAY_REQUEST_FIELDS,
            meta.get("plugin", PLUGIN_NAME),
        )

        params, _, col, line_count = core._portal_2d_params(
            request_payload, include_line_count=True
        )
        apply_fixed_acoustic_model(params, "bellhop")
        col = core._configured_sampling_size("TWO_D_RAY_TRACK", col)
        job_id, job_dir, _ = core._run_portal_job(params)
        response_payload = core._build_2d_ray_response(
            job_dir, col, line_count, params.get("freq_hz")
        )
        return response_payload, job_id

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router(openapi_extra):
    return create_portal_router(ROUTE_PATH, handle, openapi_extra)
