"""三维冰区动态声线接口。"""

from .common import apply_fixed_acoustic_model, create_portal_router, handle_portal_payload
from ..services.ray_animation import build_dynamic_3d_ray_frames


PLUGIN_NAME = "AnalyseIce3DRTRDDynamic"
ROUTE_PATH = "/AnalyseIce3DRTRDDynamic"


def handle(payload):
    def build_response(request_payload, core, meta):
        core._reject_unsupported_request_fields(
            request_payload,
            core.PORTAL_ICE_3D_RAY_REQUEST_FIELDS,
            meta.get("plugin", PLUGIN_NAME),
        )
        ice_data_path = core._portal_ice_data_path(request_payload)

        params, col, line_count, azimuths = core._portal_3d_ray_params(request_payload)
        apply_fixed_acoustic_model(params, "bellhop")
        params["ice_reflection_enabled"] = True
        params["ice_data"] = ice_data_path
        job_id, job_dir, _ = core._run_portal_job(params)
        ray_response = core._build_3d_ray_response(
            job_dir, col, line_count, azimuths, params.get("freq_hz")
        )
        ssp_depth, ssp_speed = core._read_source_ssp(job_dir)
        response = build_dynamic_3d_ray_frames(
            ray_response,
            sound_speed={
                "ssp_depth_m": ssp_depth,
                "ssp_sound_speed_mps": ssp_speed,
            },
            trail_ratio=0.0,
        )
        response["iceThicknessList"] = core._build_3d_ice_thickness_response(
            job_dir, col, len(azimuths), params
        )
        return response, job_id

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router(openapi_extra):
    return create_portal_router(ROUTE_PATH, handle, openapi_extra)
