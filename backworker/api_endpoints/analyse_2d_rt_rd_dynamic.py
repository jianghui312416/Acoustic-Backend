"""二维动态声线轨迹接口。

请求体与 ``Analyse2DRTRD`` 完全一致；响应在其静态声线结果基础上追加随时间
变化的 ``frames``：每一帧包含每条声线当前可见的水平距离数组 ``x`` 和深度数组 ``z``。
"""

from .common import apply_fixed_acoustic_model, create_portal_router, handle_portal_payload
from ..services.ray_animation import build_dynamic_ray_frames


PLUGIN_NAME = "Analyse2DRTRDDynamic"
ROUTE_PATH = "/Analyse2DRTRDDynamic"


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
        ray_response = core._build_2d_ray_response(
            job_dir, col, line_count, params.get("freq_hz")
        )
        sound_speed = core._read_tl2d_section(job_dir)
        # 动态声线采用「随传播时间从声源逐步伸长」的呈现：trail_ratio=0 使每帧
        # 从声源画到当前头位置，前端据此绘制随时间生长的完整声线。
        return build_dynamic_ray_frames(
            ray_response, sound_speed=sound_speed, trail_ratio=0.0
        ), job_id

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router(openapi_extra):
    return create_portal_router(ROUTE_PATH, handle, openapi_extra)
