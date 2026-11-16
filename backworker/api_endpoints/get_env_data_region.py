"""区域环境切片接口。"""

from .common import create_portal_router, handle_portal_payload


PLUGIN_NAME = "GetEnvDataRegion"
ROUTE_PATH = "/GetEnvDataRegion"


def handle(payload):
    def build_response(request_payload, core, meta):
        core._reject_unsupported_request_fields(
            request_payload,
            core.ENV_REGION_REQUEST_FIELDS,
            meta.get("plugin", PLUGIN_NAME),
        )

        normalised = core._env_region_request(request_payload)
        response_payload = core._build_env_region_response(normalised)
        return response_payload, None

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router(openapi_extra):
    return create_portal_router(ROUTE_PATH, handle, openapi_extra)
