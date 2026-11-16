"""航线环境剖面接口。"""

from .common import create_portal_router, handle_portal_payload


PLUGIN_NAME = "GetEnvDataRoute"
ROUTE_PATH = "/GetEnvDataRoute"


def handle(payload):
    def build_response(request_payload, core, meta):
        core._reject_unsupported_request_fields(
            request_payload,
            core.ENV_ROUTE_REQUEST_FIELDS,
            meta.get("plugin", PLUGIN_NAME),
        )

        normalised = core._env_base_request(request_payload)
        normalised["points"] = request_payload.get("points")
        response_payload = core._build_env_route_response(normalised)
        return response_payload, None

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router(openapi_extra):
    return create_portal_router(ROUTE_PATH, handle, openapi_extra)
