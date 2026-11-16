"""查询数据集列表接口。"""

from .common import create_portal_router, handle_portal_payload


PLUGIN_NAME = "GetEnvDataSourceList"
ROUTE_PATH = "/GetEnvDataSourceList"


def handle(payload):
    def build_response(request_payload, core, meta):
        core._reject_unsupported_request_fields(request_payload, set(), meta.get("plugin", PLUGIN_NAME))
        return core._query_dataset_list(), None

    return handle_portal_payload(payload, PLUGIN_NAME, build_response)


def create_router(openapi_extra):
    return create_portal_router(ROUTE_PATH, handle, openapi_extra)
