"""
后端服务 — 接收参数、执行计算、提供文件下载
启动: uvicorn backworker.backend_worker:app --host 0.0.0.0 --port 8888
"""
import os, json, uuid, time, threading, io, zipfile, math, re, queue, asyncio, shutil
from decimal import Decimal
from contextlib import asynccontextmanager
from typing import Any, Dict, Mapping, Sequence
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.gzip import GZipMiddleware
import netCDF4
import numpy as np


# =============================================================================
# 应用启动和全局配置
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    _reconcile_orphaned_jobs()
    yield


app = FastAPI(title="Bellhop Web Backend", version="backend-fastapi-v1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Bellhop-Explicit-Submit"],
)
# 海流接口包含全部深度层，JSON 体积明显大于二维天气网格；传输前压缩，
# 避免上游下载完成后又因浏览器接收大响应而超时。
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)

BASE = os.path.dirname(os.path.abspath(__file__)) # 当前文件的绝对路径
PROJECT = os.path.dirname(BASE) # 项目根目录
CONFIG_FILE = os.path.join(PROJECT, "config")


@app.middleware("http")
async def _add_private_network_cors_header(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

import sys
sys.path.insert(0, BASE)
sys.path.insert(0, PROJECT)
import ice_reflection_trc
from backworker import config_loader
from backworker import acoustic_environment
from backworker import acoustic_model_dispatcher
from backworker.services import bellhop3d_ray
from backworker.array_fusion_service import ArrayFusionSession, serve_array_fusion_websocket
from backworker.signal_transmission_service import serve_signal_transmission_websocket
from backworker.static_communication_link import (
    FrameConfig as StaticLinkFrameConfig,
    OfdmConfig as StaticLinkOfdmConfig,
    SimulationConfig as StaticLinkSimulationConfig,
    TurboConfig as StaticLinkTurboConfig,
    build_ber_diagnostics,
    build_current_link_metrics,
    channel_config_from_array_fusion,
    run_study as run_static_communication_study,
    scenarios_with_selection,
)
from backworker.bellhop_model import results as bellhop_results
from backworker.bellhop_model import runner as bellhop_runner
from backworker import catalog_queries, database_source, hycom_service
from backworker.api_endpoints.registry import PORTAL_PLUGIN_SN_MAP
_CONFIG_RELOAD_LOCK = threading.Lock()


def _read_web_config():
    """读取并合并四份分类配置；保留函数名以兼容现有内部调用。"""
    return config_loader.load_backend_config()


CONFIG = _read_web_config()
_CONFIG_MTIME_NS = config_loader.config_mtime_snapshot()


def _reload_config_if_changed():
    """请求开始前按修改时间热加载四份分类配置。"""
    global CONFIG, _CONFIG_MTIME_NS
    current_mtime_ns = config_loader.config_mtime_snapshot()
    if current_mtime_ns == _CONFIG_MTIME_NS:
        return False
    with _CONFIG_RELOAD_LOCK:
        current_mtime_ns = config_loader.config_mtime_snapshot()
        if current_mtime_ns == _CONFIG_MTIME_NS:
            return False
        updated = _read_web_config()
        CONFIG = updated
        _CONFIG_MTIME_NS = current_mtime_ns
        return True


def _project_path(value):
    path = os.path.expanduser(str(value))
    if os.path.isabs(path):
        return path
    primary = os.path.abspath(os.path.join(PROJECT, path))
    if os.path.exists(primary):
        return primary
    fallback = os.path.abspath(os.path.join(BASE, path))
    return fallback if os.path.exists(fallback) else primary


JOBS_DIR = _project_path(CONFIG["JOBS_DIR"])
os.makedirs(JOBS_DIR, exist_ok=True)
_JSON_LOCKS = {}
_JSON_LOCKS_GUARD = threading.Lock()
_ICE_FIELDS_CACHE = {}

# 每个浏览器 WebSocket 连接都有自己的小队列。后台计算线程把进度消息放进队列，
# 后端 WebSocket 处理函数再把这些消息发给页面。
_WS_SUBSCRIBERS = {}
_WS_SUBSCRIBERS_LOCK = threading.Lock()
_WS_QUEUE_SIZE = 64


REGION_FIELDS = (
    "region_lat_min", "region_lat_max", "region_lon_min", "region_lon_max",
    "source_spacing_deg",
)
SINGLE_FIELDS = ("single_lat", "single_lon")
COMPUTE_BASE_FIELDS = {
    "center_lat", "center_lon", "freq_hz", "source_depth_m",
    "receiver_depth_max_m", "receiver_depth_auto_max", "receiver_depth_auto_scope",
    "radius_km", "n_grid", "section_count",
    "selected_section_theta_deg", "selected_section_depth_m", "horizontal_tl_depth_m",
    "alpha_min", "alpha_max", "beta_min", "beta_max",
    "tl_caxis_min_db", "tl_caxis_max_db", "source_mode_selection",
    "display_3d_rays", "sound_field_view", "display_terrain", "download_nc",
    "client_post_started_ms", "submit_nonce",
    "calculation_mode", "data_source", "environment_time", "acoustic_model",
    "receiver_lat", "receiver_lon", "receiver_depth_m",
    "propagation_distance_nm", "bearing_deg",
}
NM_TO_KM = 1.852
DEFAULT_RESPONSE_ROW = 100
DEFAULT_RESPONSE_COL = 300
PORTAL_DECIMAL_PLACES = 6
# 门户协议要求 3D 计算结果必须是 4, 8, 12, 24 个扇区。
VALID_3D_SECTORS = {4, 8, 12, 24}
# 门户协议使用固定数值表示陆地或普通环境数据缺测。不能返回 null/NaN，
# 否则调用方按 size 读取矩阵时容易丢掉采样点，造成不同方位长度不一致。
PORTAL_INVALID_DATA_VALUE = 9999.0
# 没有有效传播损失结果或传播路径被陆地阻断时，按门户约定输出 200 dB。
PORTAL_INVALID_TL_DB = 200.0
PORTAL_COMMON_REQUEST_FIELDS = {
    "coordX", "coordY", "dataSource", "dataType", "time", "timeRes",
    "rdataSource", "rtime", "srcDepth", "range", "maxFreq", "minFreq",
    "grazingAngle", "beamWidth", "isWaterChanged", "receiverDepthMax",
}
PORTAL_2D_TL_REQUEST_FIELDS = PORTAL_COMMON_REQUEST_FIELDS.union({"azimuth", "rtimeRes", "seaState", "coherent"})
PORTAL_2D_RAY_REQUEST_FIELDS = PORTAL_COMMON_REQUEST_FIELDS.union({"azimuth", "lineCount", "rtimeRes"})
PORTAL_3D_TL_REQUEST_FIELDS = PORTAL_COMMON_REQUEST_FIELDS.union({"row", "col", "sector", "seaState", "coherent"})
PORTAL_3D_RAY_REQUEST_FIELDS = PORTAL_3D_TL_REQUEST_FIELDS.union(
    {"lineCount", "rtimeRes"}
)
PORTAL_NATIVE_3D_RAY_REQUEST_FIELDS = {
    "coordX", "coordY", "dataSource", "time", "srcDepth",
    "minFreq", "maxFreq", "range", "sector", "lineCount",
}
PORTAL_ICE_COMMON_REQUEST_FIELDS = {
    "coordX", "coordY", "dataSource", "dataType", "time", "timeRes",
    "rdataSource", "rtime", "rtimeRes", "srcDepth", "range", "maxFreq", "minFreq",
    "isWaterChanged", "receiverDepthMax",
}
PORTAL_ICE_2D_TL_REQUEST_FIELDS = PORTAL_ICE_COMMON_REQUEST_FIELDS.union({"azimuth", "seaState", "coherent"})
PORTAL_ICE_2D_RAY_REQUEST_FIELDS = PORTAL_ICE_COMMON_REQUEST_FIELDS.union({"azimuth", "lineCount", "rtimeRes"})
PORTAL_ICE_3D_TL_REQUEST_FIELDS = PORTAL_ICE_COMMON_REQUEST_FIELDS.union({"row", "col", "sector", "seaState", "coherent"})
PORTAL_ICE_3D_RAY_REQUEST_FIELDS = PORTAL_ICE_3D_TL_REQUEST_FIELDS.union({"lineCount"})
ENV_STATION_REQUEST_FIELDS = {
    "coordX", "coordY", "dataSource", "dataType", "time", "timeRes",
    "isReconstruct", "rdataSource", "rtime", "rtimeRes",
}
ENV_ROUTE_REQUEST_FIELDS = {"dataSource", "dataType", "time", "timeRes", "points"}
ENV_REGION_REQUEST_FIELDS = {
    "bottomLeftX", "bottomLeftY", "topRightX", "topRightY",
    "dataSource", "dataType", "time", "timeRes", "z",
}

ENV_BOTTOM_GEOACOUSTIC_FIELDS = {
    "DT_ENV_GEOACOUSTIC_CLASSID": (
        "lithology_class_id",
        "lithologyClassId",
        "class_id",
    ),
    "DT_ENV_GEOACOUSTIC_CP": (
        "cp_m_s_prior",
        "compressional_speed_mps",
        "compressional_speed",
        "cp",
    ),
    "DT_ENV_GEOACOUSTIC_ALPHA_P": (
        "alpha_p_db_per_lambda_prior",
        "compressional_attenuation",
        "alpha_p",
    ),
    "DT_ENV_GEOACOUSTIC_DENSITY": (
        "bulk_density_kg_m3",
        "density_kg_m3",
        "bulk_density",
        "density",
    ),
    "DT_ENV_GEOACOUSTIC_POROSITY": (
        "surface_porosity_fraction",
        "porosity_fraction",
        "porosity",
    ),
    "DT_ENV_GEOACOUSTIC_THICKNESS": (
        "total_sediment_thickness_m",
        "sediment_thickness_m",
        "thickness",
    ),
}

_PORTAL_COMMON_EXAMPLE = {
    "coordX": 116,
    "coordY": 16,
    "dataSource": "EnvReanalysis",
    "dataType": "DT_ENV_SOUND",
    "time": "2025-01-04 12:00:00",
    "timeRes": 0,
    "rdataSource": "EnvReanalysis",
    "rtime": "2025-01-04 12:00:00",
    "srcDepth": 150,
    "range": 10,
    "maxFreq": 100,
    "minFreq": 100,
    "grazingAngle": 0,
    "beamWidth": 180,
    "seaState": 0,
    "coherent": False,
    "isWaterChanged": True,
}


# =============================================================================
# 通用小工具和表单整理
# =============================================================================
def _text(value, default=""):
    """把配置或请求里的值整理成字符串；空值使用默认值。"""
    if value is None or value == "":
        value = default
    return str(value).strip()


def _lower_text(value, default=""):
    # 转小写
    return _text(value, default).lower()


def _is_truthy(value):
    return _lower_text(value) in ("1", "true", "yes", "on", "是")


def _compute_allowed_fields():
    fields = set(COMPUTE_BASE_FIELDS)
    fields.update(SINGLE_FIELDS)
    fields.update(REGION_FIELDS)
    return fields


def _params_from_form(form):
    params = {}
    for key in _compute_allowed_fields():
        if key in form:
            params[key] = form.get(key)
    for key, value in dict(CONFIG["TASK_DEFAULTS"]).items():
        params.setdefault(key, value)
    return params


def _apply_selection_points(params, selection_points):
    primary_point = selection_points[len(selection_points) // 2]
    params["center_lat"] = primary_point["lat"]
    params["center_lon"] = primary_point["lon"]
    params["selection_points"] = selection_points
    params["selection_point_count"] = len(selection_points)
    if len(selection_points) > 1:
        params["source_mode"] = "independent_multi_source"
    else:
        params["source_mode"] = "single_source"


# =============================================================================
# 接口文档示例构造
# =============================================================================
def _portal_request_example(plugin_name, request_fields, base_fields=None):
    request_payload = dict(_PORTAL_COMMON_EXAMPLE if base_fields is None else base_fields)
    request_payload.update(request_fields)
    return {
        "Data": {
            "SN": PORTAL_PLUGIN_SN_MAP[plugin_name],
            "Units": [{
                "Plugin": plugin_name,
                "Request": request_payload,
            }],
        },
    }


def _portal_response_example(plugin_name, response_payload):
    return {
        "Technical Support": "XWOcean",
        "Data": {
            "SN": PORTAL_PLUGIN_SN_MAP[plugin_name],
            "TaskId": "abcdef123456",
            "SelfOrder": 0,
            "Units": [{
                "Plugin": plugin_name,
                "Response": response_payload,
                "ResultCode": 0,
            }],
        },
    }


def _flat_portal_response(response_payload, result_code=0):
    """给兼容的平铺响应补充统一结果码。"""
    if isinstance(response_payload, Mapping):
        flat_response = dict(response_payload)
        flat_response["ResultCode"] = int(result_code)
        return flat_response
    # 列表等非对象业务结果无法直接添加字段，保留在 Response 中返回。
    return {
        "Response": response_payload,
        "ResultCode": int(result_code),
    }


def _portal_openapi_extra(plugin_name, request_fields, response_payload, base_fields=None):
    wrapped_request = _portal_request_example(plugin_name, request_fields, base_fields)
    flat_request = wrapped_request["Data"]["Units"][0]["Request"]
    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"type": "object"},
                    "examples": {
                        "wrapped": {
                            "summary": "统一门户包装格式",
                            "value": wrapped_request,
                        },
                        "flat": {
                            "summary": "兼容平铺 JSON 格式",
                            "value": flat_request,
                        },
                    },
                },
            },
        },
        "responses": {
            "200": {
                "description": "计算成功",
                "content": {
                    "application/json": {
                        "schema": {"type": "object"},
                        "examples": {
                            "wrapped": {
                                "summary": "统一门户包装格式",
                                "value": _portal_response_example(plugin_name, response_payload),
                            },
                            "flat": {
                                "summary": "兼容平铺 JSON 格式",
                                "value": _flat_portal_response(response_payload),
                            },
                        },
                    },
                },
            },
            "400": {
                "description": "请求字段或业务参数校验失败",
                "content": {
                    "application/json": {
                        "example": {"ok": False, "message": "参数错误说明", "ResultCode": 1},
                    },
                },
            },
            "500": {
                "description": "计算失败或服务端异常",
                "content": {
                    "application/json": {
                        "example": {"ok": False, "message": "计算失败说明", "ResultCode": 1},
                    },
                },
            },
        },
    }


# =============================================================================
# 接口文档示例定义
# =============================================================================
_OPENAPI_2D_TL = _portal_openapi_extra(
    "Analyse2DTLRD",
    {"azimuth": 90, "lineCount": 21, "rtimeRes": 0},
    {
        "calcFrequency": 100,
        "altList": {
            "size": 300,
            "x": [0, 9260, 18520],
            "y": [-1200, -1300, -1250],
        },
        "transLossData": {
            "size": 100,
            "transmissionLossVS": {
                "profile": {
                    "x": [0, 9260, 18520],
                    "z": [0, 50, 100],
                    "data": [80.1, 82.4, 200.0],
                },
            },
        },
    },
)
_OPENAPI_2D_RAY = _portal_openapi_extra(
    "Analyse2DRTRD",
    {"azimuth": 0, "lineCount": 21},
    {
        "calcFrequency": 100,
        "sampling": {
            "size": 300,
            "data": {
                "x": [0, 9260, 18520],
                "y": [-1200, -1300, -1250],
            },
        },
        "trackData": {
            "size": 2,
            "data": [
                {"x": [0, 5000, 10000], "z": [150, 220, 90]},
                {"x": [0, 5000, 10000], "z": [150, 80, 260]},
            ],
        },
    },
)
_OPENAPI_3D_TL = _portal_openapi_extra(
    "Analyse3DTLRD",
    {"row": 100, "col": 300, "sector": 4},
    {
        "calcFrequency": 100,
        "altitudeList": {
            "size": 4,
            "pairedDataVS": [
                {"size": 300, "x": [0, 9260, 18520], "y": [-1200, -1300, -1250]},
            ],
        },
        "reconstructDataVS": {
            "size": 100,
            "x": [0, 50, 100],
            "y": [1520.1, 1518.7, 1516.4],
        },
        "loadingTlv": {
            "row": 100,
            "col": 300,
            "sector": 4,
            "size": 400,
            "profile": {
                "x": [0, 9260, 18520],
                "z": [0, 50, 100],
            },
            "data": [[80.1, 82.4, 200.0]],
        },
    },
)
_OPENAPI_ICE_2D_TL = _portal_openapi_extra(
    "AnalyseIce2DTLRD",
    {"azimuth": 90, "rtimeRes": 0},
    {
        "calcFrequency": 100,
        "altList": {
            "size": 300,
            "x": [0, 9260, 18520],
            "y": [-1200, -1300, 9999.0],
        },
        "iceThickness": {
            "size": 300,
            "x": [0, 9260, 18520],
            "y": [1.72, 1.68, 9999.0],
        },
        "transLossData": {
            "size": 100,
            "transmissionLossVS": {
                "profile": {
                    "x": [0, 9260, 18520],
                    "z": [0, 50, 100],
                    "data": [80.1, 82.4, 9999.0],
                },
            },
        },
    },
)
_OPENAPI_ICE_2D_RAY = _portal_openapi_extra(
    "AnalyseIce2DRTRD",
    {"azimuth": 0, "lineCount": 21},
    {
        "calcFrequency": 100,
        "sampling": {
            "size": 300,
            "data": {
                "x": [0, 9260, 18520],
                "y": [-1200, -1300, 9999.0],
            },
        },
        "iceThickness": {
            "size": 300,
            "x": [0, 9260, 18520],
            "y": [1.72, 1.68, 9999.0],
        },
        "trackData": {
            "size": 2,
            "data": [
                {"x": [0, 5000, 10000], "z": [150, 220, 90]},
                {"x": [0, 5000, 10000], "z": [150, 80, 260]},
            ],
        },
    },
)
_OPENAPI_ICE_3D_TL = _portal_openapi_extra(
    "AnalyseIce3DTLRD",
    {"row": 100, "col": 300, "sector": 4},
    {
        "calcFrequency": 100,
        "altitudeList": {
            "size": 4,
            "pairedDataVS": [
                {"size": 300, "x": [0, 9260, 18520], "y": [-1200, -1300, 9999.0]},
            ],
        },
        "iceThicknessList": {
            "size": 4,
            "iceDataVS": [
                {"size": 300, "x": [0, 9260, 18520], "y": [1.72, 1.68, 9999.0]},
            ],
        },
        "reconstructDataVS": {
            "size": 100,
            "x": [0, 50, 100],
            "y": [1520.1, 1518.7, 1516.4],
        },
        "loadingTlv": {
            "row": 100,
            "col": 300,
            "sector": 4,
            "size": 400,
            "profile": {
                "x": [0, 9260, 18520],
                "z": [0, 50, 100],
            },
            "data": [[80.1, 82.4, 9999.0]],
        },
    },
)
_OPENAPI_ENV_STATION = _portal_openapi_extra(
    "GetEnvDataStation",
    {
        "coordX": 116,
        "coordY": 16,
        "dataSource": "EnvReanalysis",
        "dataType": "DT_ENV_SOUND",
        "time": "2025-01-04 12:00:00",
        "timeRes": 0,
        "isReconstruct": False,
        "rdataSource": "EnvReanalysis",
        "rtime": "2025-01-04 12:00:00",
        "rtimeRes": 0,
    },
    {
        "size": 3,
        "data": [{"x": 0, "y": 1520.1}, {"x": 50, "y": 1518.7}, {"x": 100, "y": 1516.4}],
    },
    base_fields={},
)
_OPENAPI_ENV_ROUTE = _portal_openapi_extra(
    "GetEnvDataRoute",
    {
        "dataSource": "EnvReanalysis",
        "dataType": "DT_ENV_TEMP",
        "time": "2025-01-04 12:00:00",
        "timeRes": 0,
        "points": [{"x": 116.0, "y": 16.0}, {"x": 116.2, "y": 16.1}],
    },
    {
        "SamplingPoints": {
            "size": 2,
            "x": [116.0, 116.2],
            "y": [16.0, 16.1],
        },
        "Depths": [1200.0, 1180.0],
        "ProfileData": {
            "size": 2,
            "data": [
                {"size": 2, "x": [0, 50], "y": [26.1, 24.8]},
                {"size": 2, "x": [0, 50], "y": [26.0, 24.7]},
            ],
        },
    },
    base_fields={},
)
_OPENAPI_ENV_REGION = _portal_openapi_extra(
    "GetEnvDataRegion",
    {
        "bottomLeftX": 115.0,
        "bottomLeftY": 15.0,
        "topRightX": 116.0,
        "topRightY": 16.0,
        "dataSource": "EnvReanalysis",
        "dataType": "DT_ENV_SALT",
        "time": "2025-01-04 12:00:00",
        "timeRes": 0,
        "z": 50.0,
    },
    {
        "row": 2,
        "col": 3,
        "size": 6,
        "data": [
            [34.71, 34.72, 34.73],
            [34.70, 34.71, 34.72],
        ],
    },
    base_fields={},
)
_OPENAPI_NOISE = _portal_openapi_extra(
    "AnalyseEnvironmentNoise",
    {
        "shipDensityLevel": 3,
        "windSpeed": 4,
        "rainIntensity": 2,
        "minFreq": 10.0,
        "maxFreq": 1000.0,
        "numPoints": 10000,
    },
    {
        "freq": [10.0, 505.0, 1000.0],
        "data": [78.165983, 65.307027, 63.584269],
    },
    base_fields={},
)
_OPENAPI_DATASET_LIST = _portal_openapi_extra(
    "GetEnvDataSourceList",
    {},
    [{
        "id": "EnvReanalysis",
        "name": "历史数据集",
        "categoryId": "environment",
        "categoryName": "环境数据",
        "pid": "EnvReanalysis",
        "datatype": [{"ids": ["DT_ENV_TEMP", "DT_ENV_SALT"], "rendertype": 0, "name": "环境要素"}],
        "datarange": {"lon": [-180, 180], "lat": [-90, 90]},
        "normalizeLayers": [0, 10, 20, 50, 100],
    }],
    base_fields={},
)
_OPENAPI_DATASET_TIME_LIST = _portal_openapi_extra(
    "FetchTimeToList",
    {"dataSource": "EnvReanalysis", "isEnv": True, "queryDataInMonth": 0},
    {
        "dataSource": "EnvReanalysis",
        "isEnv": True,
        "displayList": ["2025-01-02 15:00:00"],
        "standardTimeList": ["2025-01-02 15:00:00"],
    },
    base_fields={},
)
_OPENAPI_DATA_TYPE_META_LIST = _portal_openapi_extra(
    "GetDataTypeMetaInfo",
    {},
    [{"id": "DT_ENV_TEMP", "name": "温度", "unit": "℃", "minVal": -2.0, "maxVal": 40.0}],
    base_fields={},
)


# =============================================================================
# 页面显示参数和声源选择
# =============================================================================
def _finite_float(value, default):
    """将字符串转换为有限浮点数；转换失败时返回默认值。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _default_tl_caxis():
    """从配置中读取传播损失色轴范围；配置异常时给出安全默认值。"""
    visual = CONFIG.get("VISUALIZATION") or {}
    task_defaults = CONFIG.get("TASK_DEFAULTS") or {}
    fallback_min = _finite_float(task_defaults.get("tl_caxis_min_db"), 60.0)
    fallback_max = _finite_float(task_defaults.get("tl_caxis_max_db"), 120.0)
    if fallback_min >= fallback_max:
        fallback_min, fallback_max = 80.0, 130.0
    cmin = _finite_float(visual.get("MAIN_TL_CAXIS_MIN_DB"), fallback_min)
    cmax = _finite_float(visual.get("MAIN_TL_CAXIS_MAX_DB"), fallback_max)
    if cmin >= cmax:
        cmin, cmax = fallback_min, fallback_max
    return float(cmin), float(cmax)


def _normalise_display_params(params):
    """整理可视化参数；参数无效时给默认值，不直接拒绝计算请求。"""
    default_min, default_max = _default_tl_caxis()
    raw_min = params.get("tl_caxis_min_db")
    raw_max = params.get("tl_caxis_max_db")
    cmin = _finite_float(raw_min, math.nan)
    cmax = _finite_float(raw_max, math.nan)
    warnings = []
    if not math.isfinite(cmin) or not math.isfinite(cmax) or cmin >= cmax:
        warnings.append(
            f"TL 色标参数无效（min={raw_min!r}, max={raw_max!r}），"
            f"已使用默认 {default_min:g} ~ {default_max:g} dB。"
        )
        cmin, cmax = default_min, default_max
    params["tl_caxis_min_db"] = float(cmin)
    params["tl_caxis_max_db"] = float(cmax)
    if warnings:
        params["_parameter_warnings"] = warnings
    return warnings


def _source_points_from_form(form, primary_lat, primary_lon):
    """根据手动选择区域生成包含边界点的声源网格。"""
    mode = _lower_text(form.get("source_mode_selection"), "grid")
    if mode == "single":
        raw_lat = _text(form.get("single_lat"), primary_lat)
        raw_lon = _text(form.get("single_lon"), primary_lon)
        try:
            lat = float(raw_lat)
            lon_raw = float(raw_lon)
        except ValueError as exc:
            raise ValueError("单声源经纬度必须是数值。") from exc
        if not math.isfinite(lat) or not -90.0 <= lat <= 90.0:
            raise ValueError("单声源纬度必须位于 -90 到 90°。")
        if not math.isfinite(lon_raw) or not -180.0 <= lon_raw <= 180.0:
            raise ValueError("单声源经度必须位于 -180 到 180°。")
        return [{"lat": lat, "lon": lon_raw}]
    if mode != "grid":
        raise ValueError("未知的声源计算模式。")
    raw = {}
    for name in REGION_FIELDS:
        raw[name] = _text(form.get(name))
    if not any(raw.values()):
        # 没有提交区域时，兼容旧版直接 API：只计算主声源点。
        return [{"lat": primary_lat, "lon": primary_lon}]
    if not all(raw.values()):
        raise ValueError("请完整填写区域南北边界、东西边界和声源间隔。")
    try:
        lat_min = float(raw["region_lat_min"])
        lat_max = float(raw["region_lat_max"])
        lon_min = float(raw["region_lon_min"])
        lon_max = float(raw["region_lon_max"])
        spacing = float(raw["source_spacing_deg"])
    except ValueError as exc:
        raise ValueError("区域边界和声源间隔必须是数值。") from exc

    if not all(math.isfinite(value) for value in (lat_min, lat_max, lon_min, lon_max, spacing)):
        raise ValueError("区域边界和声源间隔必须是有限数值。")
    if not -90.0 <= lat_min < lat_max <= 90.0:
        raise ValueError("纬度范围必须位于 -90 到 90，且南边界小于北边界。")
    if not -180.0 <= lon_min < lon_max <= 180.0:
        raise ValueError("经度范围必须位于 -180 到 180，且西边界小于东边界。")
    if not 0.0 < spacing <= 10.0:
        raise ValueError("声源间隔必须大于 0 且不超过 10 度。")

    lat_count = int(math.floor((lat_max - lat_min) / spacing + 1e-9)) + 1
    lon_count = int(math.floor((lon_max - lon_min) / spacing + 1e-9)) + 1
    source_count = lat_count * lon_count
    max_sources = int(CONFIG["MAX_SOURCES"])
    if source_count > max_sources:
        raise ValueError(f"当前区域会生成 {source_count} 个声源，超过配置上限 {max_sources}。")
    return [
        {"lat": round(lat_min + row * spacing, 10), "lon": round(lon_min + column * spacing, 10)}
        for row in range(lat_count)
        for column in range(lon_count)
    ]


# =============================================================================
# 门户请求解析和参数配置
# =============================================================================
class PortalRequestError(ValueError):
    """公共计算接口中可以直接展示给用户的参数错误。"""


async def _request_payload(request: Request):
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            data = await request.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}
    form = await request.form()
    data = dict(form)
    if data:
        return data
    try:
        data = await request.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _portal_unit_request(payload, plugin_name):
    """读取 Data.Units[].Request，同时兼容平铺 JSON 请求。"""
    if not isinstance(payload, Mapping):  # 如果 payload 不是键值对类型（判断是否为正常的 JSON 对象）
        return {}, {"wrapped": False}
    data = payload.get("Data")
    if not isinstance(data, Mapping):
        return payload, {"wrapped": False}
    units = data.get("Units")
    if not isinstance(units, list):
        raise PortalRequestError("Data.Units 必须是数组。")
    selected = None
    for unit in units:
        if not isinstance(unit, Mapping):
            continue
        plugin = str(unit.get("Plugin") or "").strip()
        if plugin == plugin_name or not selected:
            selected = unit
        if plugin == plugin_name:
            break
    if not selected:
        raise PortalRequestError(f"Data.Units 中缺少 {plugin_name} 请求单元。")
    request_payload = selected.get("Request")
    if not isinstance(request_payload, Mapping):
        raise PortalRequestError(f"{plugin_name}.Request 必须是对象。")
    meta = {
        "wrapped": True,
        "technical_support": str(payload.get("Technical Support") or payload.get("TechnicalSupport") or "XWOcean"),
        "sn": data.get("SN", ""),
        "task_id": data.get("TaskId", ""),
        "self_order": data.get("SelfOrder", 0),
        "plugin": str(selected.get("Plugin") or plugin_name),
    }
    return dict(request_payload), meta


def _wrap_portal_response(response_payload, meta, plugin_name, job_id=None, result_code=0):
    if not meta.get("wrapped"):
        return _flat_portal_response(response_payload, result_code)
    return {
        "Technical Support": meta.get("technical_support") or "XWOcean",
        "Data": {
            "SN": meta.get("sn", ""),
            "TaskId": job_id if job_id is not None else meta.get("task_id", ""),
            "SelfOrder": meta.get("self_order", 0),
            "Units": [{
                "Plugin": meta.get("plugin") or plugin_name,
                "Response": response_payload,
                "ResultCode": int(result_code),
            }],
        },
    }


def _pick(data, names, default=None):
    # 从data 请求体（request）中提取names键第一个非空值，用于构建计算参数。
    # 如果请求体中没有指定的键，返回默认值。
    for name in names:
        if name not in data:
            continue
        value = data.get(name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def _api_float(data, names, default, label):
    # 从请求体中提取指定键的值，并尝试将其转换为浮点数，label用来报错提示。
    value = _pick(data, names, default)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PortalRequestError(f"{label}必须是数值。") from exc
    if not math.isfinite(number):
        raise PortalRequestError(f"{label}必须是有限数值。")
    return number


def _api_int(data, names, default, label):
    # 从请求体中提取指定键的值，并尝试将其转换为整数，label用来报错提示。
    value = _pick(data, names, default)
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise PortalRequestError(f"{label}必须是整数。") from exc
    return number


def _api_strict_bool(data, names, default, label):
    # 从请求体中提取指定键的值，并尝试将其转换为严格布尔值，label用来报错提示。
    value = _pick(data, names, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise PortalRequestError(f"{label}必须是布尔值。")


def _api_bool(data, names, default):
    # 从请求体中提取指定键的值，并尝试将其转换为布尔值，默认值为default。
    value = _pick(data, names, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _lower_text(value)
    if text in {"1", "true", "yes", "on", "y", "是"}:
        return True
    if text in {"0", "false", "no", "off", "n", "否"}:
        return False
    return bool(default)


def _normalise_lon(lon):
    # 将经度归一化到 -180 到 180 度范围内，避免跨越国际日期变更线时出现异常。
    return (float(lon) + 180.0) % 360.0 - 180.0


def _portal_request_defaults():
    # 从配置文件中（config.PORTAL_API.REQUEST_DEFAULTS）读取门户请求默认值；如果未定义则返回空字典。
    portal = CONFIG.get("PORTAL_API") or {}
    defaults = portal.get("REQUEST_DEFAULTS") if isinstance(portal, Mapping) else {}
    return defaults if isinstance(defaults, Mapping) else {}


def _portal_default(name, fallback=None):
    # 从配置文件中读取门户请求默认值；如果未定义则返回 fallback。
    return _pick(_portal_request_defaults(), (name,), fallback)


def _reject_unsupported_request_fields(data, allowed_fields, plugin_name):
    if not isinstance(data, Mapping):
        return
    # 检查请求中是否包含未定义字段，避免用户提交错误参数。
    # 遍历请求体（request）的所有键，如果有不在 allowed_fields 中的键，则从allowed_fields中append并认为是未定义字段返回给用户。
    extra = sorted(str(key) for key in data.keys() if str(key) not in allowed_fields)
    if extra:
        raise PortalRequestError(
            f"{plugin_name}.Request 包含未定义字段：{', '.join(extra)}。"
        )


def _query_dataset_list():
    return hycom_service.dataset_list()


def _local_environment_time_list(data_source, is_env, query_month):
    """从本地温盐 NetCDF 时间轴生成前端可选时次。"""
    time_sets = []
    kinds = ["temperature", "salinity"]
    if database_source.data_source_mode(CONFIG, "ice", "hycom") == "file":
        kinds.append("ice")
    for kind in kinds:
        source = database_source.data_source_config(CONFIG, kind)
        configured_files = source.get("files")
        configured = source.get("available_times")
        if isinstance(configured_files, Mapping) and configured_files:
            values = {
                str(value).strip().replace("T", " ").removesuffix("Z")
                for value in configured_files
                if str(value).strip()
            }
        elif isinstance(configured, list) and configured:
            values = {str(value).strip().replace("T", " ") for value in configured if str(value).strip()}
        else:
            path = database_source.data_source_file(CONFIG, kind)
            time_name = str(source.get("time_variable") or "time").strip()
            try:
                with netCDF4.Dataset(path, "r") as dataset:
                    variable = dataset.variables.get(time_name)
                    if variable is None:
                        raise KeyError(f"缺少时间变量 {time_name}")
                    units = getattr(variable, "units", "")
                    if not units:
                        raise ValueError(f"时间变量 {time_name} 缺少 units")
                    decoded = netCDF4.num2date(
                        variable[:],
                        units,
                        calendar=getattr(variable, "calendar", "standard"),
                    )
                    values = {
                        f"{int(value.year):04d}-{int(value.month):02d}-{int(value.day):02d} "
                        f"{int(value.hour):02d}:{int(value.minute):02d}:{int(value.second):02d}"
                        for value in np.asarray(decoded).ravel()
                    }
            except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
                raise PortalRequestError(f"本地{kind}文件时间轴读取失败：{exc}") from exc
        time_sets.append(values)
    common = sorted(set.intersection(*time_sets)) if time_sets else []
    catalog_section = CONFIG.get("DATA_CATALOG") or {}
    if query_month:
        common = [value for value in common if int(value[5:7]) == int(query_month)]
    if not common:
        raise PortalRequestError("本地温度、盐度和海冰文件没有共同可用时次。")
    return {
        "dataSource": data_source,
        "isEnv": bool(is_env),
        "displayList": common,
        "standardTimeList": list(common),
        "defaultTime": str(
            catalog_section.get("default_time")
            if isinstance(catalog_section, Mapping) and catalog_section.get("default_time") in common
            else common[-1]
        ),
        "forecastStartTime": "",
        "sourceMode": "file",
    }


def _query_dataset_time_list(data):
    data_source = str(_pick(data, ("dataSource",), "")).strip()
    if not data_source:
        raise PortalRequestError("dataSource 不能为空。")
    if "isEnv" not in data:
        raise PortalRequestError("isEnv 不能为空。")
    is_env = _api_strict_bool(data, ("isEnv",), None, "isEnv")
    raw_month = _pick(data, ("queryDataInMonth",), 0)
    if isinstance(raw_month, bool) or not re.fullmatch(r"[+]?(?:0|[1-9][0-9]*)", str(raw_month).strip()):
        raise PortalRequestError("queryDataInMonth 必须是整数。")
    query_month = int(raw_month)
    if query_month < 0 or query_month > 12:
        raise PortalRequestError("queryDataInMonth 必须为 0 到 12；0 表示查询所有月份。")
    if data_source.upper() != hycom_service.HYCOM_DATA_SOURCE:
        raise PortalRequestError("当前后端环境数据源固定为 HYCOM，dataSource 必须为 HYCOM。")
    if not is_env:
        raise PortalRequestError("HYCOM 时间查询要求 isEnv=true。")
    if (
        catalog_queries.catalog_mode(CONFIG) == "file"
        and database_source.data_source_mode(CONFIG, "temperature", "file") == "file"
        and database_source.data_source_mode(CONFIG, "salinity", "file") == "file"
    ):
        return _local_environment_time_list(data_source, is_env, query_month)
    try:
        return hycom_service.time_list(CONFIG, query_month)
    except hycom_service.HycomServiceError as exc:
        raise PortalRequestError(str(exc)) from exc


def _query_data_type_meta_list():
    if catalog_queries.catalog_mode(CONFIG) == "file":
        section = CONFIG.get("DATA_CATALOG") or {}
        values = section.get("data_type_meta", []) if isinstance(section, Mapping) else []
        return list(values) if isinstance(values, list) else []
    return catalog_queries.query_data_type_meta_list(CONFIG)


def _response_counts(data, row_default=DEFAULT_RESPONSE_ROW, col_default=DEFAULT_RESPONSE_COL):
    row = _api_int(data, ("row", "rows", "depthCount"), row_default, "垂直方向划分数量")
    col = _api_int(data, ("col", "cols", "rangeCount"), col_default, "水平方向划分数量")
    if row < 2:
        raise PortalRequestError("垂直方向划分数量 row 至少为 2。")
    if col < 2:
        raise PortalRequestError("水平方向划分数量 col 至少为 2。")
    if row > 2000 or col > 10000:
        raise PortalRequestError("row/col 过大，请控制在 row<=2000、col<=10000。")
    return row, col


def _configured_response_counts(section_name, row_default=DEFAULT_RESPONSE_ROW, col_default=DEFAULT_RESPONSE_COL):
    portal = CONFIG.get("PORTAL_API") or {}
    section = portal.get(section_name) if isinstance(portal, Mapping) else {}
    section = section if isinstance(section, Mapping) else {}
    try:
        row = int(round(float(section.get("ROW", row_default))))
        col = int(round(float(section.get("COL", col_default))))
    except (TypeError, ValueError) as exc:
        raise PortalRequestError(f"data_config.json 中 PORTAL_API.{section_name}.ROW/COL 必须是整数。") from exc
    if row < 2:
        raise PortalRequestError(f"data_config.json 中 PORTAL_API.{section_name}.ROW 至少为 2。")
    if col < 2:
        raise PortalRequestError(f"data_config.json 中 PORTAL_API.{section_name}.COL 至少为 2。")
    if row > 2000 or col > 10000:
        raise PortalRequestError(f"data_config.json 中 PORTAL_API.{section_name}.ROW/COL 过大。")
    return row, col


def _portal_section_config(section_name):
    # 从配置文件中读取门户配置节（具体某个接口的配置）；如果未定义则返回空字典。
    portal = CONFIG.get("PORTAL_API") or {}
    section = portal.get(section_name) if isinstance(portal, Mapping) else {}
    return section if isinstance(section, Mapping) else {}


def _config_int(section, names, default, label):
    # 从门户配置节中获取，并校验是否整数
    value = _pick(section, names, default)
    try:
        return int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise PortalRequestError(f"data_config.json 中 {label} 必须是整数。") from exc


def _config_float(section, names, default, label):
    value = _pick(section, names, default)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PortalRequestError(f"data_config.json 中 {label} 必须是数值。") from exc
    if not math.isfinite(number):
        raise PortalRequestError(f"data_config.json 中 {label} 必须是有限数值。")
    return number


def _config_pair(section, names, default, label):
    # 从门户配置节中获取，并校验是否两个值的数组
    value = _pick(section, names, default)
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise PortalRequestError(f"data_config.json 中 {label} 必须是长度为 2 的数组。")
    return value


def _resolve_range_km_value(value, data, label):
    # 
    if isinstance(value, str) and _lower_text(value) in {"request.range", "range"}:
        range_nm = _api_float(
            data, ("range", "propagation_distance_nm", "radius_nm"),
            # 当前函数默认值使用配置文件门户默认值，获取门户默认值时，配置文件config.TASK_DEFAULTS作为备选
            _portal_default("range", float(CONFIG["TASK_DEFAULTS"]["radius_km"]) / NM_TO_KM), "传播半径"
        )
        # 输入值为海里单位，但env文件需要的是km
        return range_nm * NM_TO_KM
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PortalRequestError(f"data_config.json 中 {label} 必须是数值或 request.range。") from exc
    if not math.isfinite(number):
        raise PortalRequestError(f"data_config.json 中 {label} 必须是有限数值。")
    return number


def _resolve_depth_m_value(value, label):
    if isinstance(value, str) and _lower_text(value) in {"auto", "actual", "bottom", "max_bottom"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PortalRequestError(f"data_config.json 中 {label} 必须是数值或 auto。") from exc
    if not math.isfinite(number):
        raise PortalRequestError(f"data_config.json 中 {label} 必须是有限数值。")
    return number


def _configured_portal_grid(
    data,
    section_name,
    run_type_names,
    allowed_run_types,
    *,
    default_depth_count=DEFAULT_RESPONSE_ROW,
    default_range_count=DEFAULT_RESPONSE_COL,
):
    section = _portal_section_config(section_name)
    depth_count = _config_int(section, ("NRD", "RECEIVER_DEPTH_COUNT", "ROW"), default_depth_count,
                              f"PORTAL_API.{section_name}.NRD")
    if depth_count < 2:
        raise PortalRequestError(f"data_config.json 中 PORTAL_API.{section_name}.NRD 至少为 2。")
    rd = _config_pair(
        section, ("RD_M", "RECEIVER_DEPTH_RANGE_M"),
        [0.0, CONFIG["TASK_DEFAULTS"].get("receiver_depth_max_m", 1500.0)],
        f"PORTAL_API.{section_name}.RD_M",
    )
    receiver_depth_min_m = _resolve_depth_m_value(rd[0], f"PORTAL_API.{section_name}.RD_M[0]")
    receiver_depth_max_m = _resolve_depth_m_value(rd[1], f"PORTAL_API.{section_name}.RD_M[1]")
    if receiver_depth_min_m is None:
        raise PortalRequestError(f"data_config.json 中 PORTAL_API.{section_name}.RD_M[0] 必须是数值。")
    if receiver_depth_min_m < 0.0:
        raise PortalRequestError(f"data_config.json 中 PORTAL_API.{section_name}.RD_M 起点不能小于 0。")
    if receiver_depth_max_m is not None and receiver_depth_max_m <= receiver_depth_min_m:
        raise PortalRequestError(f"data_config.json 中 PORTAL_API.{section_name}.RD_M 必须满足 起点 < 终点。")

    range_count = _config_int(section, ("NR", "RECEIVER_RANGE_COUNT", "COL"), default_range_count,
                              f"PORTAL_API.{section_name}.NR")
    if range_count < 2:
        raise PortalRequestError(f"data_config.json 中 PORTAL_API.{section_name}.NR 至少为 2。")
    r = _config_pair(
        section, ("R_KM", "RECEIVER_RANGE_KM"),
        [0.0, "request.range"],
        f"PORTAL_API.{section_name}.R_KM",
    )
    receiver_range_min_km = _resolve_range_km_value(r[0], data, f"PORTAL_API.{section_name}.R_KM[0]")
    receiver_range_max_km = _resolve_range_km_value(r[1], data, f"PORTAL_API.{section_name}.R_KM[1]")
    if receiver_range_min_km < 0.0 or receiver_range_max_km <= receiver_range_min_km:
        raise PortalRequestError(f"data_config.json 中 PORTAL_API.{section_name}.R_KM 必须满足 0 <= 起点 < 终点。")

    run_type = _text(_pick(section, run_type_names, allowed_run_types[0])).upper()
    if run_type not in allowed_run_types:
        allowed_text = "/".join(allowed_run_types)
        raise PortalRequestError(
            f"data_config.json 中 PORTAL_API.{section_name}.RUN_TYPE 必须是 {allowed_text}。"
        )
    nbeams = _config_int(
        section, ("NBEAMS", "N_ALPHA"),
        CONFIG.get("N_ALPHA", 21), f"PORTAL_API.{section_name}.NBEAMS"
    )
    if nbeams < 0 :
        raise PortalRequestError(f"data_config.json 中 PORTAL_API.{section_name}.NBEAMS 必须大于等于 0 。")
    nsd = _config_int(section, ("NSD",), 1, f"PORTAL_API.{section_name}.NSD")
    if nsd != 1:
        raise PortalRequestError(
            f"当前后端每次接口计算一个声源深度，PORTAL_API.{section_name}.NSD 必须为 1。"
        )
    return {
        "nsd": nsd,
        "nrd": depth_count,
        "rd_min_m": receiver_depth_min_m,
        "rd_max_m": receiver_depth_max_m,
        "rd_auto_max": receiver_depth_max_m is None,
        "nr": range_count,
        "r_min_km": receiver_range_min_km,
        "r_max_km": receiver_range_max_km,
        "run_type": run_type,
        "nbeams": nbeams,
    }


def _configured_ray_grid(data, section_name):
    depth_count = _config_int(CONFIG, ("N_RD",), DEFAULT_RESPONSE_ROW, "bellhop_config.N_RD")
    range_count = _config_int(CONFIG, ("N_RANGE",), DEFAULT_RESPONSE_COL, "bellhop_config.N_RANGE")
    if depth_count < 2 or range_count < 2:
        raise PortalRequestError("Bellhop 配置中的 N_RANGE 和 N_RD 至少为 2。")
    return _configured_portal_grid(
        data,
        section_name,
        ("RAY_RUN_TYPE", "RUN_TYPE"),
        ("R",),
        default_depth_count=depth_count,
        default_range_count=range_count,
    )


def _model_tl_grid_counts(acoustic_model):
    """从所选求解器配置读取统一命名的 TL 距离与深度点数。"""
    model = acoustic_model_dispatcher.normalise_acoustic_model(acoustic_model)
    if model == "bellhop":
        model_config = CONFIG
    else:
        model_config = CONFIG[f"{model.upper()}_CONFIG"]
    range_count = _config_int(model_config, ("TL_N_RANGE",), DEFAULT_RESPONSE_COL, f"{model}_config.TL_N_RANGE")
    depth_count = _config_int(model_config, ("TL_N_RD",), DEFAULT_RESPONSE_ROW, f"{model}_config.TL_N_RD")
    if range_count < 2 or depth_count < 2:
        raise PortalRequestError(f"{model} 配置中的 TL_N_RANGE 和 TL_N_RD 至少为 2。")
    return depth_count, range_count


def _configured_tl_grid(data, section_name, acoustic_model="bellhop"):
    depth_count, range_count = _model_tl_grid_counts(acoustic_model)
    return _configured_portal_grid(
        data,
        section_name,
        ("TL_RUN_TYPE", "RUN_TYPE"),
        ("I", "C", "S"),
        default_depth_count=depth_count,
        default_range_count=range_count,
    )


def _configured_2d_tl_grid(data, acoustic_model="bellhop"):
    return _configured_tl_grid(data, "TWO_D_TRANSMISSION_LOSS", acoustic_model)


def _configured_3d_tl_grid(data, acoustic_model="bellhop"):
    return _configured_tl_grid(data, "THREE_D_TRANSMISSION_LOSS", acoustic_model)


def _configured_sampling_size(section_name, default=DEFAULT_RESPONSE_COL):
    # 从配置文件（PORTAL_API）中读取采样点数量；如果未定义则返回默认值。
    section = _portal_section_config(section_name)
    try:
        size = int(round(float(section.get("SAMPLING_SIZE", default))))
    except (TypeError, ValueError) as exc:
        raise PortalRequestError(f"data_config.json 中 PORTAL_API.{section_name}.SAMPLING_SIZE 必须是整数。") from exc
    if size < 2 :
        raise PortalRequestError(f"data_config.json 中 PORTAL_API.{section_name}.SAMPLING_SIZE 必须大于等于 2。")
    return size


def _portal_base_params(data, row, col, acoustic_model="bellhop"):
    # 从门户请求体中提取基本参数，并进行校验和默认值处理。
    defaults = dict(CONFIG["TASK_DEFAULTS"])
    source_lon_raw = _api_float(
        data, ("coordX",),
        _portal_default("coordX", defaults.get("center_lon", 0.0)),
        "声源点经度 coordX"
    )
    source_lat = _api_float(
        data, ("coordY",),
        _portal_default("coordY", defaults.get("center_lat", 0.0)),
        "声源点纬度 coordY"
    )
    if not -90.0 <= source_lat <= 90.0:
        raise PortalRequestError("声源点纬度必须位于 -90 到 90°。")
    if not -360.0 <= source_lon_raw <= 360.0:
        raise PortalRequestError("声源点经度必须位于 -360 到 360°。")
    source_lon = _normalise_lon(source_lon_raw)

    data_source = str(_pick(data, ("dataSource",), _portal_default("dataSource", "HYCOM"))).strip()
    rdata_source = str(_pick(data, ("rdataSource",), _portal_default("rdataSource", "HYCOM"))).strip()
    data_type = str(_pick(data, ("dataType",), _portal_default("dataType", "DT_ENV_SOUND"))).strip()
    if data_source.upper() != hycom_service.HYCOM_DATA_SOURCE:
        raise PortalRequestError("dataSource 当前仅支持固定值 HYCOM。")
    if rdata_source.upper() != hycom_service.HYCOM_DATA_SOURCE:
        raise PortalRequestError("rdataSource 当前仅支持固定值 HYCOM。")
    if data_type and data_type != "DT_ENV_SOUND":
        raise PortalRequestError("dataType 当前仅支持固定值 DT_ENV_SOUND。")

    time_value = str(
        _pick(
            data,
            ("time",),
            # 默认值使用配置文件的PORTAL_API.REQUEST_DEFAULTS.time 
            _portal_default("time", database_source.default_environment_time(CONFIG)),
        )
    ).strip()
    rtime_value = str(_pick(data, ("rtime",), _portal_default("rtime", time_value))).strip()
    time_res = _api_int(data, ("timeRes",), _portal_default("timeRes", 0), "时间偏移量")
    rtime_res = _api_int(data, ("rtimeRes",), _portal_default("rtimeRes", 0), "重构时间偏移量")

    src_depth = _api_float(
        data, ("srcDepth",),
        _portal_default("srcDepth", defaults.get("source_depth_m", 150.0)),
        "声源点深度"
    )
    if src_depth <= 0.0:
        raise PortalRequestError("声源点深度必须为正数。")
    range_nm = _api_float(
        data, ("range",),
        _portal_default("range", float(defaults.get("radius_km", 100.0)) / NM_TO_KM),
        "传播半径"
    )
    if range_nm <= 0.0:
        raise PortalRequestError("传播半径必须为正数。")

    min_freq = _api_int(data, ("minFreq",), _portal_default("minFreq", 100), "最小频率")
    max_freq = _api_int(data, ("maxFreq",), _portal_default("maxFreq", 100), "最大频率")
    if min_freq <= 0 or max_freq <= 0:
        raise PortalRequestError("最大频率和最小频率必须为正整数。")
    if min_freq > max_freq:
        raise PortalRequestError("最小频率不能大于最大频率。")
    freq_hz = (float(min_freq) + float(max_freq)) * 0.5

    grazing_angle = _api_float(data, ("grazingAngle",), _portal_default("grazingAngle", 0.0), "波束角")
    beam_width = _api_float(data, ("beamWidth",), _portal_default("beamWidth", 20.0), "波束宽度")
    if beam_width <= 0.0:
        raise PortalRequestError("波束宽度必须为正数。")
    alpha_min = grazing_angle - beam_width * 0.5
    alpha_max = grazing_angle + beam_width * 0.5

    sea_state = _api_int(data, ("seaState",), _portal_default("seaState", 0), "海况")
    if sea_state < 0 or sea_state > 9:
        raise PortalRequestError("海况 seaState 必须位于 0 到 9。")
    coherent = _api_bool(data, ("coherent",), _portal_default("coherent", False))
    is_water_changed = _api_bool(data, ("isWaterChanged",), _portal_default("isWaterChanged", True))
    # 门户接口不再从请求体选择模型；具体求解器由 Plugin/SN 对应的处理器固定。
    acoustic_model = acoustic_model_dispatcher.normalise_acoustic_model(acoustic_model)
    receiver_depth_max = _api_float(
        data, ("receiverDepthMax",),
        defaults["receiver_depth_max_m"], "最大接收深度"
    )
    if receiver_depth_max <= 0.0:
        raise PortalRequestError("最大接收深度必须为正数，实际的接收深度为接收点的全海深。")

    params = dict(defaults)
    params.update({
        "center_lat": source_lat,
        "center_lon": source_lon,
        "single_lat": source_lat,
        "single_lon": source_lon,
        "selection_points": [{"lat": source_lat, "lon": source_lon}],
        "selection_point_count": 1,
        "source_mode": "single_source",
        "freq_hz": freq_hz,
        "source_depth_m": src_depth,
        "receiver_depth_max_m": max(float(receiver_depth_max), float(src_depth)),
        "radius_km": range_nm * NM_TO_KM,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "horizontal_tl_depth_m": src_depth,
        "tl_run_type": "C" if coherent else "I",
        "n_range": col,
        "n_rd": row,
        "tl_n_range": col,
        "tl_n_rd": row,
        "data_source": data_source,
        "data_type": data_type,
        "environment_time": time_value,
        "time_res": time_res,
        "rdata_source": rdata_source,
        "reconstruct_time": rtime_value,
        "rtime_res": rtime_res,
        "sea_state": sea_state,
        "coherent": coherent,
        "is_water_changed": is_water_changed,
        "acoustic_model": acoustic_model,
        "portal_request": {
            "coordX": source_lon,
            "coordY": source_lat,
            "dataSource": data_source,
            "dataType": data_type,
            "time": time_value,
            "timeRes": time_res,
            "rdataSource": rdata_source,
            "rtime": rtime_value,
            "rtimeRes": rtime_res,
            "srcDepth": src_depth,
            "range": range_nm,
            "maxFreq": max_freq,
            "minFreq": min_freq,
            "grazingAngle": grazing_angle,
            "beamWidth": beam_width,
            "seaState": sea_state,
            "coherent": coherent,
            "isWaterChanged": is_water_changed,
        },
    })
    return params


def _portal_receiver_depth_settings(data, params, env_grid, scope):
    """区分请求显式深度与配置中的自动深度。"""
    explicit = "receiverDepthMax" in data and data.get("receiverDepthMax") not in (None, "")
    if explicit:
        maximum = max(float(params["receiver_depth_max_m"]), float(params["source_depth_m"]))
        return maximum, False, scope
    maximum = max(
        float(env_grid["rd_max_m"] or CONFIG["TASK_DEFAULTS"].get("receiver_depth_max_m", 1500.0)),
        float(params["source_depth_m"]),
    )
    return maximum, bool(env_grid["rd_auto_max"]), scope


def _portal_2d_params(data, include_line_count=False):
    env_grid = _configured_ray_grid(data, "TWO_D_RAY_TRACK")
    row, col = env_grid["nrd"], env_grid["nr"]
    params = _portal_base_params(data, row, col)
    azimuth = _api_float(data, ("azimuth", "bearing_deg", "selected_section_theta_deg"), 0.0, "方位角") % 360.0
    params.update({
        "calculation_mode": "2d",
        "section_count": 1,
        "beta_min": azimuth,
        "beta_max": azimuth,
        "selected_section_theta_deg": azimuth,
        "sound_field_view": "selected_section",
        "display_3d_rays": True,
    })
    params["portal_request"]["azimuth"] = azimuth
    line_count = None
    if include_line_count:
        line_count = _api_int(data, ("lineCount", "line_count", "n_alpha"), env_grid["nbeams"], "声线条数")
        if line_count < 0 :
            raise PortalRequestError("声线条数 lineCount 必须大于等于 0 。")
        params["n_alpha"] = line_count
        params["portal_request"]["lineCount"] = line_count
    receiver_max, receiver_auto, receiver_scope = _portal_receiver_depth_settings(
        data, params, env_grid, "path"
    )
    params.update({
        "receiver_depth_min_m": env_grid["rd_min_m"],
        "receiver_depth_max_m": receiver_max,
        "receiver_depth_auto_max": receiver_auto,
        "receiver_depth_auto_scope": receiver_scope,
        "receiver_range_min_km": env_grid["r_min_km"],
        "radius_km": env_grid["r_max_km"],
        "ray_run_type": "R",
        "n_rd": row,
        "n_range": col,
        "tl_n_rd": row,
        "tl_n_range": col,
    })
    params["portal_request"].update({
        "NSD": env_grid["nsd"],
        "NRD": env_grid["nrd"],
        "RD_M": [env_grid["rd_min_m"], "auto" if env_grid["rd_auto_max"] else env_grid["rd_max_m"]],
        "NR": env_grid["nr"],
        "R_KM": [env_grid["r_min_km"], env_grid["r_max_km"]],
        "RUN_TYPE": env_grid["run_type"],
        "NBEAMS": env_grid["nbeams"],
    })
    return params, row, col, line_count


def _portal_2d_tl_params(data, acoustic_model="bellhop"):
    env_grid = _configured_2d_tl_grid(data, acoustic_model)
    row, col = env_grid["nrd"], env_grid["nr"]
    params = _portal_base_params(data, row, col, acoustic_model)
    azimuth = _api_float(data, ("azimuth", "bearing_deg", "selected_section_theta_deg"), 0.0, "方位角") % 360.0
    coherent_was_requested = any(name in data for name in ("coherent",))
    tl_run_type = params["tl_run_type"] if coherent_was_requested else env_grid["run_type"]
    receiver_max, receiver_auto, receiver_scope = _portal_receiver_depth_settings(
        data, params, env_grid, "path"
    )
    params.update({
        "calculation_mode": "2d",
        "section_count": 1,
        "beta_min": azimuth,
        "beta_max": azimuth,
        "selected_section_theta_deg": azimuth,
        "sound_field_view": "selected_section",
        "display_3d_rays": True,
        "n_alpha": env_grid["nbeams"],
        "receiver_depth_min_m": env_grid["rd_min_m"],
        "receiver_depth_max_m": receiver_max,
        "receiver_depth_auto_max": receiver_auto,
        "receiver_depth_auto_scope": receiver_scope,
        "receiver_range_min_km": env_grid["r_min_km"],
        "radius_km": env_grid["r_max_km"],
        "tl_run_type": tl_run_type,
        "n_rd": row,
        "n_range": col,
        "tl_n_rd": row,
        "tl_n_range": col,
    })
    params["portal_request"].update({
        "azimuth": azimuth,
        "NSD": env_grid["nsd"],
        "NRD": env_grid["nrd"],
        "RD_M": [env_grid["rd_min_m"], "auto" if env_grid["rd_auto_max"] else env_grid["rd_max_m"]],
        "NR": env_grid["nr"],
        "R_KM": [env_grid["r_min_km"], env_grid["r_max_km"]],
        "RUN_TYPE": tl_run_type,
        "NBEAMS": env_grid["nbeams"],
    })
    return params, row, col


def _portal_3d_params(data, acoustic_model="bellhop"):
    env_grid = _configured_3d_tl_grid(data, acoustic_model)
    row = _api_int(data, ("row",), env_grid["nrd"], "垂直方向划分数量 row")
    col = _api_int(data, ("col",), env_grid["nr"], "水平方向划分数量 col")
    if row < 2:
        raise PortalRequestError("垂直方向划分数量 row 至少为 2。")
    if col < 2:
        raise PortalRequestError("水平方向划分数量 col 至少为 2。")
    if row > 2000 or col > 10000:
        raise PortalRequestError("row/col 过大，请控制在 row<=2000、col<=10000。")
    params = _portal_base_params(data, row, col, acoustic_model)
    section = _portal_section_config("THREE_D_TRANSMISSION_LOSS")
    default_sector = _config_int(
        section, ("DEFAULT_SECTOR", "SECTOR"),
        4, "PORTAL_API.THREE_D_TRANSMISSION_LOSS.DEFAULT_SECTOR"
    )
    sector = _api_int(data, ("sector", "section_count"), default_sector, "扇面数")
    if sector not in VALID_3D_SECTORS:
        raise PortalRequestError("三维传播损失扇面数 sector 仅支持 4、8、12、24。")
    coherent_was_requested = any(name in data for name in ("coherent",))
    tl_run_type = params["tl_run_type"] if coherent_was_requested else env_grid["run_type"]
    receiver_max, receiver_auto, receiver_scope = _portal_receiver_depth_settings(
        data, params, env_grid, "area"
    )
    params.update({
        "calculation_mode": "3d",
        "section_count": sector,
        "beta_min": 0.0,
        "beta_max": 359.0,
        "selected_section_theta_deg": 0.0,
        "sound_field_view": "full_3d",
        "display_3d_rays": False,
        "receiver_depth_min_m": env_grid["rd_min_m"],
        "receiver_depth_max_m": receiver_max,
        "receiver_depth_auto_max": receiver_auto,
        "receiver_depth_auto_scope": receiver_scope,
        "receiver_range_min_km": env_grid["r_min_km"],
        "radius_km": env_grid["r_max_km"],
        "tl_run_type": tl_run_type,
        "n_alpha": env_grid["nbeams"],
        "n_rd": row,
        "n_range": col,
        "tl_n_rd": row,
        "tl_n_range": col,
    })
    params["portal_request"].update({
        "row": row,
        "col": col,
        "sector": sector,
        "NSD": env_grid["nsd"],
        "NRD": env_grid["nrd"],
        "RD_M": [env_grid["rd_min_m"], "auto" if env_grid["rd_auto_max"] else env_grid["rd_max_m"]],
        "NR": env_grid["nr"],
        "R_KM": [env_grid["r_min_km"], env_grid["r_max_km"]],
        "RUN_TYPE": tl_run_type,
        "NBEAMS": env_grid["nbeams"],
    })
    return params, row, col, sector


def _portal_3d_ray_params(data):
    """Build one multi-bearing Bellhop ray task from a portal request."""
    env_grid = _configured_ray_grid(data, "TWO_D_RAY_TRACK")
    row = _api_int(data, ("row",), env_grid["nrd"], "垂直方向划分数量 row")
    col = _api_int(data, ("col",), env_grid["nr"], "水平方向划分数量 col")
    if row < 2:
        raise PortalRequestError("垂直方向划分数量 row 至少为 2。")
    if col < 2:
        raise PortalRequestError("水平方向划分数量 col 至少为 2。")
    if row > 2000 or col > 10000:
        raise PortalRequestError("row/col 过大，请控制在 row<=2000、col<=10000。")
    params = _portal_base_params(data, row, col)
    section = _portal_section_config("THREE_D_RAY_TRACK")
    default_sector = _config_int(
        section, ("DEFAULT_SECTOR", "SECTOR"),
        4, "PORTAL_API.THREE_D_RAY_TRACK.DEFAULT_SECTOR",
    )
    sector = _api_int(data, ("sector",), default_sector, "扇面数")
    if sector not in VALID_3D_SECTORS:
        raise PortalRequestError("三维动态声线扇面数 sector 仅支持 4、8、12、24。")
    azimuths = [360.0 * index / sector for index in range(sector)]

    line_count = _api_int(
        data, ("lineCount", "line_count", "n_alpha"), env_grid["nbeams"], "每个方位的声线条数"
    )
    if line_count < 0:
        raise PortalRequestError("声线条数 lineCount 必须大于等于 0。")
    receiver_max, receiver_auto, receiver_scope = _portal_receiver_depth_settings(
        data, params, env_grid, "area"
    )
    params.update({
        "calculation_mode": "3d",
        "section_count": sector,
        "beta_min": 0.0,
        "beta_max": 359.0,
        "selected_section_theta_deg": 0.0,
        "sound_field_view": "full_3d",
        "display_3d_rays": True,
        "n_alpha": line_count,
        "receiver_depth_min_m": env_grid["rd_min_m"],
        "receiver_depth_max_m": receiver_max,
        "receiver_depth_auto_max": receiver_auto,
        "receiver_depth_auto_scope": receiver_scope,
        "receiver_range_min_km": env_grid["r_min_km"],
        "radius_km": env_grid["r_max_km"],
        "ray_run_type": "R",
        "n_rd": row,
        "n_range": col,
        "tl_n_rd": row,
        "tl_n_range": col,
    })
    params["portal_request"].update({
        "row": row,
        "col": col,
        "sector": sector,
        "lineCount": line_count,
        "NSD": env_grid["nsd"],
        "NRD": row,
        "RD_M": [env_grid["rd_min_m"], "auto" if env_grid["rd_auto_max"] else env_grid["rd_max_m"]],
        "NR": col,
        "R_KM": [env_grid["r_min_km"], env_grid["r_max_km"]],
        "RUN_TYPE": "R",
        "NBEAMS": line_count,
    })
    return params, col, line_count, azimuths


def _portal_native_3d_ray_params(data):
    """按 AnalyseRayTracing 协议构造真正的 Bellhop3D 任务。"""
    required = ("coordX", "coordY", "dataSource", "time", "srcDepth", "lineCount")
    missing = [name for name in required if name not in data or data.get(name) in (None, "")]
    if missing:
        raise PortalRequestError(
            f"AnalyseRayTracing.Request 缺少必填字段：{', '.join(missing)}。"
        )

    min_freq = _api_float(data, ("minFreq",), 500.0, "最小频率 minFreq")
    max_freq = _api_float(data, ("maxFreq",), 500.0, "最大频率 maxFreq")
    if min_freq <= 0.0 or max_freq <= 0.0:
        raise PortalRequestError("minFreq 和 maxFreq 必须为正数。")
    if min_freq > max_freq:
        raise PortalRequestError("最小频率 minFreq 不能大于最大频率 maxFreq。")
    range_nm = _api_float(data, ("range",), 50.0, "传播半径 range")
    if range_nm <= 0.0:
        raise PortalRequestError("传播半径 range 必须为正数。")
    sector_number = _api_float(data, ("sector",), 8, "扇面数 sector")
    if not sector_number.is_integer():
        raise PortalRequestError("扇面数 sector 必须是整数。")
    sector = int(sector_number)
    if sector not in {8, 12, 24}:
        raise PortalRequestError("扇面数 sector 仅支持 8、12、24，默认为 8。")
    line_count_number = _api_float(
        data, ("lineCount",), None, "每个扇面的声线条数 lineCount"
    )
    if not line_count_number.is_integer():
        raise PortalRequestError("声线条数 lineCount 必须是整数。")
    line_count = int(line_count_number)
    if line_count < 1:
        raise PortalRequestError("声线条数 lineCount 必须大于等于 1。")
    if line_count > 2000:
        raise PortalRequestError("声线条数 lineCount 过大，请控制在 2000 以内。")

    env_grid = _configured_ray_grid(data, "TWO_D_RAY_TRACK")
    normalised = dict(data)
    # 公共参数转换函数还兼容其他门户接口的字段；此处只在内部补齐，
    # 不改变 AnalyseRayTracing 对外的 10 个请求字段。
    normalised.update({
        # _portal_base_params 为历史接口按整数频率解析；此处只给它
        # 占位值，随后立即用本接口的 double 频率覆盖。
        "minFreq": 1,
        "maxFreq": 1,
        "range": range_nm,
        "dataType": "DT_ENV_SOUND",
        "timeRes": 0,
        "rdataSource": "HYCOM",
        "rtime": str(data["time"]),
        "rtimeRes": 0,
        "grazingAngle": 0.0,
        "beamWidth": float(CONFIG["TASK_DEFAULTS"].get("alpha_max", 10.0))
        - float(CONFIG["TASK_DEFAULTS"].get("alpha_min", -10.0)),
        "seaState": 0,
        "coherent": False,
        "isWaterChanged": True,
    })
    params = _portal_base_params(normalised, env_grid["nrd"], env_grid["nr"])
    params.update({
        "freq_hz": (min_freq + max_freq) * 0.5,
        "radius_km": range_nm * NM_TO_KM,
        "calculation_mode": "3d",
        "section_count": sector,
        "beta_min": 0.0,
        "beta_max": 360.0 - 360.0 / sector,
        "display_3d_rays": True,
        "sound_field_view": "full_3d",
        "n_alpha": line_count,
        "ray_run_type": "R",
    })
    params["portal_request"] = {
        "coordX": float(params["center_lon"]),
        "coordY": float(params["center_lat"]),
        "dataSource": "HYCOM",
        "time": str(data["time"]),
        "srcDepth": float(params["source_depth_m"]),
        "minFreq": min_freq,
        "maxFreq": max_freq,
        "range": range_nm,
        "sector": sector,
        "lineCount": line_count,
    }
    return params, sector, line_count


# =============================================================================
# 门户任务执行和响应整理
# =============================================================================
def _api_json_dumps(payload):
    """生成门户 JSON，并将所有浮点数固定为六位小数的数值字面量。"""
    def encode(value):
        if value is None:
            return "null"
        if isinstance(value, (bool, np.bool_)):
            return "true" if value else "false"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating, Decimal)):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("JSON 响应中不允许 NaN 或 Infinity。")
            number = round(number, PORTAL_DECIMAL_PLACES)
            if number == 0.0:
                number = 0.0
            return f"{number:.{PORTAL_DECIMAL_PLACES}f}"
        if isinstance(value, (str, np.str_)):
            return json.dumps(str(value), ensure_ascii=False)
        if isinstance(value, Mapping):
            fields = []
            for key, item in value.items():
                key_text = key if isinstance(key, str) else str(key)
                fields.append(f"{json.dumps(key_text, ensure_ascii=False)}:{encode(item)}")
            return "{" + ",".join(fields) + "}"
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return "[" + ",".join(encode(item) for item in value) + "]"
        raise TypeError(f"无法序列化为 JSON 的类型：{type(value).__name__}")

    return encode(payload)


def _api_json_response(payload, status=200):
    return Response(
        _api_json_dumps(payload),
        status_code=status,
        media_type="application/json; charset=utf-8",
    )

# =============================================================================
# 门户任务执行
# =============================================================================
def _run_portal_job(params):
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    params["job_id"] = job_id
    _normalise_display_params(params)
    _write_json(os.path.join(job_dir, "params.json"), params)
    _write_json(os.path.join(job_dir, "job_status.json"), {
        "status": "running", "progress": "准备计算", "message": "接口同步计算中。"
    })
    progress_log = []

    def collect_progress(stage, message):
        """收集计算进度"""
        progress_log.append({"stage": str(stage), "message": str(message or ""), "time": time.time()})
        if len(progress_log) > 200:
            del progress_log[:100]

    try:
        result = acoustic_model_dispatcher.run(
            job_dir, params, CONFIG, progress_callback=collect_progress
        )
    except Exception as exc:
        _write_json(os.path.join(job_dir, "job_status.json"), {
            "status": "failed",
            "progress": "失败",
            "message": str(exc),
            "ready": False,
            "progress_log": progress_log,
        })
        raise
    if result.get("status") != "ready":
        message = result.get("message") or "计算失败"
        _write_json(os.path.join(job_dir, "job_status.json"), {
            "status": "failed",
            "progress": "失败",
            "message": message,
            "ready": False,
            "progress_log": progress_log,
        })
        raise RuntimeError(message)
    # 求解器可能在环境采样后解析自动接收深度等动态参数；回写最终采用值，
    # 避免 params.json 永远停留在提交时的占位默认值。
    _write_json(os.path.join(job_dir, "params.json"), params)
    _write_json(os.path.join(job_dir, "job_status.json"), {
        "status": "ready",
        "progress": "完成",
        "message": f"完成 ({result.get('elapsed', result.get('elapsed_sec', 0.0)):.1f}s)",
        "ready": True,
        "progress_log": progress_log,
    })
    return job_id, job_dir, result


def _run_portal_native_3d_ray_job(params, sector, line_count):
    """单独执行 AnalyseRayTracing 的 Bellhop3D 任务。

    这里故意不进入 ``acoustic_model_dispatcher``，因为该分发器中的 Bellhop
    是多方位二维求解器，会产生用户明确不要的二维拼接结果。
    """
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    params["job_id"] = job_id
    _normalise_display_params(params)
    _write_json(os.path.join(job_dir, "params.json"), params)
    _write_json(os.path.join(job_dir, "job_status.json"), {
        "status": "running",
        "progress": "Bellhop3D 计算中",
        "message": "正在构建三维声速场、海底和声线扇面。",
    })
    try:
        result = bellhop3d_ray.run(job_dir, params, CONFIG, sector, line_count)
    except Exception as exc:
        _write_json(os.path.join(job_dir, "job_status.json"), {
            "status": "failed",
            "progress": "失败",
            "message": str(exc),
            "ready": False,
        })
        raise
    _write_json(os.path.join(job_dir, "params.json"), params)
    _write_json(os.path.join(job_dir, "result_summary.json"), {
        "status": "ready",
        "elapsed_sec": result.get("elapsed", 0.0),
        "compute_engine": "native Bellhop3D",
        "calculation_method": "native_bellhop3d_hexahedral_ssp",
        "manifest": result.get("manifest", {}),
    })
    _write_json(os.path.join(job_dir, "job_status.json"), {
        "status": "ready",
        "progress": "完成",
        "message": f"完成 ({result.get('elapsed', 0.0):.1f}s)",
        "ready": True,
    })
    return job_id, job_dir, result["response"]


def _nc_array(dataset, name):
    variable = dataset.variables[name]
    return np.asarray(np.ma.filled(variable[...], np.nan), dtype=np.float64)


def _json_array(values, decimals=None):
    arr = np.asarray(values)
    if arr.ndim == 0:
        if np.issubdtype(arr.dtype, np.number):
            number = float(arr)
            if not math.isfinite(number):
                return None
            return round(number, PORTAL_DECIMAL_PLACES) if decimals is not None else number
        return arr.item()
    if decimals is not None and np.issubdtype(arr.dtype, np.number):
        arr = np.round(arr.astype(np.float64, copy=False), PORTAL_DECIMAL_PLACES)
    if not np.issubdtype(arr.dtype, np.number):
        return arr.tolist()
    finite_view = arr.astype(np.float64, copy=False)
    out = arr.astype(object)
    out[~np.isfinite(finite_view)] = None
    return out.tolist()


def _portal_data_scalar(value, decimals=None):
    """序列化单个门户数据值；陆地或缺测使用 9999.0。"""
    number = _json_scalar(value, decimals)
    return PORTAL_INVALID_DATA_VALUE if number is None else number


def _portal_data_array(values, decimals=None):
    """序列化数值数组，同时保持原来的形状和长度。"""
    arr = np.asarray(values, dtype=np.float64)
    if decimals is not None:
        arr = np.round(arr, PORTAL_DECIMAL_PLACES)
    out = arr.astype(object)
    out[~np.isfinite(arr)] = PORTAL_INVALID_DATA_VALUE
    return out.tolist()


def _portal_tl_data_array(values, land_columns, below_bottom_mask=None, decimals=PORTAL_DECIMAL_PLACES):
    """序列化传播损失：无效水域和陆地为 200 dB，海底以下为 9999.0。"""
    arr = np.asarray(values, dtype=np.float64).copy()
    if arr.ndim != 2:
        raise ValueError("传播损失 data 必须是二维数组")
    land = np.asarray(land_columns, dtype=bool).ravel()
    if arr.shape[1] != land.size:
        raise ValueError("传播损失 data 的距离维度与陆地掩膜不一致")
    arr[~np.isfinite(arr)] = PORTAL_INVALID_TL_DB
    if below_bottom_mask is not None:
        below_bottom = np.asarray(below_bottom_mask, dtype=bool)
        if below_bottom.shape != arr.shape:
            raise ValueError("传播损失 data 与海底以下掩膜维度不一致")
        arr[below_bottom] = PORTAL_INVALID_DATA_VALUE
    # 陆地列覆盖海底以下掩膜：传播到岸终止，统一返回 200 dB。
    arr[:, land] = PORTAL_INVALID_TL_DB
    return _portal_data_array(arr, decimals)


def _finite_max(values, default):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if finite.size else float(default)


def _target_axis(start, stop, count):
    count = int(count)
    if count <= 1:
        return np.asarray([float(start)], dtype=np.float64)
    if not math.isfinite(float(stop)) or float(stop) <= float(start):
        stop = float(start) + 1.0
    return np.linspace(float(start), float(stop), count, dtype=np.float64)


def _sorted_xy(x_axis, values):
    x = np.asarray(x_axis, dtype=np.float64).ravel()
    y = np.asarray(values, dtype=np.float64).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    x, y = x[good], y[good]
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    return unique_x, y[unique_idx]


def _resample_series(x_axis, values, target_x):
    x, y = _sorted_xy(x_axis, values)
    target = np.asarray(target_x, dtype=np.float64)
    if len(x) == 0:
        return np.full(target.shape, np.nan, dtype=np.float64)
    if len(x) == 1:
        return np.full(target.shape, float(y[0]), dtype=np.float64)
    return np.interp(target, x, y, left=np.nan, right=np.nan)


def _resample_tl(range_km, depth_m, data_range_depth, target_range_km, target_depth_m):
    r = np.asarray(range_km, dtype=np.float64).ravel()
    z = np.asarray(depth_m, dtype=np.float64).ravel()
    values = np.asarray(data_range_depth, dtype=np.float64)
    if values.shape == (len(z), len(r)):
        values = values.T
    if values.shape != (len(r), len(z)):
        raise ValueError("传播损失数组维度与距离/深度坐标不一致")
    r_order = np.argsort(r)
    z_order = np.argsort(z)
    r = r[r_order]
    z = z[z_order]
    values = values[r_order, :][:, z_order]
    by_range = np.full((len(target_range_km), len(z)), np.nan, dtype=np.float64)
    for iz in range(len(z)):
        by_range[:, iz] = _resample_series(r, values[:, iz], target_range_km)
    out = np.full((len(target_depth_m), len(target_range_km)), np.nan, dtype=np.float64)
    for ir in range(len(target_range_km)):
        out[:, ir] = _resample_series(z, by_range[ir, :], target_depth_m)
    return out


def _read_tl2d_section(job_dir):
    path = os.path.join(job_dir, "tl2d_section.nc")
    with netCDF4.Dataset(bellhop_runner._ascii_netcdf_read_path(path), "r") as dataset:
        section = {
            "range_km": _nc_array(dataset, "range_km").ravel(),
            "depth_m": _nc_array(dataset, "depth_m").ravel(),
            "bottom_depth_m": _nc_array(dataset, "bottom_depth_m").ravel(),
            "tl_db": _nc_array(dataset, "tl_db"),
        }
        if "boundary_range_m" in dataset.variables and "boundary_bottom_m" in dataset.variables:
            section["boundary_range_m"] = _nc_array(dataset, "boundary_range_m").ravel()
            section["boundary_bottom_m"] = _nc_array(dataset, "boundary_bottom_m").ravel()
        if "ssp_depth_m" in dataset.variables and "ssp_sound_speed_mps" in dataset.variables:
            section["ssp_depth_m"] = _nc_array(dataset, "ssp_depth_m").ravel()
            section["ssp_sound_speed_mps"] = _nc_array(
                dataset, "ssp_sound_speed_mps"
            ).ravel()
        if "boundary_range_m" in dataset.variables and "section_sound_speed_mps" in dataset.variables:
            section["sound_speed_range_m"] = _nc_array(
                dataset, "boundary_range_m"
            ).ravel()
            section["section_sound_speed_mps"] = _nc_array(
                dataset, "section_sound_speed_mps"
            )
        return section


def _target_depth_axis(depth_axis, bottom_depth, row):
    depths = np.asarray(depth_axis, dtype=np.float64).ravel()
    finite_depths = depths[np.isfinite(depths)]
    depth_min = float(np.min(finite_depths)) if finite_depths.size else 0.0
    depth_max = float(np.max(finite_depths)) if finite_depths.size else 1.0
    bottom_max = _finite_max(bottom_depth, depth_max)
    # 深度轴必须覆盖所有方位中的最大海深(bottom_max)，否则像 RAM 这类自身
    # 输出深度轴较浅的模型，剖面纵轴到不了最深处，与其他模型不一致。
    # 取 bottom_max；模型未算到的更深层在重采样时为缺测(NaN)。
    stop = bottom_max if bottom_max > 0.0 else depth_max
    if stop < depth_min:
        stop = depth_min + 1.0
    return _target_axis(depth_min, stop, row)


def _build_section_sampling(section, col):
    range_axis = np.asarray(section["range_km"], dtype=np.float64)
    finite_range = range_axis[np.isfinite(range_axis)]
    start = float(np.min(finite_range)) if finite_range.size else 0.0
    target_range = _target_axis(start, _finite_max(range_axis, 0.0), col)
    bottom = _resample_series(range_axis, section["bottom_depth_m"], target_range)
    bottom_for_output = -bottom
    # 地形预处理（Bellhop）把陆地点设为 1 m 水深。门户输出中必须保留该
    # 采样位置，并用统一缺测值表示，不能删除该位置。
    bottom_for_output[np.isfinite(bottom) & (bottom <= 1.0 + 1.0e-9)] = np.nan
    return {
        "size": int(col),
        "x": _json_array(target_range * 1000.0, 3),
        "y": _portal_data_array(bottom_for_output, 3),
    }


def _portal_ice_config():
    config = CONFIG.get("ICE_REFLECTION") if isinstance(CONFIG.get("ICE_REFLECTION"), Mapping) else {}
    return config


def _portal_ice_data_path(request_payload=None):
    request_payload = request_payload if isinstance(request_payload, Mapping) else {}
    mode = database_source.data_source_mode(CONFIG, "ice", "file")
    if mode == "hycom":
        time_value = str(
            _pick(
                request_payload,
                ("time",),
                database_source.default_environment_time(CONFIG),
            )
        ).strip()
        source_lon = _api_float(
            request_payload, ("coordX",), _portal_default("coordX", 0.0), "声源经度"
        )
        source_lat = _api_float(
            request_payload, ("coordY",), _portal_default("coordY", 80.0), "声源纬度"
        )
        range_nm = _api_float(
            request_payload,
            ("range",),
            _portal_default("range", 10.0),
            "传播半径",
        )
        domain_radius = float(_portal_ice_config().get("DOMAIN_RADIUS_KM", 5.0))
        radius_km = max(1.0, range_nm * NM_TO_KM, domain_radius)
        lat_span = radius_km / 111.32
        lon_span = lat_span / max(0.05, math.cos(math.radians(source_lat)))
        query_lon = np.asarray([source_lon - lon_span, source_lon + lon_span], dtype=np.float64)
        query_lat = np.asarray([source_lat - lat_span, source_lat + lat_span], dtype=np.float64)
        try:
            return hycom_service.cache_ice_subset(
                CONFIG,
                time_value=time_value,
                query_lon=query_lon,
                query_lat=query_lat,
                cache_root=os.path.join(JOBS_DIR, "hycom_ice_cache"),
            )
        except hycom_service.HycomServiceError as exc:
            raise PortalRequestError(f"HYCOM 海冰厚度请求失败：{exc}") from exc
    if mode != "file":
        try:
            return database_source.resolve_data_file(
                CONFIG,
                mode,
                "ice",
                data_source=str(_pick(request_payload, ("dataSource",), "HYCOM")),
                time_value=str(
                    _pick(
                        request_payload,
                        ("time",),
                        database_source.default_environment_time(CONFIG),
                    )
                ),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise PortalRequestError(
                f"海冰数据源 mode={mode} 定位失败：{exc}（配置文件：{CONFIG_FILE}）"
            ) from exc
    try:
        value = database_source.data_source_file(CONFIG, "ice")
    except ValueError as exc:
        raise PortalRequestError(
            f"海冰数据源 mode=file，但 DATA_SOURCES.ice.file 未配置（配置文件：{CONFIG_FILE}）。"
        ) from exc
    path = _project_path(value)
    if not os.path.isfile(path):
        source_spec = CONFIG.get("DATA_SOURCES") or {}
        ice_spec = source_spec.get("ice") if isinstance(source_spec, Mapping) else {}
        configured_value = ice_spec.get("file", value) if isinstance(ice_spec, Mapping) else value
        raise PortalRequestError(
            "海冰数据源 mode=file，本地文件不存在："
            f"{path}（DATA_SOURCES.ice.file={configured_value!r}；配置文件：{CONFIG_FILE}）"
        )
    return path


def _portal_ice_fields(path=None, source_lat=None, source_lon=None, radius_km=None):
    path = str(path or _portal_ice_data_path())
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        raise PortalRequestError(f"无法读取海冰数据文件：{path}") from exc
    # 冰厚绘图使用 _portal_ice_data_path 返回的 HYCOM 文件；反射表数据源宏
    # 只控制 .trc 的 BUMDEN 输入，不改变这条独立的冰厚查询链。
    data_source = "HYCOM"
    cache_key = (path, mtime, data_source)
    if data_source == "HYCOM":
        if source_lat is None or source_lon is None or radius_km is None:
            raise PortalRequestError("读取 HYCOM 冰厚需要声源经纬度和计算半径")
        radius_km = max(float(radius_km), 1.0)
        cache_key += (
            round(float(source_lat), 6),
            round(float(source_lon), 6),
            round(radius_km, 3),
            int(ice_reflection_trc.HYCOM_TIME_INDEX),
        )
    cached = _ICE_FIELDS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if data_source == "HYCOM":
        fields = ice_reflection_trc.derive_hycom_ice_fields(
            path,
            source_lat=float(source_lat),
            source_lon=float(source_lon),
            radius_km=radius_km,
        )
    else:
        fields = ice_reflection_trc.derive_ice_bumden_fields(path)
    _ICE_FIELDS_CACHE.clear()
    _ICE_FIELDS_CACHE[cache_key] = fields
    return fields


def _nearest_ice_thickness(
    lon,
    lat,
    path=None,
    source_lat=None,
    source_lon=None,
    radius_km=None,
):
    fields = _portal_ice_fields(
        path,
        source_lat=source_lat,
        source_lon=source_lon,
        radius_km=radius_km,
    )
    ref_lon = np.asarray(fields["lon"], dtype=np.float64).ravel()
    ref_lat = np.asarray(fields["lat"], dtype=np.float64).ravel()
    thickness = np.asarray(fields["ice_thickness"], dtype=np.float64).ravel()
    valid = np.isfinite(ref_lon) & np.isfinite(ref_lat) & np.isfinite(thickness)
    out = np.full(np.asarray(lon, dtype=np.float64).shape, np.nan, dtype=np.float64)
    if not np.any(valid):
        return out
    ref_lon = ref_lon[valid]
    ref_lat = ref_lat[valid]
    thickness = thickness[valid]
    query_lon = np.asarray(lon, dtype=np.float64).ravel()
    query_lat = np.asarray(lat, dtype=np.float64).ravel()
    values = np.full(query_lon.shape, np.nan, dtype=np.float64)
    for index, (lon_value, lat_value) in enumerate(zip(query_lon, query_lat)):
        if not (math.isfinite(float(lon_value)) and math.isfinite(float(lat_value))):
            continue
        distance = ice_reflection_trc.haversine_km(float(lat_value), float(lon_value), ref_lat, ref_lon)
        if distance.size:
            values[index] = thickness[int(np.nanargmin(distance))]
    return values.reshape(out.shape)


def _portal_ice_thickness_on_bearing(params, target_range_km, bearing_deg, land_columns):
    ranges = np.asarray(target_range_km, dtype=np.float64)
    angle = math.radians(float(bearing_deg))
    x_km = ranges * math.sin(angle)
    y_km = ranges * math.cos(angle)
    lon, lat = bellhop_results.local_offset_to_lonlat(
        x_km,
        y_km,
        float(params["center_lon"]),
        float(params["center_lat"]),
    )
    finite_ranges = np.abs(ranges[np.isfinite(ranges)])
    radius_km = float(np.max(finite_ranges)) if finite_ranges.size else 1.0
    thickness = _nearest_ice_thickness(
        lon,
        lat,
        params.get("ice_data"),
        source_lat=float(params["center_lat"]),
        source_lon=float(params["center_lon"]),
        radius_km=max(radius_km, 1.0),
    )
    land = np.asarray(land_columns, dtype=bool)
    if land.shape == thickness.shape:
        thickness = thickness.copy()
        thickness[land] = np.nan
    return thickness


def _portal_2d_target_range(section, col):
    range_axis = np.asarray(section["range_km"], dtype=np.float64)
    finite_range = range_axis[np.isfinite(range_axis)]
    range_start = float(np.min(finite_range)) if finite_range.size else 0.0
    return _target_axis(range_start, _finite_max(section["range_km"], 0.0), col)


def _build_2d_ice_thickness_response(job_dir, col, params):
    section = _read_tl2d_section(job_dir)
    target_range = _portal_2d_target_range(section, col)
    bottom = _resample_series(section["range_km"], section["bottom_depth_m"], target_range)
    land_columns = (~np.isfinite(bottom)) | (bottom <= 1.0 + 1.0e-9)
    bearing = float(params.get("selected_section_theta_deg", params.get("bearing_deg", 0.0)))
    values = _portal_ice_thickness_on_bearing(params, target_range, bearing, land_columns)
    return {
        "size": int(col),
        "x": _json_array(target_range * 1000.0, 3),
        "y": _portal_data_array(values, 3),
    }


def _build_3d_ice_thickness_response(job_dir, col, requested_sector, params):
    path = os.path.join(job_dir, "tl3d_depth_theta_range.nc")
    with netCDF4.Dataset(bellhop_runner._ascii_netcdf_read_path(path), "r") as dataset:
        range_axis = _nc_array(dataset, "range_km").ravel()
        theta_axis = _nc_array(dataset, "theta_deg").ravel()
        bottom = _nc_array(dataset, "bottom_depth_m")
    if bottom.ndim == 3:
        bottom = bottom[0]
    sector = int(len(theta_axis))
    if sector != int(requested_sector):
        requested_sector = sector
    finite_range = range_axis[np.isfinite(range_axis)]
    range_start = float(np.min(finite_range)) if finite_range.size else 0.0
    target_range = _target_axis(range_start, _finite_max(range_axis, 0.0), col)
    ice_data = []
    for isector, bearing in enumerate(theta_axis):
        bottom_profile = _resample_series(range_axis, bottom[isector, :], target_range)
        land_columns = (~np.isfinite(bottom_profile)) | (bottom_profile <= 1.0 + 1.0e-9)
        values = _portal_ice_thickness_on_bearing(params, target_range, float(bearing), land_columns)
        ice_data.append({
            "size": int(col),
            "x": _json_array(target_range * 1000.0, 3),
            "y": _portal_data_array(values, 3),
        })
    return {
        "size": sector,
        "iceDataVS": ice_data,
    }


def _build_2d_transmission_loss_response(job_dir, row, col, calc_frequency):
    section = _read_tl2d_section(job_dir)
    target_range = _portal_2d_target_range(section, col)
    target_depth = _target_depth_axis(section["depth_m"], section["bottom_depth_m"], row)
    grid = _resample_tl(
        section["range_km"], section["depth_m"], section["tl_db"],
        target_range, target_depth,
    )
    bottom = _resample_series(section["range_km"], section["bottom_depth_m"], target_range)
    below_bottom = target_depth[:, None] > bottom[None, :]
    grid[below_bottom] = np.nan
    land_columns = (~np.isfinite(bottom)) | (bottom <= 1.0 + 1.0e-9)
    return {
        "calcFrequency": calc_frequency,
        "altList": _build_section_sampling(section, col),
        "transLossData": {
            "size": int(row),
            "transmissionLossVS": {
                "profile": {
                    "x": _json_array(target_range * 1000.0, 3),
                    "z": _json_array(target_depth, 3),
                    "data": _portal_tl_data_array(grid, land_columns, below_bottom, 4),
                }
            },
        },
    }


def _build_ice_2d_transmission_loss_response(job_dir, row, col, calc_frequency, params):
    response = _build_2d_transmission_loss_response(job_dir, row, col, calc_frequency)
    return {
        "calcFrequency": response["calcFrequency"],
        "altList": response["altList"],
        "iceThickness": _build_2d_ice_thickness_response(job_dir, col, params),
        "transLossData": response["transLossData"],
    }


def _build_2d_ray_response(job_dir, col, requested_line_count, calc_frequency):
    section = _read_tl2d_section(job_dir)
    ray_path = os.path.join(job_dir, "ray2d_section.nc")
    with netCDF4.Dataset(bellhop_runner._ascii_netcdf_read_path(ray_path), "r") as dataset:
        counts = _nc_array(dataset, "ray_point_count").astype(np.int64).ravel()
        starts = _nc_array(dataset, "ray_start_index").astype(np.int64).ravel()
        ranges = _nc_array(dataset, "range_km").ravel()
        depths = _nc_array(dataset, "depth_m").ravel()

    tracks = []
    for count, start in zip(counts, starts):
        count = int(count)
        start = int(start)
        if count <= 0:
            continue
        end = start + count
        point_count = min(len(ranges[start:end]), len(depths[start:end]))
        if point_count <= 0:
            continue
        tracks.append({
            "x": _portal_data_array(ranges[start:start + point_count] * 1000.0, 6),
            "z": _portal_data_array(depths[start:start + point_count], 6),
        })
    if requested_line_count and len(tracks) > requested_line_count:
        keep = np.unique(np.linspace(0, len(tracks) - 1, requested_line_count).round().astype(int))
        tracks = [tracks[int(index)] for index in keep]
    if requested_line_count and len(tracks) < requested_line_count:
        tracks.extend(
            {"x": [PORTAL_INVALID_DATA_VALUE], "z": [PORTAL_INVALID_DATA_VALUE]}
            for _ in range(requested_line_count - len(tracks))
        )
    sampling = _build_section_sampling(section, col)
    return {
        "calcFrequency": calc_frequency,
        "sampling": {
            "size": sampling["size"],
            "data": {
                "x": sampling["x"],
                "y": sampling["y"],
            },
        },
        "trackData": {
            "size": len(tracks),
            "data": tracks,
        },
    }


def _build_ice_2d_ray_response(job_dir, col, requested_line_count, calc_frequency, params):
    response = _build_2d_ray_response(job_dir, col, requested_line_count, calc_frequency)
    return {
        "calcFrequency": response["calcFrequency"],
        "sampling": response["sampling"],
        "iceThickness": _build_2d_ice_thickness_response(job_dir, col, params),
        "trackData": response["trackData"],
    }


def _build_3d_ray_response(
    job_dir, col, requested_line_count, requested_azimuths, calc_frequency
):
    """Read the stitched multi-bearing ray product into the portal JSON shape."""
    ray_path = os.path.join(job_dir, "ray3d_wgs84.nc")
    with netCDF4.Dataset(bellhop_runner._ascii_netcdf_read_path(ray_path), "r") as dataset:
        counts = _nc_array(dataset, "ray_point_count").astype(np.int64).ravel()
        starts = _nc_array(dataset, "ray_start_index").astype(np.int64).ravel()
        xs = _nc_array(dataset, "local_x_km").ravel() * 1000.0
        ys = _nc_array(dataset, "local_y_km").ravel() * 1000.0
        depths = _nc_array(dataset, "depth_m").ravel()
        if "ray_bearing_deg" in dataset.variables:
            ray_bearings = _nc_array(dataset, "ray_bearing_deg").ravel()
        else:
            ray_bearings = np.full(len(counts), np.nan, dtype=np.float64)

    tracks_by_azimuth = {float(angle): [] for angle in requested_azimuths}
    for ray_index, (count, start) in enumerate(zip(counts, starts)):
        count, start = int(count), int(start)
        if count <= 0:
            continue
        end = start + count
        point_count = min(len(xs[start:end]), len(ys[start:end]), len(depths[start:end]))
        if point_count <= 0:
            continue
        bearing = float(ray_bearings[ray_index]) if ray_index < len(ray_bearings) else math.nan
        if not math.isfinite(bearing):
            local_x = xs[start:start + point_count]
            local_y = ys[start:start + point_count]
            nonzero = np.flatnonzero(np.hypot(local_x, local_y) > 1.0e-6)
            sample = int(nonzero[0]) if nonzero.size else max(0, point_count - 1)
            bearing = (math.degrees(math.atan2(local_x[sample], local_y[sample])) + 360.0) % 360.0
        target = min(
            tracks_by_azimuth,
            key=lambda angle: abs(((bearing - angle + 180.0) % 360.0) - 180.0),
        )
        tracks_by_azimuth[target].append({
            "azimuth": target,
            "x": _portal_data_array(xs[start:start + point_count], 6),
            "y": _portal_data_array(ys[start:start + point_count], 6),
            "z": _portal_data_array(depths[start:start + point_count], 6),
        })

    tracks = []
    sections = []
    next_id = 0
    for azimuth in requested_azimuths:
        angle = float(azimuth)
        selected = tracks_by_azimuth[angle]
        if requested_line_count and len(selected) > requested_line_count:
            keep = np.unique(
                np.linspace(0, len(selected) - 1, requested_line_count).round().astype(int)
            )
            selected = [selected[int(index)] for index in keep]
        for track in selected:
            track["id"] = next_id
            next_id += 1
        tracks.extend(selected)
        sections.append({
            "azimuth": angle,
            "rayCount": len(selected),
            "rayIds": [track["id"] for track in selected],
        })

    tl_path = os.path.join(job_dir, "tl3d_depth_theta_range.nc")
    with netCDF4.Dataset(bellhop_runner._ascii_netcdf_read_path(tl_path), "r") as dataset:
        native_range = _nc_array(dataset, "range_km").ravel()
        native_theta = _nc_array(dataset, "theta_deg").ravel()
        bottom = _nc_array(dataset, "bottom_depth_m")
        source_lon = float(_nc_array(dataset, "source_longitude").ravel()[0])
        source_lat = float(_nc_array(dataset, "source_latitude").ravel()[0])
    if bottom.ndim == 3:
        bottom = bottom[0]
    finite_range = native_range[np.isfinite(native_range)]
    range_start = float(np.min(finite_range)) if finite_range.size else 0.0
    target_range = _target_axis(range_start, _finite_max(native_range, 0.0), col)
    sampling = []
    for azimuth in requested_azimuths:
        theta_index = int(np.argmin(np.abs(((native_theta - azimuth + 180.0) % 360.0) - 180.0)))
        bottom_profile = _resample_series(native_range, bottom[theta_index], target_range)
        angle_rad = math.radians(float(azimuth))
        radial_m = target_range * 1000.0
        sampling.append({
            "azimuth": float(azimuth),
            "size": int(col),
            "x": _portal_data_array(radial_m * math.sin(angle_rad), 3),
            "y": _portal_data_array(radial_m * math.cos(angle_rad), 3),
            "z": _portal_data_array(bottom_profile, 3),
        })

    parallel = {}
    try:
        with open(
            os.path.join(job_dir, "sources", "source_001", "multibearing_2d_manifest.json"),
            "r", encoding="utf-8",
        ) as stream:
            manifest = json.load(stream)
        parallel = {
            "mode": "bearing_thread_pool",
            "bearingThreadCount": int(manifest.get("bearing_thread_count", 1)),
            "bearingCount": int(manifest.get("bearing_count", len(requested_azimuths))),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        parallel = {"mode": "bearing_thread_pool", "bearingCount": len(requested_azimuths)}

    return {
        "calcFrequency": calc_frequency,
        "coordinateSystem": {
            "type": "local_enu_depth",
            "originLongitude": source_lon,
            "originLatitude": source_lat,
            "horizontalUnit": "m",
            "depthUnit": "m",
            "xAxis": "east",
            "yAxis": "north",
            "zAxis": "positive_down_depth",
        },
        "azimuthCount": len(requested_azimuths),
        "azimuths": [float(value) for value in requested_azimuths],
        "parallel": parallel,
        "sampling": {"size": len(sampling), "data": sampling},
        "sections": sections,
        "trackData": {"size": len(tracks), "data": tracks},
    }


def _read_source_ssp(job_dir):
    try:
        with open(os.path.join(job_dir, "interactive_panels.json"), "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        source = (payload.get("sources") or [{}])[0]
        ssp = source.get("ssp") or {}
        return (
            np.asarray(ssp.get("depth_m") or [], dtype=np.float64),
            np.asarray(ssp.get("sound_speed_mps") or [], dtype=np.float64),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)


def _build_3d_transmission_loss_response(job_dir, row, col, requested_sector, calc_frequency):
    path = os.path.join(job_dir, "tl3d_depth_theta_range.nc")
    with netCDF4.Dataset(bellhop_runner._ascii_netcdf_read_path(path), "r") as dataset:
        range_axis = _nc_array(dataset, "range_km").ravel()
        depth_axis = _nc_array(dataset, "depth_m").ravel()
        theta_axis = _nc_array(dataset, "theta_deg").ravel()
        bottom = _nc_array(dataset, "bottom_depth_m")
        tl = _nc_array(dataset, "tl_db")

    if bottom.ndim == 3:
        bottom = bottom[0]
    if tl.ndim == 4:
        tl = tl[0]
    if tl.shape != (len(depth_axis), len(theta_axis), len(range_axis)):
        raise ValueError("三维传播损失数组维度与距离/深度/扇面坐标不一致")
    sector = int(len(theta_axis))
    if sector != int(requested_sector):
        requested_sector = sector

    finite_range = range_axis[np.isfinite(range_axis)]
    range_start = float(np.min(finite_range)) if finite_range.size else 0.0
    target_range = _target_axis(range_start, _finite_max(range_axis, 0.0), col)
    target_depth = _target_depth_axis(depth_axis, bottom, row)
    range_m = target_range * 1000.0
    paired = []
    tl_rows = []
    for isector in range(sector):
        bottom_profile = _resample_series(range_axis, bottom[isector, :], target_range)
        bottom_for_output = -bottom_profile
        land_columns = (~np.isfinite(bottom_profile)) | (bottom_profile <= 1.0 + 1.0e-9)
        bottom_for_output[land_columns] = np.nan
        paired.append({
            "size": int(col),
            "x": _json_array(range_m, 3),
            "y": _portal_data_array(bottom_for_output, 3),
        })
        section = np.asarray(tl[:, isector, :], dtype=np.float64).T
        grid = _resample_tl(range_axis, depth_axis, section, target_range, target_depth)
        below_bottom = target_depth[:, None] > bottom_profile[None, :]
        grid[below_bottom] = np.nan
        tl_rows.extend(_portal_tl_data_array(grid, land_columns, below_bottom, 4))

    ssp_depth, ssp_speed = _read_source_ssp(job_dir)
    source_speed = _resample_series(ssp_depth, ssp_speed, target_depth)
    return {
        "calcFrequency": calc_frequency,
        "altitudeList": {
            "size": sector,
            "pairedDataVS": paired,
        },
        "reconstructDataVS": {
            "size": int(row),
            "x": _json_array(target_depth, 3),
            "y": _json_array(source_speed, 4),
        },
        "loadingTlv": {
            "row": int(row),
            "col": int(col),
            "sector": sector,
            "size": int(row) * sector,
            "azimuths": _json_array(theta_axis, 4),
            "profile": {
                "x": _json_array(range_m, 3),
                "z": _json_array(target_depth, 3),
            },
            "data": tl_rows,
        },
    }


def _build_ice_3d_transmission_loss_response(job_dir, row, col, requested_sector, calc_frequency, params):
    response = _build_3d_transmission_loss_response(job_dir, row, col, requested_sector, calc_frequency)
    return {
        "calcFrequency": response["calcFrequency"],
        "altitudeList": response["altitudeList"],
        "iceThicknessList": _build_3d_ice_thickness_response(job_dir, col, requested_sector, params),
        "reconstructDataVS": response["reconstructDataVS"],
        "loadingTlv": response["loadingTlv"],
    }


# =============================================================================
# 环境数据接口辅助
# =============================================================================
def _json_scalar(value, decimals=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, PORTAL_DECIMAL_PLACES) if decimals is not None else number


# --------------------------------------------------------------------------- #
# 环境数据接口
# --------------------------------------------------------------------------- #
# 三个 ENV 查询接口只负责“取环境场数据并整理返回结构”，不启动 Bellhop 计算。
# 它们按请求中的 dataType 直接查询 datasetFileCatalog；下方旧的温盐/声速读取函数
# 仍供波导等需要派生声速的声学接口使用。

def _env_base_request(data):
    data_source = str(_pick(data, ("dataSource",), _portal_default("dataSource", "HYCOM"))).strip()
    data_type = str(_pick(data, ("dataType",), _portal_default("dataType", "DT_ENV_SOUND"))).strip()
    if data_source.upper() != hycom_service.HYCOM_DATA_SOURCE:
        raise PortalRequestError("dataSource 当前仅支持固定值 HYCOM。")
    if not data_type:
        raise PortalRequestError("dataType 不能为空。")
    time_value = str(
        _pick(
            data,
            ("time",),
            _portal_default("time", database_source.default_environment_time(CONFIG)),
        )
    ).strip()
    time_res = _api_int(data, ("timeRes",), _portal_default("timeRes", 0), "时间偏移量")
    if time_res != 0:
        raise PortalRequestError("timeRes 暂没有偏移值，固定为 0。")
    return {
        "dataSource": data_source,
        "dataType": data_type,
        "time": time_value,
        "timeRes": time_res,
    }


def _env_station_request(data):
    base = _env_base_request(data)
    lon, lat = _env_lon_lat(data, ("coordX",), ("coordY",), "站点")
    is_reconstruct = _api_bool(data, ("isReconstruct",), False)
    if is_reconstruct:
        raise PortalRequestError("isReconstruct 暂不重构，固定为 false。")
    rdata_source = str(_pick(data, ("rdataSource",), "HYCOM")).strip()
    if rdata_source.upper() != hycom_service.HYCOM_DATA_SOURCE:
        raise PortalRequestError("rdataSource 当前仅支持固定值 HYCOM。")
    rtime = str(_pick(data, ("rtime",), base["time"])).strip()
    if rtime != base["time"]:
        raise PortalRequestError("rtime 需要与 time 字段保持一致。")
    rtime_res = _api_int(data, ("rtimeRes",), 0, "重构时间偏移量")
    if rtime_res != 0:
        raise PortalRequestError("rtimeRes 暂没有偏移值，固定为 0。")
    base.update({
        "lon": lon,
        "lat": lat,
        "isReconstruct": False,
        "rdataSource": rdata_source,
        "rtime": rtime,
        "rtimeRes": rtime_res,
    })
    return base


def _env_lon_lat(data, lon_names, lat_names, label):
    lon_raw = _api_float(data, lon_names, None, f"{label}经度")
    lat = _api_float(data, lat_names, None, f"{label}纬度")
    if not -360.0 <= lon_raw <= 360.0:
        raise PortalRequestError(f"{label}经度必须位于 -360 到 360°。")
    if not -90.0 <= lat <= 90.0:
        raise PortalRequestError(f"{label}纬度必须位于 -90 到 90°。")
    return _normalise_lon(lon_raw), lat


def _env_depth_axis(*source_axes):
    configured = np.asarray(CONFIG.get("DEPTH_GRID_M") or [], dtype=np.float64)
    finite_axes = [
        np.asarray(axis, dtype=np.float64)[np.isfinite(np.asarray(axis, dtype=np.float64))]
        for axis in source_axes
    ]
    finite_axes = [axis for axis in finite_axes if axis.size]
    if not finite_axes:
        raise PortalRequestError("环境数据缺少有效深度轴。")
    max_depth = min(float(np.max(axis)) for axis in finite_axes)
    if configured.ndim == 1 and configured.size:
        depth = configured[np.isfinite(configured) & (configured >= 0.0) & (configured <= max_depth)]
        depth = np.unique(depth)
        if depth.size:
            return depth.astype(np.float64)
    depth = finite_axes[0][(finite_axes[0] >= 0.0) & (finite_axes[0] <= max_depth)]
    depth = np.unique(depth)
    if not depth.size:
        raise PortalRequestError("环境数据深度轴与配置深度范围没有交集。")
    return depth.astype(np.float64)


def _env_single_depth_axis(z, *source_axes):
    depth = _env_depth_axis(*source_axes)
    if z < 0.0:
        raise PortalRequestError("查询深度 z 不能小于 0。")
    if z > float(depth[-1]):
        raise PortalRequestError(f"查询深度 z={z:g} m 超出环境数据最大深度 {float(depth[-1]):g} m。")
    return np.asarray([float(z)], dtype=np.float64)


def _env_read_profiles(lon, lat, data_type, time_value, target_depth=None, data_source="HYCOM"):
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    # 查询上下文作为本次读取参数传入，不再修改或复制全局配置。
    access = bellhop_runner._DataAccess(
        {"environment_time": time_value, "data_source": data_source},
        CONFIG,
        # 环境数据查询只读取温度、盐度或声速，不需要初始化海底地声数据源。
        bottom_geo_config={"ENABLED": False},
    )
    try:
        access.validate()
        if data_type == "DT_ENV_TEMP":
            temp_cube = access.read_hycom_cube("temperature", ["water_temp", "temperature", "thetao"], lon_arr, lat_arr)
            depth = target_depth if target_depth is not None else _env_depth_axis(temp_cube["depth"])
            values = bellhop_runner._profile_cube_to_queries(temp_cube, depth, lon_arr, lat_arr)
        elif data_type == "DT_ENV_SALT":
            salt_cube = access.read_hycom_cube("salinity", ["salinity", "so"], lon_arr, lat_arr)
            depth = target_depth if target_depth is not None else _env_depth_axis(salt_cube["depth"])
            values = bellhop_runner._profile_cube_to_queries(salt_cube, depth, lon_arr, lat_arr)
        else:
            temp_cube = access.read_hycom_cube("temperature", ["water_temp", "temperature", "thetao"], lon_arr, lat_arr)
            salt_cube = access.read_hycom_cube("salinity", ["salinity", "so"], lon_arr, lat_arr)
            depth = target_depth if target_depth is not None else _env_depth_axis(temp_cube["depth"], salt_cube["depth"])
            temp = bellhop_runner._profile_cube_to_queries(temp_cube, depth, lon_arr, lat_arr)
            salt = bellhop_runner._profile_cube_to_queries(salt_cube, depth, lon_arr, lat_arr)
            values = bellhop_runner._mackenzie_sound_speed(temp, salt, depth[:, None])
    finally:
        access.close()
    return np.asarray(depth, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _env_profile_cube_to_queries(cube, target_depth, query_lon, query_lat):
    """按最近水平网格采样任意环境字段，并兼容单层或少层数据。"""
    depth = np.asarray(cube["depth"], dtype=np.float64).ravel()
    values = np.asarray(cube["values"], dtype=np.float64)
    target = np.asarray(target_depth, dtype=np.float64).ravel()
    query_lat = np.asarray(query_lat, dtype=np.float64).ravel()
    query_lon = bellhop_runner._unwrap_lon_for_cube(cube, query_lon).ravel()
    cube_lat = np.asarray(cube["lat"], dtype=np.float64).ravel()
    cube_lon = np.asarray(cube["lon"], dtype=np.float64).ravel()
    lat_index = (
        np.zeros(query_lat.shape, dtype=np.int64)
        if len(cube_lat) == 1
        else bellhop_runner._nearest_indices(cube_lat, query_lat)
    )
    lon_index = (
        np.zeros(query_lon.shape, dtype=np.int64)
        if len(cube_lon) == 1
        else bellhop_runner._nearest_indices(cube_lon, query_lon)
    )
    result = np.full((len(target), len(query_lat)), np.nan, dtype=np.float64)

    for point, (iy, ix) in enumerate(zip(lat_index, lon_index)):
        profile = values[:, int(iy), int(ix)]
        good = np.isfinite(profile) & np.isfinite(depth)
        if not np.any(good):
            continue
        source_depth = depth[good]
        source_values = profile[good]
        order = np.argsort(source_depth)
        source_depth = source_depth[order]
        source_values = source_values[order]
        source_depth, unique_index = np.unique(source_depth, return_index=True)
        source_values = source_values[unique_index]
        if len(source_depth) == 1:
            result[:, point] = source_values[0]
        else:
            result[:, point] = np.interp(
                target,
                source_depth,
                source_values,
                left=source_values[0],
                right=source_values[-1],
            )
    return result


def _env_read_local_bottom_geoacoustics(lon, lat, data_type):
    """按配置的本地 NetCDF 文件和变量名读取航线底质参数。"""
    source = database_source.data_source_config(CONFIG, "bottom_geoacoustics")
    if database_source.data_source_mode(CONFIG, "bottom_geoacoustics", "file") != "file":
        raise PortalRequestError(
            "DATA_CATALOG.mode=file 时要求 DATA_SOURCES.bottom_geoacoustics.mode=file。"
        )
    variables = source.get("variables")
    if not isinstance(variables, Mapping):
        raise PortalRequestError(
            "data_config.json 缺少 DATA_SOURCES.bottom_geoacoustics.variables 配置。"
        )
    configured = variables.get(str(data_type))
    candidates = (
        [configured]
        if isinstance(configured, str)
        else [str(value) for value in configured]
        if isinstance(configured, Sequence)
        else []
    )
    candidates = [value.strip() for value in candidates if str(value).strip()]
    if not candidates:
        raise PortalRequestError(
            f"data_config.json 缺少本地底质变量映射：{data_type}"
        )
    try:
        path = database_source.data_source_file(CONFIG, "bottom_geoacoustics")
        values = bellhop_runner._read_scalar_grid_field(
            path,
            candidates,
            np.asarray(lon, dtype=np.float64),
            np.asarray(lat, dtype=np.float64),
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        raise PortalRequestError(f"本地底质数据 {data_type} 读取失败：{exc}") from exc
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if data_type == "DT_ENV_GEOACOUSTIC_CLASSID":
        values = np.where(np.isfinite(values) & (values > 0.0), np.rint(values), np.nan)
    return np.asarray([0.0], dtype=np.float64), values.reshape(1, -1)


def _env_read_query_profiles(
    lon,
    lat,
    data_type,
    time_value,
    target_depth=None,
    data_source="HYCOM",
):
    """温盐/声速走 HYCOM；底质可按 DATA_CATALOG 开关读本地文件。"""
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    if data_type == "DT_ENV_SOUND":
        return _env_read_profiles(
            lon_arr,
            lat_arr,
            data_type,
            time_value,
            target_depth=target_depth,
            data_source=hycom_service.HYCOM_DATA_SOURCE,
        )
    kind_by_data_type = {
        "DT_ENV_TEMP": "temperature",
        "DT_ENV_SALT": "salinity",
        "DT_ENV_ICE_THICKNESS": "ice",
    }
    source_kind = kind_by_data_type.get(data_type)
    source_mode = (
        database_source.data_source_mode(CONFIG, source_kind, "hycom")
        if source_kind
        else "api1"
    )
    catalog_mode = catalog_queries.catalog_mode(CONFIG)
    if catalog_mode == "file" and data_type in ENV_BOTTOM_GEOACOUSTIC_FIELDS:
        return _env_read_local_bottom_geoacoustics(lon_arr, lat_arr, data_type)
    if catalog_mode == "file" and source_kind is None:
        raise PortalRequestError(
            f"DATA_CATALOG.mode=file，但数据类型 {data_type} 未配置本地文件读取规则。"
        )
    reader = None
    try:
        if source_mode == "file" and source_kind:
            source = database_source.data_source_config(CONFIG, source_kind)
            path = database_source.data_source_file(CONFIG, source_kind, time_value)
            variable_name = str(source.get("hycom_variable") or data_type).strip()
            if data_type == "DT_ENV_ICE_THICKNESS":
                values = bellhop_runner._read_scalar_grid_field(
                    path, [variable_name, "sih", "ice_thickness"], lon_arr, lat_arr
                )
                return (
                    np.asarray([0.0], dtype=np.float64),
                    np.asarray(values, dtype=np.float64).reshape(1, -1),
                )
            candidates = [
                variable_name,
                "water_temp" if data_type == "DT_ENV_TEMP" else "salinity",
            ]
            cube = bellhop_runner._read_hycom_cube(path, candidates, lon_arr, lat_arr)
        else:
            reader = database_source.create_data_reader(
                CONFIG,
                source_mode,
                # postgres 目录模式仅为将来恢复目录库时保留。
                data_source=hycom_service.HYCOM_DATA_SOURCE if source_kind else "EnvReanalysis",
                time_value=time_value,
            )
            tag = (
                reader.temperature_tag
                if data_type == "DT_ENV_TEMP"
                else reader.salinity_tag
                if data_type == "DT_ENV_SALT"
                else reader.ice_tag
                if data_type == "DT_ENV_ICE_THICKNESS"
                else data_type
            )
            cube = reader.read_profile_cube(tag, lon_arr, lat_arr)
        source_depth = np.asarray(cube.get("depth", []), dtype=np.float64).ravel()
        source_depth = source_depth[np.isfinite(source_depth)]
        if not source_depth.size:
            raise ValueError("数据文件缺少有效深度轴。")
        depth = (
            np.asarray(target_depth, dtype=np.float64)
            if target_depth is not None
            else _env_depth_axis(source_depth)
        )
        values = _env_profile_cube_to_queries(cube, depth, lon_arr, lat_arr)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise PortalRequestError(f"环境数据 {data_type} 读取失败：{exc}") from exc
    finally:
        if reader is not None:
            reader.close()
    return np.asarray(depth, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _env_profile_pairs(depth, values):
    return [
        {"x": _json_scalar(z, 3), "y": _portal_data_scalar(value, 4)}
        for z, value in zip(np.asarray(depth).ravel(), np.asarray(values).ravel())
    ]


def _env_route_points(data):
    raw_points = data.get("points")
    if not isinstance(raw_points, list):
        raise PortalRequestError("points 必须是航线点数组。")
    if not raw_points:
        raise PortalRequestError("points 至少需要 1 个航线点。")
    if len(raw_points) > 10000:
        raise PortalRequestError("points 数量过大，请控制在 10000 个以内。")
    lon_values = []
    lat_values = []
    for index, point in enumerate(raw_points, 1):
        if not isinstance(point, Mapping):
            raise PortalRequestError(f"第 {index} 个航线点必须是对象。")
        lon, lat = _env_lon_lat(point, ("x",), ("y",), f"第 {index} 个航线点")
        lon_values.append(lon)
        lat_values.append(lat)
    return np.asarray(lon_values, dtype=np.float64), np.asarray(lat_values, dtype=np.float64)


def _env_topography_path(data_source="HYCOM", time_value=""):
    mode = database_source.data_source_mode(CONFIG, "topography", "file")
    if mode == "file":
        return _project_path(database_source.data_source_file(CONFIG, "topography"))
    return database_source.resolve_data_file(
        CONFIG,
        mode,
        "topography",
        data_source=str(data_source or "HYCOM"),
        time_value=str(time_value or database_source.default_environment_time(CONFIG)),
    )


def _env_bottom_depths(lon, lat, data_source="HYCOM", time_value=""):
    topo_file = _env_topography_path(data_source, time_value)
    topo = bellhop_runner.TopographyReader(topo_file)
    elevation = topo.sample(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
    depth = np.where(np.isfinite(elevation), np.maximum(0.0, -elevation), np.nan)
    return depth.astype(np.float64)


def _env_invalid_location_mask(lon, lat, data_source="HYCOM", time_value=""):
    """判断每个查询点是否为陆地，或是否缺少可用地形数据。"""
    topo_file = _env_topography_path(data_source, time_value)
    topo = bellhop_runner.TopographyReader(topo_file)
    elevation = topo.sample(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
    return (~np.isfinite(elevation)) | (elevation >= 0.0)


def _env_region_request(data):
    base = _env_base_request(data)
    bottom_left_lon, bottom_left_lat = _env_lon_lat(
        data, ("bottomLeftX",), ("bottomLeftY",), "左下角"
    )
    top_right_lon, top_right_lat = _env_lon_lat(
        data, ("topRightX",), ("topRightY",), "右上角"
    )
    if top_right_lon <= bottom_left_lon:
        raise PortalRequestError("区域经度范围必须满足 bottomLeftX < topRightX，暂不支持跨日期变更线区域。")
    if top_right_lat <= bottom_left_lat:
        raise PortalRequestError("区域纬度范围必须满足 bottomLeftY < topRightY。")
    z = _api_float(data, ("z",), None, "查询深度 z")
    base.update({
        "bottomLeftX": bottom_left_lon,
        "bottomLeftY": bottom_left_lat,
        "topRightX": top_right_lon,
        "topRightY": top_right_lat,
        "z": z,
    })
    return base


def _build_env_station_response(request_payload):
    lon = np.asarray([request_payload["lon"]], dtype=np.float64)
    lat = np.asarray([request_payload["lat"]], dtype=np.float64)
    depth, values = _env_read_query_profiles(
        lon, lat, request_payload["dataType"], request_payload["time"], data_source=request_payload["dataSource"]
    )
    profile = values[:, 0] if values.ndim == 2 else values.ravel()
    if request_payload["dataType"] == "DT_ENV_RAIN":
        # ERA5 total precipitation (tp) is stored in metres of water equivalent,
        # while the portal's precipitation value is expressed in millimetres.
        # Convert at the API boundary so every station response uses the portal
        # unit without changing the source data shared by other calculations.
        profile = np.asarray(profile, dtype=np.float64) * 1000.0
    if request_payload["dataType"] == "DT_ENV_ICE_THICKNESS":
        profile = np.asarray(profile, dtype=np.float64).copy()
        is_land = bool(
            _env_invalid_location_mask(
                lon,
                lat,
                request_payload["dataSource"],
                request_payload["time"],
            )[0]
        )
        if is_land:
            # 陆地保留门户协议的缺测标记 9999。
            profile[:] = np.nan
        else:
            # HYCOM 在海冰浓度为 0 的开阔水域通常将 sih 写成填充值。
            # 对海洋站点，这种情况以及其他冰厚缺测都按 0 m 返回。
            profile[~np.isfinite(profile)] = 0.0
    return {"size": int(len(depth)), "data": _env_profile_pairs(depth, profile)}


def _build_env_route_response(request_payload):
    lon, lat = _env_route_points(request_payload)
    depth, values = _env_read_query_profiles(
        lon, lat, request_payload["dataType"], request_payload["time"], data_source=request_payload["dataSource"]
    )
    bottom_depth = _env_bottom_depths(
        lon, lat, request_payload["dataSource"], request_payload["time"]
    )
    invalid_locations = _env_invalid_location_mask(
        lon, lat, request_payload["dataSource"], request_payload["time"]
    )
    values = np.asarray(values, dtype=np.float64).copy()
    # 底质网格本身包含陆地/缺测标记，应以底质文件为准。地形数据覆盖不足或
    # 两套网格海岸线略有差异时，不能把已读取成功的底质参数再次抹成缺测。
    if request_payload["dataType"] not in ENV_BOTTOM_GEOACOUSTIC_FIELDS:
        if values.ndim == 1:
            values[invalid_locations] = np.nan
        else:
            values[:, invalid_locations] = np.nan
    bottom_depth[invalid_locations] = np.nan
    profile_depth = _json_array(depth, 3)
    profiles = [
        {
            "size": int(len(depth)),
            "x": list(profile_depth),
            "y": _portal_data_array(values[:, index], 4),
        }
        for index in range(len(lon))
    ]
    return {
        "SamplingPoints": {
            "size": int(len(lon)),
            "x": _json_array(lon, 6),
            "y": _json_array(lat, 6),
        },
        "Depths": _portal_data_array(bottom_depth, 3),
        "ProfileData": {
            "size": int(len(lon)),
            "data": profiles,
        },
    }


def _build_env_region_response(request_payload):
    row, col = _configured_response_counts("ENV_REGION")
    lon_axis = np.linspace(request_payload["bottomLeftX"], request_payload["topRightX"], col, dtype=np.float64)
    lat_axis = np.linspace(request_payload["bottomLeftY"], request_payload["topRightY"], row, dtype=np.float64)
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)

    # 先探测字段是否带深度轴；二维标量场按单层数据返回，不要求 z=0。
    probe_depth, _ = _env_read_query_profiles(
        np.asarray([request_payload["bottomLeftX"]]),
        np.asarray([request_payload["bottomLeftY"]]),
        request_payload["dataType"],
        request_payload["time"],
        data_source=request_payload["dataSource"],
    )
    target_depth = (
        probe_depth.copy()
        if len(probe_depth) == 1
        else _env_single_depth_axis(request_payload["z"], probe_depth)
    )
    _, values = _env_read_query_profiles(
        lon_grid.ravel(), lat_grid.ravel(),
        request_payload["dataType"], request_payload["time"], target_depth,
        data_source=request_payload["dataSource"],
    )
    grid = values[0].reshape(row, col)
    invalid_locations = _env_invalid_location_mask(
        lon_grid, lat_grid, request_payload["dataSource"], request_payload["time"]
    )
    grid[invalid_locations] = np.nan
    return {
        "row": int(row),
        "col": int(col),
        "size": int(row) * int(col),
        # data 包含 row 个子数组，每个子数组包含 col 个值；
        # 陆地或缺测位置保留在原网格位置并填入 9999.0。
        "data": _portal_data_array(grid, 4),
    }


def _portal_failure(exc, status, meta=None, plugin_name=None):
    payload = {"ok": False, "message": str(exc)}
    if meta and plugin_name:
        return _api_json_response(
            _wrap_portal_response(payload, meta, plugin_name, result_code=1), status
        )
    return _api_json_response(_flat_portal_response(payload, result_code=1), status)


# =============================================================================
# 门户 API 统一入口
# =============================================================================
from backworker.api_endpoints.dispatcher import dispatch_async as _dispatch_portal_request_async
from backworker.api_endpoints.get_weather_forecast import create_router as _create_weather_forecast_router
from backworker.api_endpoints.get_regional_weather_forecast import create_router as _create_regional_weather_forecast_router
from backworker.api_endpoints.get_realtime_ais import create_router as _create_realtime_ais_router
from backworker.api_endpoints.analyse_forecast_environment_noise import create_router as _create_forecast_environment_noise_router
from backworker.api_endpoints.analyse_regional_forecast_environment_noise import create_router as _create_regional_forecast_environment_noise_router
from backworker.api_endpoints.get_regional_ocean_current import create_router as _create_regional_ocean_current_router
from backworker.api_endpoints.get_seafloor_features import create_router as _create_seafloor_features_router

@app.post("/", summary="当前前端统一门户接口")
async def _unified_portal_endpoint(request: Request):
    payload = await _request_payload(request)
    return await _dispatch_portal_request_async(payload)


app.include_router(_create_weather_forecast_router())
app.include_router(_create_regional_weather_forecast_router())
app.include_router(_create_realtime_ais_router())
app.include_router(_create_forecast_environment_noise_router())
app.include_router(_create_regional_forecast_environment_noise_router())
app.include_router(_create_regional_ocean_current_router())
app.include_router(_create_seafloor_features_router())


# =============================================================================
# 海洋混响模型接口（海面 / 海底 / 体积三个独立模块）
# =============================================================================
@app.get("/reverberation/parameters", tags=["Ocean reverberation"])
def reverberation_parameters():
    """返回参数说明和一份可以直接运行的默认请求。"""
    return {
        "ok": True,
        "model_version": _REVERB_MODULE.MODEL_VERSION,
        "capabilities": {
            "scene_endpoint": True,
            "bellhop_tl2d_section": True,
            "ray2d_geometry": True,
            "fixed_surface_bottom_cells": True,
            "fixed_volume_voxels": True,
            "bistatic_return_proxy": False,
            "causal_pulse_integration": True,
            "strict_missing_field_mask": True,
            "segment_voxel_intersections": True,
            "per_arrival_amplitudes": False,
            "boundary_tl_water_side_proxy": True,
            "time_depth_matrix": True,
            "continuous_scattering_paths": True,
            "receiver_origin_reciprocal_paths": True,
            "multipath_scattering_pairs": True,
            "dual_bellhop_bistatic_pairs": True,
            "receiver_origin_return_tl": True,
            "fractional_delay_binning": True,
            "relative_decay_output": True,
            "model_diagnostics": True,
            "noncoincident_monostatic_return_paths": True,
            "scattering_angle_fields": True,
            "scattering_paths_netcdf_output": True,
            "netcdf_time_depth_output": True,
            "response_shape": "surface_bottom_volume_total_rl_db_time_depth_scattering_paths",
        },
        "design_note": (
            "v6：单次散射固定面元/体素积分，使用双端 Bellhop 总场和保存的射线几何。"
            "按到达时延与发射脉冲窗因果积分；缺失场/射线不再用球面扩展或直线补齐。"
            "边界采用水侧总场代理，单元路径等权分配；不是逐到达幅度的精确求解。"
            "分置结果只覆盖两端共同剖面，必须检查射线覆盖率并做网格/声线数收敛验证。"
        ),
        "parameter_catalog": REVERBERATION_PARAMETER_CATALOG,
        "defaults": REVERBERATION_DEFAULT_REQUEST,
    }


def _scene_distance_and_bearing(source_lon, source_lat, receiver_lon, receiver_lat):
    x_km, y_km = bellhop_results.lonlat_to_local_offset(
        receiver_lon, receiver_lat, source_lon, source_lat
    )
    x = float(np.asarray(x_km).reshape(-1)[0])
    y = float(np.asarray(y_km).reshape(-1)[0])
    distance_km = float(math.hypot(x, y))
    bearing_deg = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
    return distance_km, bearing_deg


def _finite_median(values, default):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.median(finite)) if finite.size else float(default)

# =============================================================================
# 海洋混响模型场景同步接口
# =============================================================================
def _reverberation_scene_sync(payload):
    if not isinstance(payload, Mapping):
        raise HTTPException(status_code=400, detail="请求体必须是 JSON 对象")
    environment = payload.get("environment") if isinstance(payload.get("environment"), Mapping) else {}
    common_input = dict(payload.get("common") if isinstance(payload.get("common"), Mapping) else {})
    defaults = CONFIG["TASK_DEFAULTS"]

    source_lat = _api_float(environment, ("source_lat", "sourceLat"), defaults["center_lat"], "声源纬度")
    source_lon = _normalise_lon(_api_float(environment, ("source_lon", "sourceLon"), defaults["center_lon"], "声源经度"))
    receiver_mode = _lower_text(environment.get("receiver_mode") or environment.get("receiverMode"), "monostatic")
    receiver_depth = _api_float(
        environment, ("receiver_depth_m", "receiverDepthM"),
        common_input.get("receiver_depth_m", common_input.get("source_depth_m", defaults["source_depth_m"])),
        "接收深度",
    )
    source_depth = _api_float(
        environment, ("source_depth_m", "sourceDepthM"),
        common_input.get("source_depth_m", defaults["source_depth_m"]),
        "声源深度",
    )
    if source_depth <= 0.0 or receiver_depth <= 0.0:
        raise HTTPException(status_code=400, detail="声源深度和接收深度必须为正数")

    if receiver_mode == "remote":
        receiver_lat = _api_float(environment, ("receiver_lat", "receiverLat"), source_lat, "接收点纬度")
        receiver_lon = _normalise_lon(_api_float(environment, ("receiver_lon", "receiverLon"), source_lon, "接收点经度"))
        receiver_range_km, bearing_deg = _scene_distance_and_bearing(source_lon, source_lat, receiver_lon, receiver_lat)
        if receiver_range_km <= 1.0e-6:
            receiver_mode = "monostatic"
    else:
        receiver_mode = "monostatic"
        receiver_lat = source_lat
        receiver_lon = source_lon
        bearing_deg = _api_float(environment, ("bearing_deg", "bearingDeg", "azimuth"), 0.0, "方位角") % 360.0
        receiver_range_km = 0.0

    time_value = str(
        _pick(
            environment,
            ("time",),
            _portal_default("time", database_source.default_environment_time(CONFIG)),
        )
    ).strip()
    frequency_hz = int(round(_api_float(common_input, ("frequency_hz",), 5000.0, "频率")))
    sound_speed_mps = _api_float(common_input, ("sound_speed_mps",), 1500.0, "代表声速")
    max_time_s = _api_float(common_input, ("max_time_s",), 4.0, "仿真时长")
    pulse_width_s = _api_float(common_input, ("pulse_width_s",), 0.02, "脉冲宽度")
    time_step_s = _api_float(common_input, ("time_step_s",), 0.02, "时间步长")
    if max_time_s <= 0 or pulse_width_s <= 0 or time_step_s <= 0 or time_step_s > max_time_s or max_time_s/time_step_s > 100000:
        raise HTTPException(status_code=400, detail="时长、脉宽、时间步长须为正，且时间分箱不超过 100000")
    one_way_window_km = sound_speed_mps * max_time_s / 1000.0
    if receiver_range_km > 0.0:
        # Bistatic integration is restricted to the common section between the
        # transmitter and receiver, so the output time window must not inflate
        # the horizontal Bellhop section by many kilometres.
        bellhop_range_km = max(2.0, receiver_range_km + 2.0)
    else:
        bellhop_range_km = max(2.0, 0.5 * one_way_window_km + 2.0)

    try:
        ssp_depth, ssp_values = _env_read_profiles(
            np.asarray([source_lon], dtype=np.float64),
            np.asarray([source_lat], dtype=np.float64),
            "DT_ENV_SOUND",
            time_value,
        )
        source_ssp = np.asarray(ssp_values[:, 0] if ssp_values.ndim == 2 else ssp_values, dtype=np.float64)
        bottom_at_points = _env_bottom_depths(
            np.asarray([source_lon, receiver_lon], dtype=np.float64),
            np.asarray([source_lat, receiver_lat], dtype=np.float64),
            "HYCOM",
            time_value,
        )
    except PortalRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"环境数据读取失败：{exc}") from exc

    water_depth_m = max(_finite_median(bottom_at_points, common_input.get("water_depth_m", 300.0)), source_depth + 5.0, receiver_depth + 5.0)
    mean_sound_speed = _finite_median(source_ssp, sound_speed_mps)

    portal_request = {
        "coordX": source_lon,
        "coordY": source_lat,
        "dataSource": "HYCOM",
        "dataType": "DT_ENV_SOUND",
        "time": time_value,
        "timeRes": 0,
        "rdataSource": "HYCOM",
        "rtime": time_value,
        "rtimeRes": 0,
        "srcDepth": source_depth,
        "range": bellhop_range_km / NM_TO_KM,
        "minFreq": frequency_hz,
        "maxFreq": frequency_hz,
        "grazingAngle": 0,
        "beamWidth": _api_float(common_input, ("bellhop_beam_width_deg",), 60.0, "混响射线发射角宽度"),
        "seaState": int(round(float((payload.get("surface") or {}).get("sea_state", 3)))) if isinstance(payload.get("surface"), Mapping) else 3,
        "coherent": False,
        "isWaterChanged": True,
        "azimuth": bearing_deg,
        "lineCount": _api_int(common_input, ("bellhop_ray_count",), 401, "混响射线数"),
    }
    try:
        params, row, col = _portal_2d_tl_params(portal_request)
        params["n_alpha"] = _api_int(common_input, ("bellhop_ray_count",), 401, "混响射线数")
        params["portal_request"]["NBEAMS"] = params["n_alpha"]
        job_id, job_dir, _ = _run_portal_job(params)
        propagation = BellhopSectionPropagation.from_netcdf(os.path.join(job_dir, "tl2d_section.nc"))
        ray_section_file = os.path.join(job_dir, "ray2d_section.nc")
        if os.path.isfile(ray_section_file) and hasattr(propagation, "load_rays"):
            propagation.load_rays(ray_section_file)
        if propagation.ssp_depth_m is None:
            ssp_valid = np.isfinite(ssp_depth) & np.isfinite(source_ssp) & (source_ssp > 0)
            propagation.ssp_depth_m = np.asarray(ssp_depth)[ssp_valid]
            propagation.ssp_sound_speed_mps = source_ssp[ssp_valid]
        receiver_propagation = None
        receiver_job_id = ""
        if receiver_range_km > 1.0e-6 or abs(receiver_depth-source_depth) > 1.0e-6:
            receiver_portal_request = dict(portal_request)
            receiver_portal_request.update({
                "coordX": receiver_lon,
                "coordY": receiver_lat,
                "srcDepth": receiver_depth,
                "azimuth": (bearing_deg + (180.0 if receiver_range_km > 1.0e-6 else 0.0)) % 360.0,
            })
            receiver_params, _, _ = _portal_2d_tl_params(receiver_portal_request)
            receiver_params["n_alpha"] = params["n_alpha"]
            receiver_params["portal_request"]["NBEAMS"] = params["n_alpha"]
            receiver_job_id, receiver_job_dir, _ = _run_portal_job(receiver_params)
            receiver_propagation = BellhopSectionPropagation.from_netcdf(os.path.join(receiver_job_dir, "tl2d_section.nc"))
            receiver_ray_section_file = os.path.join(receiver_job_dir, "ray2d_section.nc")
            if os.path.isfile(receiver_ray_section_file) and hasattr(receiver_propagation, "load_rays"):
                receiver_propagation.load_rays(receiver_ray_section_file)
        section = _read_tl2d_section(job_dir)
        reverberation_request = {
            "common": {
                **common_input,
                "frequency_hz": frequency_hz,
                "source_depth_m": source_depth,
                "receiver_depth_m": receiver_depth,
                "receiver_range_m": receiver_range_km * 1000.0,
                "water_depth_m": water_depth_m,
                "sound_speed_mps": mean_sound_speed,
                "bellhop_job_id": job_id,
                "receiver_origin_bellhop_job_id": receiver_job_id,
            },
            "surface": payload.get("surface") if isinstance(payload.get("surface"), Mapping) else {},
            "bottom": payload.get("bottom") if isinstance(payload.get("bottom"), Mapping) else {},
            "volume": payload.get("volume") if isinstance(payload.get("volume"), Mapping) else {},
        }
        result = simulate_reverberation(reverberation_request, propagation=propagation, receiver_propagation=receiver_propagation)
        reverb_nc = os.path.join(job_dir, "reverberation_v6_time_depth.nc")
        paths_nc = os.path.join(job_dir, "reverberation_v6_scattering_paths.nc")
        write_reverberation_netcdf(result, reverb_nc)
        if bool((reverberation_request.get("common") or {}).get("save_scattering_paths_netcdf", True)):
            write_scattering_paths_netcdf(result, paths_nc)
        result["result_files"] = {
            "reverberation_time_depth_nc": os.path.basename(reverb_nc),
            "reverberation_time_depth_nc_endpoint": f"/file/{job_id}/{os.path.basename(reverb_nc)}",
            "scattering_paths_nc": os.path.basename(paths_nc) if os.path.isfile(paths_nc) else None,
            "scattering_paths_nc_endpoint": f"/file/{job_id}/{os.path.basename(paths_nc)}" if os.path.isfile(paths_nc) else None,
            "download_results_endpoint": f"/download_results/{job_id}",
        }
    except HTTPException:
        raise
    except PortalRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"混响参数或 Bellhop 数据不匹配：{exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Bellhop 场景计算失败：{exc}") from exc

    result["scene"] = {
        "bellhop_job_id": job_id,
        "receiver_origin_bellhop_job_id": receiver_job_id,
        "receiver_mode": receiver_mode,
        "source": {"lat": round(source_lat, 6), "lon": round(source_lon, 6), "depth_m": round(source_depth, 3)},
        "receiver": {"lat": round(receiver_lat, 6), "lon": round(receiver_lon, 6), "depth_m": round(receiver_depth, 3)},
        "receiver_range_m": round(receiver_range_km * 1000.0, 3),
        "bearing_deg": round(bearing_deg, 3),
        "bellhop_range_km": round(bellhop_range_km, 3),
        "water_depth_m": round(water_depth_m, 3),
        "mean_sound_speed_mps": round(mean_sound_speed, 3),
        "environment_time": time_value,
        "sound_speed_profile": {
            "depth_m": _json_array(ssp_depth, 3),
            "sound_speed_mps": _json_array(source_ssp, 4),
        },
        "bathymetry_profile": {
            "range_m": _json_array(np.asarray(section["range_km"], dtype=np.float64) * 1000.0, 3),
            "bottom_depth_m": _json_array(section["bottom_depth_m"], 3),
            "native_range_m": _json_array(section.get("boundary_range_m", np.asarray(section["range_km"], dtype=np.float64) * 1000.0), 3),
            "native_bottom_depth_m": _json_array(section.get("boundary_bottom_m", section["bottom_depth_m"]), 3),
            "native_boundary_source": "Bellhop boundary_range_m/boundary_bottom_m" if "boundary_bottom_m" in section else "TL grid bottom_depth_m",
        },
    }
    return result


@app.post("/reverberation/scene", tags=["Ocean reverberation"])
async def reverberation_scene(request: Request):
    """按经纬度读取环境、运行 Bellhop，再计算模块化混响。"""
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    return await asyncio.to_thread(_reverberation_scene_sync, payload)


@app.post("/reverberation/simulate", tags=["Ocean reverberation"])
async def reverberation_simulate(request: Request):
    """运行模块化混响；可选读取某个 Bellhop 任务的二维传播损失剖面。"""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="请求体必须是 JSON") from exc
    if not isinstance(payload, Mapping):
        raise HTTPException(status_code=400, detail="请求 JSON 必须是对象")

    propagation = None
    receiver_propagation = None
    common = payload.get("common") if isinstance(payload.get("common"), Mapping) else {}
    bellhop_job_id = _text(common.get("bellhop_job_id"))
    if bellhop_job_id:
        if not re.fullmatch(r"[A-Za-z0-9_-]{4,80}", bellhop_job_id):
            raise HTTPException(status_code=400, detail="Bellhop 任务号格式无效")
        section_file = os.path.join(JOBS_DIR, bellhop_job_id, "tl2d_section.nc")
        if not os.path.isfile(section_file):
            raise HTTPException(status_code=404, detail="该任务没有可用的 tl2d_section.nc")
        try:
            propagation = BellhopSectionPropagation.from_netcdf(section_file)
            ray_section_file = os.path.join(JOBS_DIR, bellhop_job_id, "ray2d_section.nc")
            if os.path.isfile(ray_section_file) and hasattr(propagation, "load_rays"):
                propagation.load_rays(ray_section_file)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"无法读取 Bellhop TL：{exc}") from exc
    receiver_origin_job_id = _text(common.get("receiver_origin_bellhop_job_id"))
    if receiver_origin_job_id:
        if not re.fullmatch(r"[A-Za-z0-9_-]{4,80}", receiver_origin_job_id):
            raise HTTPException(status_code=400, detail="接收端 Bellhop 任务号格式无效")
        section_file = os.path.join(JOBS_DIR, receiver_origin_job_id, "tl2d_section.nc")
        if not os.path.isfile(section_file):
            raise HTTPException(status_code=404, detail="接收端任务没有可用的 tl2d_section.nc")
        try:
            receiver_propagation = BellhopSectionPropagation.from_netcdf(section_file)
            ray_section_file = os.path.join(JOBS_DIR, receiver_origin_job_id, "ray2d_section.nc")
            if os.path.isfile(ray_section_file) and hasattr(receiver_propagation, "load_rays"):
                receiver_propagation.load_rays(ray_section_file)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"无法读取接收端 Bellhop TL：{exc}") from exc
    try:
        result = simulate_reverberation(payload, propagation=propagation, receiver_propagation=receiver_propagation)
        if bellhop_job_id:
            reverb_nc = os.path.join(JOBS_DIR, bellhop_job_id, "reverberation_v6_time_depth.nc")
            paths_nc = os.path.join(JOBS_DIR, bellhop_job_id, "reverberation_v6_scattering_paths.nc")
            write_reverberation_netcdf(result, reverb_nc)
            if bool(common.get("save_scattering_paths_netcdf", True)):
                write_scattering_paths_netcdf(result, paths_nc)
            result["result_files"] = {
                "reverberation_time_depth_nc": os.path.basename(reverb_nc),
                "reverberation_time_depth_nc_endpoint": f"/file/{bellhop_job_id}/{os.path.basename(reverb_nc)}",
                "scattering_paths_nc": os.path.basename(paths_nc) if os.path.isfile(paths_nc) else None,
                "scattering_paths_nc_endpoint": f"/file/{bellhop_job_id}/{os.path.basename(paths_nc)}" if os.path.isfile(paths_nc) else None,
                "download_results_endpoint": f"/download_results/{bellhop_job_id}",
            }
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/compute")
async def compute(request: Request):
    form = await request.form()
    nonce = str(form.get("submit_nonce", "")).strip()
    if len(nonce) < 12:
        return JSONResponse({"ok": False, "message": "无效提交令牌"}, status_code=400)
    job_id = uuid.uuid4().hex[:12]
    jd = os.path.join(JOBS_DIR, job_id)
    try:
        defaults = dict(CONFIG["TASK_DEFAULTS"])
        source_lat = float(form.get("center_lat", defaults["center_lat"]))
        source_lon_raw = float(form.get("center_lon", defaults["center_lon"]))
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "message": "声源经纬度必须是数值。"}, status_code=400)
    if not math.isfinite(source_lat) or not -90.0 <= source_lat <= 90.0:
        return JSONResponse({"ok": False, "message": "声源纬度必须位于 -90 到 90°。"}, status_code=400)
    if not math.isfinite(source_lon_raw) or not -360.0 <= source_lon_raw <= 360.0:
        return JSONResponse({"ok": False, "message": "声源经度必须位于 -360 到 360°。"}, status_code=400)
    source_lon = (source_lon_raw + 180.0) % 360.0 - 180.0

    params = _params_from_form(form)
    try:
        selection_points = _source_points_from_form(form, source_lat, source_lon)
    except ValueError as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
    max_sources = int(CONFIG["MAX_SOURCES"])
    if len(selection_points) > max_sources:
        return JSONResponse({"ok": False, "message": f"当前区域会生成 {len(selection_points)} 个声源，超过配置上限 {max_sources}。"}, status_code=400)
    _apply_selection_points(params, selection_points)

    calculation_mode = _lower_text(params.get("calculation_mode"), "3d")
    params["calculation_mode"] = "2d" if calculation_mode == "2d" else "3d"

    raw_auto_depth = form.get("receiver_depth_auto_max")
    params["receiver_depth_auto_max"] = _is_truthy(raw_auto_depth) if raw_auto_depth is not None else False
    auto_scope = _lower_text(params.get("receiver_depth_auto_scope"))
    if auto_scope not in ("path", "area"):
        auto_scope = "path" if params["calculation_mode"] == "2d" else "area"
    params["receiver_depth_auto_scope"] = auto_scope

    try:
        acoustic_model = acoustic_model_dispatcher.normalise_acoustic_model(
            params.get("acoustic_model", "bellhop")
        )
    except ValueError as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
    params["acoustic_model"] = acoustic_model

    if params["calculation_mode"] == "2d":
        try:
            bearing = float(params.get("bearing_deg", params.get("selected_section_theta_deg", 0.0))) % 360.0
            receiver_lat = float(params.get("receiver_lat"))
            receiver_lon = float(params.get("receiver_lon"))
            receiver_depth = float(params.get("receiver_depth_m", params.get("receiver_depth_max_m", 100.0)))
        except (TypeError, ValueError):
            return JSONResponse({"ok": False, "message": "二维声场接收位置、接收深度或方位参数无效。"}, status_code=400)
        if not -90.0 <= receiver_lat <= 90.0 or not -180.0 <= receiver_lon <= 180.0:
            return JSONResponse({"ok": False, "message": "二维声场接收位置必须是有效经纬度。"}, status_code=400)
        if not math.isfinite(receiver_depth) or receiver_depth <= 0.0:
            return JSONResponse({"ok": False, "message": "二维声场接收深度必须为正数。"}, status_code=400)

        params["bearing_deg"] = bearing
        params["selected_section_theta_deg"] = bearing
        params["beta_min"] = bearing
        params["beta_max"] = bearing
        params["section_count"] = 1
        params["receiver_lat"] = receiver_lat
        params["receiver_lon"] = receiver_lon
        params["receiver_depth_m"] = receiver_depth
        receiver_depth_max = _finite_float(params.get("receiver_depth_max_m"), receiver_depth)
        params["receiver_depth_max_m"] = max(receiver_depth, receiver_depth_max)
        if params.get("receiver_depth_auto_max"):
            params["receiver_depth_auto_scope"] = "path"
        params["horizontal_tl_depth_m"] = receiver_depth
        params["display_3d_rays"] = True
        params["sound_field_view"] = "selected_section"
    elif params.get("receiver_depth_auto_max"):
        params["receiver_depth_auto_scope"] = "area"
    for key, default in (
        ("display_3d_rays", False), ("display_terrain", True), ("download_nc", False),
    ):
        raw = form.get(key)
        if raw is None and default:
            params[key] = default
        else:
            params[key] = _is_truthy(raw)
    sound_field_view = _lower_text(form.get("sound_field_view"), "full_3d")
    if sound_field_view not in ("full_3d", "selected_section", "hidden"):
        sound_field_view = "full_3d"
    params["sound_field_view"] = sound_field_view
    params["job_id"] = job_id
    _normalise_display_params(params)
    os.makedirs(jd, exist_ok=True)
    _write_json(os.path.join(jd, "params.json"), params)
    _write_json(os.path.join(jd, "job_status.json"), {"status": "queued", "progress": "排队中", "message": ""})
    threading.Thread(target=_run_job, args=(job_id, jd, params), daemon=True).start()
    return {
        "ok": True,
        "job_id": job_id,
        "status_url": f"/status/{job_id}",
        "websocket_url": f"/ws/jobs/{job_id}",
    }


# =============================================================================
# 任务查询和文件下载路由
# =============================================================================
@app.get("/status/{job_id}")
def status(job_id):
    if not _safe_id(job_id):
        raise HTTPException(status_code=404)
    jd = os.path.join(JOBS_DIR, job_id)
    try:
        st = _read_json(os.path.join(jd, "job_status.json"))
    except:
        return {"status": "unknown"}
    st["ready"] = os.path.exists(os.path.join(jd, "result_ready.flag"))
    return st


def _saved_3d_tl_response_path(job_dir):
    """返回已保存的三维 TL 接口响应文件，避免再依赖 NetCDF 中间产品。"""
    try:
        names = os.listdir(job_dir)
    except OSError:
        return None
    for name in names:
        if (
            name.startswith("response_")
            and name.endswith(".json")
            and "3DTLRD" in name
        ):
            path = os.path.join(job_dir, name)
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return path
    return None


def _workspace_job_summary(job_id, job_dir):
    """Return the small, stable subset needed by the GeoVis work drawer."""
    try:
        params = _read_json(os.path.join(job_dir, "params.json"))
    except (OSError, ValueError, TypeError):
        params = {}
    try:
        job_status = _read_json(os.path.join(job_dir, "job_status.json"))
    except (OSError, ValueError, TypeError):
        job_status = {}
    portal = params.get("portal_request") or {}
    tl_path = os.path.join(job_dir, "tl3d_depth_theta_range.nc")
    saved_tl_path = _saved_3d_tl_response_path(job_dir)
    openable = bool(saved_tl_path) or (
        os.path.isfile(tl_path) and os.path.getsize(tl_path) > 0
    )
    candidates = [job_dir, os.path.join(job_dir, "params.json"), os.path.join(job_dir, "job_status.json"), tl_path]
    if saved_tl_path:
        candidates.append(saved_tl_path)
    modified = max((os.path.getmtime(path) for path in candidates if os.path.exists(path)), default=0.0)
    model = str(params.get("acoustic_model") or portal.get("acousticModel") or "bellhop")
    status_value = str(job_status.get("status") or ("completed" if openable else "unknown"))
    return {
        "job_id": job_id,
        "status": status_value,
        "status_label": "完成" if openable else (job_status.get("progress") or "处理中"),
        "openable": bool(openable),
        "model": model,
        "model_label": "Bellhop 三维声场" if "bellhop" in model.lower() else "三维声场",
        "frequency_hz": params.get("freq_hz", portal.get("calcFrequency")),
        "radius_km": params.get("radius_km"),
        "source": {
            "lon": params.get("center_lon", portal.get("coordX")),
            "lat": params.get("center_lat", portal.get("coordY")),
            "depth_m": params.get("source_depth_m", portal.get("srcDepth")),
        },
        "updated_at_ms": int(modified * 1000.0),
    }


@app.get("/jobs")
def list_workspace_jobs(limit: int = 40):
    limit = max(1, min(100, int(limit or 40)))
    jobs = []
    try:
        names = os.listdir(JOBS_DIR)
    except OSError:
        names = []
    for name in names:
        if not _safe_id(name):
            continue
        job_dir = os.path.join(JOBS_DIR, name)
        if not os.path.isdir(job_dir) or not os.path.isfile(os.path.join(job_dir, "params.json")):
            continue
        summary = _workspace_job_summary(name, job_dir)
        if summary["openable"]:
            jobs.append(summary)
    jobs.sort(key=lambda item: item["updated_at_ms"], reverse=True)
    return {"jobs": jobs[:limit]}


@app.get("/jobs/{job_id}/visualization")
def workspace_job_visualization(job_id):
    if not _safe_id(job_id):
        raise HTTPException(status_code=404)
    job_dir = os.path.join(JOBS_DIR, job_id)
    summary = _workspace_job_summary(job_id, job_dir)
    if not summary["openable"]:
        raise HTTPException(status_code=404, detail="该工作尚无可打开的三维声场结果。")
    try:
        params = _read_json(os.path.join(job_dir, "params.json"))
        saved_tl_path = _saved_3d_tl_response_path(job_dir)
        if saved_tl_path:
            response = _read_json(saved_tl_path)
            if not isinstance(response, dict):
                raise ValueError("已保存的三维 TL 响应格式无效")
            plugin_name = os.path.basename(saved_tl_path)[len("response_"):-len(".json")]
        else:
            tl_path = os.path.join(job_dir, "tl3d_depth_theta_range.nc")
            with netCDF4.Dataset(bellhop_runner._ascii_netcdf_read_path(tl_path), "r") as dataset:
                row = int(_nc_array(dataset, "depth_m").size)
                col = int(_nc_array(dataset, "range_km").size)
                sector = int(_nc_array(dataset, "theta_deg").size)
            calc_frequency = params.get("freq_hz") or (params.get("portal_request") or {}).get("calcFrequency") or 0
            response = _build_3d_transmission_loss_response(job_dir, row, col, sector, calc_frequency)
            plugin_name = "Analyse3DTLRD"
        portal_response = _wrap_portal_response(
            response,
            {"wrapped": True, "technical_support": "XWOcean", "sn": 15, "task_id": job_id, "self_order": 0},
            plugin_name,
            job_id=job_id,
        )
        return _api_json_response({"job": summary, "result": portal_response})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取工作声场失败：{exc}")


@app.delete("/jobs/{job_id}")
def delete_workspace_job(job_id):
    if not _safe_id(job_id):
        raise HTTPException(status_code=404)
    job_dir = os.path.abspath(os.path.join(JOBS_DIR, job_id))
    jobs_root = os.path.abspath(JOBS_DIR)
    if os.path.commonpath([jobs_root, job_dir]) != jobs_root or not os.path.isdir(job_dir):
        raise HTTPException(status_code=404)
    try:
        status_value = (_read_json(os.path.join(job_dir, "job_status.json")) or {}).get("status")
    except (OSError, ValueError, TypeError):
        status_value = None
    if status_value in ("queued", "running"):
        raise HTTPException(status_code=409, detail="工作仍在计算中，不能删除。")
    try:
        shutil.rmtree(job_dir)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"删除工作失败：{exc}")
    return {"ok": True, "job_id": job_id}


@app.post("/abandon/{job_id}")
def abandon(job_id):
    if not _safe_id(job_id):
        raise HTTPException(status_code=404)
    jd = os.path.join(JOBS_DIR, job_id)
    status_path = os.path.join(jd, "job_status.json")
    if not os.path.isdir(jd):
        raise HTTPException(status_code=404)
    try:
        current = _read_json(status_path)
    except (OSError, ValueError):
        raise HTTPException(status_code=404)
    previous = current.get("status")
    if previous in ("running", "queued"):
        _set_job_status(job_id, status_path, {
            "status": "failed",
            "progress": "已取消",
            "message": "页面已关闭，任务已取消，请重新提交。",
            "percent": 0.0,
        }, event="job_failed")
    return {"ok": True, "status": previous}


@app.get("/file/{job_id}/{filename:path}")
def job_file(job_id, filename):
    if not _safe_id(job_id):
        raise HTTPException(status_code=404)
    jd = os.path.join(JOBS_DIR, job_id)
    safe = os.path.normpath(filename).replace("\\", "/")
    if safe.startswith("../") or safe == ".." or os.path.isabs(safe):
        raise HTTPException(status_code=404)
    fp = os.path.abspath(os.path.join(jd, safe))
    if os.path.commonpath([os.path.abspath(jd), fp]) != os.path.abspath(jd):
        raise HTTPException(status_code=404)
    if not os.path.exists(fp):
        raise HTTPException(status_code=404)
    return FileResponse(fp, headers={"Cache-Control": "no-cache, max-age=0"})


@app.get("/download_results/{job_id}")
def download_results(job_id):
    if not _safe_id(job_id):
        raise HTTPException(status_code=404)
    jd = os.path.join(JOBS_DIR, job_id)
    available = []
    for root, _, files in os.walk(jd):
        for name in files:
            path = os.path.join(root, name)
            if name.lower().endswith(".nc") and os.path.getsize(path) > 0:
                available.append((os.path.relpath(path, jd), path))
    if not available:
        raise HTTPException(status_code=404, detail="当前任务尚未生成 NetCDF 结果。")
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for relative_name, path in available:
            archive.write(path, arcname=relative_name)
    memory.seek(0)
    return Response(
        memory.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="bellhop3d_{job_id}_netcdf_results.zip"'},
    )


@app.get("/job_diagnostics/{job_id}")
def job_diagnostics(job_id):
    if not _safe_id(job_id):
        raise HTTPException(status_code=404)
    jd = os.path.join(JOBS_DIR, job_id)
    chunks = [f"多方位 Bellhop 2D 拼接三维场任务诊断\n任务: {job_id}\n"]
    for root, _, files in os.walk(jd):
        for name in sorted(files):
            if name.lower().endswith((".log", ".txt", ".prt")) or name in ("job_status.json", "params.json"):
                path = os.path.join(root, name)
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as stream:
                        text = stream.read()[-30000:]
                    chunks.append(f"\n--- {os.path.relpath(path, jd)} ---\n{text}")
                except OSError:
                    pass
    return Response("".join(chunks), media_type="text/plain; charset=utf-8")


@app.get("/download_job_logs/{job_id}")
def download_job_logs(job_id):
    if not _safe_id(job_id):
        raise HTTPException(status_code=404)
    jd = os.path.join(JOBS_DIR, job_id)
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(jd):
            for name in files:
                if name.lower().endswith((".log", ".txt", ".prt")) or name in ("job_status.json", "params.json"):
                    path = os.path.join(root, name)
                    archive.write(path, arcname=os.path.relpath(path, jd))
    memory.seek(0)
    return Response(
        memory.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="bellhop3d_{job_id}_logs.zip"'},
    )


@app.get("/info")
def info():
    return {
        "version": "backend-portal-config-defaults-9-4",
        "engine": "BellhopCXX single-bearing 2D / multi-bearing stitched 3D",
        "jobs_dir": JOBS_DIR,
        "realtime_transport": "websocket",
        "websocket_url_template": "/ws/jobs/<job_id>",
        "array_fusion_websocket": "/ws/array-fusion",
        "signal_transmission_websocket": "/ws/signal-transmission",
        "array_fusion_capabilities": {
            "actual_environment_profile": True,
            "bellhop_field_binding": True,
            "bathymetry": True,
            "multipath_from_ray_geometry": True,
            "native_bellhop_arrivals": True,
            "native_arrivals_auto_generation": True,
            "native_arrivals_fallback_to_ray_geometry": True,
            "static_channel_endpoint": "/array-channel/static",
            "static_communication_endpoint": "/array-channel/communication",
            "static_complex_element_impulse_response": True,
            "static_per_element_transmission_loss": True,
            "static_finite_pulse_playback": True,
            "static_time_resolved_spatial_spectrum": True,
            "static_seekable_receive_window": True,
            "motion_kinematics": True,
            "towed_array_geometry": True,
            "physical_doppler": True,
            "per_eigenray_virtex_motion": True,
            "dynamic_per_element_bellhop_arrivals": True,
            "dynamic_per_element_complex_impulse_response": True,
            "multi_bearing_bellhop_interpolation": True,
            "time_snapshot_bellhop_interpolation": True,
            "dynamic_complex_impulse_response": True,
            "linked_signal_domains": [
                "time", "frequency", "time_frequency", "spatial_spectrum", "beam_output"
            ],
            "message_types": [
                "profile_req", "field_req", "bellhop_bind", "static_channel_req", "frame",
                "bellhop_series_bind", "motion_config", "motion_frame", "motion_pause", "motion_seek"
            ],
        },
        "status_fallback": "/status/<job_id>",
        "calculation_endpoints": {
            "2d_transmission_loss": "/Analyse2DTLRD",
            "2d_ray_track": "/Analyse2DRTRD",
            "2d_dynamic_ray_track": "/Analyse2DRTRDDynamic",
            "3d_dynamic_ray_track": "/Analyse3DRTRDDynamic",
            "3d_transmission_loss": "/Analyse3DTLRD",
            "ice_2d_transmission_loss": "/AnalyseIce2DTLRD",
            "ice_2d_ray_track": "/AnalyseIce2DRTRD",
            "ice_2d_dynamic_ray_track": "/AnalyseIce2DRTRDDynamic",
            "ice_3d_transmission_loss": "/AnalyseIce3DTLRD",
            "ice_3d_dynamic_ray_track": "/AnalyseIce3DRTRDDynamic",
            "open_water_reverberation": "/AnalyseReverberation",
            "open_water_reverberation_waveform": "/AnalyseReverberationWaveform",
            "ice_under_surface_reverberation": "/AnalyseIceUnderSurfaceReverberation",
            "bottom_surface_reverberation": "/AnalyseBottomSurfaceReverberation",
            "waveguide": "/AnalyseWaveguide",
            "ice_waveguide": "/AnalyseIceWaveguide",
            "waveguide_decay": "/AnalyseWaveguideDecay",
            "env_station": "/GetEnvDataStation",
            "env_route": "/GetEnvDataRoute",
            "env_region": "/GetEnvDataRegion",
            "noise": "/AnalyseEnvironmentNoise",
            "sound_layer": "/AnalyseSoundLayer",
            "sound_speed_class": "/AnalyseSoundSpeedClass",
            "channel_axis": "/AnalyseChannelAxis",
            "channel_impulse_response": "/AnalyseChannelImpulseResponse",
            "sound_propagation_loss": "/AnalyseSoundPropagationLoss",
            "arctic_mixed_layer": "/AnalyseArcticMixedLayer",
            "reverb_decay": "/AnalyseReverbDecay",
            "weather_forecast": "/GetWeatherForecast",
            "regional_weather_forecast": "/GetRegionalWeatherForecast",
            "realtime_ais": "/GetRealtimeAIS",
            "realtime_ais_track": "/GetRealtimeAISTrack",
            "forecast_environment_noise": "/AnalyseForecastEnvironmentNoise",
        },
        "catalog_endpoints": {
            "dataset_list": "/GetEnvDataSourceList",
            "dataset_time_list": "/FetchTimeToList",
            "data_type_meta_list": "/GetDataTypeMetaInfo",
        },
    }


# =============================================================================
# 网页实时进度推送
# =============================================================================
def _array_fusion_profile_provider(payload):
    """为线阵联动页面读取真实温盐派生 SSP 和测点海深。"""
    _reload_config_if_changed()
    lon = float(payload.get("lon", payload.get("coordX", 113.8)))
    lat = float(payload.get("lat", payload.get("coordY", 13.5)))
    if not (math.isfinite(lon) and -180.0 <= lon <= 180.0):
        raise ValueError("lon 必须位于 -180 到 180 之间")
    if not (math.isfinite(lat) and -90.0 <= lat <= 90.0):
        raise ValueError("lat 必须位于 -90 到 90 之间")
    data_source = str(payload.get("data_source", payload.get("dataSource", "HYCOM")))
    time_value = str(
        payload.get("time", payload.get("environment_time", database_source.default_environment_time(CONFIG)))
    )
    depth, values = _env_read_profiles(
        np.asarray([lon]),
        np.asarray([lat]),
        "DT_ENV_SOUND",
        time_value,
        data_source=data_source,
    )
    bottom_depth = float(_env_bottom_depths(
        np.asarray([lon]), np.asarray([lat]), data_source=data_source, time_value=time_value
    )[0])
    profile = np.asarray(values, dtype=np.float64).reshape(len(depth), -1)[:, 0]
    valid = np.isfinite(depth) & np.isfinite(profile)
    if math.isfinite(bottom_depth) and bottom_depth > 0.0:
        valid &= depth <= bottom_depth
    if np.count_nonzero(valid) < 2:
        raise ValueError("环境数据在该测点没有足够的有效声速剖面")
    return {
        "type": "profile_resp",
        "lon": lon,
        "lat": lat,
        "depth": np.round(depth[valid], 3).tolist(),
        "c": np.round(profile[valid], 4).tolist(),
        "bottom_depth_m": round(bottom_depth, 3) if math.isfinite(bottom_depth) else None,
        "data_source": data_source,
        "environment_time": time_value,
        "actual_environment": True,
    }


@app.post("/array-channel/static", tags=["Array signal processing"])
async def array_static_channel(request: Request):
    """基于已完成的 Bellhop 任务计算固定声源—固定阵列信道。"""

    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        job_id = str(payload.get("bellhop_job_id") or payload.get("job_id") or "").strip()
        if not job_id:
            raise ValueError("缺少 bellhop_job_id")
        ray_job_id = str(payload.get("ray_job_id") or "").strip() or None
        arrival_job_id = str(payload.get("arrival_job_id") or "").strip() or None
        session = ArrayFusionSession(jobs_dir=JOBS_DIR)
        await asyncio.to_thread(
            session.bind_bellhop,
            job_id,
            ray_job_id,
            arrival_job_id,
        )
        result = await asyncio.to_thread(session.static_channel, payload)
        return JSONResponse(result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _static_communication_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """把 array_fusion_service 的静态 CIR 接入可归因 BER 仿真。"""

    static_channel = payload.get("static_channel")
    communication = payload.get("communication", {})
    if not isinstance(static_channel, Mapping):
        raise ValueError("缺少 static_channel；请先在静态信道页面完成计算")
    if not isinstance(communication, Mapping):
        raise ValueError("communication 必须是对象")

    def integer(name, default, minimum, maximum):
        value = communication.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"communication.{name} 必须是整数")
        number = int(value)
        if float(value) != number or not minimum <= number <= maximum:
            raise ValueError(
                f"communication.{name} 必须是 {minimum} 到 {maximum} 的整数"
            )
        return number

    selected_element = integer("selected_element", 0, 0, 511)
    information_bits = integer("information_bits_per_frame", 512, 64, 4096)
    frames = integer("frames_per_point", 8, 1, 40)
    fft_size = integer("fft_size", 256, 8, 4096)
    if fft_size & (fft_size - 1):
        raise ValueError("communication.fft_size 必须是 2 的整数次幂")
    cp_samples = integer("cyclic_prefix_samples", 64, 0, fft_size - 1)
    default_pilot_subcarriers = min(16, max(0, fft_size // 16))
    pilot_subcarriers = integer(
        "pilot_subcarriers", default_pilot_subcarriers, 0, fft_size - 1
    )
    default_data_subcarriers = max(1, min(fft_size - pilot_subcarriers, int(fft_size * 0.75)))
    data_subcarriers = integer(
        "data_subcarriers", default_data_subcarriers, 1, fft_size
    )
    if data_subcarriers + pilot_subcarriers > fft_size:
        raise ValueError("communication.data_subcarriers 与 pilot_subcarriers 之和不能超过 FFT 点数")
    preamble_symbols = integer("preamble_symbols", 2, 0, 64)
    header_bits = integer("header_bits", 32, 0, 2048)
    guard_samples = integer("guard_samples", 32, 0, 8192)
    iterations = integer("turbo_iterations", 6, 1, 12)
    random_seed = integer("random_seed", 20261021, 0, 2_147_483_647)
    interleaver_seed = integer("interleaver_seed", 20261021, 0, 2_147_483_647)
    sample_rate = float(communication.get("sample_rate_hz", 2000.0))
    if not math.isfinite(sample_rate) or not 10.0 <= sample_rate <= 100_000.0:
        raise ValueError("communication.sample_rate_hz 必须位于 10 到 100000 Hz")
    ebn0_raw = communication.get("ebn0_db", [-2, 0, 2, 4, 6, 8])
    if not isinstance(ebn0_raw, Sequence) or isinstance(ebn0_raw, (str, bytes)):
        raise ValueError("communication.ebn0_db 必须是数组")
    ebn0_db = tuple(float(value) for value in ebn0_raw)
    if not 2 <= len(ebn0_db) <= 16 or not all(
        math.isfinite(value) and -20.0 <= value <= 40.0 for value in ebn0_db
    ):
        raise ValueError("communication.ebn0_db 需包含 2 到 16 个 -20~40 dB 数值")
    if information_bits * frames * len(ebn0_db) > 200_000:
        raise ValueError("当前码块、帧数和 Eb/N0 点数组合过大，请减少其中一项")
    ber_threshold = float(communication.get("ber_threshold", 1.0e-3))
    if not math.isfinite(ber_threshold) or not 1.0e-8 <= ber_threshold <= 0.49:
        raise ValueError("communication.ber_threshold 必须位于 1e-8 到 0.49 之间")

    channel, channel_metadata = channel_config_from_array_fusion(
        static_channel,
        element_index=selected_element,
        communication_sample_rate_hz=sample_rate,
    )
    scenarios, selected_scenario = scenarios_with_selection(
        communication.get("modulation", "QPSK"),
        communication.get("waveform", "ofdm"),
        communication.get("coding", "turbo"),
    )
    config = StaticLinkSimulationConfig(
        ebn0_db=ebn0_db,
        information_bits_per_frame=information_bits,
        frames_per_point=frames,
        random_seed=random_seed,
        channel=channel,
        ofdm=StaticLinkOfdmConfig(
            fft_size=fft_size,
            cyclic_prefix_samples=cp_samples,
            data_subcarriers=data_subcarriers,
            pilot_subcarriers=pilot_subcarriers,
        ),
        frame=StaticLinkFrameConfig(
            preamble_symbols=preamble_symbols,
            header_bits=header_bits,
            guard_samples=guard_samples,
        ),
        turbo=StaticLinkTurboConfig(
            iterations=iterations,
            interleaver_seed=interleaver_seed,
            llr_clip=60.0,
        ),
    )
    result = run_static_communication_study(config, scenarios)
    result.update({
        "type": "communication_study_resp",
        "module": "complete_static_communication",
        "selected_scenario": selected_scenario,
        "static_channel_source": channel_metadata,
    })
    result["link_metrics"] = build_current_link_metrics(
        result,
        static_channel,
        selected_scenario,
        ber_threshold=ber_threshold,
    )
    result["ber_diagnostics"] = build_ber_diagnostics(
        result, selected_scenario, result["link_metrics"]
    )
    return result


@app.post("/array-channel/communication", tags=["Array signal processing"])
async def array_static_communication(request: Request):
    """用阵列融合静态 CIR 计算 BPSK/QPSK、OFDM、Turbo/Log-MAP 的 BER。"""

    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        result = await asyncio.to_thread(_static_communication_request, payload)
        return JSONResponse(result)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws/array-fusion")
async def array_fusion_websocket(websocket: WebSocket):
    """线阵时频域、波束域与声场联动的双向实时数据流。"""
    await serve_array_fusion_websocket(
        websocket,
        jobs_dir=JOBS_DIR,
        profile_provider=_array_fusion_profile_provider,
    )


@app.websocket("/ws/signal-transmission")
async def signal_transmission_websocket(websocket: WebSocket):
    """发射信号、Bellhop 多途、水听器与 ADC 的端到端双向计算流。"""
    import sys

    await serve_signal_transmission_websocket(
        websocket,
        jobs_dir=JOBS_DIR,
        core=sys.modules[__name__],
    )


def _ws_add_subscriber(job_id):
    subscriber = queue.Queue(maxsize=_WS_QUEUE_SIZE)
    with _WS_SUBSCRIBERS_LOCK:
        _WS_SUBSCRIBERS.setdefault(job_id, set()).add(subscriber)
    return subscriber


def _ws_remove_subscriber(job_id, subscriber):
    with _WS_SUBSCRIBERS_LOCK:
        subscribers = _WS_SUBSCRIBERS.get(job_id)
        if not subscribers:
            return
        subscribers.discard(subscriber)
        if not subscribers:
            _WS_SUBSCRIBERS.pop(job_id, None)


def _publish_job_event(job_id, event, payload):
    """发布最新任务状态；不能阻塞 Bellhop 计算线程。"""
    message = dict(payload or {})
    message["event"] = str(event or "job_progress")
    message["job_id"] = job_id
    with _WS_SUBSCRIBERS_LOCK:
        subscribers = list(_WS_SUBSCRIBERS.get(job_id, ()))
    for subscriber in subscribers:
        try:
            subscriber.put_nowait(message)
        except queue.Full:
            # 队列满时丢掉一帧旧状态，优先保留最新状态。
            try:
                subscriber.get_nowait()
            except queue.Empty:
                pass
            try:
                subscriber.put_nowait(message)
            except queue.Full:
                pass


def _set_job_status(job_id, status_path, payload, event="job_progress"):
    """保存任务状态，便于刷新恢复；同时把同一份快照实时推给页面。"""
    data = dict(payload or {})
    data.setdefault("updated_at", time.time())
    _write_json(status_path, data)
    _publish_job_event(job_id, event, data)
    return data


@app.websocket("/ws/jobs/{job_id}")
async def job_websocket(websocket: WebSocket, job_id):
    """单个计算任务的后端到浏览器实时消息流。"""
    await websocket.accept()
    if not _safe_id(job_id):
        await websocket.close()
        return

    job_dir = os.path.join(JOBS_DIR, job_id)
    status_path = os.path.join(job_dir, "job_status.json")
    if not os.path.isdir(job_dir):
        try:
            await websocket.send_text(json.dumps({
                "event": "job_unknown", "job_id": job_id, "status": "unknown"
            }, ensure_ascii=False))
        except Exception:
            pass
        await websocket.close()
        return

    subscriber = _ws_add_subscriber(job_id)
    try:
        # 页面新连接或重连时，先发送最新的持久化状态，
        # 避免连接建立期间丢失关键进度。
        try:
            snapshot = _read_json(status_path)
            snapshot["ready"] = os.path.exists(os.path.join(job_dir, "result_ready.flag"))
            snapshot["event"] = "job_snapshot"
            snapshot["job_id"] = job_id
            await websocket.send_text(json.dumps(snapshot, ensure_ascii=False))
            snapshot_state = str(snapshot.get("status") or "").lower()
            if snapshot.get("ready") or snapshot_state in ("ready", "visual_ready", "failed", "error"):
                return
        except (OSError, ValueError):
            pass

        while True:
            try:
                message = await asyncio.to_thread(subscriber.get, True, 20.0)
            except queue.Empty:
                message = {
                    "event": "heartbeat",
                    "job_id": job_id,
                    "server_time": time.time(),
                }
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
            if message.get("event") in ("job_ready", "job_failed"):
                break
    except (WebSocketDisconnect, BrokenPipeError, OSError):
        pass
    finally:
        _ws_remove_subscriber(job_id, subscriber)
        try:
            await websocket.close()
        except Exception:
            pass


# =============================================================================
# 本地任务文件和后台计算
# =============================================================================
def _safe_id(jid):
    return bool(jid) and len(jid) <= 32 and all(c in "0123456789abcdef" for c in jid)


def _json_lock(path):
    """为每个 JSON 文件路径返回一个进程内共享锁。"""
    key = os.path.normcase(os.path.abspath(path))
    with _JSON_LOCKS_GUARD:
        return _JSON_LOCKS.setdefault(key, threading.Lock())


def _read_json(path):
    """读取 JSON 文件，并避开并发状态写入造成的竞争。"""
    with _json_lock(path):
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)


def _write_json(path, data):
    """原子替换 JSON 文件，并容忍 Windows 上短暂的文件占用。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    with _json_lock(path):
        try:
            with open(temporary_path, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            last_error = None
            for attempt in range(8):
                try:
                    os.replace(temporary_path, path)
                    return
                except PermissionError as exc:
                    last_error = exc
                    # 杀毒软件、资源管理器预览或外部读取程序可能短暂占用文件；
                    # 等一小会儿再重试即可。
                    time.sleep(0.025 * (attempt + 1))
            raise last_error
        finally:
            try:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
            except OSError:
                pass

def _reconcile_orphaned_jobs():
    """把上次进程崩溃后遗留的 running/queued 任务标记为失败。

    任务在线程中执行；后端进程重启后原线程已经不存在，
    因此启动时仍处于 running/queued 的任务无法继续恢复。
    """
    try:
        names = os.listdir(JOBS_DIR)
    except OSError:
        return
    for name in names:
        jd = os.path.join(JOBS_DIR, name)
        if not os.path.isdir(jd):
            continue
        status_path = os.path.join(jd, "job_status.json")
        if not os.path.isfile(status_path):
            continue
        try:
            current = _read_json(status_path)
        except (OSError, ValueError):
            continue
        if current.get("status") in ("running", "queued"):
            _set_job_status(name, status_path, {
                "status": "failed",
                "progress": "任务中断",
                "message": "后端服务已重启，任务中断，请重新提交。",
                "percent": 0.0,
            }, event="job_failed")


def _run_job(job_id, job_dir, params):
    try:
        status_path = os.path.join(job_dir, "job_status.json")
        source_count = int(params.get("selection_point_count") or len(params.get("selection_points") or []) or 1)
        progress_state = {"task_total": 0, "task_completed": 0, "last_write": 0.0}
        progress_lock = threading.Lock()

        def report_progress(stage, message):
            text = str(message or "")
            with progress_lock:
                submitted = re.search(r"提交\s+(\d+)\s+个声源×方位任务", text)
                completed = re.search(r"已完成\s+(\d+)/(\d+)\s+个声源×方位任务", text)
                source_match = re.search(r"\[(\d+)/(\d+)\]", text)
                if submitted:
                    progress_state["task_total"] = int(submitted.group(1))
                if completed:
                    progress_state["task_completed"] = int(completed.group(1))
                    progress_state["task_total"] = int(completed.group(2))
                try:
                    phase_number = int(str(stage).split("/", 1)[0])
                except (ValueError, IndexError):
                    phase_number = 0
                shared_match = re.search(
                    r"Shared environment ready:\s*([\d,]+)\s+local-grid samples",
                    text,
                    re.IGNORECASE,
                )
                if progress_state["task_total"]:
                    percent = 15.0 + 75.0 * progress_state["task_completed"] / progress_state["task_total"]
                elif shared_match:
                    percent = 15.0
                elif source_match:
                    source_index = int(source_match.group(1))
                    percent = 15.0 * ((source_index - 1) + phase_number / 7.0) / max(1, source_count)
                else:
                    percent = min(15.0, phase_number / 7.0 * 15.0)
                if phase_number >= 7:
                    percent = 100.0
                now = time.time()
                should_write = completed is not None or phase_number in (4, 7) or now - progress_state["last_write"] >= 0.25
                if not should_write:
                    return
                progress_state["last_write"] = now
                progress_payload = {
                    "status": "running", "progress": str(stage), "message": text,
                    "source_count": source_count,
                    "completed_tasks": progress_state["task_completed"],
                    "total_tasks": progress_state["task_total"],
                    "percent": round(max(0.0, min(100.0, percent)), 1),
                    "updated_at": now,
                }
                try:
                    _set_job_status(job_id, status_path, progress_payload, event="job_progress")
                except OSError:
                    # 文件锁只短暂保护写入，不能卡住实时页面或 Bellhop。
                    _publish_job_event(job_id, "job_progress", progress_payload)

        _set_job_status(job_id, status_path, {
            "status": "running", "progress": "准备计算", "message": "正在校验声源与环境文件。",
            "source_count": source_count, "completed_tasks": 0, "total_tasks": 0, "percent": 0.0,
        }, event="job_progress")
        for warning in params.get("_parameter_warnings", []) if isinstance(params.get("_parameter_warnings"), list) else []:
            report_progress("0/7", f"参数提示：{warning}")
        result = acoustic_model_dispatcher.run(job_dir, params, CONFIG, progress_callback=report_progress)
        if result.get("status") == "ready":
            _set_job_status(job_id, status_path, {
                "status": "ready", "progress": "完成", "message": f"完成 ({result.get('elapsed',0):.1f}s)",
                "source_count": source_count, "completed_tasks": progress_state["task_completed"],
                "total_tasks": progress_state["task_total"], "percent": 100.0, "ready": True,
            }, event="job_ready")
        else:
            _set_job_status(job_id, status_path, {
                "status": "failed", "progress": "出错", "message": result.get("message", "未知错误"),
                "source_count": source_count, "completed_tasks": progress_state["task_completed"],
                "total_tasks": progress_state["task_total"], "percent": progress_state["task_completed"] / max(1, progress_state["task_total"]) * 90.0,
            }, event="job_failed")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        status_path = os.path.join(job_dir, "job_status.json")
        payload = {"status": "failed", "progress": "异常", "message": f"{e}\n{tb[-500:]}", "percent": 0.0}
        try:
            _set_job_status(job_id, status_path, payload, event="job_failed")
        except OSError:
            _publish_job_event(job_id, "job_failed", payload)


# 仅暴露当前前端实际调用的路由。保留同文件中的公共计算辅助函数，供门户
# Bellhop/RAM/Kraken、环境查询和任务工作区复用。
_FRONTEND_ROUTE_PATHS = {
    "",  # APIRouter 根路由在 include_router 后内部记为空字符串。
    "/",
    "/GetWeatherForecast",
    "/GetRegionalWeatherForecast",
    "/GetRealtimeAIS",
    "/GetRealtimeAISTrack",
    "/AnalyseForecastEnvironmentNoise",
    "/AnalyseRegionalForecastEnvironmentNoise",
    "/GetRegionalOceanCurrent",
    "/GetSeafloorFeatures",
    "/jobs",
    "/jobs/{job_id}/visualization",
    "/jobs/{job_id}",
    "/file/{job_id}/{filename:path}",
    "/array-channel/static",
    "/array-channel/communication",
    "/ws/array-fusion",
    "/ws/signal-transmission",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
}
app.router.routes[:] = [
    route for route in app.router.routes
    if getattr(route, "path", None) in _FRONTEND_ROUTE_PATHS
]


if __name__ == "__main__":
    import uvicorn

    print("backend __file__ =", __file__)
    print("BASE =", BASE)
    print("PROJECT =", PROJECT)
    print("CONFIG_FILE =", CONFIG_FILE)
    print("runner =", bellhop_runner.__file__)
    uvicorn.run(app, host="0.0.0.0", port=8888)
