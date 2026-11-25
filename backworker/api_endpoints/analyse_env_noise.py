"""非冰区环境噪声门户接口。"""

from pathlib import Path

from ..services.environment_noise import build_response
from ..services.environment_noise_20hz import (
    MODEL_ID as CURRENT_20HZ_MODEL_ID,
    native_frequency_axis,
)
from ..services.environment_noise_time_frequency import (
    build_time_frequency,
    parse_options as parse_time_frequency_options,
)
from ..services.wind_noise_receiver import distributed_response
from .common import create_portal_router, handle_portal_payload
from .distributed_noise import (
    build_current_20hz_levels,
    build_distributed_levels,
    data_config,
)


PLUGIN_NAME = "AnalyseEnvironmentNoise"
ROUTE_PATH = "/AnalyseEnvironmentNoise"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANUAL_LEVEL_FIELDS = {"shipDensityLevel", "windSpeed", "rainIntensity"}
MANUAL_CALCULATION_MODE = "manual_levels"
GRID_CALCULATION_MODE = "source_grid"
CALCULATION_MODES = {MANUAL_CALCULATION_MODE, GRID_CALCULATION_MODE}
MANUAL_REQUEST_FIELDS = MANUAL_LEVEL_FIELDS | {
    "minFreq", "maxFreq", "numPoints", "timeFrequency",
}
DATA_REQUEST_FIELDS = {
    "coordX", "coordY", "srcDepth", "range", "time", "timeRes",
    "minFreq", "maxFreq", "numPoints", "windSourceModel", "propagationModel",
    "sourceGridModel", "timeFrequency",
} | MANUAL_LEVEL_FIELDS
REQUEST_FIELDS = MANUAL_REQUEST_FIELDS | DATA_REQUEST_FIELDS


def _is_present(data, name):
    value = data.get(name)
    return value is not None and not (isinstance(value, str) and not value.strip())


def _strict_int(data, name, label, core):
    value = data.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise core.PortalRequestError(f"{label}必须是整数。")
    return value


def _validate_frequency(min_freq, max_freq, num_points, core, maximum_points):
    if min_freq <= 0.0:
        raise core.PortalRequestError("minFreq 必须大于 0 Hz。")
    if max_freq <= min_freq:
        raise core.PortalRequestError("maxFreq 必须大于 minFreq。")
    if max_freq > 100000.0:
        raise core.PortalRequestError("maxFreq 不能超过 100000 Hz。")
    if not 2 <= num_points <= maximum_points:
        raise core.PortalRequestError(f"numPoints 必须位于 2 到 {maximum_points} 之间。")


def _normalise_manual_request(data, core):
    core._reject_unsupported_request_fields(data, MANUAL_REQUEST_FIELDS, PLUGIN_NAME)
    ship_level = _strict_int(data, "shipDensityLevel", "航运等级", core)
    wind_level = _strict_int(data, "windSpeed", "海况等级", core)
    rain_level = _strict_int(data, "rainIntensity", "降雨等级", core)
    min_freq = core._api_float(data, ["minFreq"], 10.0, "最小频率")
    max_freq = core._api_float(data, ["maxFreq"], 1000.0, "最大频率")
    # 按要求保留原手动模式的 numPoints 行为。
    num_points = core._api_int(data, ["numPoints"], 10000, "频率采样点数")
    if not 1 <= ship_level <= 5:
        raise core.PortalRequestError("shipDensityLevel 必须位于 1 到 5 级。")
    if not 0 <= wind_level <= 12:
        raise core.PortalRequestError("windSpeed 必须位于 0 到 12 级。")
    if not 0 <= rain_level <= 6:
        raise core.PortalRequestError("rainIntensity 必须位于 0 到 6 级。")
    _validate_frequency(min_freq, max_freq, num_points, core, 100000)
    return {
        "mode": "manual_mixed",
        "shipDensityLevel": ship_level,
        "windSpeed": wind_level,
        "rainIntensity": rain_level,
        "minFreq": min_freq,
        "maxFreq": max_freq,
        "numPoints": num_points,
    }


def _normalise_data_request(data, core):
    core._reject_unsupported_request_fields(data, DATA_REQUEST_FIELDS, PLUGIN_NAME)
    missing = [
        name
        for name in ("coordX", "coordY", "srcDepth", "time")
        if not _is_present(data, name)
    ]
    if missing:
        raise core.PortalRequestError(
            "分布式噪声源网格请求缺少字段：" + ", ".join(missing) + "。"
        )
    longitude = core._api_float(data, ["coordX"], None, "接收点经度")
    latitude = core._api_float(data, ["coordY"], None, "接收点纬度")
    receiver_depth = core._api_float(data, ["srcDepth"], None, "工作深度")
    if not -360.0 <= longitude <= 360.0:
        raise core.PortalRequestError("接收点经度必须位于 -360 到 360°。")
    if not -90.0 <= latitude <= 90.0:
        raise core.PortalRequestError("接收点纬度必须位于 -90 到 90°。")
    if receiver_depth <= 0.0:
        raise core.PortalRequestError("工作深度必须为正数。")
    if core._api_int(data, ["timeRes"], 0, "时间偏移量") != 0:
        raise core.PortalRequestError("分布式噪声源网格当前要求 timeRes=0。")

    config = data_config(core)
    source_grid_model = str(
        data.get(
            "sourceGridModel",
            config.get("DEFAULT_SOURCE_GRID_MODEL", "legacy"),
        )
    ).strip().lower()
    if source_grid_model not in {"legacy", CURRENT_20HZ_MODEL_ID}:
        raise core.PortalRequestError(
            "sourceGridModel 仅支持 legacy 或 ainslie_urick_jomopans_20hz。"
        )
    current_20hz = source_grid_model == CURRENT_20HZ_MODEL_ID
    min_freq = core._api_float(
        data, ["minFreq"], 20.0 if current_20hz else 10.0, "最小频率"
    )
    max_freq = core._api_float(data, ["maxFreq"], 1000.0, "最大频率")
    num_points = core._api_int(
        data, ["numPoints"], 50 if current_20hz else 100, "频率采样点数"
    )
    maximum_points = int(config.get("MAX_RESPONSE_POINTS", 2000))
    _validate_frequency(min_freq, max_freq, num_points, core, maximum_points)
    minimum_supported = 20.0 if current_20hz else float(config.get("MIN_FREQUENCY_HZ", 10.0))
    maximum_supported = 1000.0 if current_20hz else float(config.get("MAX_FREQUENCY_HZ", 1000.0))
    if min_freq < minimum_supported or max_freq > maximum_supported:
        raise core.PortalRequestError(
            f"分布式噪声频率范围为 {minimum_supported:g} 到 {maximum_supported:g} Hz。"
        )
    if current_20hz:
        try:
            native_frequency = native_frequency_axis(min_freq, max_freq)
        except ValueError as exc:
            raise core.PortalRequestError(str(exc)) from exc
        if num_points != int(native_frequency.size):
            raise core.PortalRequestError(
                "当前 20 Hz 模型的 numPoints 必须与原生频率点数一致；"
                f"{min_freq:g}–{max_freq:g} Hz 应为 {native_frequency.size}。"
            )
        if _is_present(data, "windSourceModel") and str(data["windSourceModel"]).strip().lower() not in {
            "ainslie2010", "ainslie_2010"
        }:
            raise core.PortalRequestError(
                "20 Hz 模型固定使用 Ainslie 2010 风噪声源谱，不能选择其他 windSourceModel。"
            )
        requested_propagation = str(data.get("propagationModel", "bellhop")).strip().lower()
        if requested_propagation != "bellhop":
            raise core.PortalRequestError("20 Hz 模型固定使用经典 Bellhop，propagationModel 必须为 bellhop。")
        wind_source_model = "ainslie2010"
        propagation_model = "bellhop"
    else:
        wind_source_model = str(data.get("windSourceModel", "hildebrand2021")).strip().lower()
        if wind_source_model not in {"hildebrand2021", "liu2022", "liu2022_refined"}:
            raise core.PortalRequestError(
                "windSourceModel 仅支持 hildebrand2021、liu2022 或 liu2022_refined。"
            )
        propagation_model = str(data.get("propagationModel", "hybrid")).strip().lower()
        if propagation_model not in {"hybrid", "ram", "bellhop"}:
            raise core.PortalRequestError(
                "propagationModel 仅支持 hybrid、ram 或 bellhop。"
            )
    default_radius_km = float(
        config.get("CURRENT_20HZ_RADIUS_KM", 60.0)
        if current_20hz
        else config.get("DEFAULT_RADIUS_KM", 100.0)
    )
    range_nm = core._api_float(data, ["range"], default_radius_km / core.NM_TO_KM, "噪声传播半径")
    radius_km = range_nm * core.NM_TO_KM
    maximum_radius_km = float(
        config.get("CURRENT_20HZ_MAX_RADIUS_KM", 60.0)
        if current_20hz
        else config.get("MAX_RADIUS_KM", 150.0)
    )
    if range_nm <= 0.0:
        raise core.PortalRequestError("range 必须为正数，单位为海里。")
    if radius_km > maximum_radius_km:
        raise core.PortalRequestError(
            f"range 不能超过 {maximum_radius_km / core.NM_TO_KM:.3f} 海里。"
        )
    return {
        "mode": "distributed_data",
        "minFreq": min_freq,
        "maxFreq": max_freq,
        "numPoints": num_points,
        "receiverLon": ((longitude + 180.0) % 360.0) - 180.0,
        "receiverLat": latitude,
        "receiverDepthM": receiver_depth,
        "radiusKm": radius_km,
        "requestedTime": data.get("time"),
        "acousticModel": (
            "bellhop"
            if current_20hz
            else str(config.get("ACOUSTIC_MODEL", "bellhop")).strip().lower()
        ),
        "sourceGridModel": source_grid_model,
        "windSourceModel": wind_source_model,
        "propagationModel": propagation_model,
    }


def _configured_calculation_mode(core):
    calculation_mode = str(
        data_config(core).get("CALCULATION_MODE", "")
    ).strip().lower()
    if calculation_mode not in CALCULATION_MODES:
        raise ValueError(
            "data_config.json 中 ENVIRONMENT_NOISE_DATA.CALCULATION_MODE "
            "仅支持 manual_levels 或 source_grid。"
        )
    return calculation_mode


def _normalise_request(data, core):
    calculation_mode = _configured_calculation_mode(core)
    present = {name for name in MANUAL_LEVEL_FIELDS if _is_present(data, name)}
    if calculation_mode == MANUAL_CALCULATION_MODE and present != MANUAL_LEVEL_FIELDS:
        missing = sorted(MANUAL_LEVEL_FIELDS - present)
        raise core.PortalRequestError(
            "manual_levels 模式必须同时传入 shipDensityLevel、windSpeed 和 "
            "rainIntensity；缺少：" + ", ".join(missing) + "。"
        )
    if calculation_mode == MANUAL_CALCULATION_MODE:
        return _normalise_manual_request(data, core)
    # source_grid 模式兼容接收前端遗留的三个等级字段，但网格计算链不会读取它们。
    return _normalise_data_request(data, core)


def _build_distributed_response(normalised, core):
    builder = (
        build_current_20hz_levels
        if normalised.get("sourceGridModel") == CURRENT_20HZ_MODEL_ID
        else build_distributed_levels
    )
    selections, levels, job_ids, contributing_counts, _ = builder(normalised, core)
    response = distributed_response(normalised, selections, levels, job_ids)
    response["propagation"]["contributingSourceCellsByFrequency"] = contributing_counts
    return response, (job_ids[0] if job_ids else None)


def _attach_time_frequency(response, request_payload, normalised, core):
    """请求带 timeFrequency 时追加时频矩阵，否则保持原有响应结构。"""
    try:
        options = parse_time_frequency_options(
            request_payload.get("timeFrequency"), allow_ice=False
        )
    except ValueError as exc:
        raise core.PortalRequestError(str(exc)) from exc
    if options is None:
        return response

    frequency = response.get("freq")
    reference = response.get("data")
    if not isinstance(frequency, list) or not isinstance(reference, list):
        raise core.PortalRequestError("当前响应缺少频率或噪声谱，无法生成时频图。")

    try:
        response["timeFrequency"] = build_time_frequency(
            frequency,
            reference,
            options,
            ice=False,
        )
    except ValueError as exc:
        raise core.PortalRequestError(str(exc)) from exc
    return response


def handle(payload):
    def build_portal_response(request_payload, core, meta):
        core._reject_unsupported_request_fields(
            request_payload,
            REQUEST_FIELDS,
            meta.get("plugin", PLUGIN_NAME),
        )
        normalised = _normalise_request(request_payload, core)
        if normalised["mode"] == "manual_mixed":
            response = _attach_time_frequency(
                build_response(normalised), request_payload, normalised, core
            )
            # 手动噪声不启动声学求解器，但仍登记任务目录，供工作区直接恢复结果。
            job_id, _job_dir = core._save_result_job(
                {"noise_request": request_payload, "noise_normalised": normalised},
                PLUGIN_NAME,
            )
            return response, job_id
        response, job_id = _build_distributed_response(normalised, core)
        return _attach_time_frequency(response, request_payload, normalised, core), job_id

    return handle_portal_payload(payload, PLUGIN_NAME, build_portal_response)


def create_router(openapi_extra):
    return create_portal_router(ROUTE_PATH, handle, openapi_extra)
