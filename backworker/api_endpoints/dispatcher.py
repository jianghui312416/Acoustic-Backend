"""当前前端使用的统一门户入口。"""

from __future__ import annotations

import asyncio
import atexit
import multiprocessing
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable

from fastapi import APIRouter, Request

from .analyse_2d_rt_rd import handle as handle_analyse_2d_rt
from .analyse_2d_rt_rd_dynamic import handle as handle_analyse_2d_rt_dynamic
from .analyse_2d_tl_rd import handle as handle_analyse_2d_tl
from .analyse_3d_rt_rd_dynamic import handle as handle_analyse_3d_rt_dynamic
from .analyse_3d_tl_rd import handle as handle_analyse_3d_tl
from .analyse_env_noise import handle as handle_environment_noise
from .analyse_forecast_environment_noise import handle as handle_forecast_environment_noise
from .analyse_regional_forecast_environment_noise import handle as handle_regional_forecast_environment_noise
from .get_dataset_list import handle as handle_dataset_list
from .get_dataset_time_list import handle as handle_dataset_time_list
from .get_env_data_region import handle as handle_env_data_region
from .get_env_data_route import handle as handle_env_data_route
from .get_realtime_ais import handle as handle_realtime_ais
from .get_regional_weather_forecast import handle as handle_regional_weather_forecast
from .get_regional_ocean_current import handle as handle_regional_ocean_current
from .get_seafloor_features import handle as handle_seafloor_features
from .get_weather_forecast import handle as handle_weather_forecast
from .ice_2d_rt import handle as handle_ice_2d_rt
from .ice_2d_rt_dynamic import handle as handle_ice_2d_rt_dynamic
from .ice_2d_tl import handle as handle_ice_2d_tl
from .ice_3d_rt_rd_dynamic import handle as handle_ice_3d_rt_dynamic
from .ice_3d_tl import handle as handle_ice_3d_tl
from .ice_kraken_2d_tl import handle as handle_ice_kraken_2d_tl
from .ice_kraken_3d_tl import handle as handle_ice_kraken_3d_tl
from .ice_ram_2d_tl import handle as handle_ice_ram_2d_tl
from .ice_ram_3d_tl import handle as handle_ice_ram_3d_tl
from .kraken_2d_tl import handle as handle_kraken_2d_tl
from .kraken_3d_tl import handle as handle_kraken_3d_tl
from .ram_2d_tl import handle as handle_ram_2d_tl
from .ram_3d_tl import handle as handle_ram_3d_tl
from .registry import PORTAL_PLUGIN_SN_MAP


PortalHandler = Callable[[Any], Any]

# NetCDF4/HDF5 的 Windows 原生库不能安全地在线程间共享。声场任务固定交给
# 最多四个 spawn 子进程；每个进程拥有独立的原生库状态。
ACOUSTIC_PROCESS_COUNT = 4
_ACOUSTIC_PLUGIN_MARKERS = ("TLRD", "RTRD", "ImpulseResponse")
_ACOUSTIC_PROCESS_POOL: ProcessPoolExecutor | None = None


def _requires_acoustic_process(plugin_name: str) -> bool:
    return any(marker in plugin_name for marker in _ACOUSTIC_PLUGIN_MARKERS)


def _payload_plugin_name(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""
    data = payload.get("Data")
    if not isinstance(data, Mapping):
        return ""
    units = data.get("Units")
    if not isinstance(units, list) or not units or not isinstance(units[0], Mapping):
        return ""
    return str(units[0].get("Plugin") or "").strip()


def _get_acoustic_process_pool() -> ProcessPoolExecutor:
    global _ACOUSTIC_PROCESS_POOL
    if _ACOUSTIC_PROCESS_POOL is None:
        _ACOUSTIC_PROCESS_POOL = ProcessPoolExecutor(
            max_workers=ACOUSTIC_PROCESS_COUNT,
            mp_context=multiprocessing.get_context("spawn"),
        )
    return _ACOUSTIC_PROCESS_POOL


def _discard_acoustic_process_pool(pool: ProcessPoolExecutor | None = None) -> None:
    global _ACOUSTIC_PROCESS_POOL
    current = _ACOUSTIC_PROCESS_POOL
    if current is None or (pool is not None and current is not pool):
        return
    _ACOUSTIC_PROCESS_POOL = None
    current.shutdown(wait=False, cancel_futures=True)


atexit.register(_discard_acoustic_process_pool)

PORTAL_HANDLERS: dict[str, PortalHandler] = {
    "Analyse2DTLRD": handle_analyse_2d_tl,
    "Analyse3DTLRD": handle_analyse_3d_tl,
    "Analyse2DRTRD": handle_analyse_2d_rt,
    "Analyse2DRTRDDynamic": handle_analyse_2d_rt_dynamic,
    "Analyse3DRTRDDynamic": handle_analyse_3d_rt_dynamic,
    "AnalyseIce2DTLRD": handle_ice_2d_tl,
    "AnalyseIce3DTLRD": handle_ice_3d_tl,
    "AnalyseIce2DRTRD": handle_ice_2d_rt,
    "AnalyseIce2DRTRDDynamic": handle_ice_2d_rt_dynamic,
    "AnalyseIce3DRTRDDynamic": handle_ice_3d_rt_dynamic,
    "AnalyseKraken2DTLRD": handle_kraken_2d_tl,
    "AnalyseKraken3DTLRD": handle_kraken_3d_tl,
    "AnalyseIceKraken2DTLRD": handle_ice_kraken_2d_tl,
    "AnalyseIceKraken3DTLRD": handle_ice_kraken_3d_tl,
    "Analyse2DTLRDPE": handle_ram_2d_tl,
    "AnalyseRam3DTLRD": handle_ram_3d_tl,
    "AnalyseIce2DTLRDPE": handle_ice_ram_2d_tl,
    "AnalyseIceRam3DTLRD": handle_ice_ram_3d_tl,
    "GetEnvDataRoute": handle_env_data_route,
    "GetEnvDataRegion": handle_env_data_region,
    "GetEnvDataSourceList": handle_dataset_list,
    "FetchTimeToList": handle_dataset_time_list,
    "AnalyseEnvironmentNoise": handle_environment_noise,
    "GetWeatherForecast": handle_weather_forecast,
    "GetRealtimeAIS": handle_realtime_ais,
    "AnalyseForecastEnvironmentNoise": handle_forecast_environment_noise,
    "GetRegionalWeatherForecast": handle_regional_weather_forecast,
    "AnalyseRegionalForecastEnvironmentNoise": handle_regional_forecast_environment_noise,
    "GetRegionalOceanCurrent": handle_regional_ocean_current,
    "GetSeafloorFeatures": handle_seafloor_features,
}

PORTAL_ENDPOINTS = {
    name: (PORTAL_PLUGIN_SN_MAP[name], handler)
    for name, handler in PORTAL_HANDLERS.items()
}


def _error_meta(payload: Any, plugin_name: str = "") -> dict[str, Any] | None:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("Data"), Mapping):
        return None
    data = payload["Data"]
    if not plugin_name:
        units = data.get("Units")
        if isinstance(units, list) and units and isinstance(units[0], Mapping):
            plugin_name = str(units[0].get("Plugin") or "").strip()
    return {
        "wrapped": True,
        "technical_support": str(payload.get("Technical Support") or payload.get("TechnicalSupport") or "XWOcean"),
        "sn": data.get("SN", ""),
        "task_id": data.get("TaskId", ""),
        "self_order": data.get("SelfOrder", 0),
        "plugin": plugin_name,
    }


def _parse_sn(value: Any) -> int:
    if isinstance(value, bool) or value is None or value == "":
        raise ValueError
    number = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError
    if isinstance(value, str) and value.strip() != str(number):
        raise ValueError
    return number


def resolve_portal_handler(payload: Any) -> tuple[int, str, PortalHandler]:
    from .. import backend_worker as core

    if not isinstance(payload, Mapping):
        raise core.PortalRequestError("请求体必须是 JSON 对象。")
    data = payload.get("Data")
    if not isinstance(data, Mapping):
        raise core.PortalRequestError("统一入口请求体缺少 Data 对象。")
    try:
        sn = _parse_sn(data.get("SN"))
    except (TypeError, ValueError) as exc:
        raise core.PortalRequestError("Data.SN 必须是已登记的整数接口编号。") from exc
    units = data.get("Units")
    if not isinstance(units, list) or len(units) != 1 or not isinstance(units[0], Mapping):
        raise core.PortalRequestError("Data.Units 必须且只能包含一个对象。")
    plugin_name = str(units[0].get("Plugin") or "").strip()
    endpoint = PORTAL_ENDPOINTS.get(plugin_name)
    if endpoint is None:
        raise core.PortalRequestError(f"Plugin='{plugin_name}' 不属于当前前端后端接口。")
    expected_sn, handler = endpoint
    if sn != expected_sn:
        raise core.PortalRequestError(f"Plugin='{plugin_name}' 必须对应 SN={expected_sn}，当前 SN={sn}。")
    return sn, plugin_name, handler


def dispatch(payload: Any):
    from .. import backend_worker as core

    plugin_name = ""
    try:
        _sn, plugin_name, handler = resolve_portal_handler(payload)
        return handler(payload)
    except core.PortalRequestError as exc:
        meta = _error_meta(payload, plugin_name)
        error_plugin = str(meta.get("plugin") or "") if meta else ""
        return core._portal_failure(exc, 400, meta, error_plugin)


async def dispatch_async(payload: Any):
    """声场任务进四进程池；普通门户请求继续使用轻量工作线程。"""
    if not _requires_acoustic_process(_payload_plugin_name(payload)):
        return await asyncio.to_thread(dispatch, payload)

    loop = asyncio.get_running_loop()
    for attempt in range(2):
        pool = _get_acoustic_process_pool()
        try:
            return await loop.run_in_executor(pool, dispatch, payload)
        except BrokenProcessPool:
            # 原生求解器异常退出只淘汰子进程池，不再带崩主后端；重建后重试一次。
            _discard_acoustic_process_pool(pool)
            if attempt:
                raise
    raise RuntimeError("声场计算进程池不可用。")


def create_router():
    router = APIRouter()

    @router.post("/", summary="当前前端统一门户接口")
    async def unified_portal_endpoint(request: Request):
        from .. import backend_worker as core

        payload = await core._request_payload(request)
        return await dispatch_async(payload)

    return router
