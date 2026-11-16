# -*- coding: utf-8 -*-
"""Bellhop、Kraken 与 RAM 三套独立 runner 的统一任务分发层。"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from backworker import acoustic_environment
from backworker.bellhop_model import adapter as bellhop_adapter
from backworker.bellhop_model import results as bellhop_results
from backworker.bellhop_model import runner as bellhop_runner
from backworker.ocean_environment_source import DEFAULT_SOURCE as environment_source
from backworker.kraken_model import adapter as kraken_adapter
from backworker.kraken_model.config import load_config as load_kraken_config
from backworker.kraken_model.results import load_field_result
from backworker.kraken_model.runner import KrakenRunner
from backworker.ram_model.config import load_config as load_ram_config
from backworker.ram_model.results import read_tl_grid
from backworker.ram_model.runner import RamRunner


MODEL_ALIASES = {
    "auto": "bellhop",
    "bellhop": "bellhop",
    "ray": "bellhop",
    "kraken": "kraken",
    "ram": "ram",
    "pe": "ram",
}


def normalise_acoustic_model(value: Any) -> str:
    """把接口中的模型名称整理为 bellhop、kraken 或 ram。"""
    model = str(value or "bellhop").strip().lower()
    if model not in MODEL_ALIASES:
        raise ValueError("acoustic_model 只能是 bellhop、kraken 或 ram。")
    return MODEL_ALIASES[model]


def _resolve_auto_receiver_depth_max(
    params: Dict[str, Any],
    context: Dict[str, Any],
    sections: list[Mapping[str, Any]],
) -> None:
    """把门户的自动接收深度解析为有效水域内的最大海深。"""
    if not bool(params.get("receiver_depth_auto_max")):
        return
    radius_km = float(params["radius_km"])
    valid_depths: list[np.ndarray] = []
    for section in sections:
        ranges = np.asarray(section["range_km"], dtype=np.float64)
        bottom = np.asarray(section["bottom_depth_m"], dtype=np.float64)
        keep = ranges <= radius_km + 1.0e-9
        raw_land = section.get("land_mask")
        land = np.zeros(ranges.shape, dtype=bool) if raw_land is None else np.asarray(raw_land, dtype=bool)
        if bottom.shape != ranges.shape or land.shape != ranges.shape:
            raise ValueError("自动接收深度所用的距离、地形或陆地掩膜形状不一致。")
        selected = np.flatnonzero(keep)
        if selected.size:
            selected_land = land[selected]
            first_land = np.flatnonzero(selected_land)
            if first_land.size:
                selected = selected[: int(first_land[0])]
        values = bottom[selected]
        values = values[np.isfinite(values) & (values > 1.0)]
        if values.size:
            valid_depths.append(values)
    if not valid_depths:
        raise ValueError("自动接收深度无法找到有效水域海深。")

    maximum_bottom = max(float(np.max(values)) for values in valid_depths)
    environment_depths = np.asarray(context["environment"]["depth_m"], dtype=np.float64)
    model_depth = float(np.nanmax(environment_depths))
    depth_guard = float(context["settings"].get("depth_guard_m", 10.0))
    source_depth = float(params["source_depth_m"])
    lower_bound = max(float(params.get("receiver_depth_min_m", 0.0)) + 1.0, source_depth)
    model_limit = model_depth - depth_guard
    if model_limit <= lower_bound:
        raise ValueError("环境数据最大深度无法覆盖声源深度或自动接收深度。")
    resolved = min(max(maximum_bottom, lower_bound), model_limit)
    params["receiver_depth_max_m"] = resolved
    context["tl_depth_max_m"] = resolved


def _prepare_sections(
    params: Dict[str, Any],
    config: Mapping[str, Any],
    progress,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """分别返回任务共享数据和各方位剖面，避免共享数据反向包含 section。"""
    params, points = acoustic_environment.prepare_common_task(params, config, max_sources=1)
    settings = acoustic_environment.environment_settings(params, config)
    model = str(params.get("acoustic_model", "")).strip().lower()
    kraken_config = None
    ram_config = None
    extra_range_km = float(settings["profile_extra_km"])
    if model == "kraken":
        # 每个求解器只从自己的配置文件读取统一命名的距离采样点数。
        config_file = config.get("CONFIG_FILES", {}).get("kraken")
        kraken_config = load_kraken_config(config_file)
        profile_count = int(kraken_config["N_2D_PROFILES"])
        extra_range_km = 0.0
    elif model == "ram":
        config_file = config.get("CONFIG_FILES", {}).get("ram")
        ram_config = load_ram_config(config_file)
        profile_count = int(ram_config["N_2D_PROFILES"])
    else:
        raise ValueError("公共二维环境准备只支持 Kraken 或 RAM。")
    environment = acoustic_environment.acquire(params, config, settings, points, environment_source)
    try:
        progress("1/4", "读取温盐、地形、波浪和海底声学数据。")
        context = environment.build_common_source_context(0)
        sections = environment.sample_sections(
            context,
            float(params["radius_km"]),
            extra_range_km=extra_range_km,
            profile_count=profile_count,
        )
        _resolve_auto_receiver_depth_max(params, context, sections)
        shared = {
            "params": params,
            "context": context,
            "config": config,
            "kraken_config": kraken_config,
            "ram_config": ram_config,
        }
        return shared, sections
    finally:
        environment.close()


def _bottom_values(
    section: Mapping[str, Any],
    within_calculation_range: np.ndarray | None = None,
) -> Dict[str, float]:
    """从计算半径内的海底声学字段中提取稳健代表值。"""
    fields = section.get("bottom_geoacoustics", {}).get("fields", {})
    defaults = {
        "sound_speed_mps": 1700.0,
        "shear_speed_mps": 0.0,
        "density_g_cm3": 1.8,
        "attenuation_db_per_wavelength": 0.5,
        "shear_attenuation_db_per_wavelength": 0.0,
    }
    mapping = {
        "sound_speed_mps": "compressional_speed_mps",
        "shear_speed_mps": "shear_speed_mps",
        "density_g_cm3": "density_g_cm3",
        "attenuation_db_per_wavelength": "compressional_attenuation",
        "shear_attenuation_db_per_wavelength": "shear_attenuation",
    }
    for output_name, field_name in mapping.items():
        values = np.asarray(fields.get(field_name, []), dtype=np.float64)
        if within_calculation_range is not None and values.size:
            mask = np.asarray(within_calculation_range, dtype=bool)
            if values.shape != mask.shape:
                raise ValueError(f"海底声学字段 {field_name!r} 与距离采样轴形状不一致。")
            values = values[mask]
        finite = values[np.isfinite(values)]
        if finite.size:
            defaults[output_name] = float(np.median(finite))
    return defaults


def _bottom_values_at(section: Mapping[str, Any], index: int) -> Dict[str, float]:
    """读取单个 Kraken 沿程剖面处的海底声学参数。"""
    fallback = _bottom_values(section)
    fields = section.get("bottom_geoacoustics", {}).get("fields", {})
    mapping = {
        "sound_speed_mps": "compressional_speed_mps",
        "shear_speed_mps": "shear_speed_mps",
        "density_g_cm3": "density_g_cm3",
        "attenuation_db_per_wavelength": "compressional_attenuation",
        "shear_attenuation_db_per_wavelength": "shear_attenuation",
    }
    result = dict(fallback)
    for output_name, field_name in mapping.items():
        values = np.asarray(fields.get(field_name, []), dtype=np.float64)
        if values.ndim == 1 and index < values.size and np.isfinite(values[index]):
            result[output_name] = float(values[index])
    return result


def _bottom_geoacoustics_changed(
    section: Mapping[str, Any],
    indices: np.ndarray,
) -> bool:
    """判断连续水体路径上的任一海底声学字段是否发生变化。"""
    fields = section.get("bottom_geoacoustics", {}).get("fields", {})
    selected_indices = np.asarray(indices, dtype=np.int64)
    for field_name in (
        "compressional_speed_mps",
        "shear_speed_mps",
        "density_g_cm3",
        "compressional_attenuation",
        "shear_attenuation",
    ):
        values = np.asarray(fields.get(field_name, []), dtype=np.float64)
        if not values.size:
            continue
        if values.ndim != 1 or np.any(selected_indices >= values.size):
            raise ValueError(f"海底声学字段 {field_name!r} 与距离采样轴形状不一致。")
        finite = values[selected_indices]
        finite = finite[np.isfinite(finite)]
        if finite.size > 1 and float(np.ptp(finite)) > 1.0e-9:
            return True
    return False


def _kraken_receiver_range_spec(
    params: Mapping[str, Any],
    receiver_count: int,
    default_range_count: int,
) -> Dict[str, float | int]:
    """返回覆盖请求半径的 Kraken/FIELD 公共距离网格。"""
    range_count = min(
        max(int(params.get("tl_n_range", default_range_count)), 2),
        max(2, 210000 // receiver_count),
        4093,
    )
    maximum_range = float(params["radius_km"])
    return {
        "MIN": max(0.001, maximum_range / range_count),
        "MAX": maximum_range,
        "COUNT": range_count,
    }


def _ram_native_grid_model(
    params: Mapping[str, Any],
    ram_config: Mapping[str, Any],
    maximum_range_m: float,
    output_depth_max_m: float,
) -> Dict[str, float | int]:
    """把门户 TL 点数转换成 RAM 原生步长和整数输出抽取倍数。

    RAM 在第一个正距离/深度输出数据，原生点数因此等于
    ``extent / (step * decimation)``，而不是包含零点的区间数。
    内部计算步长只允许相对配置默认值缩小，避免为了减少输出点数而
    降低求解精度。
    """
    model = ram_config["MODEL"]
    range_count = max(int(params.get("tl_n_range", ram_config["TL_N_RANGE"])), 2)
    depth_count = max(int(params.get("tl_n_rd", ram_config["TL_N_RD"])), 2)
    maximum_range = float(maximum_range_m)
    maximum_depth = float(output_depth_max_m)
    if maximum_range <= 0.0 or maximum_depth <= 0.0:
        raise ValueError("RAM 原生输出范围和深度必须为正数。")

    def axis_settings(extent: float, count: int, configured_step: float) -> tuple[float, int]:
        # RAM 1.5 用浮点除法后截断输出记录数；若间隔恰好等于 extent/count，
        # 输入文本的十进制舍入可能使最后一条记录少 1。极小幅度缩短间隔，
        # 保证请求的最后一个正坐标仍落在输出范围内。
        output_spacing = (extent / count) * (1.0 - 1.0e-7)
        decimation = max(1, int(math.ceil(output_spacing / configured_step)))
        calculation_step = output_spacing / decimation
        return calculation_step, decimation

    range_step, range_decimation = axis_settings(
        maximum_range,
        range_count,
        float(model["RANGE_STEP_M"]),
    )
    depth_step, depth_decimation = axis_settings(
        maximum_depth,
        depth_count,
        float(model["DEPTH_STEP_M"]),
    )
    return {
        "RANGE_STEP_M": range_step,
        "RANGE_OUTPUT_DECIMATION": range_decimation,
        "DEPTH_STEP_M": depth_step,
        "DEPTH_OUTPUT_DECIMATION": depth_decimation,
    }


def _kraken_profile_configs(
    shared: Mapping[str, Any],
    section: Mapping[str, Any],
    ice: bool,
) -> tuple[list[Dict[str, Any]], np.ndarray, np.ndarray, float | None, np.ndarray]:
    """把一个方位的距离相关环境转换为 Kraken 分段绝热剖面。"""
    params = shared["params"]
    kraken_config = shared.get("kraken_config")
    if kraken_config is None:
        config_file = shared["config"].get("CONFIG_FILES", {}).get("kraken")
        kraken_config = load_kraken_config(config_file)
    range_options = kraken_config["RANGE_DEPENDENT"]
    all_ranges = np.asarray(section["range_km"], dtype=np.float64)
    # 半径范围限制
    within = all_ranges <= float(params["radius_km"]) + 1.0e-9
    within_indices = np.flatnonzero(within)
    ranges = all_ranges[within]
    bottom = np.asarray(section["bottom_depth_m"], dtype=np.float64)[within]
    speed = np.asarray(section["sound_speed_mps"], dtype=np.float64)[:, within]
    if ranges.size < 1 or speed.shape[1] != ranges.size or bottom.shape != ranges.shape:
        raise ValueError("Kraken 沿程环境的距离、声速或地形数组形状无效。")
    raw_land = section.get("land_mask")
    if raw_land is None:
        land = np.zeros(all_ranges.shape, dtype=bool)[within]
    else:
        land_array = np.asarray(raw_land, dtype=bool)
        if land_array.shape != all_ranges.shape:
            raise ValueError("Kraken 沿程陆地掩膜与距离轴形状不一致。")
        land = land_array[within]
    if bool(land[0]):
        raise ValueError("Kraken 声源位置位于陆地，无法计算水下传播。")
    land_start_km: float | None = None
    # np.flatnonzero返回非0/true的索引
    land_indices = np.flatnonzero(land)
    if land_indices.size:
        first_land_index = int(land_indices[0])
        land_start_km = float(ranges[first_land_index])
        # Kraken 不能让同一条绝热传播路径越过陆地后再进入另一片水体。
        # 只把声源到首次陆地点之前的连续水体交给模态求解器。
        ranges = ranges[:first_land_index]
        bottom = bottom[:first_land_index]
        speed = speed[:, :first_land_index]
        within_indices = within_indices[:first_land_index]
    if ranges.size < 1:
        raise ValueError("Kraken 声源位置之后没有可计算的连续水体距离。")
    if not np.all(np.isfinite(bottom)):
        raise ValueError("Kraken 连续水体路径内包含无效水深。")
    source_depth = float(params["source_depth_m"])
    if float(bottom[0]) <= source_depth:
        raise ValueError("Kraken 声源位置水深不大于声源深度。")
    # np.ptp计算极差（极大值-极小值）
    bathymetry_changed = float(np.ptp(bottom)) > float(range_options["BATHYMETRY_TOLERANCE_M"])
    ssp_amplitude = float(np.nanmax(np.ptp(speed, axis=1)))
    ssp_changed = ssp_amplitude > float(range_options["SOUND_SPEED_TOLERANCE_MPS"])
    water_requested = str(shared["context"]["settings"].get("ssp_mode", "fixed")) == "range_dependent"
    bottom_geo_changed = _bottom_geoacoustics_changed(section, within_indices)
    # 接口或公共环境已明确指定“水体变化”时必须分段；ENABLED
    # 只控制在未明确指定时，是否按实际 SSP/地形变化自动分段。
    crosses_source_depth = bool(np.any(bottom[1:] <= source_depth))
    use_multiple = water_requested or crosses_source_depth or bottom_geo_changed or (
        bool(range_options.get("ENABLED", True)) and (ssp_changed or bathymetry_changed)
    )
    profile_indices = list(range(ranges.size)) if use_multiple else [0]
    profile_ranges = ranges if use_multiple else np.asarray([0.0], dtype=np.float64)

    configured_receivers = np.asarray(
        kraken_config["FIELD"]["RECEIVER_DEPTHS_M"], dtype=np.float64
    )
    requested_receiver_max = float(
        params.get("receiver_depth_max_m", np.max(configured_receivers))
    )
    receiver_count = min(
        max(int(params.get("tl_n_rd", kraken_config["TL_N_RD"])), 2),
        200,
    )
    # FIELD 输出使用请求的统一深度轴；各 ENV 在下面按当地水深生成自己的
    # 合法模态接收深度。这样局部浅水不会截掉其他深水距离处的有效结果。
    output_receivers = np.linspace(0.0, requested_receiver_max, receiver_count)
    full_receiver_ranges = _kraken_receiver_range_spec(
        params,
        receiver_count,
        int(kraken_config["TL_N_RANGE"]),
    )
    receiver_ranges = dict(full_receiver_ranges)
    if land_start_km is not None:
        full_grid = np.linspace(
            float(full_receiver_ranges["MIN"]),
            float(full_receiver_ranges["MAX"]),
            int(full_receiver_ranges["COUNT"]),
        )
        first_blocked = min(int(np.searchsorted(full_grid, land_start_km, side="left")), full_grid.size - 1)
        if first_blocked == 0:
            # FIELD 至少需要两个不同的接收距离；这种极近岸情形的输出会全部被掩膜。
            receiver_ranges["MAX"] = max(land_start_km, float(full_grid[0]) * 2.0)
            receiver_ranges["COUNT"] = 2
        else:
            # 多算到第一个落在陆地上的公共网格点，读取后立即掩膜；其后不再计算。
            receiver_ranges["MAX"] = float(full_grid[first_blocked])
            receiver_ranges["COUNT"] = first_blocked + 1
    depth_axis = np.asarray(shared["context"]["environment"]["depth_m"], dtype=np.float64)
    profiles: list[Dict[str, Any]] = []
    config_file = shared["config"].get("CONFIG_FILES", {}).get("kraken")
    for profile_number, section_index in enumerate(profile_indices, 1):
        # 未分段时用全程最浅水深代表该范围无关环境；分段时使用当地水深。
        water_depth = float(bottom[section_index]) if use_multiple else float(np.min(bottom))
        local_receiver_max = min(requested_receiver_max, water_depth - 1.0e-3)
        local_receivers = np.linspace(0.0, local_receiver_max, receiver_count).tolist()
        bottom_values = _bottom_values_at(section, int(within_indices[section_index]))
        # 物理声源只位于首剖面。后续剖面若浅于真实源深，则复用共同接收
        # 深度轴中的最深点作为局部模态采样点，避免把物理声源复制到海底中。
        mode_sample_depth = source_depth
        if profile_number > 1 and mode_sample_depth >= water_depth:
            mode_sample_depth = float(local_receivers[-1])
        overrides = kraken_adapter.build_overrides_from_backend(
            params,
            depth_axis,
            speed[:, section_index],
            water_depth_m=water_depth,
            mode_sample_depth_m=mode_sample_depth,
            receiver_depths_m=local_receivers,
            bottom={
                "COMPRESSIONAL_SPEED_MPS": bottom_values["sound_speed_mps"],
                "SHEAR_SPEED_MPS": bottom_values["shear_speed_mps"],
                "DENSITY_G_CM3": bottom_values["density_g_cm3"],
                "COMPRESSIONAL_ATTENUATION": bottom_values["attenuation_db_per_wavelength"],
                "SHEAR_ATTENUATION": bottom_values["shear_attenuation_db_per_wavelength"],
            },
        )
        overrides["ENVIRONMENT"]["TITLE"] = f"Kraken adiabatic profile {profile_number:03d}"
        overrides["ENVIRONMENT"]["MAX_RANGE_KM"] = float(receiver_ranges["MAX"])
        overrides["ENVIRONMENT"]["PHASE_SPEED_MAX_MPS"] = max(
            float(kraken_config["ENVIRONMENT"].get("PHASE_SPEED_MAX_MPS", 0.0)),
            float(bottom_values["sound_speed_mps"]),
        )
        overrides["FIELD"].update({
            "MODE_THEORY": "A",
            "RECEIVER_RANGES_KM": receiver_ranges,
        })
        if ice:
            ice_config = kraken_config.get("ICE", {})
            overrides["ENVIRONMENT"].update({
                "TOP_BOUNDARY": ice_config.get("TOP_BOUNDARY", "T"),
                "TWERSKY": {
                    "BUMDEN": float(shared["context"]["settings"].get("ice_reflection_bumden_domain") or ice_config.get("BUMDEN", 0.01)),
                    "ETA_M": float(ice_config.get("ETA_M", 8.2)),
                    "XI_M": float(ice_config.get("XI_M", 5.1)),
                },
            })
        profile_config = load_kraken_config(config_file, overrides)
        profiles.append({
            "range_km": float(profile_ranges[profile_number - 1]),
            "config": profile_config,
        })
    return profiles, ranges, bottom, land_start_km, output_receivers


def _kraken_section(shared: Mapping[str, Any], section: Mapping[str, Any], bearing_index: int, job_dir: Path, ice: bool) -> Dict[str, Any]:
    """运行一个方位的 Kraken/FIELD 并返回统一二维 TL 数据。"""
    profiles, environment_ranges, bottom, land_start_km, output_depths = _kraken_profile_configs(
        shared, section, ice
    )
    # profiles 中的 ENV 使用各自局部合法深度；单独的 FIELD 配置负责声明
    # 统一输出深度轴。KrakenRunner 写 ENV 时读取 profiles，写 FLP 时读取 cfg。
    field_config = copy.deepcopy(profiles[0]["config"])
    field_config["FIELD"]["RECEIVER_DEPTHS_M"] = output_depths.tolist()
    result = KrakenRunner(field_config, profiles=profiles).run(
        output_dir=job_dir / "native" / f"bearing_{bearing_index:03d}"
    )
    field = load_field_result(result["shade_file"])
    field_ranges = np.asarray(field["range_km"], dtype=np.float64)
    depths = np.asarray(field["receiver_depth_m"], dtype=np.float64)
    native_tl = np.asarray(field["tl"], dtype=np.float64)[0, 0]
    output_ranges = field_ranges
    tl = native_tl.copy()
    output_bottom = np.interp(output_ranges, environment_ranges, bottom)
    if land_start_km is not None:
        kraken_config = shared.get("kraken_config", field_config)
        range_spec = _kraken_receiver_range_spec(
            shared["params"],
            depths.size,
            int(kraken_config.get("TL_N_RANGE", 1001)),
        )
        output_ranges = np.linspace(
            float(range_spec["MIN"]),
            float(range_spec["MAX"]),
            int(range_spec["COUNT"]),
        )
        water_output = output_ranges < land_start_km - 1.0e-9
        water_native = field_ranges < land_start_km - 1.0e-9
        tl = np.full((native_tl.shape[0], output_ranges.size), 200.0, dtype=np.float64)
        if np.any(water_output) and np.any(water_native):
            for depth_index, row in enumerate(native_tl):
                tl[depth_index, water_output] = np.interp(
                    output_ranges[water_output],
                    field_ranges[water_native],
                    row[water_native],
                )
        output_bottom = np.interp(output_ranges, environment_ranges, bottom)
        blocked = ~water_output
        # 保留陆地地形标记；传播损失在首次陆地点及其后方统一为 200 dB。
        output_bottom[blocked] = 1.0
    land_columns = output_bottom <= 1.0 + 1.0e-9
    below_local_bottom = depths[:, None] >= output_bottom[None, :] - 1.0e-9
    tl[below_local_bottom & ~land_columns[None, :]] = np.nan
    tl[:, land_columns] = 200.0
    return {
        "range_km": output_ranges,
        "depth_m": depths,
        "tl_db": tl,
        "bottom_depth_m": output_bottom,
        "land_start_km": land_start_km,
    }


def _ram_section(shared: Mapping[str, Any], section: Mapping[str, Any], bearing_index: int, job_dir: Path) -> Dict[str, Any]:
    """运行一个方位的距离相关 RAM 并返回统一二维 TL 数据。"""
    params = shared["params"]
    ranges_km = np.asarray(section["range_km"], dtype=np.float64)
    keep = ranges_km <= float(params["radius_km"]) + 1.0e-9
    ranges_m = ranges_km[keep] * 1000.0
    source_indices = np.flatnonzero(keep)
    bottom = np.asarray(section["bottom_depth_m"], dtype=np.float64)[keep]
    speed = np.asarray(section["sound_speed_mps"], dtype=np.float64)[:, keep]
    depth = np.asarray(shared["context"]["environment"]["depth_m"], dtype=np.float64)
    local_bottom_max_m = float(np.nanmax(bottom))
    # 门户会把所有方位合并到同一个 depth × theta × range 网格。RAM 的
    # tl.grid 深度轴由 OUTPUT_DEPTH_MAX_M 决定，因此各方位不能再使用各自
    # 的局部最大海深；否则合并时第一个较浅方位会把其他深水方位截断。
    # 自动深度在 _resolve_auto_receiver_depth_max 中已经解析为全区域可达的
    # 最大海深；显式 receiverDepthMax 则按用户指定的统一输出深度处理。
    output_depth_max_m = float(
        params.get("receiver_depth_max_m", local_bottom_max_m)
    )
    if not math.isfinite(output_depth_max_m) or output_depth_max_m <= 0.0:
        raise ValueError("RAM 统一输出深度必须为正的有限数值。")
    # 求解域既要容纳本方位全部地形，也要覆盖统一输出深度，并在其下方
    # 保留 RAM 海底吸收层；输出深度本身不包含这段数值缓冲区。
    physical_depth_max_m = max(local_bottom_max_m, output_depth_max_m)
    max_depth = max(physical_depth_max_m + 300.0, physical_depth_max_m * 1.5)
    maximum_range_m = float(params["radius_km"]) * 1000.0
    if ranges_m[-1] < maximum_range_m:
        ranges_m = np.append(ranges_m, maximum_range_m)
        bottom = np.append(bottom, bottom[-1])
        speed = np.column_stack((speed, speed[:, -1]))
        source_indices = np.append(source_indices, source_indices[-1])
    profiles = []
    for index in range(max(1, len(ranges_m) - 1)):
        bottom_values = _bottom_values_at(section, int(source_indices[index]))
        profiles.append({
            "START_RANGE_M": float(ranges_m[index]),
            "WATER_SOUND_SPEED_MPS": [
                [float(z), float(c)] for z, c in zip(depth, speed[:, index])
            ],
            "BOTTOM_SOUND_SPEED_MPS": [[0.0, bottom_values["sound_speed_mps"]]],
            "BOTTOM_DENSITY_G_CM3": [[0.0, bottom_values["density_g_cm3"]]],
            "BOTTOM_ATTENUATION_DB_PER_WAVELENGTH": [
                [0.0, bottom_values["attenuation_db_per_wavelength"]],
                [max_depth - 100.0, bottom_values["attenuation_db_per_wavelength"]],
                [max_depth, 10.0],
            ],
        })
    ram_config = shared.get("ram_config")
    if ram_config is None:
        ram_config = load_ram_config(shared["config"]["CONFIG_FILES"]["ram"])
    native_grid = _ram_native_grid_model(
        params,
        ram_config,
        maximum_range_m,
        output_depth_max_m,
    )
    overrides = {"MODEL": {"FREQUENCY_HZ": float(params["freq_hz"]), "SOURCE_DEPTH_M": float(params["source_depth_m"]), "RECEIVER_DEPTH_M": float(params.get("horizontal_tl_depth_m", params["source_depth_m"])), "MAX_RANGE_M": maximum_range_m, "MAX_DEPTH_M": max_depth, "OUTPUT_DEPTH_MAX_M": output_depth_max_m, **native_grid}, "ENVIRONMENT": {"BATHYMETRY": [[float(r), float(z)] for r, z in zip(ranges_m, bottom)], "PROFILES": profiles}}
    cfg = load_ram_config(shared["config"]["CONFIG_FILES"]["ram"], overrides)
    result = RamRunner(cfg).run(output_dir=job_dir / "native" / f"bearing_{bearing_index:03d}")
    grid = read_tl_grid(result["grid_file"])
    grid_range = np.asarray(grid["range_m"], dtype=np.float64) / 1000.0
    return {"range_km": grid_range, "depth_m": np.asarray(grid["depth_m"]), "tl_db": np.asarray(grid["tl_db"]), "bottom_depth_m": np.interp(grid_range * 1000.0, ranges_m, bottom)}


def _export(job_dir: Path, shared: Mapping[str, Any], solved: list[Mapping[str, Any]], model: str, started: float) -> Dict[str, Any]:
    """把 Kraken/RAM 多方位结果转换成现有门户响应所需的标准 NetCDF。"""
    first = solved[0]
    ranges = np.asarray(first["range_km"], dtype=np.float64)
    depths = np.asarray(first["depth_m"], dtype=np.float64)
    layers, bottoms = [], []
    for item in solved:
        tl = RegularGridInterpolator(
            (np.asarray(item["depth_m"]), np.asarray(item["range_km"])),
            np.asarray(item["tl_db"]),
            bounds_error=False,
            fill_value=np.nan,
        )(np.stack(np.meshgrid(depths, ranges, indexing="ij"), axis=-1))
        layers.append(tl)
        bottoms.append(np.interp(ranges, item["range_km"], item["bottom_depth_m"]))
    context = shared["context"]
    source = {"index":1,"lat":context["source_lat"],"lon":context["source_lon"],"x_km":0.0,"y_km":0.0,"source_depth_m":float(shared["params"]["source_depth_m"]),"ssp_depth_m":context["environment"]["depth_m"],"ssp_sound_speed_mps":context["environment"]["sound_speed_mps"][:, int(context["environment"]["sound_speed_mps"].shape[1]//2), int(context["environment"]["sound_speed_mps"].shape[2]//2)],"reverberation_boundaries":[],"tl":{"theta_deg":np.asarray(context["bearings"]),"depth_m":depths,"range_km":ranges,"layers":np.stack(layers,axis=1),"bottom_depth_m":np.stack(bottoms)}}
    bellhop_results.write_tl3d_nc([source], job_dir / "tl3d_depth_theta_range.nc")
    bellhop_results.write_tl_section_nc(source, shared["params"], job_dir / "tl2d_section.nc")
    (job_dir / "interactive_panels.json").write_text(json.dumps({"sources":[{"ssp":{"depth_m":source["ssp_depth_m"].tolist(),"sound_speed_mps":source["ssp_sound_speed_mps"].tolist()}}]}, ensure_ascii=False), encoding="utf-8")
    elapsed = time.time() - started
    bottom_geo = context.get("bottom_geoacoustics", {})
    used_bottom_fields = (
        ["compressional_speed_mps", "shear_speed_mps", "density_g_cm3", "compressional_attenuation", "shear_attenuation"]
        if model == "kraken"
        else ["compressional_speed_mps", "density_g_cm3", "compressional_attenuation"]
    )
    summary = {
        "status": "ready",
        "acoustic_model": model,
        "elapsed": elapsed,
        "bearing_count": len(solved),
        "bottom_geoacoustics": {
            "mode": bottom_geo.get("mode", "fixed"),
            "source": bottom_geo.get("source", ""),
            "used_fields": used_bottom_fields,
            "unsupported_fields": ["shear_speed_mps", "shear_attenuation"] if model == "ram" else [],
        },
        "ice_physics_note": "RAM 1.5 冰区仅附加冰厚输出，不包含冰盖边界物理。" if model == "ram" and shared["params"].get("ice_reflection_enabled") else "",
    }
    (job_dir / "result_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("visual_ready.flag","backend_export_done.flag","result_ready.flag"):
        (job_dir / name).write_text(json.dumps({"status":"ready","elapsed":elapsed}), encoding="utf-8")
    return {"status":"ready","elapsed":elapsed,"files":["tl2d_section.nc","tl3d_depth_theta_range.nc"]}


def run(job_dir: str, params: Dict[str, Any], config: Mapping[str, Any], progress_callback=None) -> Dict[str, Any]:
    """按 acoustic_model 选择独立 Bellhop、Kraken 或 RAM runner。"""
    model = normalise_acoustic_model(params.get("acoustic_model"))
    params["acoustic_model"] = model
    if model == "bellhop":
        bellhop_params = bellhop_adapter.build_overrides_from_backend(
            params,
            bellhop_runner._task_defaults(config),
        )
        return bellhop_runner.run(job_dir, bellhop_params, config, progress_callback=progress_callback)

    if model not in {"kraken", "ram"}:
        raise ValueError(f"不支持的声学模型：{model}")

    progress = progress_callback or (lambda stage, message: None)
    started = time.time()
    path = Path(job_dir)
    shared, sections = _prepare_sections(params, config, progress)
    if "receiver_depth_max_m" in shared["params"]:
        params["receiver_depth_max_m"] = float(shared["params"]["receiver_depth_max_m"])
    ice = bool(params.get("ice_reflection_enabled"))
    solved = []
    for index, section in enumerate(sections, 1):
        progress("2/4", f"{model} 正在计算方位 {index}/{len(sections)}。")
        if model == "kraken":
            result = _kraken_section(shared, section, index, path, ice)
        elif model == "ram":
            result = _ram_section(shared, section, index, path)
        else:
            raise ValueError(f"不支持的声学模型：{model}")
        solved.append(result)
    progress("4/4", "正在导出统一二维/三维传播损失结果。")
    return _export(path, shared, solved, model, started)
