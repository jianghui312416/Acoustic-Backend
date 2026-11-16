# -*- coding: utf-8 -*-
"""RAM 独立配置的读取、覆盖合并、路径解析与参数校验。"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = MODULE_DIR.parent.parent / "config" / "ram_config.json"


def _deep_merge(target: MutableMapping[str, Any], override: Mapping[str, Any]) -> None:
    """递归合并配置覆盖值，并保护调用方传入的对象不被修改。"""
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _resolve_path(value: Any, config_file: Path) -> str:
    """把相对路径按配置文件所在目录转换为绝对路径。"""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_file.parent / path
    return str(path.resolve())


def _finite(value: Any, name: str) -> float:
    """读取有限浮点数并提供包含字段名的中文错误。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数值。") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值。")
    return number


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    """读取不小于指定下限的整数。"""
    number = _finite(value, name)
    integer = int(round(number))
    if abs(number - integer) > 1.0e-9 or integer < minimum:
        raise ValueError(f"{name} 必须是不小于 {minimum} 的整数。")
    return integer


def _validate_pairs(values: Any, name: str, *, increasing_depth: bool = True) -> list[tuple[float, float]]:
    """校验 RAM 使用的 `[坐标, 数值]` 二元数组。"""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"{name} 必须是非空二元数组。")
    pairs = []
    for index, row in enumerate(values, 1):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) < 2:
            raise ValueError(f"{name} 第 {index} 行必须包含两个数值。")
        pairs.append((_finite(row[0], f"{name} 第 {index} 行坐标"), _finite(row[1], f"{name} 第 {index} 行数值")))
    if increasing_depth and any(pairs[index][0] <= pairs[index - 1][0] for index in range(1, len(pairs))):
        raise ValueError(f"{name} 的第一列必须严格递增。")
    return pairs


def validate_config(config: Mapping[str, Any]) -> None:
    """校验 RAM 网格、环境、剖面以及求解器数值参数。"""
    if _integer(config.get("N_2D_PROFILES"), "N_2D_PROFILES") < 3:
        raise ValueError("N_2D_PROFILES 至少为 3。")
    if _integer(config.get("TL_N_RANGE"), "TL_N_RANGE") < 2 or _integer(config.get("TL_N_RD"), "TL_N_RD") < 2:
        raise ValueError("TL_N_RANGE 和 TL_N_RD 至少为 2。")
    model = config.get("MODEL")
    environment = config.get("ENVIRONMENT")
    runtime = config.get("RUNTIME")
    if not isinstance(model, Mapping) or not isinstance(environment, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("配置必须包含 MODEL、ENVIRONMENT 和 RUNTIME 对象。")
    positive_fields = (
        "FREQUENCY_HZ", "MAX_RANGE_M", "RANGE_STEP_M", "MAX_DEPTH_M",
        "DEPTH_STEP_M", "OUTPUT_DEPTH_MAX_M", "REFERENCE_SOUND_SPEED_MPS",
    )
    for key in positive_fields:
        if _finite(model.get(key), f"MODEL.{key}") <= 0.0:
            raise ValueError(f"MODEL.{key} 必须为正数。")
    source_depth = _finite(model.get("SOURCE_DEPTH_M"), "MODEL.SOURCE_DEPTH_M")
    receiver_depth = _finite(model.get("RECEIVER_DEPTH_M"), "MODEL.RECEIVER_DEPTH_M")
    maximum_depth = float(model["MAX_DEPTH_M"])
    if source_depth <= 0.0 or source_depth >= maximum_depth or receiver_depth < 0.0 or receiver_depth >= maximum_depth:
        raise ValueError("声源和接收深度必须位于计算深度范围内。")
    if float(model["OUTPUT_DEPTH_MAX_M"]) > maximum_depth:
        raise ValueError("OUTPUT_DEPTH_MAX_M 不能超过 MAX_DEPTH_M。")
    _integer(model.get("RANGE_OUTPUT_DECIMATION"), "MODEL.RANGE_OUTPUT_DECIMATION")
    _integer(model.get("DEPTH_OUTPUT_DECIMATION"), "MODEL.DEPTH_OUTPUT_DECIMATION")
    _integer(model.get("PADE_TERMS"), "MODEL.PADE_TERMS")
    if int(model["PADE_TERMS"]) > 16:
        raise ValueError("当前 ram.exe 的 PADE_TERMS 不能超过 16。")
    _integer(model.get("STABILITY_CONSTRAINTS"), "MODEL.STABILITY_CONSTRAINTS", minimum=0)
    _integer(runtime.get("TIMEOUT_SECONDS"), "RUNTIME.TIMEOUT_SECONDS")
    bathymetry = _validate_pairs(environment.get("BATHYMETRY"), "ENVIRONMENT.BATHYMETRY")
    if bathymetry[0][0] > 0.0 or bathymetry[-1][0] < float(model["MAX_RANGE_M"]):
        raise ValueError("BATHYMETRY 必须从 0 m 以前开始并覆盖 MAX_RANGE_M。")
    if any(depth <= 0.0 or depth >= maximum_depth for _, depth in bathymetry):
        raise ValueError("海底深度必须大于 0 且小于 MAX_DEPTH_M。")
    profiles = environment.get("PROFILES")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("ENVIRONMENT.PROFILES 必须是非空数组。")
    previous_range = -math.inf
    for index, profile in enumerate(profiles, 1):
        if not isinstance(profile, Mapping):
            raise ValueError(f"第 {index} 个环境剖面必须是对象。")
        start_range = _finite(profile.get("START_RANGE_M"), f"第 {index} 个剖面 START_RANGE_M")
        if start_range < 0.0 or start_range <= previous_range:
            raise ValueError("环境剖面的 START_RANGE_M 必须从 0 开始并严格递增。")
        if index == 1 and abs(start_range) > 1.0e-6:
            raise ValueError("第一个环境剖面的 START_RANGE_M 必须为 0。")
        if start_range >= float(model["MAX_RANGE_M"]) and index > 1:
            raise ValueError("环境剖面起始距离必须小于 MAX_RANGE_M。")
        previous_range = start_range
        water = _validate_pairs(profile.get("WATER_SOUND_SPEED_MPS"), f"第 {index} 个剖面 WATER_SOUND_SPEED_MPS")
        bottom_speed = _validate_pairs(profile.get("BOTTOM_SOUND_SPEED_MPS"), f"第 {index} 个剖面 BOTTOM_SOUND_SPEED_MPS")
        density = _validate_pairs(profile.get("BOTTOM_DENSITY_G_CM3"), f"第 {index} 个剖面 BOTTOM_DENSITY_G_CM3")
        attenuation = _validate_pairs(profile.get("BOTTOM_ATTENUATION_DB_PER_WAVELENGTH"), f"第 {index} 个剖面 BOTTOM_ATTENUATION_DB_PER_WAVELENGTH")
        if water[0][0] != 0.0 or bottom_speed[0][0] != 0.0 or density[0][0] != 0.0 or attenuation[0][0] != 0.0:
            raise ValueError("RAM 的所有深度剖面都必须从 0 m 开始。")
        if any(value <= 0.0 for _, value in water + bottom_speed + density):
            raise ValueError("声速和海底密度必须为正数。")
        if any(value < 0.0 for _, value in attenuation):
            raise ValueError("海底衰减不能为负数。")


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """读取 RAM 独立配置，并应用本次任务的递归覆盖值。"""
    path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_FILE
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if overrides:
        _deep_merge(config, overrides)
    executables = config.get("EXECUTABLES")
    output = config.get("OUTPUT")
    if not isinstance(executables, MutableMapping) or not isinstance(output, MutableMapping):
        raise ValueError("配置必须包含 EXECUTABLES 和 OUTPUT 对象。")
    executables["RAM"] = _resolve_path(executables.get("RAM", ""), path)
    output["DIRECTORY"] = _resolve_path(output.get("DIRECTORY", "jobs_ram"), path)
    config["_CONFIG_FILE"] = str(path)
    validate_config(config)
    return config
