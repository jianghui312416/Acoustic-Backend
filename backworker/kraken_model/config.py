# -*- coding: utf-8 -*-
"""Kraken 独立配置读取、合并和校验。"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = MODULE_DIR.parent.parent / "config" / "kraken_config.json"


def _deep_merge(target: MutableMapping[str, Any], override: Mapping[str, Any]) -> None:
    """把用户参数递归合并到配置副本中，不修改调用方传入的数据。"""
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), MutableMapping):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _resolve_path(value: Any, config_file: Path) -> str:
    """把相对路径按配置文件所在目录解析成绝对路径。"""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_file.parent / path
    return str(path.resolve())


def _finite_number(value: Any, name: str) -> float:
    """把配置值转换为有限浮点数，并在失败时给出明确的中文错误。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数值。") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值。")
    return number


def _positive_integer(value: Any, name: str) -> int:
    """读取正整数配置值。"""
    number = _finite_number(value, name)
    integer = int(round(number))
    if integer <= 0 or abs(number - integer) > 1.0e-9:
        raise ValueError(f"{name} 必须是正整数。")
    return integer


def _validate_profile(environment: Mapping[str, Any]) -> None:
    """校验水体声速剖面的深度顺序和边界覆盖范围。"""
    profile = environment.get("SOUND_SPEED_PROFILE")
    if not isinstance(profile, list) or len(profile) < 2:
        raise ValueError("ENVIRONMENT.SOUND_SPEED_PROFILE 至少需要两个 [深度, 声速] 点。")
    points = []
    for index, row in enumerate(profile, 1):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            raise ValueError(f"声速剖面第 {index} 行必须是 [深度_m, 声速_m/s]。")
        depth = _finite_number(row[0], f"声速剖面第 {index} 行深度")
        speed = _finite_number(row[1], f"声速剖面第 {index} 行声速")
        if depth < 0.0 or speed <= 0.0:
            raise ValueError(f"声速剖面第 {index} 行的深度不能为负，声速必须为正。")
        points.append((depth, speed))
    if any(points[index][0] <= points[index - 1][0] for index in range(1, len(points))):
        raise ValueError("声速剖面深度必须严格递增，不能重复或倒序。")
    water_depth = _finite_number(environment.get("WATER_DEPTH_M"), "ENVIRONMENT.WATER_DEPTH_M")
    if water_depth <= 0.0:
        raise ValueError("ENVIRONMENT.WATER_DEPTH_M 必须为正数。")
    if abs(points[0][0]) > 1.0e-6:
        raise ValueError("Kraken 水体声速剖面必须从 0 m 开始。")
    if abs(points[-1][0] - water_depth) > 1.0e-6:
        raise ValueError("声速剖面的最后一个深度必须等于 WATER_DEPTH_M。")


def validate_config(config: Mapping[str, Any]) -> None:
    """校验 Kraken 与 FIELD 运行所需的关键配置。"""
    environment = config.get("ENVIRONMENT")
    field = config.get("FIELD")
    runtime = config.get("RUNTIME")
    if not isinstance(environment, Mapping) or not isinstance(field, Mapping):
        raise ValueError("配置必须包含 ENVIRONMENT 和 FIELD 对象。")
    if not isinstance(runtime, Mapping):
        raise ValueError("配置必须包含 RUNTIME 对象。")
    range_dependent = config.get("RANGE_DEPENDENT")
    if not isinstance(range_dependent, Mapping):
        raise ValueError("配置必须包含 RANGE_DEPENDENT 对象。")
    profile_count = _positive_integer(config.get("N_2D_PROFILES"), "N_2D_PROFILES")
    if profile_count < 3:
        raise ValueError("N_2D_PROFILES 至少为 3。")
    tl_range_count = _positive_integer(config.get("TL_N_RANGE"), "TL_N_RANGE")
    tl_depth_count = _positive_integer(config.get("TL_N_RD"), "TL_N_RD")
    if tl_range_count < 2 or tl_depth_count < 2:
        raise ValueError("TL_N_RANGE 和 TL_N_RD 至少为 2。")
    if tl_range_count >= 4094 or tl_depth_count >= 201 or tl_range_count * tl_depth_count > 210000:
        raise ValueError("Kraken TL 网格超过限制：TL_N_RANGE<4094、TL_N_RD<201，且两者乘积不超过 210000。")
    for key in ("BATHYMETRY_TOLERANCE_M", "SOUND_SPEED_TOLERANCE_MPS"):
        if _finite_number(range_dependent.get(key), f"RANGE_DEPENDENT.{key}") < 0.0:
            raise ValueError(f"RANGE_DEPENDENT.{key} 不能为负数。")
    _validate_profile(environment)
    if _finite_number(environment.get("FREQUENCY_HZ"), "ENVIRONMENT.FREQUENCY_HZ") <= 0.0:
        raise ValueError("ENVIRONMENT.FREQUENCY_HZ 必须为正数。")
    if _finite_number(environment.get("MAX_RANGE_KM"), "ENVIRONMENT.MAX_RANGE_KM") <= 0.0:
        raise ValueError("ENVIRONMENT.MAX_RANGE_KM 必须为正数。")
    interpolation = str(environment.get("INTERPOLATION", "C")).upper()
    if interpolation not in {"C", "N", "S"}:
        raise ValueError("ENVIRONMENT.INTERPOLATION 只能是 C、N 或 S。")
    top_boundary = str(environment.get("TOP_BOUNDARY", "V")).upper()
    if top_boundary not in {"V", "R", "F", "S", "H", "T", "I"}:
        raise ValueError(
            "TOP_BOUNDARY 只支持 V、R、反射系数文件 F "
            "或 Kraken Twersky 冰面选项 S/H/T/I。"
        )
    if top_boundary == "F" and not str(environment.get("TOP_REFLECTION_FILE") or "").strip():
        raise ValueError("F 顶边界需要 ENVIRONMENT.TOP_REFLECTION_FILE。")
    if top_boundary in {"S", "H", "T", "I"}:
        twersky = environment.get("TWERSKY")
        if not isinstance(twersky, Mapping):
            raise ValueError("冰面选项需要 ENVIRONMENT.TWERSKY 参数。")
        for key in ("BUMDEN", "ETA_M", "XI_M"):
            if _finite_number(twersky.get(key), f"ENVIRONMENT.TWERSKY.{key}") <= 0.0:
                raise ValueError(f"ENVIRONMENT.TWERSKY.{key} 必须为正数。")
    attenuation_units = str(environment.get("ATTENUATION_UNITS", "W")).upper()
    if attenuation_units not in {"N", "F", "M", "W", "Q", "T"}:
        raise ValueError("ENVIRONMENT.ATTENUATION_UNITS 配置无效。")
    water_depth = float(environment["WATER_DEPTH_M"])
    for key in ("SOURCE_DEPTHS_M", "RECEIVER_DEPTHS_M"):
        values = field.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"FIELD.{key} 必须是非空数组。")
        for value in values:
            depth = _finite_number(value, f"FIELD.{key}")
            if depth < 0.0 or depth >= water_depth:
                raise ValueError(f"FIELD.{key} 中的深度必须位于 [0, {water_depth}) m。")
    ranges = field.get("RECEIVER_RANGES_KM")
    if not isinstance(ranges, Mapping):
        raise ValueError("FIELD.RECEIVER_RANGES_KM 必须是对象。")
    range_min = _finite_number(ranges.get("MIN"), "FIELD.RECEIVER_RANGES_KM.MIN")
    range_max = _finite_number(ranges.get("MAX"), "FIELD.RECEIVER_RANGES_KM.MAX")
    effective_range_count = _positive_integer(
        ranges.get("COUNT", tl_range_count),
        "FIELD.RECEIVER_RANGES_KM.COUNT",
    )
    if range_min < 0.0 or range_max <= range_min:
        raise ValueError("FIELD 接收距离必须满足 0 <= MIN < MAX。")
    if effective_range_count >= 4094:
        raise ValueError("FIELD 接收距离点数必须小于 4094。")
    receiver_count = len(field["RECEIVER_DEPTHS_M"])
    if receiver_count >= 201 or receiver_count * effective_range_count > 210000:
        raise ValueError("FIELD 网格超过限制：深度点少于 201，且深度点数×距离点数不超过 210000。")
    _positive_integer(runtime.get("TIMEOUT_SECONDS"), "RUNTIME.TIMEOUT_SECONDS")


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """读取独立 Kraken 配置，并可用请求参数覆盖其中任意字段。"""
    path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_FILE
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if overrides:
        _deep_merge(config, overrides)
    executables = config.get("EXECUTABLES")
    output = config.get("OUTPUT")
    if not isinstance(executables, MutableMapping) or not isinstance(output, MutableMapping):
        raise ValueError("配置必须包含 EXECUTABLES 和 OUTPUT 对象。")
    executables["KRAKEN"] = _resolve_path(executables.get("KRAKEN", ""), path)
    executables["FIELD"] = _resolve_path(executables.get("FIELD", ""), path)
    output["DIRECTORY"] = _resolve_path(output.get("DIRECTORY", "jobs_kraken"), path)
    environment = config.get("ENVIRONMENT")
    if isinstance(environment, MutableMapping):
        top_boundary = str(environment.get("TOP_BOUNDARY", "V")).upper()
        reflection_file = str(environment.get("TOP_REFLECTION_FILE") or "").strip()
        if top_boundary == "F" and reflection_file:
            environment["TOP_REFLECTION_FILE"] = _resolve_path(reflection_file, path)
    config["_CONFIG_FILE"] = str(path)
    validate_config(config)
    return config
